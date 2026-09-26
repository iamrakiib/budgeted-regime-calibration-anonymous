
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path
from dependency_paths import third_party_path

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR100
from torchvision.models import vit_b_16, ViT_B_16_Weights

from brc_core import (
    BUDGET_MULTIPLIERS,
    UNIFORM_BINS,
    aps_score_matrix_strict,
    aps_true_scores,
    assign_global,
    assign_mixed,
    calibrate_allocation,
    class_coverage_stats,
    classification_uncertainty,
    conformal_quantile,
    global_U,
    global_edges,
    group_stats,
    learn_patterns,
    mixed_edges,
    pattern_relative_U,
    precompute_local_costs,
    sample_random_legal_allocations,
    select_global_B,
    solve_budget_dp,
    standardize_three,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(third_party_path("class-conditional-conformal-main")))

from utils.conformal_utils import (
    clustered_conformal,
    get_APS_scores,
    get_APS_scores_all,
)

ALPHA = 0.10
REPORTING_SEEDS = [42, 999, 0, 123, 12345]
PCA_DIM = 64
MIN_EVAL_CELL = 40
MIN_CAL_LEAF = 50
MIN_STRUCTURE_SUPPORT = 100
RANDOM_DRAWS = 20
FEATURE_SHARD_SIZE = 1024


def set_all_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cache_dir_for(cache_root: Path, split: str):
    return cache_root / f"cifar100_vit_b16_imagenet1k_v1_{split}_shards_v2"


def load_complete_shards(shard_dir: Path):
    shards = sorted(shard_dir.glob("shard_*.npz"))
    feats, ys = [], []
    for p in shards:
        z = np.load(p, allow_pickle=False)
        feats.append(z["features"].astype(np.float32))
        ys.append(z["y"].astype(int))
    if not feats:
        return None, None
    return np.concatenate(feats), np.concatenate(ys)


@torch.no_grad()
def extract_or_resume_split(
    split: str,
    cache_root: Path,
    batch_size: int,
):
    """
    Extract ViT-B/16 penultimate features for CIFAR-100 with the official
    torchvision ImageNet preprocessing.

    Completed feature shards are written directly to the persistent cache so an interrupted
    disconnect does not erase finished extraction.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    train_flag = split == "train"
    ds = CIFAR100(
        root="./data/cifar100",
        train=train_flag,
        download=True,
        transform=transform,
    )

    shard_dir = cache_dir_for(cache_root, split)
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(shard_dir.glob("shard_*.npz"))
    done = 0
    for p in existing:
        z = np.load(p, allow_pickle=False)
        done += len(z["y"])

    if done > len(ds):
        raise RuntimeError(
            f"Cache {shard_dir} has {done} examples but dataset has {len(ds)}."
        )

    print(f"[ViT-B/16] {split} persistent cache: {shard_dir}")
    print(f"[ViT-B/16] {split} already cached: {done}/{len(ds)}")

    if done < len(ds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = vit_b_16(weights=weights).to(device).eval()

        feat_box = {}

        def hook_fn(module, inputs, output):
            feat_box["x"] = inputs[0].detach()

        handle = model.heads.register_forward_hook(hook_fn)

        subset = Subset(ds, list(range(done, len(ds))))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        buf_f, buf_y = [], []
        nbuf = 0
        shard_id = len(existing)

        def flush():
            nonlocal buf_f, buf_y, nbuf, shard_id
            if nbuf == 0:
                return
            f = np.concatenate(buf_f, axis=0)
            y = np.concatenate(buf_y, axis=0)
            p = shard_dir / f"shard_{shard_id:05d}.npz"
            np.savez_compressed(
                p,
                features=f.astype(np.float16),
                y=y.astype(np.int16),
            )
            print(f"Saved {split} shard {p.name}: {len(y)} examples")
            shard_id += 1
            buf_f, buf_y, nbuf = [], [], 0

        for bi, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            _ = model(x)
            f = feat_box["x"]

            buf_f.append(f.cpu().numpy().astype(np.float16))
            buf_y.append(y.numpy().astype(np.int16))
            nbuf += len(y)

            if nbuf >= FEATURE_SHARD_SIZE:
                flush()

            if bi % 25 == 0:
                print(f"[ViT-B/16] {split} extraction batch {bi}/{len(loader)}")

        flush()
        handle.remove()

    features, labels = load_complete_shards(shard_dir)
    if features is None or len(labels) != len(ds):
        raise RuntimeError(
            f"Incomplete {split} cache: "
            f"{0 if labels is None else len(labels)} / {len(ds)}"
        )

    manifest = {
        "split": split,
        "n": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "backbone": "ViT_B_16_IMAGENET1K_V1",
        "feature_shard_size": FEATURE_SHARD_SIZE,
    }
    (shard_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    return features, labels


def cifar_split(labels, seed):
    """
    Matches the frozen CIFAR-100 ResNet protocol:
      TRAIN 30,000
      VAL 5,000
      STRUCTURE 7,500
      CAL 7,500
      TEST = official CIFAR-100 test set (10,000)
    """
    idx = np.arange(len(labels))
    train, rest = train_test_split(
        idx,
        train_size=30000,
        random_state=seed,
        stratify=labels,
    )
    val, rest = train_test_split(
        rest,
        train_size=5000,
        random_state=seed + 1,
        stratify=labels[rest],
    )
    structure, cal = train_test_split(
        rest,
        train_size=7500,
        random_state=seed + 2,
        stratify=labels[rest],
    )
    return {
        "train": train,
        "val": val,
        "structure": structure,
        "cal": cal,
    }


class LinearHead(torch.nn.Module):
    def __init__(self, d, classes=100):
        super().__init__()
        self.fc = torch.nn.Linear(d, classes)

    def forward(self, x):
        return self.fc(x)


def train_head(Xtr, ytr, Xv, yv, seed):
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LinearHead(Xtr.shape[1], 100).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=1e-4,
    )
    lossfn = torch.nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(ytr.astype(np.int64)),
    )
    loader = DataLoader(
        ds,
        batch_size=512,
        shuffle=True,
        num_workers=0,
    )

    Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
    yv_t = torch.from_numpy(yv.astype(np.int64)).to(device)

    best_state = None
    best_acc = -1.0
    bad = 0

    for epoch in range(60):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = float((model(Xv_t).argmax(1) == yv_t).float().mean().item())

        print(
            f"CIFAR ViT head seed={seed} "
            f"epoch={epoch+1:02d} val_acc={acc:.4f}"
        )

        if acc > best_acc + 1e-5:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, device, best_acc


@torch.no_grad()
def head_probs(model, device, X, batch_size=2048):
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(
            X[start:start + batch_size].astype(np.float32)
        ).to(device)
        out.append(
            torch.softmax(model(xb), 1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return np.concatenate(out)


def build_bundle(
    train_features,
    train_labels,
    test_features,
    test_labels,
    seed,
):
    sp = cifar_split(train_labels, seed)

    Xtrain = l2_normalize(train_features)
    Xtest = l2_normalize(test_features)

    head, device, best_val_acc = train_head(
        Xtrain[sp["train"]],
        train_labels[sp["train"]],
        Xtrain[sp["val"]],
        train_labels[sp["val"]],
        seed,
    )

    ps = head_probs(head, device, Xtrain[sp["structure"]])
    pc = head_probs(head, device, Xtrain[sp["cal"]])
    pt = head_probs(head, device, Xtest)

    ys = train_labels[sp["structure"]]
    yc = train_labels[sp["cal"]]
    yt = test_labels

    # Internal representation summary for regime construction.
    pca = PCA(
        n_components=min(PCA_DIM, train_features.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    )
    phi_s = pca.fit_transform(
        train_features[sp["structure"]]
    ).astype(np.float32)
    phi_c = pca.transform(
        train_features[sp["cal"]]
    ).astype(np.float32)
    phi_t = pca.transform(
        test_features
    ).astype(np.float32)

    u1s, u2s = classification_uncertainty(ps)
    u1c, u2c = classification_uncertainty(pc)
    u1t, u2t = classification_uncertainty(pt)

    sig_s = np.concatenate([phi_s, np.stack([u1s, u2s], axis=1)], axis=1)
    sig_c = np.concatenate([phi_c, np.stack([u1c, u2c], axis=1)], axis=1)
    sig_t = np.concatenate([phi_t, np.stack([u1t, u2t], axis=1)], axis=1)

    sig_s, sig_c, sig_t, _ = standardize_three(sig_s, sig_c, sig_t)

    km, k_table = learn_patterns(
        sig_s,
        min_structure_support=MIN_STRUCTURE_SUPPORT,
        seed=seed,
    )
    K = int(km.n_clusters)

    pat_s = km.predict(sig_s).astype(int)
    pat_c = km.predict(sig_c).astype(int)
    pat_t = km.predict(sig_t).astype(int)

    U_s = pattern_relative_U(u1s, u2s, pat_s, u1s, u2s, pat_s, K)
    U_c = pattern_relative_U(u1s, u2s, pat_s, u1c, u2c, pat_c, K)
    U_t = pattern_relative_U(u1s, u2s, pat_s, u1t, u2t, pat_t, K)

    Ug_s = global_U(u1s, u2s, u1s, u2s)
    Ug_t = global_U(u1s, u2s, u1t, u2t)

    # Canonical paper score: deterministic nonrandomized APS with cumulative
    # probability strictly before the candidate class.  The same convention is
    # used by the matched Clustered-CP comparison below.
    score_s = aps_true_scores(ps, ys)
    score_c = aps_true_scores(pc, yc)
    score_test_all = aps_score_matrix_strict(pt)

    return {
        "dataset": "cifar100",
        "backbone": "vit_b_16_imagenet1k_v1_features_linear_head",
        "seed": int(seed),
        "K": K,
        "score_name": "Deterministic_StrictBeforeCandidate_APS",
        "score_structure": score_s,
        "score_cal": score_c,
        "score_test_all": score_test_all,
        "probs_structure": ps,
        "probs_cal": pc,
        "probs_test": pt,
        "y_structure": ys,
        "y_cal": yc,
        "y_test": yt,
        "pat_s": pat_s,
        "pat_c": pat_c,
        "pat_t": pat_t,
        "U_s": U_s,
        "U_c": U_c,
        "U_t": U_t,
        "Ug_s": Ug_s,
        "Ug_t": Ug_t,
        "k_selection": k_table,
        "min_eval_cell": MIN_EVAL_CELL,
        "min_cal_leaf": MIN_CAL_LEAF,
        "val_accuracy": float(best_val_acc),
        "test_accuracy": float(np.mean(pt.argmax(1) == yt)),
    }


def eval_groups(bundle):
    K = int(bundle["K"])
    pt_edges = mixed_edges(bundle["U_s"], bundle["pat_s"], K, [10] * K)
    eval_pt = assign_mixed(bundle["U_t"], bundle["pat_t"], pt_edges)

    gt_edges = global_edges(bundle["Ug_s"], 10)
    eval_gt = assign_global(bundle["Ug_t"], gt_edges)

    return eval_pt, eval_gt


def mask_from_q(score_test_all, q):
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 0:
        q = np.full(len(score_test_all), float(q), dtype=np.float32)
    return score_test_all <= q[:, None]


def evaluate_mask(method, mask, bundle, extra=None):
    eval_pt, eval_gt = eval_groups(bundle)
    y = bundle["y_test"]
    mask = np.asarray(mask, dtype=bool)

    covered = mask[np.arange(len(y)), y]
    sizes = mask.sum(axis=1)

    pst, pdf = group_stats(
        covered, eval_pt, bundle["min_eval_cell"]
    )
    gst, gdf = group_stats(
        covered, eval_gt, bundle["min_eval_cell"]
    )
    cst, cdf = class_coverage_stats(covered, y)

    row = {
        "dataset": bundle["dataset"],
        "backbone": bundle["backbone"],
        "score_name": bundle["score_name"],
        "seed": bundle["seed"],
        "method": method,
        "K": bundle["K"],
        "marginal_coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "empty_rate": float((sizes == 0).mean()),
        "worst_eval_pattern_U": pst["worst"],
        "spread_eval_pattern_U": pst["spread"],
        "max_gap_eval_pattern_U": pst["max_gap"],
        "weighted_gap_eval_pattern_U": pst["weighted_gap"],
        "worst_eval_global_U": gst["worst"],
        "worst_class_coverage": cst["worst_class_coverage"],
        "bottom5_class_coverage": cst["bottom5_class_coverage"],
        "bottom10_class_coverage": cst["bottom10_class_coverage"],
        "val_accuracy": bundle["val_accuracy"],
        "test_accuracy": bundle["test_accuracy"],
    }
    if extra:
        row.update(extra)

    details = []
    if len(pdf):
        d = pdf.copy()
        d["method"] = method
        d["family"] = "pattern_U10"
        details.append(d)
    if len(gdf):
        d = gdf.copy()
        d["method"] = method
        d["family"] = "global_U10"
        details.append(d)
    if len(cdf):
        d = cdf.copy()
        d["group"] = d.pop("class")
        d["method"] = method
        d["family"] = "class"
        details.append(d)

    det = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return row, det


def run_core(bundle):
    K = int(bundle["K"])
    rows, details, random_rows, support_rows = [], [], [], []

    local, _ = precompute_local_costs(
        bundle["score_structure"],
        bundle["U_s"],
        bundle["pat_s"],
        K,
        bundle["min_eval_cell"],
        bundle["seed"],
    )

    def add(method, qtest, complexity, allocation=None, extra=None):
        ex = {"complexity_cells": int(complexity)}
        if allocation is not None:
            ex["allocation"] = "-".join(map(str, allocation))
        if extra:
            ex.update(extra)

        mask = mask_from_q(bundle["score_test_all"], qtest)
        row, det = evaluate_mask(method, mask, bundle, ex)
        rows.append(row)
        if len(det):
            det["dataset"] = bundle["dataset"]
            det["backbone"] = bundle["backbone"]
            det["seed"] = bundle["seed"]
            details.append(det)

    qg = conformal_quantile(bundle["score_cal"])
    add("GlobalCP", qg, 1)

    for b in [1, *UNIFORM_BINS]:
        alloc = tuple([int(b)] * K)
        q, sup, active = calibrate_allocation(
            bundle["score_cal"],
            bundle["pat_c"],
            bundle["U_c"],
            bundle["pat_t"],
            bundle["U_t"],
            bundle["U_s"],
            bundle["pat_s"],
            K,
            alloc,
            bundle["min_cal_leaf"],
        )
        method = "PatternCP" if b == 1 else f"Uniform_q{b}"
        add(method, q, sum(alloc), alloc, {"active_cells": active})

        sup = sup.copy()
        sup["method"] = method
        sup["dataset"] = bundle["dataset"]
        sup["backbone"] = bundle["backbone"]
        sup["seed"] = bundle["seed"]
        support_rows.append(sup)

    selected_B, selected_table = select_global_B(local, K)
    alloc = tuple([selected_B] * K)
    q, sup, active = calibrate_allocation(
        bundle["score_cal"],
        bundle["pat_c"],
        bundle["U_c"],
        bundle["pat_t"],
        bundle["U_t"],
        bundle["U_s"],
        bundle["pat_s"],
        K,
        alloc,
        bundle["min_cal_leaf"],
    )
    add(
        "StructureSelected_GlobalB",
        q,
        sum(alloc),
        alloc,
        {
            "selected_global_B": selected_B,
            "active_cells": active,
        },
    )

    for mult in BUDGET_MULTIPLIERS:
        budget = mult * K
        state = solve_budget_dp(local, K, budget)
        alloc = tuple(state[2])

        q, sup, active = calibrate_allocation(
            bundle["score_cal"],
            bundle["pat_c"],
            bundle["U_c"],
            bundle["pat_t"],
            bundle["U_t"],
            bundle["U_s"],
            bundle["pat_s"],
            K,
            alloc,
            bundle["min_cal_leaf"],
        )
        add(
            f"BRC_{mult}K",
            q,
            budget,
            alloc,
            {
                "structure_objective_max_gap": state[0],
                "structure_objective_weighted_sum": state[1],
                "active_cells": active,
            },
        )

        sup = sup.copy()
        sup["method"] = f"BRC_{mult}K"
        sup["dataset"] = bundle["dataset"]
        sup["backbone"] = bundle["backbone"]
        sup["seed"] = bundle["seed"]
        support_rows.append(sup)

        draws = sample_random_legal_allocations(
            K,
            budget,
            RANDOM_DRAWS,
            bundle["seed"] + 1000 * mult,
        )
        per_draw = []

        for di, da in enumerate(draws):
            qd, _, active_d = calibrate_allocation(
                bundle["score_cal"],
                bundle["pat_c"],
                bundle["U_c"],
                bundle["pat_t"],
                bundle["U_t"],
                bundle["U_s"],
                bundle["pat_s"],
                K,
                da,
                bundle["min_cal_leaf"],
            )
            mask = mask_from_q(bundle["score_test_all"], qd)
            row, _ = evaluate_mask(
                f"Random_{mult}K_draw",
                mask,
                bundle,
                {
                    "complexity_cells": budget,
                    "allocation": "-".join(map(str, da)),
                    "active_cells": active_d,
                    "draw": di,
                },
            )
            per_draw.append(row)

        rdf = pd.DataFrame(per_draw)
        random_rows.append(rdf)

        mean_row = {
            "dataset": bundle["dataset"],
            "backbone": bundle["backbone"],
            "score_name": bundle["score_name"],
            "seed": bundle["seed"],
            "method": f"Random_{mult}K_mean",
            "K": bundle["K"],
            "complexity_cells": budget,
            "val_accuracy": bundle["val_accuracy"],
            "test_accuracy": bundle["test_accuracy"],
        }
        for c in [
            "marginal_coverage",
            "avg_set_size",
            "empty_rate",
            "worst_eval_pattern_U",
            "spread_eval_pattern_U",
            "max_gap_eval_pattern_U",
            "weighted_gap_eval_pattern_U",
            "worst_eval_global_U",
            "worst_class_coverage",
            "bottom5_class_coverage",
            "bottom10_class_coverage",
        ]:
            mean_row[c] = float(rdf[c].mean())
        rows.append(mean_row)

    return (
        pd.DataFrame(rows),
        details,
        random_rows,
        support_rows,
        selected_table,
    )


def run_clustered_cp(bundle):
    ps = bundle["probs_structure"]
    pc = bundle["probs_cal"]

    ys = bundle["y_structure"]
    yc = bundle["y_cal"]

    scores_s_all = get_APS_scores_all(
        ps, randomize=False, seed=bundle["seed"] + 101
    ).astype(np.float32)
    scores_c_all = get_APS_scores_all(
        pc, randomize=False, seed=bundle["seed"] + 102
    ).astype(np.float32)

    pool_scores = np.concatenate([scores_s_all, scores_c_all], axis=0)
    pool_labels = np.concatenate([ys, yc], axis=0)

    _, preds, class_metrics, _ = clustered_conformal(
        pool_scores,
        pool_labels,
        ALPHA,
        val_scores_all=bundle["score_test_all"],
        val_labels=bundle["y_test"],
        frac_clustering="auto",
        num_clusters="auto",
        split="random",
        exact_coverage=False,
        seed=bundle["seed"],
    )

    n = len(preds)
    C = bundle["score_test_all"].shape[1]
    mask = np.zeros((n, C), dtype=bool)
    for i, pred in enumerate(preds):
        mask[i, np.asarray(pred, dtype=int)] = True

    row, det = evaluate_mask(
        "ClusteredCP_Official_NonrandomizedAPS",
        mask,
        bundle,
        {
            "clusteredcp_total_pool_size": int(len(pool_labels)),
            "clusteredcp_source_split": "random_auto",
            "clusteredcp_num_unclustered_classes": class_metrics.get(
                "num_unclustered_classes", np.nan
            ),
        },
    )
    if len(det):
        det["dataset"] = bundle["dataset"]
        det["backbone"] = bundle["backbone"]
        det["seed"] = bundle["seed"]

    return row, det


def aggregate(df):
    id_cols = {
        "dataset",
        "backbone",
        "score_name",
        "seed",
        "method",
        "allocation",
        "clusteredcp_source_split",
    }
    numeric_cols = [
        c for c in df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for key, g in df.groupby(
        ["dataset", "backbone", "score_name", "method"],
        dropna=False,
        sort=False,
    ):
        row = dict(
            zip(
                ["dataset", "backbone", "score_name", "method"],
                key,
            )
        )
        row["n_seeds"] = int(g["seed"].nunique())

        for c in numeric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals):
                row[c + "_mean"] = float(vals.mean())
                row[c + "_std"] = (
                    float(vals.std(ddof=1))
                    if len(vals) > 1 else 0.0
                )
        rows.append(row)

    return pd.DataFrame(rows)


def save_partial(outdir, rows, details, randoms, supports):
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            outdir / "ALL_RESULTS_PARTIAL.csv",
            index=False,
        )
    if details:
        pd.concat(details, ignore_index=True).to_csv(
            outdir / "GROUP_DETAILS_PARTIAL.csv",
            index=False,
        )
    if randoms:
        pd.concat(randoms, ignore_index=True).to_csv(
            outdir / "RANDOM_DRAWS_PARTIAL.csv",
            index=False,
        )
    if supports:
        pd.concat(supports, ignore_index=True).to_csv(
            outdir / "CAL_SUPPORT_PARTIAL.csv",
            index=False,
        )


def existing_completed_seeds(outdir: Path):
    p = outdir / "ALL_RESULTS_PARTIAL.csv"
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    if "seed" not in df.columns:
        return set()
    completed = set()
    for seed, g in df.groupby("seed"):
        methods = set(g["method"].astype(str))
        required = {"BRC_2K", "BRC_3K", "BRC_4K", "Uniform_q2", "Uniform_q3", "Uniform_q4"}
        if required.issubset(methods):
            completed.add(int(seed))
    return completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=REPORTING_SEEDS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-clustered-cp", action="store_true")
    args = ap.parse_args()

    if sorted(map(int, args.seeds)) != sorted(REPORTING_SEEDS) or len(args.seeds) != len(REPORTING_SEEDS):
        raise RuntimeError(f"Paper reporting protocol uses exactly {REPORTING_SEEDS}; received {args.seeds}.")

    outdir = Path(args.out).resolve()
    cache_root = Path(args.cache_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "alpha": ALPHA,
        "dataset": "CIFAR-100",
        "backbone": "torchvision ViT-B/16 IMAGENET1K_V1 frozen features + trained linear head",
        "split": {
            "train": 30000,
            "val": 5000,
            "structure": 7500,
            "cal": 7500,
            "test": 10000,
        },
        "reporting_seeds": args.seeds,
                "score": (
            "deterministic nonrandomized strict-before-candidate APS for BRC; matched convention for Clustered CP"
        ),
        "core_suite": [
            "GlobalCP",
            "PatternCP",
            "Uniform_q2",
            "Uniform_q3",
            "Uniform_q4",
            "Uniform_q6",
            "StructureSelected_GlobalB",
            "Random_2K",
            "Random_3K",
            "Random_4K",
            "BRC_2K",
            "BRC_3K",
            "BRC_4K",
        ],
        "external_baseline": (
            "ClusteredCP_Official_NonrandomizedAPS unless --skip-clustered-cp"
        ),
        "drive_safety": (
            "ViT feature extraction is stored in persistent shards under --cache-dir. "
            "Results are checkpointed after every completed seed."
        ),
    }
    (outdir / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    train_features, train_labels = extract_or_resume_split(
        "train",
        cache_root,
        args.batch_size,
    )
    test_features, test_labels = extract_or_resume_split(
        "test",
        cache_root,
        args.batch_size,
    )

    # Load any partial results so rerunning after a disconnect does not erase them.
    all_rows, all_details, all_randoms, all_supports = [], [], [], []
    if (outdir / "ALL_RESULTS_PARTIAL.csv").exists():
        all_rows.append(pd.read_csv(outdir / "ALL_RESULTS_PARTIAL.csv"))
    if (outdir / "GROUP_DETAILS_PARTIAL.csv").exists():
        all_details.append(pd.read_csv(outdir / "GROUP_DETAILS_PARTIAL.csv"))
    if (outdir / "RANDOM_DRAWS_PARTIAL.csv").exists():
        all_randoms.append(pd.read_csv(outdir / "RANDOM_DRAWS_PARTIAL.csv"))
    if (outdir / "CAL_SUPPORT_PARTIAL.csv").exists():
        all_supports.append(pd.read_csv(outdir / "CAL_SUPPORT_PARTIAL.csv"))

    completed = existing_completed_seeds(outdir)
    if completed:
        print("Already completed seeds:", sorted(completed))

    for seed in args.seeds:
        if int(seed) in completed:
            print(f"Skipping completed seed={seed}")
            continue

        print("\n" + "=" * 120)
        print(f"CIFAR-100 ViT BRC V2 | seed={seed}")
        print("=" * 120)

        bundle = build_bundle(
            train_features,
            train_labels,
            test_features,
            test_labels,
            int(seed),
        )

        core, details, randoms, supports, selected_B = run_core(bundle)
        all_rows.append(core)
        all_details.extend(details)
        all_randoms.extend(randoms)
        all_supports.extend(supports)

        bundle["k_selection"].to_csv(
            outdir / f"seed{seed}_K_SELECTION.csv",
            index=False,
        )
        selected_B.to_csv(
            outdir / f"seed{seed}_SELECTED_GLOBAL_B.csv",
            index=False,
        )

        if not args.skip_clustered_cp:
            ext_row, ext_det = run_clustered_cp(bundle)
            all_rows.append(pd.DataFrame([ext_row]))
            if len(ext_det):
                all_details.append(ext_det)

        save_partial(
            outdir,
            all_rows,
            all_details,
            all_randoms,
            all_supports,
        )
        gc.collect()

    result = pd.concat(all_rows, ignore_index=True)
    result = result.drop_duplicates(
        subset=["dataset", "backbone", "seed", "method"],
        keep="last",
    )
    result.to_csv(outdir / "ALL_RESULTS.csv", index=False)

    agg = aggregate(result)
    agg.to_csv(outdir / "AGGREGATE_MEAN_STD.csv", index=False)

    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(
            outdir / "GROUP_DETAILS.csv",
            index=False,
        )
    if all_randoms:
        pd.concat(all_randoms, ignore_index=True).to_csv(
            outdir / "RANDOM_DRAWS.csv",
            index=False,
        )
    if all_supports:
        pd.concat(all_supports, ignore_index=True).to_csv(
            outdir / "CAL_SUPPORT.csv",
            index=False,
        )

    print("\nAGGREGATE SUMMARY")
    cols = [
        c for c in [
            "dataset",
            "backbone",
            "method",
            "marginal_coverage_mean",
            "worst_eval_pattern_U_mean",
            "avg_set_size_mean",
            "empty_rate_mean",
            "K_mean",
            "test_accuracy_mean",
        ]
        if c in agg.columns
    ]
    print(agg[cols].to_string(index=False))

    zip_path = shutil.make_archive(
        str(outdir),
        "zip",
        root_dir=outdir,
    )
    print("\nDONE")
    print("Persistent result ZIP:", zip_path)


if __name__ == "__main__":
    main()
