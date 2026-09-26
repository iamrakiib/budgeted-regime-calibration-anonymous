
#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)

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
TARGET = 0.90
REPORTING_SEEDS = [42, 999, 0, 123, 12345]
RANDOM_DRAWS = 20
PCA_DIM = 64
MIN_EVAL_CELL = 20
MIN_CAL_LEAF = 20
MIN_STRUCTURE_SUPPORT = 100
EXTRACT_SHARD_SIZE = 1024


def set_all_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(backbone: str):
    if backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        head_module = model.fc
        display = "ResNet50_IMAGENET1K_V2"
    elif backbone == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        head_module = model.heads
        display = "ViT_B_16_IMAGENET1K_V1"
    else:
        raise ValueError(backbone)
    return model, weights, head_module, display


def cache_shard_dir(cache_root: Path, backbone: str):
    return cache_root / f"imagenet1k_val_{backbone}_shards_v2"


def completed_cache_count(shard_dir: Path):
    shards = sorted(shard_dir.glob("shard_*.npz"))
    total = 0
    for p in shards:
        z = np.load(p, allow_pickle=False)
        total += len(z["y"])
    return total, shards


@torch.no_grad()
def extract_or_resume_imagenet(
    imagenet_root: str,
    cache_root: Path,
    backbone: str,
    batch_size: int = 128,
):
    """
    Extract features/probabilities from pretrained torchvision ImageNet models.

    IMPORTANT: each ~1024-image shard is written directly to CACHE_ROOT.
    If a run disconnects, completed shards remain in the persistent cache and the next
    run resumes after the last complete shard.
    """
    model, weights, head_module, display = make_model(backbone)
    transform = weights.transforms()

    ds = ImageFolder(imagenet_root, transform=transform)
    if len(ds) < 49000 or len(ds.classes) != 1000:
        raise RuntimeError(
            "Expected ImageNet-1K validation in ImageFolder layout with "
            f"~50,000 images and 1,000 class folders. Found n={len(ds)}, "
            f"classes={len(ds.classes)} at {imagenet_root}"
        )

    shard_dir = cache_shard_dir(cache_root, backbone)
    shard_dir.mkdir(parents=True, exist_ok=True)

    done, shards = completed_cache_count(shard_dir)
    if done > len(ds):
        raise RuntimeError("Cache has more examples than the dataset. Delete the cache directory.")

    print(f"[{display}] persistent cache: {shard_dir}")
    print(f"[{display}] already cached: {done}/{len(ds)} images")

    if done < len(ds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        feat_box = {}

        def hook_fn(module, inputs, output):
            feat_box["x"] = inputs[0].detach()

        handle = head_module.register_forward_hook(hook_fn)

        remaining = Subset(ds, list(range(done, len(ds))))
        loader = DataLoader(
            remaining,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        buf_feat, buf_prob, buf_y = [], [], []
        shard_id = len(shards)
        n_buffer = 0

        def flush():
            nonlocal shard_id, n_buffer, buf_feat, buf_prob, buf_y
            if not n_buffer:
                return
            f = np.concatenate(buf_feat, axis=0)
            p = np.concatenate(buf_prob, axis=0)
            y = np.concatenate(buf_y, axis=0)
            path = shard_dir / f"shard_{shard_id:05d}.npz"
            np.savez_compressed(
                path,
                features=f.astype(np.float16),
                probs=p.astype(np.float16),
                y=y.astype(np.int16),
            )
            print(f"Saved persistent shard {path.name}: {len(y)} images")
            shard_id += 1
            n_buffer = 0
            buf_feat, buf_prob, buf_y = [], [], []

        for bi, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            feat = feat_box["x"]
            probs = torch.softmax(logits, dim=1)

            buf_feat.append(feat.cpu().numpy().astype(np.float16))
            buf_prob.append(probs.cpu().numpy().astype(np.float16))
            buf_y.append(y.numpy().astype(np.int16))
            n_buffer += len(y)

            if n_buffer >= EXTRACT_SHARD_SIZE:
                flush()

            if bi % 20 == 0:
                print(f"[{display}] extraction batch {bi}/{len(loader)}")

        flush()
        handle.remove()

    # Load the persistent shards.
    _, shards = completed_cache_count(shard_dir)
    feats, probs, labels = [], [], []
    for pth in shards:
        z = np.load(pth, allow_pickle=False)
        feats.append(z["features"].astype(np.float32))
        probs.append(z["probs"].astype(np.float32))
        labels.append(z["y"].astype(int))

    features = np.concatenate(feats, axis=0)
    probs = np.concatenate(probs, axis=0)
    y = np.concatenate(labels, axis=0)

    if len(y) != len(ds):
        raise RuntimeError(
            f"Cache incomplete after extraction: {len(y)} / {len(ds)}"
        )

    top1 = float(np.mean(probs.argmax(1) == y))
    print(f"[{display}] ImageNet class-index sanity top1={top1:.4f}")
    if top1 < 0.50:
        raise RuntimeError(
            "Top-1 < 50%. This usually means the ImageFolder class ordering "
            "does not match torchvision ImageNet class indices."
        )

    manifest = {
        "backbone": backbone,
        "display": display,
        "n": int(len(y)),
        "num_classes": int(probs.shape[1]),
        "feature_dim": int(features.shape[1]),
        "top1": top1,
        "shard_size": EXTRACT_SHARD_SIZE,
    }
    (shard_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    return features, probs, y, display, top1


def imagenet_split(y, seed: int):
    """
    No predictor training is performed here. The pretrained ImageNet model is fixed.
    The 50k ImageNet validation set is split into:
      STRUCTURE 30% / CAL 30% / TEST 40%
    stratified by class.
    """
    idx = np.arange(len(y))
    structure, rest = train_test_split(
        idx,
        train_size=0.30,
        random_state=seed,
        stratify=y,
    )
    cal, test = train_test_split(
        rest,
        train_size=3/7,
        random_state=seed + 1,
        stratify=y[rest],
    )
    return {
        "structure": structure,
        "cal": cal,
        "test": test,
    }


def build_bundle(features, probs, y, backbone, display, seed):
    set_all_seeds(seed)
    sp = imagenet_split(y, seed)

    fs = features[sp["structure"]]
    fc = features[sp["cal"]]
    ft = features[sp["test"]]

    ps = probs[sp["structure"]]
    pc = probs[sp["cal"]]
    pt = probs[sp["test"]]

    ys = y[sp["structure"]]
    yc = y[sp["cal"]]
    yt = y[sp["test"]]

    # Model-internal regime representation, but allocation remains the paper's core.
    pca = PCA(
        n_components=min(PCA_DIM, fs.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    )
    phi_s = pca.fit_transform(fs).astype(np.float32)
    phi_c = pca.transform(fc).astype(np.float32)
    phi_t = pca.transform(ft).astype(np.float32)

    u1s, u2s = classification_uncertainty(ps)
    u1c, u2c = classification_uncertainty(pc)
    u1t, u2t = classification_uncertainty(pt)

    sig_s = np.concatenate([phi_s, np.stack([u1s, u2s], 1)], axis=1)
    sig_c = np.concatenate([phi_c, np.stack([u1c, u2c], 1)], axis=1)
    sig_t = np.concatenate([phi_t, np.stack([u1t, u2t], 1)], axis=1)

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
    # probability strictly before the candidate class.
    score_s = aps_true_scores(ps, ys)
    score_c = aps_true_scores(pc, yc)
    score_t_all = aps_score_matrix_strict(pt)

    return {
        "dataset": "imagenet1k",
        "backbone": backbone,
        "backbone_display": display,
        "seed": int(seed),
        "K": K,
        "score_name": "Deterministic_StrictBeforeCandidate_APS",
        "score_structure": score_s,
        "score_cal": score_c,
        "score_test_all": score_t_all,
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
        "min_cal_leaf": MIN_CAL_LEAF,
        "min_eval_cell": MIN_EVAL_CELL,
        "split_sizes": {k: int(len(v)) for k, v in sp.items()},
        "test_accuracy": float(np.mean(pt.argmax(1) == yt)),
    }


def eval_groups(bundle):
    K = int(bundle["K"])
    pt_edges = mixed_edges(bundle["U_s"], bundle["pat_s"], K, [10] * K)
    eval_pt = assign_mixed(bundle["U_t"], bundle["pat_t"], pt_edges)
    gt_edges = global_edges(bundle["Ug_s"], 10)
    eval_gt = assign_global(bundle["Ug_t"], gt_edges)
    return eval_pt, eval_gt


def evaluate_mask(method, mask, bundle, extra=None):
    eval_pt, eval_gt = eval_groups(bundle)
    y = bundle["y_test"]
    mask = np.asarray(mask, dtype=bool)

    covered = mask[np.arange(len(y)), y]
    sizes = mask.sum(axis=1)

    pst, pdf = group_stats(covered, eval_pt, bundle["min_eval_cell"])
    gst, gdf = group_stats(covered, eval_gt, bundle["min_eval_cell"])
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


def mask_from_q(score_test_all, q):
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 0:
        q = np.full(len(score_test_all), float(q), dtype=np.float32)
    return np.asarray(score_test_all) <= q[:, None]


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

    # Global CP.
    qg = conformal_quantile(bundle["score_cal"])
    add("GlobalCP", qg, 1)

    # Pattern and uniform refinements.
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
        bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
        bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
        K, alloc, bundle["min_cal_leaf"]
    )
    add(
        "StructureSelected_GlobalB",
        q,
        sum(alloc),
        alloc,
        {"selected_global_B": selected_B, "active_cells": active},
    )

    for mult in BUDGET_MULTIPLIERS:
        budget = mult * K
        state = solve_budget_dp(local, K, budget)
        alloc = tuple(state[2])

        q, sup, active = calibrate_allocation(
            bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
            bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
            K, alloc, bundle["min_cal_leaf"]
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
            K, budget, RANDOM_DRAWS, bundle["seed"] + 1000 * mult
        )
        per_draw = []
        for di, da in enumerate(draws):
            qd, _, active_d = calibrate_allocation(
                bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
                bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
                K, da, bundle["min_cal_leaf"]
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
            "test_accuracy": bundle["test_accuracy"],
        }
        for c in [
            "marginal_coverage", "avg_set_size", "empty_rate",
            "worst_eval_pattern_U", "spread_eval_pattern_U",
            "max_gap_eval_pattern_U", "weighted_gap_eval_pattern_U",
            "worst_eval_global_U", "worst_class_coverage",
            "bottom5_class_coverage", "bottom10_class_coverage",
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


def run_clustered_cp_official_aps(bundle):
    """
    Clustered CP is given the same official nonrandomized APS score definition
    used by BRC in this ImageNet package.

    It receives the same total STRUCTURE+CAL pool. Its own random class-clustering
    / proper-calibration split is preserved from the official source.
    """
    ps = bundle["probs_structure"]
    pc = bundle["probs_cal"]
    pt = bundle["probs_test"]

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

    qhats, preds, class_metrics, set_metrics = clustered_conformal(
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
        "dataset", "backbone", "score_name", "seed", "method",
        "allocation", "clusteredcp_source_split"
    }
    numeric_cols = [
        c for c in df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for key, g in df.groupby(
        ["dataset", "backbone", "score_name", "method"],
        dropna=False, sort=False
    ):
        row = dict(zip(["dataset", "backbone", "score_name", "method"], key))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in numeric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals):
                row[c + "_mean"] = float(vals.mean())
                row[c + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_partial(outdir, rows, details, randoms, supports):
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            outdir / "ALL_RESULTS_PARTIAL.csv", index=False
        )
    if details:
        pd.concat(details, ignore_index=True).to_csv(
            outdir / "GROUP_DETAILS_PARTIAL.csv", index=False
        )
    if randoms:
        pd.concat(randoms, ignore_index=True).to_csv(
            outdir / "RANDOM_DRAWS_PARTIAL.csv", index=False
        )
    if supports:
        pd.concat(supports, ignore_index=True).to_csv(
            outdir / "CAL_SUPPORT_PARTIAL.csv", index=False
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument(
        "--backbones",
        nargs="+",
        choices=["resnet50", "vit_b_16"],
        default=["resnet50", "vit_b_16"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=REPORTING_SEEDS)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--skip-clustered-cp",
        action="store_true",
        help="Run only the core BRC suite if Clustered CP is too slow.",
    )
    args = ap.parse_args()

    if sorted(map(int, args.seeds)) != sorted(REPORTING_SEEDS) or len(args.seeds) != len(REPORTING_SEEDS):
        raise RuntimeError(f"Paper reporting protocol uses exactly {REPORTING_SEEDS}; received {args.seeds}.")

    outdir = Path(args.out).resolve()
    cache_root = Path(args.cache_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "alpha": ALPHA,
        "dataset": "ImageNet-1K validation",
        "predictor_training": "none; fixed torchvision pretrained model",
        "split": "STRUCTURE 30% / CAL 30% / TEST 40%, stratified",
        "reporting_seeds": args.seeds,
                "score": (
            "Deterministic nonrandomized strict-before-candidate APS implemented in brc_core.py"
        ),
        "core_suite": [
            "GlobalCP", "PatternCP", "Uniform_q2", "Uniform_q3",
            "Uniform_q4", "Uniform_q6", "StructureSelected_GlobalB",
            "Random_2K", "Random_3K", "Random_4K",
            "BRC_2K", "BRC_3K", "BRC_4K",
        ],
        "external_baseline": (
            "ClusteredCP_Official_NonrandomizedAPS unless --skip-clustered-cp"
        ),
        "drive_safety": (
            "Feature/probability extraction is saved in ~1024-image shards "
            "directly under --cache-dir, and result CSVs are checkpointed after every seed."
        ),
    }
    (outdir / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))

    all_rows, all_details, all_randoms, all_supports = [], [], [], []

    for backbone in args.backbones:
        features, probs, y, display, full_top1 = extract_or_resume_imagenet(
            args.imagenet_root,
            cache_root,
            backbone,
            batch_size=args.batch_size,
        )

        for seed in args.seeds:
            print("\n" + "=" * 120)
            print(f"IMAGENET BRC V2 | backbone={backbone} | seed={seed}")
            print("=" * 120)

            bundle = build_bundle(
                features, probs, y, backbone, display, int(seed)
            )

            core, details, randoms, supports, selected_B = run_core(bundle)
            all_rows.append(core)
            all_details.extend(details)
            all_randoms.extend(randoms)
            all_supports.extend(supports)

            bundle["k_selection"].to_csv(
                outdir / f"{backbone}_seed{seed}_K_SELECTION.csv", index=False
            )
            selected_B.to_csv(
                outdir / f"{backbone}_seed{seed}_SELECTED_GLOBAL_B.csv", index=False
            )

            if not args.skip_clustered_cp:
                ext_row, ext_det = run_clustered_cp_official_aps(bundle)
                all_rows.append(pd.DataFrame([ext_row]))
                if len(ext_det):
                    all_details.append(ext_det)

            save_partial(
                outdir, all_rows, all_details, all_randoms, all_supports
            )
            gc.collect()

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(outdir / "ALL_RESULTS.csv", index=False)

    agg = aggregate(result)
    agg.to_csv(outdir / "AGGREGATE_MEAN_STD.csv", index=False)

    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(
            outdir / "GROUP_DETAILS.csv", index=False
        )
    if all_randoms:
        pd.concat(all_randoms, ignore_index=True).to_csv(
            outdir / "RANDOM_DRAWS.csv", index=False
        )
    if all_supports:
        pd.concat(all_supports, ignore_index=True).to_csv(
            outdir / "CAL_SUPPORT.csv", index=False
        )

    print("\nAGGREGATE SUMMARY")
    cols = [
        c for c in [
            "dataset", "backbone", "method",
            "marginal_coverage_mean", "worst_eval_pattern_U_mean",
            "avg_set_size_mean", "empty_rate_mean", "K_mean"
        ] if c in agg.columns
    ]
    print(agg[cols].to_string(index=False))

    zip_path = shutil.make_archive(
        str(outdir), "zip", root_dir=outdir
    )
    print("\nDONE")
    print("Persistent result ZIP:", zip_path)


if __name__ == "__main__":
    main()
