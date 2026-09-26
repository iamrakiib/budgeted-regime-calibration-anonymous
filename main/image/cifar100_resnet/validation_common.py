from __future__ import annotations

import copy
import gc
import json
import random
import sys
from pathlib import Path
from dependency_paths import third_party_path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR100
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)

import brc_core as bc
from data_models import maybe_pca_phi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(third_party_path("class-conditional-conformal-main")))
from utils.conformal_utils import (  # noqa: E402
    clustered_conformal,
    get_APS_scores,
    get_APS_scores_all,
)

ALPHA = 0.10
TARGET = 0.90
REPORTING_SEEDS = [42, 999, 0, 123, 12345]
PCA_DIM = 64
MIN_EVAL_CELL_CIFAR = 40
MIN_EVAL_CELL_IMAGENET = 20
MIN_EVAL_CELL_TABULAR = 30
MIN_CAL_LEAF = 50
MIN_STRUCTURE_SUPPORT = 100
FEATURE_SHARD_SIZE = 1024


def set_all_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refuse_reserved(seeds: Iterable[int]):
    """Compatibility name: enforce the paper's five reporting seeds."""
    got = [int(s) for s in seeds]
    if sorted(got) != sorted(REPORTING_SEEDS) or len(got) != len(REPORTING_SEEDS):
        raise RuntimeError(
            f"Paper reporting protocol uses exactly {REPORTING_SEEDS}; received {got}."
        )


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cifar_split(labels, seed: int):
    """Frozen image protocol: TRAIN 30k / VAL 5k / STRUCTURE 7.5k / CAL 7.5k / official TEST 10k."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    train, rest = train_test_split(
        idx, train_size=30000, random_state=seed, stratify=labels
    )
    val, rest = train_test_split(
        rest, train_size=5000, random_state=seed + 1, stratify=labels[rest]
    )
    structure, cal = train_test_split(
        rest, train_size=7500, random_state=seed + 2, stratify=labels[rest]
    )
    return {"train": train, "val": val, "structure": structure, "cal": cal}


def _cifar_cache_dir(cache_root: Path, backbone: str, split: str) -> Path:
    if backbone == "resnet50":
        # This exactly reuses the cache produced by the EnergyAPS ResNet run.
        return cache_root / f"cifar100_resnet50_imagenet1k_v2_{split}_shards_energy_v1"
    if backbone == "vit_b_16":
        # This exactly reuses the corrected CIFAR-ViT BRC cache.
        return cache_root / f"cifar100_vit_b16_imagenet1k_v1_{split}_shards_v2"
    raise ValueError(backbone)


def _make_cifar_feature_model(backbone: str):
    if backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        head = model.fc
        display = "ResNet50_IMAGENET1K_V2"
    elif backbone == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        head = model.heads
        display = "ViT_B_16_IMAGENET1K_V1"
    else:
        raise ValueError(backbone)
    return model, weights, head, display


def _load_shards(shard_dir: Path):
    feats, ys = [], []
    for p in sorted(shard_dir.glob("shard_*.npz")):
        z = np.load(p, allow_pickle=False)
        feats.append(z["features"].astype(np.float32))
        ys.append(z["y"].astype(int))
    if not feats:
        return None, None
    return np.concatenate(feats), np.concatenate(ys)


@torch.no_grad()
def extract_or_resume_cifar_features(
    backbone: str,
    split: str,
    cache_root: str | Path,
    batch_size: int = 64,
):
    cache_root = Path(cache_root)
    model, weights, head, display = _make_cifar_feature_model(backbone)
    transform = weights.transforms()
    ds = CIFAR100(
        root="./data/cifar100",
        train=(split == "train"),
        download=True,
        transform=transform,
    )
    shard_dir = _cifar_cache_dir(cache_root, backbone, split)
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(shard_dir.glob("shard_*.npz"))
    done = sum(len(np.load(p, allow_pickle=False)["y"]) for p in existing)
    print(f"[{display}] {split}: cached {done}/{len(ds)} at {shard_dir}")
    if done > len(ds):
        raise RuntimeError(f"Cache contains {done} rows for a {len(ds)}-row dataset.")

    if done < len(ds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        box = {}

        def hook(_module, inputs, _output):
            box["x"] = inputs[0].detach()

        handle = head.register_forward_hook(hook)
        subset = Subset(ds, list(range(done, len(ds))))
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        bf, by, nbuf = [], [], 0
        shard_id = len(existing)

        def flush():
            nonlocal bf, by, nbuf, shard_id
            if nbuf == 0:
                return
            f = np.concatenate(bf)
            y = np.concatenate(by)
            path = shard_dir / f"shard_{shard_id:05d}.npz"
            np.savez_compressed(path, features=f.astype(np.float16), y=y.astype(np.int16))
            print(f"Saved {path.name}: {len(y)}")
            shard_id += 1
            bf, by, nbuf = [], [], 0

        for bi, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            _ = model(x)
            bf.append(box["x"].cpu().numpy().astype(np.float16))
            by.append(y.numpy().astype(np.int16))
            nbuf += len(y)
            if nbuf >= FEATURE_SHARD_SIZE:
                flush()
            if bi % 25 == 0:
                print(f"[{display}] {split} extraction batch {bi}/{len(loader)}")
        flush()
        handle.remove()

    X, y = _load_shards(shard_dir)
    if X is None or len(y) != len(ds):
        raise RuntimeError(f"Incomplete cache {shard_dir}: {0 if y is None else len(y)}/{len(ds)}")
    return X, y


class LinearHead(torch.nn.Module):
    def __init__(self, d: int, classes: int = 100):
        super().__init__()
        self.fc = torch.nn.Linear(d, classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_head(Xtr, ytr, Xv, yv, seed: int):
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearHead(Xtr.shape[1], 100).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    lossfn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr.astype(np.float32)),
            torch.from_numpy(np.asarray(ytr, dtype=np.int64)),
        ),
        batch_size=512, shuffle=True, num_workers=0,
    )
    Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
    yv_t = torch.from_numpy(np.asarray(yv, dtype=np.int64)).to(device)

    best_state, best_acc, bad = None, -1.0, 0
    for epoch in range(60):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xv_t).argmax(1) == yv_t).float().mean().item())
        print(f"head seed={seed} epoch={epoch+1:02d} val_acc={acc:.4f}")
        if acc > best_acc + 1e-5:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state is None:
        raise RuntimeError("Linear head produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, device, best_acc


@torch.no_grad()
def head_probs(model, device, X, batch_size: int = 2048):
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
        out.append(torch.softmax(model(xb), 1).cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def build_cifar_prediction_bundle(
    backbone: str,
    cache_root: str | Path,
    seed: int,
    batch_size: int = 64,
):
    Xall, yall = extract_or_resume_cifar_features(backbone, "train", cache_root, batch_size)
    Xt, yt = extract_or_resume_cifar_features(backbone, "test", cache_root, batch_size)
    sp = cifar_split(yall, seed)
    Xn = l2_normalize(Xall)
    Xtn = l2_normalize(Xt)
    head, device, val_acc = train_linear_head(
        Xn[sp["train"]], yall[sp["train"]], Xn[sp["val"]], yall[sp["val"]], seed
    )
    ps = head_probs(head, device, Xn[sp["structure"]])
    pc = head_probs(head, device, Xn[sp["cal"]])
    pt = head_probs(head, device, Xtn)
    if backbone == "resnet50":
        bname = "resnet50_imagenet1k_v2_features_linear_head"
    else:
        bname = "vit_b_16_imagenet1k_v1_features_linear_head"
    return {
        "dataset": "cifar100",
        "backbone": bname,
        "backbone_accuracy": float((pt.argmax(1) == yt).mean()),
        "val_accuracy": float(val_acc),
        "probs_s": ps, "probs_c": pc, "probs_t": pt,
        "phi_s": Xall[sp["structure"]].astype(np.float32),
        "phi_c": Xall[sp["cal"]].astype(np.float32),
        "phi_t": Xt.astype(np.float32),
        "y_s": yall[sp["structure"]], "y_c": yall[sp["cal"]], "y_t": yt,
        "split_sizes": {"train": 30000, "val": 5000, "structure": 7500, "cal": 7500, "test": 10000},
    }


def load_imagenet_prediction_bundle(cache_root: str | Path, backbone: str, seed: int):
    cache_root = Path(cache_root)
    shard_dir = cache_root / f"imagenet1k_val_{backbone}_shards_v2"
    feats, probs, ys = [], [], []
    for p in sorted(shard_dir.glob("shard_*.npz")):
        z = np.load(p, allow_pickle=False)
        feats.append(z["features"].astype(np.float32))
        probs.append(z["probs"].astype(np.float32))
        ys.append(z["y"].astype(int))
    if not feats:
        raise FileNotFoundError(
            f"No ImageNet v2 cache found at {shard_dir}. Run the earlier ImageNet BRC extraction first."
        )
    X, P, y = np.concatenate(feats), np.concatenate(probs), np.concatenate(ys)
    if len(y) != 50000:
        raise RuntimeError(f"Expected 50,000 ImageNet rows, found {len(y)}")
    sanity = float((P.argmax(1) == y).mean())
    print(f"ImageNet {backbone} full-cache top1 sanity={sanity:.4f}")
    if sanity < 0.50:
        raise RuntimeError("ImageNet class-index sanity failed.")
    idx = np.arange(len(y))
    st, rest = train_test_split(idx, train_size=0.30, random_state=seed, stratify=y)
    ca, te = train_test_split(rest, train_size=3/7, random_state=seed + 1, stratify=y[rest])
    return {
        "dataset": "imagenet1k",
        "backbone": backbone,
        "backbone_accuracy": float((P[te].argmax(1) == y[te]).mean()),
        "probs_s": P[st], "probs_c": P[ca], "probs_t": P[te],
        "phi_s": X[st], "phi_c": X[ca], "phi_t": X[te],
        "y_s": y[st], "y_c": y[ca], "y_t": y[te],
        "split_sizes": {"train": 0, "val": 0, "structure": len(st), "cal": len(ca), "test": len(te)},
    }


def _fixed_k_fit(sig_s, sig_c, sig_t, K: int, seed: int):
    ari = bc.kmeans_cv_ari(sig_s, int(K), int(seed))
    km = KMeans(n_clusters=int(K), n_init=20, random_state=int(seed)).fit(sig_s)
    counts = np.bincount(km.labels_, minlength=int(K))
    table = pd.DataFrame([{
        "K": int(K),
        "CV_ARI": float(ari),
        "min_structure_count": int(counts.min()),
        "support_ok": bool(counts.min() >= MIN_STRUCTURE_SUPPORT),
        "forced_for_sensitivity": True,
    }])
    return km, table


def build_regime_bundle(pred_bundle: dict, seed: int, fixed_k: Optional[int] = None):
    probs_s, probs_c, probs_t = pred_bundle["probs_s"], pred_bundle["probs_c"], pred_bundle["probs_t"]
    phi_s, phi_c, phi_t = maybe_pca_phi(
        pred_bundle["phi_s"], pred_bundle["phi_c"], pred_bundle["phi_t"], seed, max_dim=PCA_DIM
    )
    u1s, u2s = bc.classification_uncertainty(probs_s)
    u1c, u2c = bc.classification_uncertainty(probs_c)
    u1t, u2t = bc.classification_uncertainty(probs_t)
    sig_s = np.concatenate([phi_s, np.stack([u1s, u2s], 1)], 1)
    sig_c = np.concatenate([phi_c, np.stack([u1c, u2c], 1)], 1)
    sig_t = np.concatenate([phi_t, np.stack([u1t, u2t], 1)], 1)
    sig_s, sig_c, sig_t, _ = bc.standardize_three(sig_s, sig_c, sig_t)

    if fixed_k is None:
        km, ktable = bc.learn_patterns(sig_s, MIN_STRUCTURE_SUPPORT, seed)
    else:
        km, ktable = _fixed_k_fit(sig_s, sig_c, sig_t, int(fixed_k), seed)
    K = int(km.n_clusters)
    pat_s = km.predict(sig_s).astype(int)
    pat_c = km.predict(sig_c).astype(int)
    pat_t = km.predict(sig_t).astype(int)
    U_s = bc.pattern_relative_U(u1s, u2s, pat_s, u1s, u2s, pat_s, K)
    U_c = bc.pattern_relative_U(u1s, u2s, pat_s, u1c, u2c, pat_c, K)
    U_t = bc.pattern_relative_U(u1s, u2s, pat_s, u1t, u2t, pat_t, K)
    Ug_s = bc.global_U(u1s, u2s, u1s, u2s)
    Ug_t = bc.global_U(u1s, u2s, u1t, u2t)
    out = dict(pred_bundle)
    out.update({
        "seed": int(seed), "K": K,
        "pat_s": pat_s, "pat_c": pat_c, "pat_t": pat_t,
        "U_s": U_s, "U_c": U_c, "U_t": U_t,
        "Ug_s": Ug_s, "Ug_t": Ug_t,
        "k_selection": ktable,
    })
    return out


def add_score(bundle: dict, score_type: str):
    out = dict(bundle)
    score_type = score_type.lower()
    if score_type == "aps":
        # Canonical paper convention: deterministic nonrandomized APS, cumulative
        # probability strictly before the candidate class.
        score_s = bc.aps_true_scores(bundle["probs_s"], bundle["y_s"])
        score_c = bc.aps_true_scores(bundle["probs_c"], bundle["y_c"])
        score_t_all = bc.aps_score_matrix_strict(bundle["probs_t"])
        score_name = "Deterministic_StrictBeforeCandidate_APS"
    elif score_type == "lac":
        ps, pc, pt = bundle["probs_s"], bundle["probs_c"], bundle["probs_t"]
        ys, yc = np.asarray(bundle["y_s"], int), np.asarray(bundle["y_c"], int)
        score_s = (1.0 - ps[np.arange(len(ys)), ys]).astype(np.float32)
        score_c = (1.0 - pc[np.arange(len(yc)), yc]).astype(np.float32)
        score_t_all = (1.0 - pt).astype(np.float32)
        score_name = "LAC_1_minus_probability"
    else:
        raise ValueError("score_type must be 'aps' or 'lac'")
    out.update({
        "score_type": score_type,
        "score_name": score_name,
        "score_structure": score_s,
        "score_cal": score_c,
        "score_test_all": score_t_all,
    })
    return out


def default_min_cal_leaf(bundle: dict) -> int:
    """Paper protocol final leaf support by dataset family."""
    if bundle["dataset"] == "cifar100":
        return 50
    if bundle["dataset"] in {"imagenet1k", "imagenet"}:
        return 20
    return 30


def default_min_eval(bundle: dict) -> int:
    if bundle["dataset"] == "cifar100":
        return MIN_EVAL_CELL_CIFAR
    if bundle["dataset"] == "imagenet1k":
        return MIN_EVAL_CELL_IMAGENET
    return MIN_EVAL_CELL_TABULAR


def eval_groups(bundle: dict):
    K = int(bundle["K"])
    edges = bc.mixed_edges(bundle["U_s"], bundle["pat_s"], K, [bc.EVAL_BINS] * K)
    eval_pt = bc.assign_mixed(bundle["U_t"], bundle["pat_t"], edges)
    gedges = bc.global_edges(bundle["Ug_s"], bc.EVAL_BINS)
    eval_gt = bc.assign_global(bundle["Ug_t"], gedges)
    return eval_pt, eval_gt


def mask_from_q(score_test_all, q):
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 0:
        q = np.full(len(score_test_all), float(q), dtype=np.float32)
    return np.asarray(score_test_all) <= q[:, None]


def evaluate_mask(method: str, mask, bundle: dict, extra: Optional[dict] = None):
    y = np.asarray(bundle["y_t"], dtype=int)
    mask = np.asarray(mask, dtype=bool)
    covered = mask[np.arange(len(y)), y]
    sizes = mask.sum(1)
    eval_pt, eval_gt = eval_groups(bundle)
    min_eval = default_min_eval(bundle)
    pst, pdf = bc.group_stats(covered, eval_pt, min_eval)
    gst, gdf = bc.group_stats(covered, eval_gt, min_eval)
    cst, cdf = bc.class_coverage_stats(covered, y)
    row = {
        "dataset": bundle["dataset"], "backbone": bundle["backbone"],
        "score_type": bundle["score_type"], "score_name": bundle["score_name"],
        "seed": int(bundle["seed"]), "K": int(bundle["K"]), "method": method,
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
        "backbone_accuracy": float(bundle.get("backbone_accuracy", np.nan)),
        "val_accuracy": float(bundle.get("val_accuracy", np.nan)),
    }
    if extra:
        row.update(extra)
    details = []
    for fam, df in [("pattern_U10", pdf), ("global_U10", gdf)]:
        if len(df):
            d = df.copy(); d["family"] = fam; d["method"] = method; details.append(d)
    if len(cdf):
        d = cdf.copy(); d["group"] = d.pop("class"); d["family"] = "class"; d["method"] = method; details.append(d)
    det = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return row, det


def run_brc_core(bundle: dict, include_random: bool = True):
    K = int(bundle["K"])
    min_eval = default_min_eval(bundle)
    local, _ = bc.precompute_local_costs(
        bundle["score_structure"], bundle["U_s"], bundle["pat_s"], K, min_eval, bundle["seed"]
    )
    demand_rows, demand_summary = bc.demand_heterogeneity_table(local, K)

    rows, details, support_rows, allocation_rows, random_rows = [], [], [], [], []

    def add(method, q, complexity, allocation=None, extra=None):
        ex = {"complexity_cells": int(complexity)}
        if allocation is not None:
            ex["allocation"] = "-".join(map(str, allocation))
        if extra:
            ex.update(extra)
        row, det = evaluate_mask(method, mask_from_q(bundle["score_test_all"], q), bundle, ex)
        rows.append(row)
        if len(det):
            details.append(det)

    add("GlobalCP", bc.conformal_quantile(bundle["score_cal"]), 1)

    for b in [1, 2, 3, 4, 6]:
        alloc = tuple([int(b)] * K)
        q, sup, active = bc.calibrate_allocation(
            bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
            bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
            K, alloc, MIN_CAL_LEAF,
        )
        method = "PatternCP" if b == 1 else f"Uniform_q{b}"
        add(method, q, sum(alloc), alloc, {"active_cells": active})
        sup = sup.copy(); sup["method"] = method; support_rows.append(sup)

    Bstar, Btable = bc.select_global_B(local, K)
    alloc = tuple([Bstar] * K)
    q, sup, active = bc.calibrate_allocation(
        bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
        bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
        K, alloc, MIN_CAL_LEAF,
    )
    add("StructureSelected_GlobalB", q, sum(alloc), alloc, {"selected_global_B": Bstar, "active_cells": active})
    sup = sup.copy(); sup["method"] = "StructureSelected_GlobalB"; support_rows.append(sup)

    for mult in bc.BUDGET_MULTIPLIERS:
        budget = int(mult * K)
        state = bc.solve_budget_dp(local, K, budget)
        alloc = tuple(int(x) for x in state[2])
        q, sup, active = bc.calibrate_allocation(
            bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
            bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
            K, alloc, MIN_CAL_LEAF,
        )
        method = f"BRC_{mult}K"
        add(method, q, budget, alloc, {
            "active_cells": active,
            "structure_objective_max_gap": float(state[0]),
            "structure_objective_weighted_sum": float(state[1]),
        })
        allocation_rows.append({
            "dataset": bundle["dataset"], "backbone": bundle["backbone"],
            "score_type": bundle["score_type"], "seed": bundle["seed"], "K": K,
            "budget_multiplier": mult, "budget_cells": budget,
            "allocation": "-".join(map(str, alloc)),
            "structure_objective_max_gap": float(state[0]),
            "structure_objective_weighted_sum": float(state[1]),
        })
        sup = sup.copy(); sup["method"] = method; support_rows.append(sup)

        if include_random:
            draws = bc.sample_random_legal_allocations(K, budget, bc.RANDOM_DRAWS, bundle["seed"] + 700000 + 101 * mult)
            per = []
            for draw_id, a in enumerate(draws):
                qd, _, active_d = bc.calibrate_allocation(
                    bundle["score_cal"], bundle["pat_c"], bundle["U_c"],
                    bundle["pat_t"], bundle["U_t"], bundle["U_s"], bundle["pat_s"],
                    K, a, MIN_CAL_LEAF,
                )
                row, _ = evaluate_mask(
                    f"Random_{mult}K_draw", mask_from_q(bundle["score_test_all"], qd), bundle,
                    {"complexity_cells": budget, "allocation": "-".join(map(str, a)), "draw_id": draw_id, "active_cells": active_d},
                )
                per.append(row)
            if per:
                rdf = pd.DataFrame(per); random_rows.append(rdf)
                mean_row = {
                    "dataset": bundle["dataset"], "backbone": bundle["backbone"],
                    "score_type": bundle["score_type"], "score_name": bundle["score_name"],
                    "seed": bundle["seed"], "K": K, "method": f"Random_{mult}K_mean",
                    "complexity_cells": budget,
                    "backbone_accuracy": float(bundle.get("backbone_accuracy", np.nan)),
                    "val_accuracy": float(bundle.get("val_accuracy", np.nan)),
                }
                for c in [
                    "marginal_coverage", "avg_set_size", "empty_rate", "worst_eval_pattern_U",
                    "spread_eval_pattern_U", "max_gap_eval_pattern_U", "weighted_gap_eval_pattern_U",
                    "worst_eval_global_U", "worst_class_coverage", "bottom5_class_coverage", "bottom10_class_coverage",
                ]:
                    mean_row[c] = float(rdf[c].mean())
                rows.append(mean_row)

    return {
        "results": pd.DataFrame(rows),
        "details": pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        "support": pd.concat(support_rows, ignore_index=True) if support_rows else pd.DataFrame(),
        "allocations": pd.DataFrame(allocation_rows),
        "random_draws": pd.concat(random_rows, ignore_index=True) if random_rows else pd.DataFrame(),
        "demand_rows": demand_rows,
        "demand_summary": demand_summary,
        "global_B_table": Btable,
        "local": local,
    }


def run_clustered_cp(bundle: dict, randomize: bool, label: str):
    ps, pc = bundle["probs_s"], bundle["probs_c"]
    ys, yc = bundle["y_s"], bundle["y_c"]
    sseed = int(bundle["seed"])
    scores_s_all = get_APS_scores_all(ps, randomize=randomize, seed=sseed + 101).astype(np.float32)
    scores_c_all = get_APS_scores_all(pc, randomize=randomize, seed=sseed + 102).astype(np.float32)
    test_scores = get_APS_scores_all(bundle["probs_t"], randomize=randomize, seed=sseed + 103).astype(np.float32)
    pool_scores = np.concatenate([scores_s_all, scores_c_all], 0)
    pool_labels = np.concatenate([ys, yc], 0)
    _, preds, class_metrics, _ = clustered_conformal(
        pool_scores, pool_labels, ALPHA,
        val_scores_all=test_scores,
        val_labels=bundle["y_t"],
        frac_clustering="auto", num_clusters="auto", split="random",
        exact_coverage=False, seed=sseed,
    )
    C = bundle["probs_t"].shape[1]
    mask = np.zeros((len(preds), C), dtype=bool)
    for i, pred in enumerate(preds):
        mask[i, np.asarray(pred, dtype=int)] = True
    row, det = evaluate_mask(
        label, mask, bundle,
        {"complexity_cells": np.nan, "clustered_cp_randomized_aps": bool(randomize)},
    )
    row["score_name"] = (
        "ClusteredCP_SourceNative_RandomizedAPS" if randomize
        else "Official_Nonrandomized_APS"
    )
    return row, det


def aggregate_results(df: pd.DataFrame):
    id_cols = {"dataset", "backbone", "score_type", "score_name", "seed", "method", "allocation"}
    group_cols = ["dataset", "backbone", "score_type", "score_name", "method"]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key)); row["n_seeds"] = int(g["seed"].nunique())
        for c in g.columns:
            if c in id_cols or c in group_cols:
                continue
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals):
                row[c + "_mean"] = float(vals.mean())
                row[c + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_seed_outputs(outdir: Path, bundle: dict, core: dict, prefix: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)
    p = (prefix + "_") if prefix else ""
    bundle["k_selection"].to_csv(outdir / f"{p}k_selection.csv", index=False)
    core["results"].to_csv(outdir / f"{p}results.csv", index=False)
    core["details"].to_csv(outdir / f"{p}group_details.csv", index=False)
    core["support"].to_csv(outdir / f"{p}calibration_support.csv", index=False)
    core["allocations"].to_csv(outdir / f"{p}brc_allocations.csv", index=False)
    core["random_draws"].to_csv(outdir / f"{p}random_allocation_draws.csv", index=False)
    core["demand_rows"].to_csv(outdir / f"{p}regime_resolution_demand.csv", index=False)
    core["demand_summary"].to_csv(outdir / f"{p}regime_resolution_demand_summary.csv", index=False)
    core["global_B_table"].to_csv(outdir / f"{p}structure_selected_global_B.csv", index=False)
    (outdir / f"{p}split_sizes.json").write_text(json.dumps(bundle.get("split_sizes", {}), indent=2))


def make_mechanism_plot(demand_df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for pat, g in demand_df.groupby("pattern"):
        gg = g.groupby("B", as_index=False)["oof_max_gap"].mean().sort_values("B")
        ax.plot(gg["B"], gg["oof_max_gap"], marker="o", label=f"Regime {int(pat)}")
    ax.set_xlabel("Allocated calibration cells $B_k$")
    ax.set_ylabel("OOF max absolute coverage gap")
    ax.set_title(title)
    ax.set_xticks(sorted(demand_df["B"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
