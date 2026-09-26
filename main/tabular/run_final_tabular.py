#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path


def ensure(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        pkg = pip_name or import_name
        print(f"Installing missing dependency: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for a, b in [
    ("numpy", None), ("pandas", None), ("sklearn", "scikit-learn"),
    ("torch", None), ("pytorch_tabnet", "pytorch-tabnet"),
    ("folktables", None), ("rtdl_revisiting_models", "rtdl_revisiting_models"),
]:
    ensure(a, b)

import numpy as np
import pandas as pd
import torch

import brc_core as bc
from data_models import build_tabular_prediction_bundle, finalize_brc_bundle

LOCKED_SEEDS = [42, 999, 0, 123, 12345]
SETTINGS = {
    # Requested final-table settings.
    "adult_ft": ("adult", "fttransformer", "lac"),
    "covertype_tabnet": ("covertype", "tabnet", "lac"),
    "covertype_ft": ("covertype", "fttransformer", "lac"),
    "folktables_ft": ("folktables", "fttransformer", "lac"),
    # Remaining paper settings, included in the default six-setting suite.
    "adult_tabnet": ("adult", "tabnet", "lac"),
    "folktables_tabnet": ("folktables", "tabnet", "lac"),
}
DEFAULT_SETTINGS = [
    "adult_tabnet", "adult_ft",
    "covertype_tabnet", "covertype_ft",
    "folktables_tabnet", "folktables_ft",
]


def run_one_setting(bundle, outdir: Path, min_cal_leaf: int = 30, min_eval_cell: int = 30):
    dataset, backbone, score_type, seed, K = (
        bundle["dataset"], bundle["backbone"], bundle["score_type"], int(bundle["seed"]), int(bundle["K"])
    )
    print("\n" + "=" * 120)
    print(f"FINAL TABULAR BRC | {dataset} | {backbone} | {score_type} | seed={seed} | K={K}")
    print("=" * 120)

    seed_dir = outdir / f"{dataset}_{backbone}_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    bundle["k_selection"].to_csv(seed_dir / "k_selection.csv", index=False)
    (seed_dir / "split_sizes.json").write_text(json.dumps(bundle["split_sizes"], indent=2), encoding="utf-8")
    (seed_dir / "split_audit.json").write_text(json.dumps({
        "protocol_version": bundle.get("protocol_version"),
        "split_protocol": bundle.get("split_protocol"),
        "split_hashes": bundle.get("split_hashes"),
        "split_sizes": bundle.get("split_sizes"),
    }, indent=2), encoding="utf-8")

    # Frozen conditional audit grid comes from STRUCTURE only.
    eval_edges = bc.mixed_edges(bundle["U_s"], bundle["pat_s"], K, [bc.EVAL_BINS] * K)
    eval_pt = bc.assign_mixed(bundle["U_t"], bundle["pat_t"], eval_edges)
    gedges = bc.global_edges(bundle["Ug_s"], bc.EVAL_BINS)
    eval_gt = bc.assign_global(bundle["Ug_t"], gedges)

    # Allocation costs are estimated from STRUCTURE only with OOF folds.
    local, _ = bc.precompute_local_costs(
        bundle["score_structure"], bundle["U_s"], bundle["pat_s"], K, min_eval_cell, seed
    )
    demand_rows, demand_summary = bc.demand_heterogeneity_table(local, K)
    demand_rows.to_csv(seed_dir / "regime_resolution_demand.csv", index=False)
    demand_summary.to_csv(seed_dir / "regime_resolution_demand_summary.csv", index=False)

    common_extra = {
        "dataset": dataset,
        "backbone": backbone,
        "score_type": score_type,
        "seed": seed,
        "K": K,
        "backbone_accuracy": float(bundle["backbone_accuracy"]),
        "n_structure": int(bundle["split_sizes"]["structure"]),
        "n_cal": int(bundle["split_sizes"]["cal"]),
        "n_test": int(bundle["split_sizes"]["test"]),
        "split_protocol": str(bundle.get("split_protocol", "")),
    }
    rows, details, support_tables, allocation_rows, random_rows = [], [], [], [], []

    # Global CP: CAL quantile only, TEST evaluation only.
    qg = bc.conformal_quantile(bundle["score_cal"])
    qtest = np.full(len(bundle["y_t"]), qg)
    row, det = bc.evaluate_classification(
        "GlobalCP", bundle["probs_t"], bundle["y_t"], qtest, score_type,
        eval_pt, eval_gt, min_eval_cell, 1, common_extra
    )
    rows.append(row); details.append(det)

    # Pattern-only and fixed uniform-resolution controls.
    for b in [1, 2, 3, 4, 6]:
        allocation = tuple([b] * K)
        qtest, sup, active = bc.calibrate_allocation(
            bundle["score_cal"], bundle["pat_c"], bundle["U_c"], bundle["pat_t"], bundle["U_t"],
            bundle["U_s"], bundle["pat_s"], K, allocation, min_cal_leaf
        )
        method = "PatternCP" if b == 1 else f"Uniform_q{b}"
        extra = dict(common_extra)
        extra.update({"allocation": "-".join(map(str, allocation)), "active_cells": active})
        row, det = bc.evaluate_classification(
            method, bundle["probs_t"], bundle["y_t"], qtest, score_type,
            eval_pt, eval_gt, min_eval_cell, sum(allocation), extra
        )
        rows.append(row); details.append(det)
        sup["method"] = method; sup["seed"] = seed; support_tables.append(sup)

    # STRUCTURE-selected common resolution, transparent control only.
    Bstar, Btable = bc.select_global_B(local, K)
    Btable.to_csv(seed_dir / "structure_selected_global_B.csv", index=False)
    allocation = tuple([Bstar] * K)
    qtest, sup, active = bc.calibrate_allocation(
        bundle["score_cal"], bundle["pat_c"], bundle["U_c"], bundle["pat_t"], bundle["U_t"],
        bundle["U_s"], bundle["pat_s"], K, allocation, min_cal_leaf
    )
    extra = dict(common_extra)
    extra.update({"selected_B": Bstar, "allocation": "-".join(map(str, allocation)), "active_cells": active})
    row, det = bc.evaluate_classification(
        "StructureSelected_GlobalB", bundle["probs_t"], bundle["y_t"], qtest, score_type,
        eval_pt, eval_gt, min_eval_cell, sum(allocation), extra
    )
    rows.append(row); details.append(det)
    sup["method"] = "StructureSelected_GlobalB"; sup["seed"] = seed; support_tables.append(sup)

    # BRC and random matched-budget controls.
    for mult in bc.BUDGET_MULTIPLIERS:
        budget = int(mult * K)
        state = bc.solve_budget_dp(local, K, budget)
        allocation = tuple(int(x) for x in state[2])
        qtest, sup, active = bc.calibrate_allocation(
            bundle["score_cal"], bundle["pat_c"], bundle["U_c"], bundle["pat_t"], bundle["U_t"],
            bundle["U_s"], bundle["pat_s"], K, allocation, min_cal_leaf
        )
        extra = dict(common_extra)
        extra.update({
            "allocation": "-".join(map(str, allocation)),
            "active_cells": active,
            "structure_objective_max_gap": float(state[0]),
            "structure_objective_weighted_sum": float(state[1]),
        })
        method = f"BRC_{mult}K"
        row, det = bc.evaluate_classification(
            method, bundle["probs_t"], bundle["y_t"], qtest, score_type,
            eval_pt, eval_gt, min_eval_cell, budget, extra
        )
        rows.append(row); details.append(det)
        sup["method"] = method; sup["seed"] = seed; support_tables.append(sup)
        allocation_rows.append({
            "dataset": dataset, "backbone": backbone, "seed": seed, "K": K,
            "budget_multiplier": mult, "budget_cells": budget,
            "allocation": "-".join(map(str, allocation)),
            "structure_objective_max_gap": float(state[0]),
            "structure_objective_weighted_sum": float(state[1]),
        })

        draws = bc.sample_random_legal_allocations(
            K, budget, bc.RANDOM_DRAWS, seed + 700000 + 101 * mult
        )
        for draw_id, a in enumerate(draws):
            qrand, _, _ = bc.calibrate_allocation(
                bundle["score_cal"], bundle["pat_c"], bundle["U_c"], bundle["pat_t"], bundle["U_t"],
                bundle["U_s"], bundle["pat_s"], K, a, min_cal_leaf
            )
            rextra = dict(common_extra)
            rextra.update({"draw_id": draw_id, "allocation": "-".join(map(str, a))})
            rrow, _ = bc.evaluate_classification(
                f"Random_{mult}K_draw", bundle["probs_t"], bundle["y_t"], qrand, score_type,
                eval_pt, eval_gt, min_eval_cell, budget, rextra
            )
            random_rows.append(rrow)

    if random_rows:
        rdf = pd.DataFrame(random_rows)
        for mult in bc.BUDGET_MULTIPLIERS:
            g = rdf[rdf.method == f"Random_{mult}K_draw"]
            if not len(g):
                continue
            metric_cols = [c for c in g.columns if c not in {
                "method", "dataset", "backbone", "score_type", "seed", "K", "allocation", "draw_id", "split_protocol"
            }]
            mean_row = dict(common_extra)
            mean_row["method"] = f"Random_{mult}K_mean"
            mean_row["complexity_cells"] = int(mult * K)
            for c in metric_cols:
                vals = pd.to_numeric(g[c], errors="coerce")
                if vals.notna().any():
                    mean_row[c] = float(vals.mean())
            rows.append(mean_row)
        rdf.to_csv(seed_dir / "random_allocation_draws.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(seed_dir / "results.csv", index=False)
    if details:
        pd.concat([d for d in details if len(d)], ignore_index=True).to_csv(seed_dir / "group_details.csv", index=False)
    if support_tables:
        pd.concat(support_tables, ignore_index=True).to_csv(seed_dir / "calibration_support.csv", index=False)
    if allocation_rows:
        pd.DataFrame(allocation_rows).to_csv(seed_dir / "brc_allocations.csv", index=False)

    return (
        res, details, support_tables, allocation_rows, random_rows,
        demand_rows.assign(dataset=dataset, backbone=backbone, seed=seed),
        demand_summary.assign(dataset=dataset, backbone=backbone, seed=seed),
    )


def make_compact_final_table(agg: pd.DataFrame) -> pd.DataFrame:
    keep_methods = [
        "GlobalCP", "PatternCP", "Uniform_q2", "Uniform_q3", "Uniform_q4", "Uniform_q6",
        "BRC_2K", "BRC_3K", "BRC_4K", "Random_2K_mean", "Random_3K_mean", "Random_4K_mean",
    ]
    cols = [
        "dataset", "backbone", "score_type", "method",
        "marginal_coverage_mean", "marginal_coverage_std",
        "worst_eval_pattern_U_mean", "worst_eval_pattern_U_std",
        "avg_set_size_mean", "avg_set_size_std",
        "empty_rate_mean", "empty_rate_std",
        "K_mean", "backbone_accuracy_mean",
    ]
    have = [c for c in cols if c in agg.columns]
    out = agg[agg["method"].isin(keep_methods)][have].copy()
    order = {m: i for i, m in enumerate(keep_methods)}
    out["_order"] = out["method"].map(order)
    out = out.sort_values(["dataset", "backbone", "_order"]).drop(columns="_order")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=DEFAULT_SETTINGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=LOCKED_SEEDS)
    ap.add_argument("--out", default="./outputs/main_tabular")
    ap.add_argument("--cache-dir", default="./cache/main_tabular")
    ap.add_argument("--data-home", default="./data")
    ap.add_argument("--folktables-root", default="./data/folktables")
    ap.add_argument("--folktables-state", default="CA")
    ap.add_argument("--folktables-year", type=int, default=2018)
    args = ap.parse_args()

    if sorted(args.seeds) != sorted(LOCKED_SEEDS) or len(args.seeds) != 5:
        raise RuntimeError(
            f"Final 5-seed protocol is locked to exactly {LOCKED_SEEDS}. "
            f"Received {args.seeds}."
        )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_home).mkdir(parents=True, exist_ok=True)
    Path(args.folktables_root).mkdir(parents=True, exist_ok=True)

    all_results, all_details, all_support, all_alloc, all_random = [], [], [], [], []
    all_demand, all_demand_sum, split_audit_rows = [], [], []
    paired_hashes = {}

    for setting in args.settings:
        ds, backbone, score_type = SETTINGS[setting]
        for seed in args.seeds:
            pb = build_tabular_prediction_bundle(
                ds, backbone, seed, args.cache_dir, args.data_home, args.folktables_root,
                args.folktables_state, args.folktables_year
            )
            if pb.get("protocol_version") != "final_tabular_5seed_v1":
                raise RuntimeError("Prediction cache protocol version mismatch. Refusing stale cache.")

            # Pairing audit: within a dataset+seed, all backbones must share exactly the same split hashes.
            key = (ds, int(seed))
            hashes = pb.get("split_hashes", {})
            if key in paired_hashes and paired_hashes[key] != hashes:
                raise RuntimeError(
                    f"Paired-backbone split audit failed for {ds}, seed={seed}. "
                    "Different backbones received different data roles."
                )
            paired_hashes[key] = hashes
            split_audit_rows.append({
                "setting": setting, "dataset_key": ds, "dataset_output": pb["dataset"],
                "backbone": backbone, "seed": seed, "split_protocol": pb.get("split_protocol"),
                **{f"hash_{k}": v for k, v in hashes.items()},
                **{f"n_{k}": int(v) for k, v in pb["split_sizes"].items()},
            })

            bundle = finalize_brc_bundle(pb, seed, score_type, min_structure_support=100)
            result = run_one_setting(bundle, outdir, min_cal_leaf=30, min_eval_cell=30)
            res, det, sup, alloc, rnd, dem, dems = result
            all_results.append(res)
            all_details.extend(det)
            all_support.extend(sup)
            all_alloc.extend(alloc)
            all_random.extend(rnd)
            all_demand.append(dem)
            all_demand_sum.append(dems)
            pd.concat(all_results, ignore_index=True).to_csv(outdir / "ALL_RESULTS_PARTIAL.csv", index=False)
            pd.DataFrame(split_audit_rows).to_csv(outdir / "SPLIT_AUDIT.csv", index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(outdir / "ALL_RESULTS.csv", index=False)
    agg = bc.aggregate_results(df)
    agg.to_csv(outdir / "AGGREGATE_MEAN_STD.csv", index=False)
    make_compact_final_table(agg).to_csv(outdir / "FINAL_TABLE_COMPACT.csv", index=False)

    if all_details:
        pd.concat([d for d in all_details if len(d)], ignore_index=True).to_csv(outdir / "ALL_GROUP_DETAILS.csv", index=False)
    if all_support:
        pd.concat(all_support, ignore_index=True).to_csv(outdir / "ALL_CALIBRATION_SUPPORT.csv", index=False)
    if all_alloc:
        pd.DataFrame(all_alloc).to_csv(outdir / "ALL_BRC_ALLOCATIONS.csv", index=False)
    if all_random:
        pd.DataFrame(all_random).to_csv(outdir / "ALL_RANDOM_DRAWS.csv", index=False)
    if all_demand:
        pd.concat(all_demand, ignore_index=True).to_csv(outdir / "ALL_REGIME_RESOLUTION_DEMAND.csv", index=False)
    if all_demand_sum:
        pd.concat(all_demand_sum, ignore_index=True).to_csv(outdir / "ALL_REGIME_RESOLUTION_DEMAND_SUMMARY.csv", index=False)
    pd.DataFrame(split_audit_rows).to_csv(outdir / "SPLIT_AUDIT.csv", index=False)

    # Matched-budget causal comparison: same regimes, same total cells, same CAL/TEST.
    piv = agg.pivot_table(
        index=["dataset", "backbone", "score_type"], columns="method",
        values="worst_eval_pattern_U_mean", aggfunc="first"
    )
    win_rows = []
    for idx, r in piv.iterrows():
        for mult in bc.BUDGET_MULTIPLIERS:
            b, q = f"BRC_{mult}K", f"Uniform_q{mult}"
            if b in r.index and q in r.index and pd.notna(r[b]) and pd.notna(r[q]):
                delta = float(r[b] - r[q])
                win_rows.append({
                    "dataset": idx[0], "backbone": idx[1], "score_type": idx[2],
                    "budget": f"{mult}K", "BRC_WPC": float(r[b]), "uniform_WPC": float(r[q]),
                    "delta_WPC": delta,
                    "numeric_outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
                })
    pd.DataFrame(win_rows).to_csv(outdir / "MATCHED_BUDGET_WIN_LOSS.csv", index=False)

    protocol = {
        "package": "brc_final_tabular_backbones_5seed_v1",
        "method": "Budgeted Regime Calibration (BRC)",
        "reporting_seeds": LOCKED_SEEDS,
        "settings_default": DEFAULT_SETTINGS,
        "settings_requested": args.settings,
        "score": "LAC for all tabular settings",
        "alpha": float(bc.ALPHA),
        "roles": {
            "TRAIN": "predictor fit and preprocessing fit only",
            "VAL": "predictor early stopping only",
            "STRUCTURE": "PCA/signature standardization, regime discovery, uncertainty ranks/bin edges, OOF allocation objective",
            "CAL": "final conformal quantiles only",
            "TEST": "evaluation only; never used for selection",
        },
        "tabular_split_fractions": "55/10/15/10/10 TRAIN/VAL/STRUCTURE/CAL/TEST",
        "adult_pairing": "Adult TabNet and Adult FT-Transformer use the same deterministic dataset+seed split generator for all five reporting seeds.",
        "covertype_folktables_pairing": "Backbones share deterministic dataset+seed split hashes; runtime aborts on mismatch.",
        "preprocessing": "numeric medians/scalers and categorical encoders fit on TRAIN only",
        "representation": "FT final-head input or TabNet explanation features; PCA, if needed, fit on STRUCTURE only",
        "B_choices": list(bc.B_CHOICES),
        "budget_multipliers": list(bc.BUDGET_MULTIPLIERS),
        "min_structure_support": 100,
        "min_cal_leaf": 30,
        "min_eval_cell": 30,
        "oof_folds": int(bc.OOF_FOLDS),
        "audit_grid": f"frozen P x U{bc.EVAL_BINS} plus global U{bc.EVAL_BINS}, built from STRUCTURE",
        "random_draws": int(bc.RANDOM_DRAWS),
        "test_selection": False,
        "legacy_cache_reuse": False,
    }
    (outdir / "PROTOCOL_LOCK.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("FINAL AGGREGATE SUMMARY")
    print("=" * 120)
    print(make_compact_final_table(agg).to_string(index=False))
    zip_path = shutil.make_archive(str(outdir), "zip", root_dir=outdir)
    print("\nDONE. Upload this ZIP:\n", zip_path)
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
