
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dependency_paths import third_party_path

import numpy as np
import pandas as pd

from brc_core import (
    B_CHOICES,
    BUDGET_MULTIPLIERS,
    UNIFORM_BINS,
    assign_global,
    assign_mixed,
    calibrate_allocation,
    class_coverage_stats,
    conformal_quantile,
    global_edges,
    group_stats,
    mixed_edges,
    precompute_local_costs,
    sample_random_legal_allocations,
    select_global_B,
    solve_budget_dp,
)
from matched_data_models import build_regression_bundle

ALPHA = 0.10
TARGET = 0.90
REPORTING_SEEDS = [42, 999, 0, 123, 12345]
RANDOM_DRAWS = 20


def ensure_dependency(import_name, pip_name=None):
    try:
        __import__(import_name)
        return
    except Exception:
        pkg = pip_name or import_name
        print(f"Installing missing dependency: {pkg}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg]
        )


def setup_sources(root: Path):
    """Load the paper-facing regression baseline sources used in this runner."""
    clover_root = third_party_path("clover-main")
    featurecp_root = third_party_path("FeatureCP-main")

    if not (clover_root / "clover" / "locart.py").exists():
        raise FileNotFoundError(clover_root)
    if not (featurecp_root / "datasets" / "bike_train.csv").exists():
        raise FileNotFoundError(featurecp_root)

    sys.path.insert(0, str(clover_root))
    from clover.locart import LocartSplit
    from clover.scores import RegressionScore
    return featurecp_root, LocartSplit, RegressionScore


def eval_groups(bundle):
    K = int(bundle["K"])
    pt_edges = mixed_edges(
        bundle["U_s"],
        bundle["pat_s"],
        K,
        [10] * K,
    )
    eval_pt = assign_mixed(
        bundle["U_t"],
        bundle["pat_t"],
        pt_edges,
    )
    gt_edges = global_edges(
        bundle["Ug_s"],
        10,
    )
    eval_gt = assign_global(
        bundle["Ug_t"],
        gt_edges,
    )
    return eval_pt, eval_gt


def regression_eval(
    method,
    lower,
    upper,
    y,
    eval_pt,
    eval_gt,
    min_eval_cell,
    extra=None,
):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)

    covered = (y >= lower) & (y <= upper)
    length = upper - lower

    pst, pdf = group_stats(
        covered, eval_pt, min_eval_cell
    )
    gst, gdf = group_stats(
        covered, eval_gt, min_eval_cell
    )

    row = {
        "method": method,
        "marginal_coverage": float(covered.mean()),
        "avg_interval_length": float(length.mean()),
        "worst_eval_pattern_U": pst["worst"],
        "spread_eval_pattern_U": pst["spread"],
        "max_gap_eval_pattern_U": pst["max_gap"],
        "weighted_gap_eval_pattern_U": pst["weighted_gap"],
        "worst_eval_global_U": gst["worst"],
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

    return (
        row,
        pd.concat(details, ignore_index=True)
        if details
        else pd.DataFrame(),
    )








def run_core_regression(bundle):
    eval_pt, eval_gt = eval_groups(bundle)
    K = int(bundle["K"])
    rows, details, random_draws, supports = [], [], [], []

    local, _ = precompute_local_costs(
        bundle["score_structure"],
        bundle["U_s"],
        bundle["pat_s"],
        K,
        bundle["min_eval_cell"],
        bundle["seed"],
    )

    def add(method, qtest, complexity, allocation=None, extra=None):
        x = {
            "complexity_cells": complexity,
        }
        if allocation is not None:
            x["allocation"] = "-".join(
                map(str, allocation)
            )
        if extra:
            x.update(extra)

        row, det = regression_eval(
            method,
            bundle["prediction_test"] - qtest,
            bundle["prediction_test"] + qtest,
            bundle["y_test"],
            eval_pt,
            eval_gt,
            bundle["min_eval_cell"],
            extra=x,
        )
        rows.append(row)
        if len(det):
            details.append(det)

    qg = conformal_quantile(
        bundle["score_cal"]
    )
    add(
        "GlobalCP",
        np.full(len(bundle["y_test"]), qg),
        1,
    )

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
        add(
            method,
            q,
            sum(alloc),
            allocation=alloc,
            extra={"active_cells": active},
        )
        sup = sup.copy()
        sup["method"] = method
        supports.append(sup)

    selected_B, selected_table = select_global_B(
        local, K
    )
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
        allocation=alloc,
        extra={
            "selected_global_B": selected_B,
            "active_cells": active,
        },
    )

    for mult in BUDGET_MULTIPLIERS:
        budget = mult * K
        state = solve_budget_dp(
            local, K, budget
        )
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
            allocation=alloc,
            extra={
                "structure_objective_max_gap": state[0],
                "structure_objective_weighted_sum": state[1],
                "active_cells": active,
            },
        )
        sup = sup.copy()
        sup["method"] = f"BRC_{mult}K"
        supports.append(sup)

        draws = sample_random_legal_allocations(
            K,
            budget,
            RANDOM_DRAWS,
            bundle["seed"] + 1000 * mult,
        )
        temp = []

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
            row, _ = regression_eval(
                f"Random_{mult}K_draw",
                bundle["prediction_test"] - qd,
                bundle["prediction_test"] + qd,
                bundle["y_test"],
                eval_pt,
                eval_gt,
                bundle["min_eval_cell"],
                extra={
                    "complexity_cells": budget,
                    "allocation": "-".join(map(str, da)),
                    "active_cells": active_d,
                    "draw": di,
                },
            )
            temp.append(row)

        if temp:
            rdf = pd.DataFrame(temp)
            random_draws.append(rdf)

            mean_row = {
                "method": f"Random_{mult}K_mean",
                "complexity_cells": budget,
            }
            for c in [
                "marginal_coverage",
                "avg_interval_length",
                "worst_eval_pattern_U",
                "spread_eval_pattern_U",
                "max_gap_eval_pattern_U",
                "weighted_gap_eval_pattern_U",
                "worst_eval_global_U",
            ]:
                mean_row[c] = float(rdf[c].mean())
            rows.append(mean_row)

    return (
        pd.DataFrame(rows),
        details,
        random_draws,
        supports,
        selected_table,
    )




def run_locart(
    bundle,
    LocartSplit,
    RegressionScore,
):
    """
    Official LOCART algorithm on the exact same fitted TabNet ensemble.

    LOCART's source-native split_calib=True is retained. To make the resource
    pool fair, LOCART receives the same combined STRUCTURE+CAL examples BRC
    receives. LOCART then performs its own internal 50/50 partition/cutoff
    split, as designed by the source.
    """
    eval_pt, eval_gt = eval_groups(bundle)

    total_X = np.concatenate(
        [bundle["X_structure"], bundle["X_cal"]],
        axis=0,
    )
    total_y = np.concatenate(
        [bundle["y_structure"], bundle["y_cal"]],
        axis=0,
    )

    locart = LocartSplit(
        RegressionScore,
        bundle["base_model"],
        alpha=ALPHA,
        is_fitted=True,
        cart_type="CART",
        split_calib=True,
        weighting=False,
    )

    locart.fit(
        bundle["X_train"],
        bundle["y_train"],
    )

    locart.calib(
        total_X,
        total_y,
        random_seed=bundle["seed"],
        prune_tree=True,
        prune_seed=bundle["seed"] + 701,
        cart_train_size=0.5,
    )

    intervals = locart.predict(
        bundle["X_test"]
    )

    n_leaves = int(
        locart.cart.get_n_leaves()
    )

    row, det = regression_eval(
        "LOCART_Official_SourcePort",
        intervals[:, 0],
        intervals[:, 1],
        bundle["y_test"],
        eval_pt,
        eval_gt,
        bundle["min_eval_cell"],
        extra={
            "complexity_cells": n_leaves,
            "locart_tree_leaves": n_leaves,
            "locart_split_calib": True,
            "locart_total_pool_size": len(total_y),
        },
    )

    return row, det






def annotate(df, bundle):
    if not len(df):
        return df

    out = df.copy()
    out.insert(0, "dataset", bundle["dataset"])
    out.insert(1, "backbone", bundle["backbone"])
    out.insert(2, "score_type", bundle["score_type"])
    out.insert(3, "seed", bundle["seed"])
    out["K"] = int(bundle["K"])

    if "backbone_accuracy" in bundle:
        out["backbone_accuracy"] = bundle["backbone_accuracy"]

    if "backbone_metric" in bundle:
        out[bundle["backbone_metric_name"]] = bundle[
            "backbone_metric"
        ]

    return out


def aggregate(df):
    numeric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "dataset",
            "backbone",
            "score_type",
            "seed",
            "method",
            "allocation",
            "clusteredcp_source_split",
        }
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []

    for key, g in df.groupby(
        ["dataset", "backbone", "score_type", "method"],
        dropna=False,
        sort=False,
    ):
        row = dict(
            zip(
                ["dataset", "backbone", "score_type", "method"],
                key,
            )
        )
        row["n_seeds"] = int(g["seed"].nunique())

        for c in numeric_cols:
            vals = pd.to_numeric(
                g[c],
                errors="coerce",
            ).dropna()

            if len(vals):
                row[c + "_mean"] = float(vals.mean())
                row[c + "_std"] = (
                    float(vals.std(ddof=1))
                    if len(vals) > 1
                    else 0.0
                )

        rows.append(row)

    return pd.DataFrame(rows)


def checkpoint(
    outdir,
    all_rows,
    all_details,
    all_random,
    all_support,
):
    if all_rows:
        pd.concat(
            all_rows,
            ignore_index=True,
        ).to_csv(
            outdir / "ALL_RESULTS_PARTIAL.csv",
            index=False,
        )

    if all_details:
        pd.concat(
            all_details,
            ignore_index=True,
        ).to_csv(
            outdir / "GROUP_DETAILS_PARTIAL.csv",
            index=False,
        )

    if all_random:
        pd.concat(
            all_random,
            ignore_index=True,
        ).to_csv(
            outdir / "RANDOM_DRAWS_PARTIAL.csv",
            index=False,
        )

    if all_support:
        pd.concat(
            all_support,
            ignore_index=True,
        ).to_csv(
            outdir / "CAL_SUPPORT_PARTIAL.csv",
            index=False,
        )


def main():
    ap = argparse.ArgumentParser(description="Five-seed matched BRC regression transfer and LOCART context baseline.")
    ap.add_argument("--tasks", nargs="+", choices=["housing", "bike"], default=["housing", "bike"])
    ap.add_argument("--seeds", nargs="+", type=int, default=REPORTING_SEEDS)
    ap.add_argument("--out", default="./outputs/main_regression")
    ap.add_argument("--cache-dir", default="./cache/main_regression")
    args = ap.parse_args()

    if sorted(args.seeds) != sorted(REPORTING_SEEDS) or len(args.seeds) != 5:
        raise RuntimeError(f"Paper reporting protocol uses exactly {REPORTING_SEEDS}; received {args.seeds}.")

    ensure_dependency("pytorch_tabnet", "pytorch-tabnet")

    root = Path(__file__).resolve().parent
    outdir = Path(args.out).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    featurecp_root, LocartSplit, RegressionScore = setup_sources(root)

    all_rows, all_details, all_random, all_support = [], [], [], []

    protocol = {
        "alpha": ALPHA,
        "reporting_seeds": args.seeds,
        "tasks": args.tasks,
        "split": "55/10/15/10/10 TRAIN/VAL/STRUCTURE/CAL/TEST",
        "predictor": "three-member TabNet regression ensemble",
        "conformity_score": "absolute residual",
        "regime_signature": "first-member TabNet mask summary + log1p(ensemble std) + log1p(ensemble range)",
        "within_regime_uncertainty": "equal-weight within-regime percentile ranks of ensemble std and range",
        "min_structure_support": 50,
        "min_cal_leaf": 20,
        "min_eval_cell": 20,
        "external_baseline": {
            "method": "LOCART_Official_SourcePort",
            "same_fitted_predictor_as_BRC": True,
            "same_total_structure_plus_cal_pool": True,
            "method_native_internal_split": "LOCART split_calib=True, cart_train_size=0.5",
        },
        "test_selection": False,
    }
    (outdir / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for task in args.tasks:
        for seed in args.seeds:
            print("\n" + "=" * 110)
            print(f"REGRESSION BRC | task={task} | seed={seed}")
            print("=" * 110)

            bundle = build_regression_bundle(task, int(seed), featurecp_root)
            core, dets, rdraws, supports, selected = run_core_regression(bundle)
            row_ext, det_ext = run_locart(bundle, LocartSplit, RegressionScore)

            core = annotate(core, bundle)
            ext = annotate(pd.DataFrame([row_ext]), bundle)
            all_rows.extend([core, ext])

            for d in dets:
                d = d.copy()
                d["dataset"] = bundle["dataset"]
                d["backbone"] = bundle["backbone"]
                d["score_type"] = bundle["score_type"]
                d["seed"] = bundle["seed"]
                all_details.append(d)
            if len(det_ext):
                d = det_ext.copy()
                d["dataset"] = bundle["dataset"]
                d["backbone"] = bundle["backbone"]
                d["score_type"] = bundle["score_type"]
                d["seed"] = bundle["seed"]
                all_details.append(d)

            for rdf in rdraws:
                all_random.append(annotate(rdf, bundle))
            for sup in supports:
                sup = sup.copy()
                sup["dataset"] = bundle["dataset"]
                sup["backbone"] = bundle["backbone"]
                sup["seed"] = bundle["seed"]
                all_support.append(sup)

            bundle["k_selection"].to_csv(outdir / f"{task}_seed{seed}_K_SELECTION.csv", index=False)
            selected.to_csv(outdir / f"{task}_seed{seed}_SELECTED_GLOBAL_B.csv", index=False)
            checkpoint(outdir, all_rows, all_details, all_random, all_support)
            gc.collect()

    result_df = pd.concat(all_rows, ignore_index=True)
    result_df.to_csv(outdir / "ALL_RESULTS.csv", index=False)
    agg = aggregate(result_df)
    agg.to_csv(outdir / "AGGREGATE_MEAN_STD.csv", index=False)
    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(outdir / "GROUP_DETAILS.csv", index=False)
    if all_random:
        pd.concat(all_random, ignore_index=True).to_csv(outdir / "RANDOM_DRAWS.csv", index=False)
    if all_support:
        pd.concat(all_support, ignore_index=True).to_csv(outdir / "CAL_SUPPORT.csv", index=False)

    print("\n" + "=" * 110)
    print("AGGREGATE SUMMARY")
    print("=" * 110)
    print(agg.to_string(index=False))
    print("\nDONE:", shutil.make_archive(str(outdir), "zip", root_dir=outdir))


if __name__ == "__main__":
    main()
