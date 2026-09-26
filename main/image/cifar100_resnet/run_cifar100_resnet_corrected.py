#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from validation_common import (
    REPORTING_SEEDS,
    add_score,
    aggregate_results,
    build_cifar_prediction_bundle,
    build_regime_bundle,
    refuse_reserved,
    run_brc_core,
    run_clustered_cp,
    save_seed_outputs,
    set_all_seeds,
)


def main():
    ap = argparse.ArgumentParser(description="Final corrected-APS CIFAR-100 ResNet-50 BRC paper-aligned rerun.")
    ap.add_argument("--cache-dir", required=True,
                    help="Persistent cache root. Can reuse the EnergyAPS ResNet cache directory.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=REPORTING_SEEDS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-clustered", action="store_true")
    args = ap.parse_args()

    refuse_reserved(args.seeds)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "CIFAR-100 ResNet-50 corrected APS BRC rerun",
        "predictor": "torchvision ResNet50 IMAGENET1K_V2 frozen features + seed-specific linear CIFAR-100 head",
        "split": "TRAIN 30000 / VAL 5000 / STRUCTURE 7500 / CAL 7500 / official TEST 10000",
        "score": "deterministic nonrandomized strict-before-candidate APS implemented in brc_core.py",
        "alpha": 0.10,
        "budgets": ["2K", "3K", "4K"],
        "uniform_controls": ["q2", "q3", "q4", "q6"],
        "random_controls": True,
        "clustered_cp_rows": [
            "matched nonrandomized APS",
            "source-native randomized APS",
        ] if not args.skip_clustered else [],
        "selection_data": "STRUCTURE only",
        "final_threshold_data": "independent CAL",
        "test_selection": False,
        "seeds": args.seeds,
            }
    (out / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))

    all_rows, all_details, all_demand, all_alloc = [], [], [], []
    for seed in args.seeds:
        print("\n" + "=" * 120)
        print(f"CORRECTED CIFAR-100 RESNET | seed={seed}")
        print("=" * 120)
        set_all_seeds(seed)
        pred = build_cifar_prediction_bundle("resnet50", args.cache_dir, seed, args.batch_size)
        bundle = add_score(build_regime_bundle(pred, seed), "aps")
        core = run_brc_core(bundle, include_random=True)
        seed_dir = out / f"seed{seed}"
        save_seed_outputs(seed_dir, bundle, core)

        rows = core["results"].copy()
        det = core["details"].copy()
        if not args.skip_clustered:
            for randomized, label in [
                (False, "ClusteredCP_Matched_NonrandomizedAPS"),
                (True, "ClusteredCP_SourceNative_RandomizedAPS"),
            ]:
                crow, cdet = run_clustered_cp(bundle, randomized, label)
                rows = pd.concat([rows, pd.DataFrame([crow])], ignore_index=True)
                if len(cdet):
                    det = pd.concat([det, cdet], ignore_index=True)
        rows.to_csv(seed_dir / "results_with_clustered.csv", index=False)
        det.to_csv(seed_dir / "group_details_with_clustered.csv", index=False)

        d = core["demand_rows"].copy(); d["seed"] = seed; d["K"] = bundle["K"]
        a = core["allocations"].copy()
        all_rows.append(rows); all_details.append(det); all_demand.append(d); all_alloc.append(a)

    results = pd.concat(all_rows, ignore_index=True)
    aggregate = aggregate_results(results)
    results.to_csv(out / "ALL_RESULTS.csv", index=False)
    aggregate.to_csv(out / "AGGREGATE_MEAN_STD.csv", index=False)
    pd.concat(all_details, ignore_index=True).to_csv(out / "GROUP_DETAILS.csv", index=False)
    pd.concat(all_demand, ignore_index=True).to_csv(out / "REGIME_RESOLUTION_DEMAND.csv", index=False)
    pd.concat(all_alloc, ignore_index=True).to_csv(out / "BRC_ALLOCATIONS.csv", index=False)

    print("\nAGGREGATE")
    show = [c for c in [
        "method", "marginal_coverage_mean", "worst_eval_pattern_U_mean",
        "avg_set_size_mean", "empty_rate_mean", "worst_class_coverage_mean",
    ] if c in aggregate.columns]
    print(aggregate[show].to_string(index=False))
    zp = shutil.make_archive(str(out), "zip", root_dir=out)
    print("\nDONE:", zp)


if __name__ == "__main__":
    main()
