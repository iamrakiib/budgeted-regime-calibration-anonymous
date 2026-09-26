
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from brc_core import (
    global_U,
    learn_patterns,
    pattern_relative_U,
    standardize_three,
)

ALPHA = 0.10
PCA_DIM = 64


def set_all_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def regression_split(n: int, seed: int):
    """
    Exact split proportions and order used by the frozen Housing BRC pipeline:
    TRAIN / VAL / STRUCTURE / CAL / TEST = 55 / 10 / 15 / 10 / 10.
    """
    idx = np.arange(n)
    rest, test = train_test_split(idx, test_size=0.10, random_state=seed)
    rest, cal = train_test_split(
        rest, test_size=0.10 / 0.90, random_state=seed + 1
    )
    rest, structure = train_test_split(
        rest, test_size=0.15 / 0.80, random_state=seed + 2
    )
    train, val = train_test_split(
        rest, test_size=0.10 / 0.65, random_state=seed + 3
    )
    return {
        "train": train,
        "val": val,
        "structure": structure,
        "cal": cal,
        "test": test,
    }


def tabnet_mask_phi(model, X, tau=0.01):
    mask, _ = model.explain(X)
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.clip(mask, 0, None)
    row_sum = mask.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
    P = mask / row_sum
    ent = -np.sum(P * np.log(P + 1e-12), axis=1)
    ent_norm = ent / np.log(P.shape[1] + 1e-12)
    effective = np.exp(ent)
    concentration = P.max(axis=1)
    support = (P > tau).sum(axis=1) / P.shape[1]
    return np.stack(
        [ent_norm, effective, concentration, support], axis=1
    ).astype(np.float32)


def load_bike_from_featurecp(featurecp_root: Path):
    """
    Exact Bike preprocessing copied from the uploaded FeatureCP repository's
    datasets/datasets.py.
    """
    csv_path = featurecp_root / "datasets" / "bike_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    season = pd.get_dummies(df["season"], prefix="season")
    df = pd.concat([df, season], axis=1)

    weather = pd.get_dummies(df["weather"], prefix="weather")
    df = pd.concat([df, weather], axis=1)

    df.drop(["season", "weather"], inplace=True, axis=1)

    dt = pd.DatetimeIndex(df.datetime)
    df["hour"] = [t.hour for t in dt]
    df["day"] = [t.dayofweek for t in dt]
    df["month"] = [t.month for t in dt]
    df["year"] = [t.year for t in dt]
    df["year"] = df["year"].map({2011: 0, 2012: 1})

    df.drop("datetime", axis=1, inplace=True)
    df.drop(["casual", "registered"], axis=1, inplace=True)

    X = df.drop("count", axis=1).values.astype(np.float32)
    y = df["count"].values.astype(np.float32)
    return X, y


class TabNetEnsembleMean:
    """Fitted predict-only wrapper for Clover/LOCART."""

    def __init__(self, models, ymean: float, ystd: float):
        self.models = list(models)
        self.ymean = float(ymean)
        self.ystd = float(ystd)

    def predict(self, X):
        pp = []
        for model in self.models:
            p = model.predict(np.asarray(X, dtype=np.float32)).reshape(-1)
            pp.append(p * self.ystd + self.ymean)
        return np.stack(pp, axis=1).mean(axis=1)


def build_regression_bundle(
    dataset: str,
    seed: int,
    featurecp_root: Path,
):
    """
    Canonical matched regression setting used here:
      - same 55/10/15/10/10 role split
      - same 3-member TabNet ensemble design as frozen Housing BRC
      - absolute-residual conformity score
      - first TabNet member mask summary as phi
      - ensemble std/range as predictive uncertainty
    """
    from pytorch_tabnet.tab_model import TabNetRegressor

    set_all_seeds(seed)

    if dataset == "housing":
        data = fetch_california_housing(as_frame=True)
        X = data.data.to_numpy(np.float32)
        y = data.target.to_numpy(np.float32)
    elif dataset == "bike":
        X, y = load_bike_from_featurecp(featurecp_root)
    else:
        raise ValueError(dataset)

    sp = regression_split(len(y), seed)

    xsc = StandardScaler().fit(X[sp["train"]])
    Xz = xsc.transform(X).astype(np.float32)

    ymean = float(y[sp["train"]].mean())
    ystd = float(y[sp["train"]].std() + 1e-8)
    yn = ((y - ymean) / ystd).astype(np.float32)

    models = []
    split_predictions = {
        "structure": [],
        "cal": [],
        "test": [],
    }

    for mseed in [seed, seed + 1, seed + 2]:
        print(f"Training {dataset} TabNet ensemble member seed={mseed}")
        model = TabNetRegressor(
            n_steps=5,
            n_d=16,
            n_a=16,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            optimizer_params=dict(lr=2e-2),
            mask_type="sparsemax",
            seed=mseed,
            verbose=0,
        )
        model.fit(
            X_train=Xz[sp["train"]],
            y_train=yn[sp["train"]][:, None],
            eval_set=[(Xz[sp["val"]], yn[sp["val"]][:, None])],
            eval_name=["val"],
            eval_metric=["rmse"],
            max_epochs=250,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        models.append(model)

        for part in ["structure", "cal", "test"]:
            p = model.predict(Xz[sp[part]]).reshape(-1)
            p = p * ystd + ymean
            split_predictions[part].append(p.astype(np.float32))

    Ps = np.stack(split_predictions["structure"], axis=1)
    Pc = np.stack(split_predictions["cal"], axis=1)
    Pt = np.stack(split_predictions["test"], axis=1)

    mean_s = Ps.mean(axis=1)
    mean_c = Pc.mean(axis=1)
    mean_t = Pt.mean(axis=1)

    u1s = Ps.std(axis=1)
    u2s = Ps.max(axis=1) - Ps.min(axis=1)
    u1c = Pc.std(axis=1)
    u2c = Pc.max(axis=1) - Pc.min(axis=1)
    u1t = Pt.std(axis=1)
    u2t = Pt.max(axis=1) - Pt.min(axis=1)

    phi_s = tabnet_mask_phi(models[0], Xz[sp["structure"]])
    phi_c = tabnet_mask_phi(models[0], Xz[sp["cal"]])
    phi_t = tabnet_mask_phi(models[0], Xz[sp["test"]])

    sig_s = np.concatenate(
        [phi_s, np.stack([np.log1p(u1s), np.log1p(u2s)], axis=1)],
        axis=1,
    )
    sig_c = np.concatenate(
        [phi_c, np.stack([np.log1p(u1c), np.log1p(u2c)], axis=1)],
        axis=1,
    )
    sig_t = np.concatenate(
        [phi_t, np.stack([np.log1p(u1t), np.log1p(u2t)], axis=1)],
        axis=1,
    )

    sig_s, sig_c, sig_t, _ = standardize_three(sig_s, sig_c, sig_t)
    km, ktable = learn_patterns(sig_s, min_structure_support=50, seed=seed)
    K = int(km.n_clusters)

    pat_s = km.predict(sig_s).astype(int)
    pat_c = km.predict(sig_c).astype(int)
    pat_t = km.predict(sig_t).astype(int)

    U_s = pattern_relative_U(
        u1s, u2s, pat_s, u1s, u2s, pat_s, K
    )
    U_c = pattern_relative_U(
        u1s, u2s, pat_s, u1c, u2c, pat_c, K
    )
    U_t = pattern_relative_U(
        u1s, u2s, pat_s, u1t, u2t, pat_t, K
    )

    Ug_s = global_U(u1s, u2s, u1s, u2s)
    Ug_t = global_U(u1s, u2s, u1t, u2t)

    wrapper = TabNetEnsembleMean(models, ymean, ystd)

    pred_val = wrapper.predict(Xz[sp["val"]])
    val_rmse = float(
        np.sqrt(np.mean((pred_val - y[sp["val"]]) ** 2))
    )

    return {
        "dataset": dataset,
        "backbone": "tabnet_ensemble3",
        "task": "regression",
        "score_type": "absolute_residual",
        "seed": int(seed),
        "K": K,
        "score_structure": np.abs(y[sp["structure"]] - mean_s),
        "score_cal": np.abs(y[sp["cal"]] - mean_c),
        "prediction_test": mean_t,
        "y_test": y[sp["test"]],
        "pat_s": pat_s,
        "pat_c": pat_c,
        "pat_t": pat_t,
        "U_s": U_s,
        "U_c": U_c,
        "U_t": U_t,
        "Ug_s": Ug_s,
        "Ug_t": Ug_t,
        "k_selection": ktable,
        "min_cal_leaf": 20,
        "min_eval_cell": 20,
        "X_train": Xz[sp["train"]],
        "y_train": y[sp["train"]],
        "X_structure": Xz[sp["structure"]],
        "y_structure": y[sp["structure"]],
        "X_cal": Xz[sp["cal"]],
        "y_cal": y[sp["cal"]],
        "X_test": Xz[sp["test"]],
        "base_model": wrapper,
        "backbone_metric_name": "val_rmse",
        "backbone_metric": val_rmse,
        "split_sizes": {k: int(len(v)) for k, v in sp.items()},
    }
