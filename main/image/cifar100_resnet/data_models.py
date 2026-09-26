from __future__ import annotations

import copy
import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from brc_core import classification_uncertainty, learn_patterns, pattern_relative_U, global_U, standardize_three


def set_all_seeds(seed: int):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def five_way_stratified_split(y, seed: int):
    """55/10/15/10/10 = TRAIN/VAL/STRUCTURE/CAL/TEST."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, train_size=0.55, random_state=seed, stratify=y)
    val, rest = train_test_split(rest, train_size=10/45, random_state=seed + 1, stratify=y[rest])
    structure, rest = train_test_split(rest, train_size=15/35, random_state=seed + 2, stratify=y[rest])
    cal, test = train_test_split(rest, train_size=0.50, random_state=seed + 3, stratify=y[rest])
    return {"train": train, "val": val, "structure": structure, "cal": cal, "test": test}


@dataclass
class PreparedTabular:
    dataset: str
    X_dense: np.ndarray
    X_num: np.ndarray
    X_cat: np.ndarray
    y: np.ndarray
    cat_idxs_dense: List[int]
    cat_dims: List[int]
    split: Dict[str, np.ndarray]
    n_classes: int


def _ordinalize(train_raw, all_raw):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_raw)
    z = enc.transform(all_raw).astype(np.int64) + 1
    card = [int(z[:, j].max()) + 1 for j in range(z.shape[1])]
    return z, card


def load_covertype(seed: int, data_home: str):
    from sklearn.datasets import fetch_covtype
    X, y = fetch_covtype(data_home=data_home, return_X_y=True, as_frame=False)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64) - 1
    split = five_way_stratified_split(y, seed)
    sc = StandardScaler().fit(X[split["train"]])
    Xn = sc.transform(X).astype(np.float32)
    # Covertype's 44 indicator columns are already one-hot; keep all 54 as numerical tokens.
    return PreparedTabular(
        dataset="covertype", X_dense=Xn, X_num=Xn, X_cat=np.zeros((len(Xn), 0), np.int64),
        y=y, cat_idxs_dense=[], cat_dims=[], split=split, n_classes=int(y.max()) + 1,
    )


def load_folktables(seed: int, data_root: str, state: str = "CA", year: int = 2018):
    from folktables import ACSDataSource, ACSIncome
    src = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=data_root)
    df = src.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_numpy(df)
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    good = np.isfinite(X.astype(float)).all(axis=1)
    X, y = X[good], y[good]
    split = five_way_stratified_split(y, seed)

    # ACSIncome feature order in folktables:
    # AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P
    num_idx = [0, 7]
    cat_idx = [1, 2, 3, 4, 5, 6, 8, 9]
    sc = StandardScaler().fit(X[split["train"]][:, num_idx].astype(np.float32))
    X_num = sc.transform(X[:, num_idx].astype(np.float32)).astype(np.float32)
    X_cat, cat_dims = _ordinalize(X[split["train"]][:, cat_idx], X[:, cat_idx])

    X_dense = np.zeros((len(X), len(num_idx) + len(cat_idx)), dtype=np.float32)
    X_dense[:, :len(num_idx)] = X_num
    X_dense[:, len(num_idx):] = X_cat.astype(np.float32)
    cat_idxs_dense = list(range(len(num_idx), len(num_idx) + len(cat_idx)))
    return PreparedTabular(
        dataset=f"folktables_acs_income_{state}_{year}", X_dense=X_dense, X_num=X_num,
        X_cat=X_cat, y=y, cat_idxs_dense=cat_idxs_dense, cat_dims=cat_dims,
        split=split, n_classes=2,
    )


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
    return np.stack([ent_norm, effective, concentration, support], 1).astype(np.float32)


def train_tabnet_classifier(data: PreparedTabular, seed: int):
    from pytorch_tabnet.tab_model import TabNetClassifier
    set_all_seeds(seed)
    sp = data.split
    kwargs = dict(
        n_steps=5, n_d=16, n_a=16, gamma=1.5,
        n_independent=2, n_shared=2,
        optimizer_params=dict(lr=2e-2), mask_type="sparsemax",
        seed=seed, verbose=0,
    )
    if data.cat_dims:
        kwargs.update(cat_idxs=data.cat_idxs_dense, cat_dims=data.cat_dims, cat_emb_dim=2)
    model = TabNetClassifier(**kwargs)
    model.fit(
        X_train=data.X_dense[sp["train"]], y_train=data.y[sp["train"]],
        eval_set=[(data.X_dense[sp["val"]], data.y[sp["val"]])],
        eval_name=["val"], eval_metric=["accuracy"],
        max_epochs=150, patience=20, batch_size=4096, virtual_batch_size=256,
        num_workers=0, drop_last=False,
    )
    probs, phi = {}, {}
    for part in ["structure", "cal", "test"]:
        probs[part] = model.predict_proba(data.X_dense[sp[part]]).astype(np.float32)
        phi[part] = tabnet_mask_phi(model, data.X_dense[sp[part]])
    val_pred = model.predict(data.X_dense[sp["val"]]).reshape(-1)
    val_acc = float(np.mean(val_pred == data.y[sp["val"]]))
    return probs, phi, val_acc


def _last_linear(model):
    linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    if not linear:
        raise RuntimeError("FT-Transformer contains no Linear module.")
    return linear[-1]


def train_fttransformer_classifier(data: PreparedTabular, seed: int):
    from rtdl_revisiting_models import FTTransformer
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = data.split
    model = FTTransformer(
        n_cont_features=int(data.X_num.shape[1]),
        cat_cardinalities=list(data.cat_dims),
        d_out=int(data.n_classes),
        n_blocks=3,
        d_block=192,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4/3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    lossfn = torch.nn.CrossEntropyLoss()

    def make_loader(ix, shuffle, batch=1024):
        xn = torch.from_numpy(data.X_num[ix]).float()
        xc = torch.from_numpy(data.X_cat[ix]).long()
        yy = torch.from_numpy(data.y[ix]).long()
        return DataLoader(TensorDataset(xn, xc, yy), batch_size=batch, shuffle=shuffle,
                          num_workers=0, pin_memory=torch.cuda.is_available())

    train_loader = make_loader(sp["train"], True)
    val_loader = make_loader(sp["val"], False, 4096)
    best, best_acc, bad = None, -1.0, 0
    for epoch in range(50):
        model.train()
        for xn, xc, yy in train_loader:
            xn, xc, yy = xn.to(device), xc.to(device), yy.to(device)
            xcat = xc if xc.shape[1] else None
            opt.zero_grad(set_to_none=True)
            logits = model(xn, xcat)
            loss = lossfn(logits, yy)
            loss.backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for xn, xc, yy in val_loader:
                xn, xc, yy = xn.to(device), xc.to(device), yy.to(device)
                logits = model(xn, xc if xc.shape[1] else None)
                correct += int((logits.argmax(1) == yy).sum().item()); total += len(yy)
        acc = correct / max(1, total)
        print(f"FT seed={seed} epoch={epoch+1:02d} val_acc={acc:.5f}")
        if acc > best_acc + 1e-5:
            best_acc = acc; best = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= 8:
                break
    if best is None:
        raise RuntimeError("FT-Transformer training failed to create a checkpoint.")
    model.load_state_dict(best); model.eval()

    head = _last_linear(model)
    capture = {"x": None}
    def hook(_module, inputs):
        capture["x"] = inputs[0].detach()
    h = head.register_forward_pre_hook(hook)

    probs, phi = {}, {}
    with torch.no_grad():
        for part in ["structure", "cal", "test"]:
            loader = make_loader(sp[part], False, 4096)
            pp, hh = [], []
            for xn, xc, yy in loader:
                xn, xc = xn.to(device), xc.to(device)
                logits = model(xn, xc if xc.shape[1] else None)
                if capture["x"] is None or capture["x"].shape[0] != logits.shape[0]:
                    raise RuntimeError("FT-Transformer representation hook did not capture the final-head input correctly.")
                pp.append(torch.softmax(logits, 1).cpu().numpy().astype(np.float32))
                hh.append(capture["x"].cpu().numpy().astype(np.float32))
            probs[part] = np.concatenate(pp)
            phi[part] = np.concatenate(hh)
    h.remove()
    return probs, phi, float(best_acc)


def maybe_pca_phi(phi_s, phi_c, phi_t, seed: int, max_dim: int = 64):
    if phi_s.shape[1] <= max_dim:
        return phi_s.astype(np.float32), phi_c.astype(np.float32), phi_t.astype(np.float32)
    pca = PCA(n_components=max_dim, random_state=seed)
    return (
        pca.fit_transform(phi_s).astype(np.float32),
        pca.transform(phi_c).astype(np.float32),
        pca.transform(phi_t).astype(np.float32),
    )


def save_tabular_bundle_cache(path: Path, data: PreparedTabular, probs, phi, val_acc, backbone: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    sp = data.split
    np.savez_compressed(
        path,
        dataset=np.asarray(data.dataset), backbone=np.asarray(backbone), val_acc=np.asarray(val_acc),
        probs_structure=probs["structure"].astype(np.float16),
        probs_cal=probs["cal"].astype(np.float16), probs_test=probs["test"].astype(np.float16),
        phi_structure=phi["structure"].astype(np.float16), phi_cal=phi["cal"].astype(np.float16),
        phi_test=phi["test"].astype(np.float16), y_structure=data.y[sp["structure"]].astype(np.int16),
        y_cal=data.y[sp["cal"]].astype(np.int16), y_test=data.y[sp["test"]].astype(np.int16),
        n_train=np.asarray(len(sp["train"])), n_val=np.asarray(len(sp["val"])),
        n_structure=np.asarray(len(sp["structure"])), n_cal=np.asarray(len(sp["cal"])), n_test=np.asarray(len(sp["test"])),
    )


def load_tabular_bundle_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    return {
        "dataset": str(z["dataset"]), "backbone": str(z["backbone"]), "backbone_accuracy": float(z["val_acc"]),
        "probs_s": z["probs_structure"].astype(np.float32), "probs_c": z["probs_cal"].astype(np.float32),
        "probs_t": z["probs_test"].astype(np.float32), "phi_s": z["phi_structure"].astype(np.float32),
        "phi_c": z["phi_cal"].astype(np.float32), "phi_t": z["phi_test"].astype(np.float32),
        "y_s": z["y_structure"].astype(int), "y_c": z["y_cal"].astype(int), "y_t": z["y_test"].astype(int),
        "split_sizes": {k: int(z[f"n_{k}"]) for k in ["train", "val", "structure", "cal", "test"]},
    }


def build_tabular_prediction_bundle(dataset: str, backbone: str, seed: int, cache_dir: str,
                                    data_home: str, folktables_root: str, folktables_state="CA", folktables_year=2018):
    cache = Path(cache_dir) / f"{dataset}_{backbone}_seed{seed}_predictions.npz"
    if cache.exists():
        print(f"Reusing cache: {cache}")
        return load_tabular_bundle_cache(cache)

    if dataset == "covertype":
        data = load_covertype(seed, data_home)
    elif dataset == "folktables":
        data = load_folktables(seed, folktables_root, folktables_state, folktables_year)
    else:
        raise ValueError(dataset)

    if backbone == "tabnet":
        probs, phi, acc = train_tabnet_classifier(data, seed)
    elif backbone == "fttransformer":
        probs, phi, acc = train_fttransformer_classifier(data, seed)
    else:
        raise ValueError(backbone)

    save_tabular_bundle_cache(cache, data, probs, phi, acc, backbone)
    return load_tabular_bundle_cache(cache)


@torch.no_grad()
def _extract_imagenet_cache(root: str, cache_path: Path, batch_size=128):
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    tfm = weights.transforms()
    ds = ImageFolder(root, transform=tfm)
    if len(ds) < 49000 or len(ds.classes) != 1000:
        raise RuntimeError(
            f"Expected ImageNet-1K validation in class-subdirectory ImageFolder format. "
            f"Found n={len(ds)}, classes={len(ds.classes)} at {root}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=weights).to(device).eval()
    fc = model.fc
    model.fc = torch.nn.Identity()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=torch.cuda.is_available())
    feats, probs, ys = [], [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        f = model(x)
        logits = fc(f)
        feats.append(f.cpu().numpy().astype(np.float16))
        probs.append(torch.softmax(logits, 1).cpu().numpy().astype(np.float16))
        ys.append(y.numpy().astype(np.int16))
        if i % 25 == 0:
            print(f"ImageNet feature extraction batch {i}/{len(loader)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=np.concatenate(feats), probs=np.concatenate(probs), y=np.concatenate(ys))


def build_imagenet_prediction_bundle(seed: int, imagenet_root: str, cache_dir: str):
    if not imagenet_root:
        raise ValueError("--imagenet-root is required for ImageNet.")
    cache = Path(cache_dir) / "imagenet_val_resnet50_v2_features_probs_float16.npz"
    if not cache.exists():
        _extract_imagenet_cache(imagenet_root, cache)
    else:
        print(f"Reusing ImageNet feature cache: {cache}")
    z = np.load(cache, allow_pickle=False)
    features = z["features"].astype(np.float32)
    probs = z["probs"].astype(np.float32)
    y = z["y"].astype(int)

    # A plain ImageFolder uses lexicographic folder indices.  Standard ILSVRC
    # wnid folders normally align with the canonical ImageNet-1K class order,
    # but we never silently assume that a user's copy is arranged correctly.
    # A severe label-index mismatch would make all conformal results meaningless,
    # so fail early if the frozen pretrained model is nowhere near a plausible
    # ImageNet accuracy.
    overall_acc = float(np.mean(probs.argmax(1) == y))
    print(f"ImageNet label/index sanity top1={overall_acc:.4f}")
    if overall_acc < 0.50:
        raise RuntimeError(
            "ImageNet class-index sanity check failed (top-1 < 50%). "
            "Your validation subfolder ordering/labels are likely not aligned "
            "with canonical ImageNet-1K class indices. Do not use these results."
        )

    # 30/30/40 STRUCTURE/CAL/TEST from ImageNet validation. The pretrained model is frozen.
    idx = np.arange(len(y))
    structure, rest = train_test_split(idx, train_size=0.30, random_state=seed, stratify=y)
    cal, test = train_test_split(rest, train_size=3/7, random_state=seed + 1, stratify=y[rest])
    pred = probs.argmax(1)
    acc_test = float(np.mean(pred[test] == y[test]))
    return {
        "dataset": "imagenet", "backbone": "resnet50_imagenet1k_v2", "backbone_accuracy": acc_test,
        "probs_s": probs[structure], "probs_c": probs[cal], "probs_t": probs[test],
        "phi_s": features[structure], "phi_c": features[cal], "phi_t": features[test],
        "y_s": y[structure], "y_c": y[cal], "y_t": y[test],
        "split_sizes": {"train": 0, "val": 0, "structure": len(structure), "cal": len(cal), "test": len(test)},
    }


def finalize_brc_bundle(pred_bundle, seed: int, score_type: str, min_structure_support=100):
    probs_s, probs_c, probs_t = pred_bundle["probs_s"], pred_bundle["probs_c"], pred_bundle["probs_t"]
    phi_s, phi_c, phi_t = maybe_pca_phi(pred_bundle["phi_s"], pred_bundle["phi_c"], pred_bundle["phi_t"], seed)
    u1s, u2s = classification_uncertainty(probs_s)
    u1c, u2c = classification_uncertainty(probs_c)
    u1t, u2t = classification_uncertainty(probs_t)
    sig_s = np.concatenate([phi_s, np.stack([u1s, u2s], 1)], 1)
    sig_c = np.concatenate([phi_c, np.stack([u1c, u2c], 1)], 1)
    sig_t = np.concatenate([phi_t, np.stack([u1t, u2t], 1)], 1)
    sig_s, sig_c, sig_t, _ = standardize_three(sig_s, sig_c, sig_t)
    km, ktable = learn_patterns(sig_s, min_structure_support, seed)
    K = int(km.n_clusters)
    pat_s, pat_c, pat_t = km.predict(sig_s), km.predict(sig_c), km.predict(sig_t)
    U_s = pattern_relative_U(u1s, u2s, pat_s, u1s, u2s, pat_s, K)
    U_c = pattern_relative_U(u1s, u2s, pat_s, u1c, u2c, pat_c, K)
    U_t = pattern_relative_U(u1s, u2s, pat_s, u1t, u2t, pat_t, K)
    Ug_s = global_U(u1s, u2s, u1s, u2s)
    Ug_t = global_U(u1s, u2s, u1t, u2t)
    from brc_core import lac_true_scores, aps_true_scores
    if score_type == "lac":
        score_s = lac_true_scores(probs_s, pred_bundle["y_s"])
        score_c = lac_true_scores(probs_c, pred_bundle["y_c"])
    elif score_type == "aps":
        score_s = aps_true_scores(probs_s, pred_bundle["y_s"])
        score_c = aps_true_scores(probs_c, pred_bundle["y_c"])
    else:
        raise ValueError(score_type)
    out = dict(pred_bundle)
    out.update({
        "seed": int(seed), "score_type": score_type, "K": K,
        "score_structure": score_s, "score_cal": score_c,
        "pat_s": pat_s.astype(int), "pat_c": pat_c.astype(int), "pat_t": pat_t.astype(int),
        "U_s": U_s, "U_c": U_c, "U_t": U_t, "Ug_s": Ug_s, "Ug_t": Ug_t,
        "k_selection": ktable,
    })
    return out
