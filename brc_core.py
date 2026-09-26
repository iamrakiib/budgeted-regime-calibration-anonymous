from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ALPHA = 0.10
TARGET = 1.0 - ALPHA
B_CHOICES = (1, 2, 3, 4, 6)
BUDGET_MULTIPLIERS = (2, 3, 4)
UNIFORM_BINS = (2, 3, 4, 6)
EVAL_BINS = 10
OOF_FOLDS = 5
K_CANDIDATES = (2, 3, 5, 7, 10)
ARI_THRESHOLD = 0.70
MIN_STRUCTURE_SUPPORT = 100
MIN_CAL_PATTERN = 30
MIN_TRAIN_CELL = 80
RANDOM_DRAWS = 20
EPS = 1e-12
REPORTING_SEEDS = (42, 999, 0, 123, 12345)


def conformal_quantile(scores, alpha=None) -> float:
    """Finite-sample split-conformal quantile. If alpha is omitted, use current module ALPHA.

    Making alpha dynamic is required only for the appendix alpha-sensitivity runner;
    default behavior remains alpha=0.10.
    """
    if alpha is None:
        alpha = ALPHA
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0:
        raise RuntimeError("Cannot calibrate from zero scores.")
    j = int(np.ceil((len(s) + 1) * (1.0 - float(alpha))))
    if j > len(s):
        return float("inf")
    return float(np.sort(s)[j - 1])


def standardize_three(structure, cal, test):
    scaler = StandardScaler().fit(structure)
    return (
        scaler.transform(structure).astype(np.float32),
        scaler.transform(cal).astype(np.float32),
        scaler.transform(test).astype(np.float32),
        scaler,
    )


def classification_uncertainty(probs):
    probs = np.asarray(probs, dtype=np.float64)
    p = np.clip(probs, EPS, 1.0)
    ent = -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
    sorted_probs = -np.sort(-probs, axis=1)
    inv_margin = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
    return ent.astype(np.float32), inv_margin.astype(np.float32)


def lac_true_scores(probs, y):
    """LAC true-label scores: 1 - p_y."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return (1.0 - probs[np.arange(len(y)), y]).astype(np.float32)


def lac_score_matrix(probs):
    """All-candidate LAC score matrix."""
    return (1.0 - np.asarray(probs, dtype=np.float64)).astype(np.float32)


def lac_sets(probs, q):
    probs = np.asarray(probs, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 0:
        q = np.full(len(probs), float(q))
    return lac_score_matrix(probs) <= q[:, None]


def aps_true_scores(probs, y):
    """Deterministic nonrandomized APS using the strict-before-candidate convention.

    For a candidate class y with rank r in descending predicted probability,
    the score is the cumulative probability mass of classes ranked strictly
    before y.  This is the APS convention used by the paper's image BRC runs.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    before = np.cumsum(sorted_p, axis=1) - sorted_p
    inv = np.empty_like(order)
    inv[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
    rank = inv[np.arange(len(y)), y]
    return before[np.arange(len(y)), rank].astype(np.float32)


def aps_score_matrix_strict(probs):
    """All-candidate deterministic APS scores, strict before candidate."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    before_sorted = np.cumsum(sorted_p, axis=1) - sorted_p
    out = np.empty_like(before_sorted)
    out[np.arange(len(probs))[:, None], order] = before_sorted
    return out.astype(np.float32)


# Explicit alias used by the constructor-control runner.
aps_true_scores_strict = aps_true_scores


def aps_sets(probs, q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 0:
        q = np.full(len(probs), float(q))
    return aps_score_matrix_strict(probs) <= q[:, None]


def kmeans_cv_ari(X, K: int, seed: int, n_folds: int = 5):
    reference = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(X)
    ref_labels = reference.labels_
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aris = []
    for tr, va in kf.split(X):
        km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(X[tr])
        aris.append(adjusted_rand_score(ref_labels[va], km.predict(X[va])))
    return float(np.mean(aris))


def learn_patterns(signature_structure, min_structure_support: int, seed: int):
    rows, feasible = [], []
    for K in K_CANDIDATES:
        ari = kmeans_cv_ari(signature_structure, K, seed)
        km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(signature_structure)
        counts = np.bincount(km.labels_, minlength=K)
        min_count = int(counts.min())
        support_ok = min_count >= int(min_structure_support)
        ok = ari >= ARI_THRESHOLD and support_ok
        rows.append({
            "K": int(K),
            "CV_ARI": float(ari),
            "min_structure_count": min_count,
            "support_ok": bool(support_ok),
            "feasible": bool(ok),
        })
        if ok:
            feasible.append((ari, min_count, -K, km))
    table = pd.DataFrame(rows)
    if not feasible:
        raise RuntimeError("No K satisfies the frozen ARI/support rule.\n" + table.to_string(index=False))
    feasible.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
    return feasible[0][3], table


def percentile_rank(target_values, reference_values):
    ref = np.sort(np.asarray(reference_values, dtype=np.float64))
    tar = np.asarray(target_values, dtype=np.float64)
    return np.searchsorted(ref, tar, side="right").astype(np.float64) / len(ref)


def pattern_relative_U(u1s, u2s, pat_s, u1t, u2t, pat_t, K, w1=0.5, w2=0.5):
    out = np.zeros(len(u1t), dtype=np.float64)
    for k in range(int(K)):
        ri = np.where(np.asarray(pat_s) == k)[0]
        ti = np.where(np.asarray(pat_t) == k)[0]
        if not len(ti):
            continue
        p1 = percentile_rank(np.asarray(u1t)[ti], np.asarray(u1s)[ri])
        p2 = percentile_rank(np.asarray(u2t)[ti], np.asarray(u2s)[ri])
        den = float(w1) + float(w2)
        if den <= 0:
            raise ValueError("Uncertainty weights must have positive sum.")
        out[ti] = (float(w1) * p1 + float(w2) * p2) / den
    return out.astype(np.float32)


def global_U(u1s, u2s, u1t, u2t, w1=0.5, w2=0.5):
    p1 = percentile_rank(u1t, u1s)
    p2 = percentile_rank(u2t, u2s)
    den = float(w1) + float(w2)
    if den <= 0:
        raise ValueError("Uncertainty weights must have positive sum.")
    return ((float(w1) * p1 + float(w2) * p2) / den).astype(np.float32)


def mixed_edges(U_s, pat_s, K: int, allocation: Sequence[int]):
    U_s = np.asarray(U_s, dtype=float)
    pat_s = np.asarray(pat_s, dtype=int)
    edges = {}
    for k in range(int(K)):
        b = int(allocation[k])
        vals = U_s[pat_s == k]
        if b <= 1:
            edges[k] = np.asarray([], dtype=float)
        else:
            edges[k] = np.quantile(vals, np.arange(1, b, dtype=float) / float(b))
    return edges


def assign_mixed(U, pat, edges):
    U = np.asarray(U, dtype=float)
    pat = np.asarray(pat, dtype=int)
    out = np.empty(len(U), dtype=int)
    for k in np.unique(pat):
        ix = np.where(pat == int(k))[0]
        out[ix] = int(k) * 100 + np.digitize(U[ix], edges[int(k)])
    return out


def global_edges(U_s, bins=EVAL_BINS):
    return np.quantile(np.asarray(U_s, dtype=float), np.arange(1, bins) / float(bins))


def assign_global(U, edges):
    return np.digitize(np.asarray(U, dtype=float), np.asarray(edges, dtype=float)).astype(int)


def group_stats(covered, groups, min_n: int):
    covered = np.asarray(covered, dtype=bool)
    groups = np.asarray(groups, dtype=int)
    rows = []
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        if len(ix) < int(min_n):
            continue
        cov = float(np.mean(covered[ix]))
        rows.append({"group": int(g), "n": int(len(ix)), "coverage": cov, "abs_gap": abs(cov - TARGET)})
    if not rows:
        return {
            "worst": np.nan,
            "spread": np.nan,
            "max_gap": np.nan,
            "weighted_gap": np.nan,
            "n_groups": 0,
        }, pd.DataFrame()
    df = pd.DataFrame(rows)
    return {
        "worst": float(df.coverage.min()),
        "spread": float(df.coverage.max() - df.coverage.min()),
        "max_gap": float(df.abs_gap.max()),
        "weighted_gap": float(np.average(df.abs_gap, weights=df.n)),
        "n_groups": int(len(df)),
    }, df


def class_coverage_stats(covered, y):
    covered = np.asarray(covered, dtype=bool)
    y = np.asarray(y, dtype=int)
    vals = []
    rows = []
    for c in np.unique(y):
        ix = np.where(y == c)[0]
        cov = float(np.mean(covered[ix]))
        vals.append(cov)
        rows.append({"class": int(c), "n": int(len(ix)), "coverage": cov})
    vals = np.sort(np.asarray(vals, dtype=float))
    out = {
        "worst_class_coverage": float(vals[0]),
        "bottom5_class_coverage": float(vals[:5].mean()) if len(vals) >= 5 else np.nan,
        "bottom10_class_coverage": float(vals[:10].mean()) if len(vals) >= 10 else np.nan,
    }
    return out, pd.DataFrame(rows)


def balanced_fold_ids(eval_groups, seed: int):
    eval_groups = np.asarray(eval_groups, dtype=int)
    rng = np.random.default_rng(int(seed))
    fold_id = np.empty(len(eval_groups), dtype=int)
    for g in np.unique(eval_groups):
        ix = np.where(eval_groups == g)[0]
        perm = rng.permutation(len(ix))
        local = np.empty(len(ix), dtype=int)
        local[perm] = np.arange(len(ix)) % OOF_FOLDS
        fold_id[ix] = local
    return fold_id


def precompute_local_costs(scores_s, U_s, pat_s, K: int, min_eval_cell: int, seed: int, support_safe: bool = False):
    scores_s = np.asarray(scores_s, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    pat_s = np.asarray(pat_s, dtype=int)

    eval_edges = mixed_edges(U_s, pat_s, K, [EVAL_BINS] * K)
    eval_groups = assign_mixed(U_s, pat_s, eval_edges)
    fold_id = balanced_fold_ids(eval_groups, int(seed) + 9173)

    # Parent/global thresholds are allocation-independent, so compute them once.
    fold_q_global = {}
    fold_q_pattern = {}
    for f in range(OOF_FOLDS):
        tr = np.where(fold_id != f)[0]
        qg = conformal_quantile(scores_s[tr])
        fold_q_global[f] = qg
        fold_q_pattern[f] = {}
        for k in range(K):
            ix = tr[pat_s[tr] == k]
            fold_q_pattern[f][k] = conformal_quantile(scores_s[ix]) if len(ix) >= MIN_TRAIN_CELL else qg

    local = {k: {} for k in range(K)}
    for k in range(K):
        sample_k = np.where(pat_s == k)[0]
        for b in B_CHOICES:
            if b <= 1:
                e = np.asarray([], dtype=float)
            else:
                e = np.quantile(U_s[sample_k], np.arange(1, b) / float(b))
            cell_all = np.digitize(U_s[sample_k], e)
            covered = np.zeros(len(sample_k), dtype=bool)

            for f in range(OOF_FOLDS):
                local_tr_mask = fold_id[sample_k] != f
                local_va_mask = fold_id[sample_k] == f
                for cell in range(int(b)):
                    it_local = np.where(local_tr_mask & (cell_all == cell))[0]
                    iv_local = np.where(local_va_mask & (cell_all == cell))[0]
                    if not len(iv_local):
                        continue
                    if len(it_local) >= MIN_TRAIN_CELL:
                        q = conformal_quantile(scores_s[sample_k[it_local]])
                    else:
                        q = fold_q_pattern[f][k]
                    covered[iv_local] = scores_s[sample_k[iv_local]] <= q

            local_eval = eval_groups[sample_k]
            stats, df = group_stats(covered, local_eval, min_eval_cell)
            audit_fallback = False
            if (not len(df)) and support_safe:
                # Sensitivity-only support-safe audit fallback: if a forced-K regime
                # is too small for the frozen U10/min-n audit, evaluate that regime
                # at its parent level rather than making the optimization undefined.
                cov = float(covered.mean()) if len(covered) else np.nan
                gap = abs(cov - TARGET) if np.isfinite(cov) else np.inf
                weighted_num = float(len(covered) * gap) if np.isfinite(gap) else np.inf
                worst = cov
                n_eval = int(len(covered))
                audit_fallback = True
            else:
                gap = float(stats["max_gap"])
                weighted_num = float(np.sum(df.n * df.abs_gap)) if len(df) else np.inf
                worst = float(stats["worst"])
                n_eval = int(df.n.sum()) if len(df) else 0
            local[k][int(b)] = {
                "max_gap": gap,
                "weighted_numerator": weighted_num,
                "worst": worst,
                "n_eval": n_eval,
                "oof_marginal": float(covered.mean()),
                "audit_support_fallback": bool(audit_fallback),
            }
    return local, eval_groups


def _lex_better(a, b, tol=1e-12):
    if b is None:
        return True
    for i in (0, 1):
        av, bv = float(a[i]), float(b[i])
        if np.isfinite(av) and not np.isfinite(bv):
            return True
        if not np.isfinite(av) and np.isfinite(bv):
            return False
        if np.isfinite(av) and np.isfinite(bv):
            if av < bv - tol:
                return True
            if av > bv + tol:
                return False
    return tuple(a[2]) < tuple(b[2])


def solve_budget_dp(local, K: int, budget: int):
    """Exact DP for the lexicographic BRC objective.

    Objective:
      1. minimize the maximum common-grid absolute coverage gap;
      2. among those allocations, minimize the weighted absolute gap sum;
      3. deterministic lexicographic allocation tie-break.

    A naive DP that keeps only one partial state for each (k, used_cells) is
    not generally exact for a max-then-sum objective: a partial state with a
    slightly worse current max can later become preferable after the future
    max dominates both paths.  We therefore solve the primary minimax problem
    by enumerating the *finite set of attainable local max-gap thresholds*,
    and inside each threshold run an additive DP for the secondary objective.
    The first feasible threshold is the globally minimal max-gap.  This is
    exact and remains polynomial in K, budget and the finite candidate set.
    """
    tol = 1e-12
    thresholds = sorted({
        float(local[k][int(b)]["max_gap"])
        for k in range(int(K))
        for b in B_CHOICES
        if np.isfinite(float(local[k][int(b)]["max_gap"]))
    })
    if not thresholds:
        raise RuntimeError("No finite BRC local objective values are available.")

    for threshold in thresholds:
        # dp[used_cells] = (weighted_sum, allocation_tuple)
        dp = {0: (0.0, tuple())}
        for k in range(int(K)):
            new = {}
            for used, state in dp.items():
                for b in B_CHOICES:
                    item = local[k][int(b)]
                    gap = float(item["max_gap"])
                    if (not np.isfinite(gap)) or gap > threshold + tol:
                        continue
                    used2 = int(used) + int(b)
                    if used2 > int(budget):
                        continue
                    cand = (
                        float(state[0]) + float(item["weighted_numerator"]),
                        tuple(state[1]) + (int(b),),
                    )
                    prev = new.get(used2)
                    if prev is None:
                        new[used2] = cand
                    else:
                        if cand[0] < prev[0] - tol or (
                            abs(cand[0] - prev[0]) <= tol
                            and tuple(cand[1]) < tuple(prev[1])
                        ):
                            new[used2] = cand
            dp = new
            if not dp:
                break

        if int(budget) in dp:
            weighted, allocation = dp[int(budget)]
            actual_max = max(
                float(local[k][int(allocation[k])]["max_gap"])
                for k in range(int(K))
            )
            return (float(actual_max), float(weighted), tuple(allocation))

    raise RuntimeError(f"No legal allocation for K={K}, budget={budget}")


def solve_budget_exhaustive(local, K: int, budget: int):
    """Audit-only exhaustive solver for small synthetic/verification cases."""
    import itertools
    best = None
    for a in itertools.product(B_CHOICES, repeat=int(K)):
        if sum(a) != int(budget):
            continue
        max_gap = max(float(local[k][int(a[k])]["max_gap"]) for k in range(int(K)))
        weighted = sum(float(local[k][int(a[k])]["weighted_numerator"]) for k in range(int(K)))
        state = (float(max_gap), float(weighted), tuple(int(x) for x in a))
        if _lex_better(state, best):
            best = state
    if best is None:
        raise RuntimeError(f"No exhaustive allocation for K={K}, budget={budget}")
    return best


def select_global_B(local, K: int):
    rows = []
    best = None
    for b in B_CHOICES:
        max_gap = max(local[k][b]["max_gap"] for k in range(K))
        weighted = sum(local[k][b]["weighted_numerator"] for k in range(K))
        row = {"B": int(b), "max_gap": float(max_gap), "weighted_sum": float(weighted)}
        rows.append(row)
        cand = (float(max_gap), float(weighted), (int(b),))
        if _lex_better(cand, best):
            best = cand
    return int(best[2][0]), pd.DataFrame(rows)


@lru_cache(None)
def _count_legal_completions(n_left: int, remaining: int) -> int:
    if n_left == 0:
        return int(remaining == 0)
    total = 0
    for b in B_CHOICES:
        if b <= remaining:
            total += _count_legal_completions(n_left - 1, remaining - b)
    return total


def sample_random_legal_allocations(K: int, budget: int, n_draws: int, seed: int):
    total = _count_legal_completions(K, budget)
    if total <= 0:
        return []
    target = min(int(n_draws), int(total))
    rng = np.random.default_rng(seed)
    draws = set()
    while len(draws) < target:
        remaining = int(budget)
        a = []
        for pos in range(K):
            n_left = K - pos - 1
            vals, weights = [], []
            for b in B_CHOICES:
                if b > remaining:
                    continue
                c = _count_legal_completions(n_left, remaining - b)
                if c > 0:
                    vals.append(int(b)); weights.append(int(c))
            probs = np.asarray(weights, dtype=float); probs /= probs.sum()
            chosen = int(rng.choice(vals, p=probs))
            a.append(chosen); remaining -= chosen
        draws.add(tuple(a))
    return sorted(draws)


def calibrate_allocation(scores_c, pat_c, U_c, pat_t, U_t, U_s, pat_s, K: int,
                          allocation: Sequence[int], min_cal_leaf: int):
    scores_c = np.asarray(scores_c, dtype=float)
    pat_c = np.asarray(pat_c, dtype=int)
    q_global = conformal_quantile(scores_c)
    q_pattern = {}
    for k in range(K):
        ix = np.where(pat_c == k)[0]
        q_pattern[k] = conformal_quantile(scores_c[ix]) if len(ix) >= MIN_CAL_PATTERN else q_global

    edges = mixed_edges(U_s, pat_s, K, allocation)
    cells_c = assign_mixed(U_c, pat_c, edges)
    cells_t = assign_mixed(U_t, pat_t, edges)
    q_cell, support = {}, []
    for k in range(K):
        for b in range(int(allocation[k])):
            code = k * 100 + b
            ix = np.where(cells_c == code)[0]
            if len(ix) >= int(min_cal_leaf):
                q = conformal_quantile(scores_c[ix]); src = "cell"
            else:
                q = q_pattern[k]; src = "pattern_support_fallback"
            q_cell[code] = float(q)
            n_test = int(np.sum(cells_t == code))
            support.append({
                "pattern": int(k), "u_bin": int(b), "allocation_Bk": int(allocation[k]),
                "cell_code": int(code), "n_cal": int(len(ix)), "n_test": n_test,
                "q": float(q), "source": src,
            })
    qtest = np.asarray([q_cell.get(int(c), q_pattern[int(k)]) for c, k in zip(cells_t, pat_t)], dtype=float)
    support_df = pd.DataFrame(support)
    active = int(np.sum(support_df.source == "cell")) if len(support_df) else 0
    return qtest, support_df, active


def evaluate_classification(method: str, probs_t, y_t, qtest, score_type: str,
                            eval_pt, eval_gt, min_eval_cell: int, complexity_cells,
                            extra=None):
    if score_type == "lac":
        mask = lac_sets(probs_t, qtest)
    elif score_type == "aps":
        mask = aps_sets(probs_t, qtest)
    else:
        raise ValueError(score_type)
    y_t = np.asarray(y_t, dtype=int)
    covered = mask[np.arange(len(y_t)), y_t]
    sizes = mask.sum(axis=1)
    pst, pdf = group_stats(covered, eval_pt, min_eval_cell)
    gst, gdf = group_stats(covered, eval_gt, min_eval_cell)
    cst, cdf = class_coverage_stats(covered, y_t)
    row = {
        "method": method,
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
        "complexity_cells": complexity_cells,
    }
    if extra:
        row.update(extra)
    details = []
    if len(pdf):
        d = pdf.copy(); d["family"] = "pattern_U10"; d["method"] = method; details.append(d)
    if len(gdf):
        d = gdf.copy(); d["family"] = "global_U10"; d["method"] = method; details.append(d)
    if len(cdf):
        d = cdf.copy(); d["family"] = "class"; d["group"] = d.pop("class"); d["method"] = method; details.append(d)
    return row, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def mask_from_scores(score_test_all, qtest):
    """Threshold an all-candidate score matrix with per-example thresholds."""
    score_test_all = np.asarray(score_test_all, dtype=float)
    q = np.asarray(qtest, dtype=float)
    if q.ndim == 0:
        q = np.full(len(score_test_all), float(q))
    return score_test_all <= q[:, None]


def evaluate_mask(mask, y_t, eval_groups, min_eval_cell: int):
    """Evaluate a prediction-set mask on the common supported audit groups."""
    mask = np.asarray(mask, dtype=bool)
    y_t = np.asarray(y_t, dtype=int)
    covered = mask[np.arange(len(y_t)), y_t]
    sizes = mask.sum(axis=1)
    stats, detail = group_stats(covered, eval_groups, min_eval_cell)
    cstats, _ = class_coverage_stats(covered, y_t)
    row = {
        "marginal_coverage": float(covered.mean()),
        "wpc": float(stats["worst"]),
        "max_gap": float(stats["max_gap"]),
        "weighted_gap": float(stats["weighted_gap"]),
        "avg_set_size": float(sizes.mean()),
        "empty_rate": float((sizes == 0).mean()),
        **cstats,
    }
    return row, detail


def aggregate_results(df: pd.DataFrame):
    numeric = [
        "marginal_coverage", "avg_set_size", "empty_rate", "worst_eval_pattern_U",
        "spread_eval_pattern_U", "max_gap_eval_pattern_U", "weighted_gap_eval_pattern_U",
        "worst_eval_global_U", "worst_class_coverage", "bottom5_class_coverage",
        "bottom10_class_coverage", "complexity_cells", "backbone_accuracy",
    ]
    rows = []
    group_cols = ["dataset", "backbone", "score_type", "method"]
    for key, g in df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key)); row["n_seeds"] = int(g.seed.nunique())
        for m in numeric:
            if m not in g:
                continue
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            if len(vals):
                row[m + "_mean"] = float(vals.mean())
                row[m + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def demand_heterogeneity_table(local: Dict[int, Dict[int, dict]], K: int):
    rows = []
    for k in range(K):
        base = float(local[k][1]["max_gap"])
        for b in B_CHOICES:
            mg = float(local[k][b]["max_gap"])
            rows.append({
                "pattern": int(k), "B": int(b), "oof_max_gap": mg,
                "improvement_vs_B1": base - mg,
                "oof_worst": float(local[k][b]["worst"]),
                "oof_marginal": float(local[k][b]["oof_marginal"]),
                "audit_support_fallback": bool(local[k][b].get("audit_support_fallback", False)),
            })
    df = pd.DataFrame(rows)
    summary = []
    for b, g in df.groupby("B"):
        vals = g.improvement_vs_B1.to_numpy(float)
        summary.append({
            "B": int(b),
            "demand_gain_mean": float(vals.mean()),
            "demand_gain_std_across_regimes": float(vals.std(ddof=0)),
            "demand_gain_range_across_regimes": float(vals.max() - vals.min()),
        })
    return df, pd.DataFrame(summary)
