"""
Significance tests for the performance and shift tables.

Design notes
------------
All tests use a unified paired bootstrap approach:

* For per-pair P-Acc: bootstrap the mean of per-pair differences
  ``cond_correct - base_correct``. For natural conditions ``cond_correct``
  is averaged across variants → values lie in [0, 1].

* For F1: bootstrap the difference F1(cond) - F1(base) by resampling
  claim indices. F1 is non-decomposable so the difference cannot be
  expressed as a mean of per-claim terms; resampling indices preserves
  the pairing between base and condition predictions.

* For Δp_sup / Δp_ref: bootstrap the mean of per-pair shift values that
  ``compute_shifts`` produces (already averaged across variants by
  ``average_shifts``).

All three follow the same logic: resample with replacement N=1,000 times,
compute the statistic on each resample, report a two-sided p-value as
  2 * min(P(stat >= 0), P(stat <= 0)).

Multiplicity is handled by Benjamini-Hochberg FDR over all populated
cells in each table.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def bh_fdr(pvals: Iterable[float]) -> list[float]:
    """
    Benjamini-Hochberg-adjusted q-values for a list of p-values.

    NaN inputs are passed through unchanged. q-values are clipped to [0, 1]
    and monotonised from the top so the sequence is non-decreasing in p.
    """
    p_arr = np.array(list(pvals), dtype=float)
    out = np.full_like(p_arr, np.nan, dtype=float)
    valid_mask = ~np.isnan(p_arr)
    p_valid = p_arr[valid_mask]
    m = p_valid.size
    if m == 0:
        return out.tolist()

    order = np.argsort(p_valid)
    ranked = p_valid[order]
    adj = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity walking from largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    q_valid = np.empty(m)
    q_valid[order] = adj
    out[valid_mask] = q_valid
    return out.tolist()


def stars(q: float | None) -> str:
    """Map a q-value to a star string ('', '*', '**', '***')."""
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Unified bootstrap: mean of differences
# ---------------------------------------------------------------------------


def bootstrap_mean_p(
    diffs: Iterable[float],
    n_iter: int = 1_000,
    seed: int = 0,
    chunk: int = 200,
) -> float:
    """
    Two-sided bootstrap p-value that mean(diffs) != 0.

    Resamples ``diffs`` with replacement ``n_iter`` times and computes the
    mean on each resample.  The p-value is:
        2 * min(P(mean >= 0), P(mean <= 0))

    Resampling is done in row-chunks of ``chunk`` to keep peak memory low
    (avoids allocating an n_iter × n matrix all at once).

    Returns NaN if fewer than 2 finite observations are present.
    """
    d = np.asarray(list(diffs), dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    n_pos = 0
    n_neg = 0
    done = 0
    while done < n_iter:
        rows = min(chunk, n_iter - done)
        idx = rng.integers(0, n, size=(rows, n))
        means = d[idx].mean(axis=1)
        n_pos += int(np.sum(means >= 0))
        n_neg += int(np.sum(means <= 0))
        done += rows

    p_pos = n_pos / n_iter
    p_neg = n_neg / n_iter
    return float(min(1.0, 2.0 * min(p_pos, p_neg)))


def bootstrap_mean_ci(
    diffs: Iterable[float],
    n_iter: int = 1_000,
    seed: int = 0,
    alpha: float = 0.05,
    chunk: int = 200,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for the mean of ``diffs``.

    Uses the *same* resampling scheme as ``bootstrap_mean_p`` (resample the
    values with replacement, ``n_iter`` times, same seed/chunking) so the CI and
    the two-sided p-value are mutually consistent. Returns
    ``(mean, lo, hi)`` where ``[lo, hi]`` is the central ``1 - alpha`` interval.

    Returns ``(mean, nan, nan)`` if fewer than 2 finite observations are present.
    """
    d = np.asarray(list(diffs), dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        m = float(d.mean()) if n else float("nan")
        return m, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.empty(n_iter, dtype=float)
    done = 0
    while done < n_iter:
        rows = min(chunk, n_iter - done)
        idx = rng.integers(0, n, size=(rows, n))
        means[done:done + rows] = d[idx].mean(axis=1)
        done += rows

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Bootstrap p-value for F1 differences (index-resampling)
# ---------------------------------------------------------------------------


def _f1_from_arrays(pred: np.ndarray, true: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p == 0 or denom_r == 0:
        return 0.0
    p = tp / denom_p
    r = tp / denom_r
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def paired_bootstrap_f1_diff_p(
    base_pred: np.ndarray,
    base_true: np.ndarray,
    cond_preds: list[np.ndarray],
    n_iter: int = 1_000,
    seed: int = 0,
) -> float:
    """
    Two-sided bootstrap p-value for the paired difference F1(cond) - F1(base).

    Resamples claim indices (not per-claim diffs) so that F1, which is
    non-decomposable, is recomputed correctly on each resample. Both base
    and condition use the same resampled indices (true pairing). For natural
    conditions, F1 is averaged across variants per resample.

    Returns NaN if fewer than 5 claims are available.
    """
    n = base_pred.size
    if n < 5 or not cond_preds:
        return float("nan")
    rng = np.random.default_rng(seed)
    v = len(cond_preds)
    diffs = np.empty(n_iter, dtype=float)
    for it in range(n_iter):
        idx = rng.integers(0, n, n)
        bp = base_pred[idx]
        bt = base_true[idx]
        f1_base = _f1_from_arrays(bp, bt)
        f1_cond_sum = sum(_f1_from_arrays(cp[idx], bt) for cp in cond_preds)
        diffs[it] = f1_cond_sum / v - f1_base

    p_pos = float(np.sum(diffs >= 0)) / n_iter
    p_neg = float(np.sum(diffs <= 0)) / n_iter
    return float(min(1.0, 2.0 * min(p_pos, p_neg)))


# ---------------------------------------------------------------------------
# Builders for per-pair / per-claim vectors used by the tests
# ---------------------------------------------------------------------------


def pair_correctness(preds: dict, complete_pairs: dict) -> dict[str, int]:
    """
    Per-pair Bernoulli correctness (1 if both items correctly classified).
    Pairs missing predictions on either side are excluded.
    """
    out: dict[str, int] = {}
    for pid, ids in complete_pairs.items():
        sup_id, ref_id = ids["Supported"], ids["Refuted"]
        if sup_id not in preds or ref_id not in preds:
            continue
        ok = (
            preds[sup_id]["pred_label"] == "Supported"
            and preds[ref_id]["pred_label"] == "Refuted"
        )
        out[pid] = 1 if ok else 0
    return out


def paired_pair_acc_diffs(
    base_preds: dict,
    variant_preds_list: list[dict],
    complete_pairs: dict,
) -> np.ndarray:
    """
    Per-pair (cond_correct - base_correct) differences, where ``cond_correct``
    is averaged across variants (in [0, 1]) and ``base_correct`` is 0/1.
    Only pairs present in every variant and in base are used.
    """
    if not variant_preds_list:
        return np.array([])
    base_corr = pair_correctness(base_preds, complete_pairs)
    per_variant = [pair_correctness(vp, complete_pairs) for vp in variant_preds_list]
    common = set(base_corr)
    for pc in per_variant:
        common &= set(pc)
    diffs = [
        float(np.mean([pc[pid] for pc in per_variant])) - base_corr[pid]
        for pid in common
    ]
    return np.asarray(diffs, dtype=float)


def claim_label_arrays(
    preds: dict, claim_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-claim (pred=1 if Supported, true=1 if Supported) over a fixed id order."""
    pred = np.array(
        [1 if preds.get(cid, {}).get("pred_label") == "Supported" else 0
         for cid in claim_ids],
        dtype=np.int8,
    )
    true = np.array(
        [1 if preds.get(cid, {}).get("label") == "Supported" else 0
         for cid in claim_ids],
        dtype=np.int8,
    )
    return pred, true


def common_claim_ids(base_preds: dict, variant_preds_list: list[dict]) -> list[str]:
    """Claim ids present in base and in every variant — required for paired bootstrap."""
    if not variant_preds_list:
        return []
    common = set(base_preds)
    for vp in variant_preds_list:
        common &= set(vp)
    return sorted(common)


# ---------------------------------------------------------------------------
# Top-level per-cell test wrappers
# ---------------------------------------------------------------------------


def pair_acc_pvalue(
    base_preds: dict,
    variant_preds_list: list[dict],
    complete_pairs: dict,
    n_iter: int = 1_000,
    seed: int = 0,
) -> tuple[float, int]:
    """Bootstrap p-value for paired pair-accuracy difference + n_pairs used."""
    diffs = paired_pair_acc_diffs(base_preds, variant_preds_list, complete_pairs)
    if diffs.size == 0:
        return float("nan"), 0
    return bootstrap_mean_p(diffs, n_iter=n_iter, seed=seed), int(diffs.size)


def f1_pvalue(
    base_preds: dict,
    variant_preds_list: list[dict],
    n_iter: int = 1_000,
    seed: int = 0,
) -> tuple[float, int]:
    """Bootstrap p-value for F1 difference (index-resampling) + n_claims used."""
    ids = common_claim_ids(base_preds, variant_preds_list)
    if not ids:
        return float("nan"), 0
    base_pred, base_true = claim_label_arrays(base_preds, ids)
    variant_pred_arrs = [claim_label_arrays(vp, ids)[0] for vp in variant_preds_list]
    p = paired_bootstrap_f1_diff_p(
        base_pred, base_true, variant_pred_arrs, n_iter=n_iter, seed=seed,
    )
    return p, len(ids)


def shift_pvalue(
    shifts: list[dict],
    key: str,
    n_iter: int = 1_000,
    seed: int = 0,
) -> tuple[float, int]:
    """
    Bootstrap p-value (two-sided) on per-pair shift values.

    ``key`` is 'dp_sup' or 'dp_ref'. ``shifts`` is the list produced by
    ``compute_shifts`` (or ``average_shifts`` for natural conditions).
    """
    if not shifts:
        return float("nan"), 0
    diffs = np.array([s[key] for s in shifts if s.get(key) is not None], dtype=float)
    if diffs.size == 0:
        return float("nan"), 0
    return bootstrap_mean_p(diffs, n_iter=n_iter, seed=seed), int(diffs.size)
