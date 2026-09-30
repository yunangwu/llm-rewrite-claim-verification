"""
Pure metric functions — no matplotlib, no file I/O.

All functions that consume predictions accept `pred_key` and `prob_key`
so the caller controls which eval mode (consistency / direct / self_reported)
is used. The default is consistency mode.

Data model
----------
pair_mapping:
    complete_pairs : {pair_id: {"Supported": claim_id, "Refuted": claim_id}}
    valid_ids      : set of claim_ids that belong to complete pairs

predictions:
    {claim_id: {"pred_label": str, "p_supported": float, "label": str}}
    (field names depend on which pred_key / prob_key was requested)

shifts:
    list of {pair_id, p_sup, p_ref, p_sup_new, p_ref_new, dp_sup, dp_ref}
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Rewrite claim validation
# ---------------------------------------------------------------------------


def validate_rewrite_claims(
    rewrite_items: list[dict],
    base_claims: dict[str, str],
    complete_pairs: dict,
    *,
    min_length: int = 10,
) -> tuple[set[str], dict]:
    """
    Check rewrite items for broken/empty/identical claims.

    A claim is considered broken if:
      - The ``claim`` field is missing, empty, or None
      - The claim is shorter than ``min_length`` characters after stripping

    If either member of a pair is broken, **both** claim IDs are removed so
    pair integrity is preserved.

    Parameters
    ----------
    rewrite_items : list of dicts from a rewrite JSON file
    base_claims   : {claim_id: original_claim} for comparison
    complete_pairs: pair mapping from ``build_pair_mapping``
    min_length    : minimum character length for a non-broken claim

    Returns
    -------
    clean_ids : set of claim_ids that passed validation (safe to use)
    stats     : dict with counts for reporting
    """
    broken_ids: set[str] = set()
    n_empty = 0
    n_too_short = 0

    for item in rewrite_items:
        cid   = item.get("claim_id")
        claim = item.get("claim")
        if not claim or not claim.strip():
            broken_ids.add(cid)
            n_empty += 1
        elif len(claim.strip()) < min_length:
            broken_ids.add(cid)
            n_too_short += 1

    # Propagate: if one member of a pair is broken, drop the whole pair.
    pairs_dropped = 0
    pair_propagated: set[str] = set()
    if complete_pairs:
        for pid, ids in complete_pairs.items():
            sup_id = ids.get("Supported")
            ref_id = ids.get("Refuted")
            if sup_id in broken_ids or ref_id in broken_ids:
                pair_propagated.update(cid for cid in ids.values() if cid not in broken_ids)
                pairs_dropped += 1
        broken_ids |= pair_propagated

    all_ids = {item["claim_id"] for item in rewrite_items if item.get("claim_id")}
    clean_ids = all_ids - broken_ids

    return clean_ids, {
        "n_total":        len(all_ids),
        "n_broken":       len(broken_ids),
        "n_empty":        n_empty,
        "n_too_short":    n_too_short,
        "n_pair_dropped": pairs_dropped,
        "clean_ids":      clean_ids,
    }


# ---------------------------------------------------------------------------
# Pair mapping (built from the base dataset)
# ---------------------------------------------------------------------------


def build_pair_mapping(data: list[dict]) -> tuple[dict, set]:
    """
    Returns:
        complete_pairs : {pair_id: {"Supported": cid, "Refuted": cid}}
        valid_ids      : set of claim_ids to use in downstream analysis

    For datasets without ``claim_id_pair`` (pairless), complete_pairs is empty
    and valid_ids contains every claim_id so that per-item metrics (F1, drift,
    strip chart) still run over the full dataset.
    """
    if not any("claim_id_pair" in entry for entry in data):
        return {}, {entry["claim_id"] for entry in data}

    pair_to_ids: dict[Any, dict[str, str]] = defaultdict(dict)
    for entry in data:
        cid      = entry["claim_id"]
        pair_id  = entry["claim_id_pair"]
        label    = entry["label"]
        pair_to_ids[pair_id][label] = cid

    complete_pairs = {
        pid: ids
        for pid, ids in pair_to_ids.items()
        if "Supported" in ids and "Refuted" in ids
    }
    valid_ids = {
        cid
        for ids in complete_pairs.values()
        for cid in ids.values()
    }
    return complete_pairs, valid_ids


# ---------------------------------------------------------------------------
# Load predictions from an inference output file
# ---------------------------------------------------------------------------


def load_predictions(
    path: Path | str,
    valid_ids: set,
    *,
    pred_key: str = "pred_label_consistency",
    prob_key: str = "p_supported_consistency",
) -> dict[str, dict]:
    """
    Load one inference output JSON and return:
        {claim_id: {"pred_label": str, "p_supported": float, "label": str}}
    Keys are normalised so analysis code is independent of mode.
    Only entries whose claim_id is in valid_ids are returned.
    """
    import json

    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    preds: dict[str, dict] = {}
    for entry in data:
        cid = entry.get("claim_id")
        if cid not in valid_ids:
            continue
        pred_val = entry.get(pred_key, "")
        prob_val = entry.get(prob_key)
        if prob_val is not None:
            try:
                prob_val = float(prob_val)
            except (TypeError, ValueError):
                prob_val = None
        preds[cid] = {
            "pred_label":  str(pred_val) if pred_val else "",
            "p_supported": prob_val,
            "label":       entry.get("label", ""),
        }
    return preds


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_performance(preds: dict, complete_pairs: dict) -> dict[str, float]:
    """
    Returns:
        f1        : macro-binary F1 (Supported = positive class)
        pair_acc  : fraction of pairs where both claims are correctly classified
    """
    y_true, y_pred = [], []
    for pred in preds.values():
        y_true.append(1 if pred["label"] == "Supported" else 0)
        y_pred.append(1 if pred["pred_label"] == "Supported" else 0)

    n = len(y_true)
    if n == 0:
        return {"f1": float("nan"), "pair_acc": float("nan")}

    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    n_pairs = n_correct = 0
    for ids in complete_pairs.values():
        sup_id, ref_id = ids["Supported"], ids["Refuted"]
        if sup_id not in preds or ref_id not in preds:
            continue
        n_pairs += 1
        if (preds[sup_id]["pred_label"] == "Supported"
                and preds[ref_id]["pred_label"] == "Refuted"):
            n_correct += 1

    return {
        "f1":       round(f1, 6),
        "pair_acc": round(n_correct / n_pairs, 6) if n_pairs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Shift analysis
# ---------------------------------------------------------------------------


def compute_shifts(
    base_preds: dict,
    rewrite_preds: dict,
    complete_pairs: dict,
) -> list[dict]:
    """
    Per-pair Δp(supported) between base and one rewrite condition.
    Returns list of shift dicts.
    """
    shifts = []
    for pid, ids in complete_pairs.items():
        sup_id, ref_id = ids["Supported"], ids["Refuted"]
        if not all(cid in base_preds and cid in rewrite_preds for cid in (sup_id, ref_id)):
            continue
        p_sup     = base_preds[sup_id]["p_supported"]
        p_ref     = base_preds[ref_id]["p_supported"]
        p_sup_new = rewrite_preds[sup_id]["p_supported"]
        p_ref_new = rewrite_preds[ref_id]["p_supported"]
        if any(v is None for v in (p_sup, p_ref, p_sup_new, p_ref_new)):
            continue
        shifts.append({
            "pair_id":   pid,
            "p_sup":     p_sup,
            "p_ref":     p_ref,
            "p_sup_new": p_sup_new,
            "p_ref_new": p_ref_new,
            "dp_sup":    p_sup_new - p_sup,
            "dp_ref":    p_ref_new - p_ref,
        })
    return shifts


def average_shifts(shifts_per_variant: list[list[dict]]) -> list[dict]:
    """Average per-pair shifts across multiple variants (natural rewrite groups)."""
    accum: dict[Any, dict] = defaultdict(lambda: {
        "dp_sup": [], "dp_ref": [], "p_sup_new": [], "p_ref_new": [],
        "p_sup": None, "p_ref": None,
    })
    for shifts in shifts_per_variant:
        for s in shifts:
            pid = s["pair_id"]
            accum[pid]["dp_sup"].append(s["dp_sup"])
            accum[pid]["dp_ref"].append(s["dp_ref"])
            accum[pid]["p_sup_new"].append(s["p_sup_new"])
            accum[pid]["p_ref_new"].append(s["p_ref_new"])
            accum[pid]["p_sup"] = s["p_sup"]
            accum[pid]["p_ref"] = s["p_ref"]
    result = []
    for pid, data in accum.items():
        if not data["dp_sup"]:
            continue
        result.append({
            "pair_id":   pid,
            "p_sup":     data["p_sup"],
            "p_ref":     data["p_ref"],
            "p_sup_new": float(np.mean(data["p_sup_new"])),
            "p_ref_new": float(np.mean(data["p_ref_new"])),
            "dp_sup":    float(np.mean(data["dp_sup"])),
            "dp_ref":    float(np.mean(data["dp_ref"])),
        })
    return result


def summarize_shifts(shifts: list[dict]) -> dict[str, float]:
    if not shifts:
        return {k: float("nan") for k in
                ("p_sup", "p_ref", "p_sup_new", "p_ref_new", "dp_sup", "dp_ref")}
    return {
        "p_sup":     float(np.mean([s["p_sup"]     for s in shifts])),
        "p_ref":     float(np.mean([s["p_ref"]     for s in shifts])),
        "p_sup_new": float(np.mean([s["p_sup_new"] for s in shifts])),
        "p_ref_new": float(np.mean([s["p_ref_new"] for s in shifts])),
        "dp_sup":    float(np.mean([s["dp_sup"]    for s in shifts])),
        "dp_ref":    float(np.mean([s["dp_ref"]    for s in shifts])),
    }


def compute_quadrant_proportions(shifts: list[dict]) -> dict[str, float]:
    """Fraction of pairs falling in each compass quadrant."""
    if not shifts:
        return {"optimistic": 0.0, "conservative": 0.0, "sharper": 0.0, "duller": 0.0}
    n = len(shifts)
    return {
        "optimistic":   sum(s["dp_sup"] > 0 and s["dp_ref"] > 0 for s in shifts) / n,
        "conservative": sum(s["dp_sup"] < 0 and s["dp_ref"] < 0 for s in shifts) / n,
        "sharper":      sum(s["dp_sup"] > 0 and s["dp_ref"] < 0 for s in shifts) / n,
        "duller":       sum(s["dp_sup"] < 0 and s["dp_ref"] > 0 for s in shifts) / n,
    }


# ---------------------------------------------------------------------------
# Per-claim Δp for strip chart
# ---------------------------------------------------------------------------


def compute_claim_deltas(
    base_preds: dict,
    rewrite_preds_list: list[dict],
    valid_ids: set,
) -> list[float]:
    """
    Average per-claim Δp(supported) over a list of rewrite prediction dicts.
    Used for strip chart — one value per claim.
    """
    base_p = {cid: pred["p_supported"] for cid, pred in base_preds.items()
              if pred["p_supported"] is not None}
    per_claim: dict[str, list[float]] = defaultdict(list)
    for preds in rewrite_preds_list:
        for cid, pred in preds.items():
            if cid in base_p and pred["p_supported"] is not None:
                per_claim[cid].append(pred["p_supported"] - base_p[cid])
    return [float(np.mean(dps)) for dps in per_claim.values() if dps]


# ---------------------------------------------------------------------------
# Label-group shifts (pairless datasets)
# ---------------------------------------------------------------------------


def compute_label_shifts(base_preds: dict, rewrite_preds: dict) -> list[dict]:
    """
    Per-item Δp(supported), grouped by label — used for datasets without pair structure.

    Mathematically equivalent to averaging compute_shifts over complete pairs when
    every item belongs to exactly one pair (the sum/mean commutes). For pairless
    datasets this is the only available estimate of the rewrite effect.

    Returns a list of dicts with keys: label, p_orig, dp.
    """
    shifts = []
    for cid, base in base_preds.items():
        rew = rewrite_preds.get(cid)
        if rew is None or base["p_supported"] is None or rew["p_supported"] is None:
            continue
        shifts.append({
            "label":  base["label"],
            "p_orig": base["p_supported"],
            "dp":     rew["p_supported"] - base["p_supported"],
        })
    return shifts


def summarize_label_shifts(shifts: list[dict]) -> dict[str, float]:
    """
    Summarize label-group shifts into the same key structure as summarize_shifts,
    so the shift table and console output work unchanged for pairless datasets.

        p_sup / p_ref     : mean original p(supported) for Supported / Refuted items
        dp_sup / dp_ref   : mean Δp for Supported / Refuted items
        p_sup_new / p_ref_new : p_sup + dp_sup  (derived)
    """
    nan = float("nan")
    if not shifts:
        return {"p_sup": nan, "p_ref": nan, "p_sup_new": nan,
                "p_ref_new": nan, "dp_sup": nan, "dp_ref": nan}

    sup = [s for s in shifts if s["label"] == "Supported"]
    ref = [s for s in shifts if s["label"] == "Refuted"]

    p_sup  = float(np.mean([s["p_orig"] for s in sup])) if sup else nan
    p_ref  = float(np.mean([s["p_orig"] for s in ref])) if ref else nan
    dp_sup = float(np.mean([s["dp"]     for s in sup])) if sup else nan
    dp_ref = float(np.mean([s["dp"]     for s in ref])) if ref else nan

    import math
    p_sup_new = (p_sup + dp_sup) if not (math.isnan(p_sup) or math.isnan(dp_sup)) else nan
    p_ref_new = (p_ref + dp_ref) if not (math.isnan(p_ref) or math.isnan(dp_ref)) else nan

    return {
        "p_sup":     p_sup,
        "p_ref":     p_ref,
        "p_sup_new": p_sup_new,
        "p_ref_new": p_ref_new,
        "dp_sup":    dp_sup,
        "dp_ref":    dp_ref,
    }


# ---------------------------------------------------------------------------
# Drift binning
# ---------------------------------------------------------------------------


def compute_binned_drift(
    p_orig: list[float],
    dp: list[float],
    num_bins: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin (p_orig, Δp) pairs and return (centers, mean_dp, counts)."""
    if len(p_orig) < 5:
        empty = np.array([])
        return empty, empty, empty
    po = np.array(p_orig)
    d  = np.array(dp)
    edges = np.linspace(0, 1, num_bins + 1)
    centers, means, counts = [], [], []
    for i in range(num_bins):
        mask = (po >= edges[i]) & (po <= edges[i + 1] if i == num_bins - 1 else po < edges[i + 1])
        if mask.sum() >= 1:
            centers.append((edges[i] + edges[i + 1]) / 2)
            means.append(d[mask].mean())
            counts.append(int(mask.sum()))
    return np.array(centers), np.array(means), np.array(counts)
