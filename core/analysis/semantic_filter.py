"""
Semantic-change robustness filter (rebuttal experiment).

Reviewer concern
----------------
A hedge/approximation placed *before/adjacent to* a precise element
("achieves 95%" -> "achieves approximately 95%") does not merely restyle the
claim, it weakens its factual content. If the CT / C_1 effect were driven by
such rewrites it would be a semantic artifact, not a style bias.

Approach (rule-based, narrow adjacency reading)
-----------------------------------------------
For each condition we compare every original claim with its rewrite and flag the
claim as *potentially semantically changed* iff the rewrite **introduces** a
hedge/approximation token (present in the rewrite, absent from the original) that
sits within ``window`` tokens of a *precise element*:

  Tier 1  numbers, %, decimals, units (mg, ms, %, ...), p-values, comparison
          operators (>, <, vs, than)
  Tier 2  rankings / superlatives / comparison words (best, outperforms, higher, ...)

(Method / dataset names — Tier 3 — are intentionally out of scope: hedges rarely
attach to proper nouns, and reliable detection needs NER/gazetteers.)

Claims with no such pattern form the **pure-style** set; the rest are **flagged**.
The mask is derived from the rewrite text only, so it is *model-agnostic* and
shared across every inference model. We then recompute the paper's shift metric
(signed Δp for Supported / Refuted claims) on the pure and flagged subsets. If
CT / C_1 keep strong, significant effects on the pure set, the style-bias claim
survives the reviewer's filter.

Recall-first design: false positives only shrink the pure set, so the rule flags
aggressively (any inserted hedge near any precise element) to keep the pure set
clean rather than maximal.

Usage (standalone — reads existing runs/, writes nothing unless --out-dir)
--------------------------------------------------------------------------
  python -m core.analysis.semantic_filter --dataset sciclaimeval --split dev test
  python -m core.analysis.semantic_filter --dataset sciclaimeval --split dev test \
      --conditions CT C_1 --per-model
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Hedge / approximation tokens whose *insertion* may weaken a precise claim.
# Superset of the conservative injection pool plus common natural-rewrite hedges.
HEDGE_LEXICON: set[str] = {
    # conservative injection pool (core/rewrite/inject.py)
    "acknowledges", "allegedly", "apparently", "approximately", "arguably",
    "basically", "broadly", "comparatively", "conceivably", "essentially",
    "generally", "hinting", "inadequately", "largely", "mainly", "marginally",
    "moderately", "mostly", "often", "ostensibly", "partially", "perhaps",
    "plausibly", "possibly", "potential", "presumably", "primarily",
    "purportedly", "relatively", "roughly", "seemingly", "slightly", "somewhat",
    "supposedly", "tentatively", "typically", "underexplored", "unexplored",
    "usually",
    # common approximation / degree / epistemic hedges from natural rewrites
    "about", "around", "nearly", "almost", "approximate", "circa", "some",
    "close", "upwards", "somewhat", "fairly", "quite", "rather", "considerably",
    "substantially", "notably", "appreciably", "may", "might", "could", "would",
    "appears", "appear", "seems", "seem", "suggests", "suggest", "indicates",
    "indicate", "tends", "tend", "likely", "potentially", "possible", "probable",
    "probably", "reportedly", "putatively",
}

# Tier-1 precise elements handled by regex on the raw token.
_TIER1_DIGIT = re.compile(r"\d")
_TIER1_PVAL = re.compile(r"^p[<>=]", re.IGNORECASE)

# Comparison operators / function words (Tier 1 + Tier 2 boundary).
COMPARISON_OPS: set[str] = {">", "<", ">=", "<=", "≥", "≤", "vs", "vs.", "versus", "than"}

# Tier-2 ranking / superlative / comparison words.
RANK_WORDS: set[str] = {
    "best", "worst", "highest", "lowest", "largest", "smallest", "first",
    "second", "third", "top", "leading", "lead", "superior", "inferior",
    "outperform", "outperforms", "outperformed", "outperforming", "exceed",
    "exceeds", "exceeded", "surpass", "surpasses", "surpassed", "better",
    "worse", "higher", "lower", "greater", "less", "fewer", "most", "least",
    "fewest", "state-of-the-art", "sota", "optimal", "maximum", "minimum",
    "dominant", "strongest", "fastest", "slowest", "only", "superiority",
}

# Tier-1 unit tokens: spelled-out units and multi-char abbreviations that denote
# a precise measured quantity. Single-letter abbreviations (m, s, g, l, x) are
# intentionally excluded — they are too ambiguous, and a unit glued to a number
# ("5mg", "20%") is already caught by the digit regex. This only adds the
# space-separated / spelled-out cases ("5 mg", "milligrams", "percent").
UNIT_WORDS: set[str] = {
    # generic
    "percent", "percentage", "pp",
    # mass
    "mg", "kg", "ug", "µg", "ng", "gram", "grams", "milligram", "milligrams",
    "kilogram", "kilograms", "microgram", "micrograms", "nanogram", "nanograms",
    # volume
    "ml", "µl", "ul", "litre", "litres", "liter", "liters", "milliliter",
    "milliliters", "millilitre", "millilitres",
    # length
    "mm", "cm", "km", "nm", "um", "µm", "meter", "meters", "metre", "metres",
    "millimeter", "millimeters", "centimeter", "centimeters", "kilometer",
    "kilometers", "micrometer", "micrometers", "nanometer", "nanometers",
    # time
    "ms", "ns", "ps", "millisecond", "milliseconds", "microsecond",
    "microseconds", "nanosecond", "nanoseconds", "second", "seconds",
    "minute", "minutes", "hour", "hours",
    # frequency
    "hz", "khz", "mhz", "ghz", "hertz", "kilohertz", "megahertz", "gigahertz",
    # data
    "kb", "mb", "gb", "tb", "byte", "bytes", "kilobyte", "kilobytes",
    "megabyte", "megabytes", "gigabyte", "gigabytes", "terabyte", "terabytes",
    "bit", "bits", "flops", "flop",
    # chemistry / energy
    "mol", "mole", "moles", "molar", "kcal", "cal", "calorie", "calories",
    "kilocalorie", "kilocalories", "joule", "joules", "watt", "watts", "volt",
    "volts", "ampere", "amperes", "pascal", "newton", "newtons",
    # temperature
    "celsius", "fahrenheit", "kelvin",
    # ML / experimental
    "px", "pixel", "pixels", "fps", "epoch", "epochs", "bp", "fold",
}


def _norm(tok: str) -> str:
    return tok.lower().strip(string.punctuation + "≥≤×")


# ---------------------------------------------------------------------------
# Flagging logic
# ---------------------------------------------------------------------------


def _precise_tier(ctx_tokens: list[str], skip_idx: int) -> str | None:
    """Return 'num'/'rank' if a precise element appears in ctx (excluding skip_idx)."""
    for j, t in enumerate(ctx_tokens):
        if j == skip_idx:
            continue
        if _TIER1_DIGIT.search(t) or _TIER1_PVAL.match(t):
            return "num"
        tl = _norm(t)
        if tl in UNIT_WORDS:
            return "num"
        if t in COMPARISON_OPS or tl in COMPARISON_OPS:
            return "rank"
        if tl in RANK_WORDS:
            return "rank"
    return None


def flag_reasons(
    original: str,
    rewrite: str,
    window: int = 2,
    direction: str = "before",
) -> list[tuple[str, str]]:
    """
    Return [(hedge, tier), ...] for every newly-inserted hedge adjacent to a
    precise element. Empty list ⇒ pure style.

    direction:
      "before" — flag only when the precise element *follows* the hedge, i.e. the
                 hedge is placed before it ("approximately 95%"). This matches the
                 reviewer's wording and avoids the "Table N + suggests" false-
                 positive class (table/figure index numbers preceding an epistemic
                 verb).  Default.
      "after"  — flag only when the precise element precedes the hedge
                 ("outperforms largely", "95%, roughly").
      "both"   — either side (symmetric ±window window).
    """
    orig_counts = Counter(_norm(w) for w in original.split())
    rw_tokens = rewrite.split()
    rw_counts = Counter(_norm(w) for w in rw_tokens)

    reasons: list[tuple[str, str]] = []
    for i, w in enumerate(rw_tokens):
        nw = _norm(w)
        if not nw or nw not in HEDGE_LEXICON:
            continue
        # "newly inserted" = appears more often in rewrite than original
        if rw_counts[nw] <= orig_counts.get(nw, 0):
            continue
        if direction == "before":       # precise element to the right of the hedge
            ctx = rw_tokens[i + 1: i + 1 + window]
            tier = _precise_tier(ctx, skip_idx=-1)
        elif direction == "after":      # precise element to the left of the hedge
            ctx = rw_tokens[max(0, i - window): i]
            tier = _precise_tier(ctx, skip_idx=-1)
        else:                            # both sides
            lo = max(0, i - window)
            ctx = rw_tokens[lo: min(len(rw_tokens), i + window + 1)]
            tier = _precise_tier(ctx, skip_idx=i - lo)
        if tier:
            reasons.append((nw, tier))
    return reasons


# ---------------------------------------------------------------------------
# Condition resolution (mirrors run.py analyze conventions)
# ---------------------------------------------------------------------------


def _condition_variants(cond: str, mix_seed: int | None) -> list[tuple[str, str]]:
    """Return [(variant_key, method), ...] backing a condition."""
    from core.config import NATURAL_GROUPS

    def method(v: str) -> str:
        return "inject" if v in ("D_1", "D_2", "C_1", "C_2") else "natural"

    if cond in NATURAL_GROUPS:
        if mix_seed is not None:
            v = f"{cond}-mix-{mix_seed}"
            return [(v, "natural")]
        return [(v, "natural") for v in NATURAL_GROUPS[cond]]
    return [(cond, method(cond))]


# ---------------------------------------------------------------------------
# Mask construction (model-agnostic, from rewrite text)
# ---------------------------------------------------------------------------


def build_condition_mask(
    dataset: str,
    splits: Sequence[str],
    cond: str,
    mix_seed: int | None,
    originals: dict[str, str],
    valid_ids: set,
    *,
    window: int = 2,
    direction: str = "before",
) -> tuple[set, set, dict]:
    """Partition valid claim_ids into (pure_ids, flagged_ids, stats) for one condition."""
    from core.io import load_rewrite

    rewrites_by_cid: dict[str, list[str]] = defaultdict(list)
    for variant, method in _condition_variants(cond, mix_seed):
        for sp in splits:
            try:
                items = load_rewrite(dataset, sp, method, variant)
            except FileNotFoundError:
                continue
            for it in items:
                cid = it.get("claim_id")
                claim = it.get("claim")
                if cid in valid_ids and claim:
                    rewrites_by_cid[cid].append(claim)

    pure: set = set()
    flagged: set = set()
    tiers: Counter = Counter()
    for cid, rws in rewrites_by_cid.items():
        og = originals.get(cid)
        if not og:
            continue
        reasons: list[tuple[str, str]] = []
        for rw in rws:
            reasons.extend(flag_reasons(og, rw, window=window, direction=direction))
        if reasons:
            flagged.add(cid)
            for _, t in reasons:
                tiers[t] += 1
        else:
            pure.add(cid)

    n_total = len(pure) + len(flagged)
    stats = {
        "n_total": n_total,
        "n_pure": len(pure),
        "n_flagged": len(flagged),
        "flag_rate": (len(flagged) / n_total) if n_total else float("nan"),
        "tiers": dict(tiers),
    }
    return pure, flagged, stats


# ---------------------------------------------------------------------------
# Δp aggregation
# ---------------------------------------------------------------------------


def _load_merged(dataset, splits, slug, variant, valid_ids, *, pred_key, prob_key, id_to_label):
    from core.analysis.metrics import load_predictions
    from core.io import run_exists, run_path

    merged: dict = {}
    for sp in splits:
        if run_exists(dataset, sp, slug, variant):
            merged.update(load_predictions(
                run_path(dataset, sp, slug, variant), valid_ids,
                pred_key=pred_key, prob_key=prob_key,
            ))
    for cid, pred in merged.items():
        if not pred.get("label") and cid in id_to_label:
            pred["label"] = id_to_label[cid]
    return merged


def _label_deltas(base_preds, variant_preds_list, id_subset) -> dict[str, list[float]]:
    """Per-claim mean Δp within id_subset, grouped by true label."""
    base_p = {cid: p["p_supported"] for cid, p in base_preds.items()
              if p["p_supported"] is not None}
    per_claim: dict[str, list[float]] = defaultdict(list)
    for preds in variant_preds_list:
        for cid, pred in preds.items():
            if cid in id_subset and cid in base_p and pred["p_supported"] is not None:
                per_claim[cid].append(pred["p_supported"] - base_p[cid])
    out: dict[str, list[float]] = {"Supported": [], "Refuted": []}
    for cid, dps in per_claim.items():
        if dps:
            lab = base_preds[cid]["label"]
            if lab in out:
                out[lab].append(float(np.mean(dps)))
    return out


def _summ(deltas: dict[str, list[float]], *, n_boot: int = 1000) -> dict:
    from core.analysis.stats import bootstrap_mean_ci, bootstrap_mean_p

    sup, ref = deltas["Supported"], deltas["Refuted"]
    sup_ci = bootstrap_mean_ci(sup, n_iter=n_boot) if sup else (float("nan"),) * 3
    ref_ci = bootstrap_mean_ci(ref, n_iter=n_boot) if ref else (float("nan"),) * 3
    return {
        "n_sup": len(sup), "n_ref": len(ref),
        "dp_sup": float(np.mean(sup)) if sup else float("nan"),
        "dp_ref": float(np.mean(ref)) if ref else float("nan"),
        "p_sup": bootstrap_mean_p(sup, n_iter=n_boot) if sup else float("nan"),
        "p_ref": bootstrap_mean_p(ref, n_iter=n_boot) if ref else float("nan"),
        "ci_sup": [sup_ci[1], sup_ci[2]],
        "ci_ref": [ref_ci[1], ref_ci[2]],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_analysis(
    dataset: str,
    splits: Sequence[str],
    conditions: Sequence[str],
    *,
    mix_seed: int | None,
    eval_mode: str,
    models: Sequence[str] | None,
    window: int = 2,
    direction: str = "before",
    n_boot: int = 2000,
    verbose: bool = True,
) -> dict:
    from core.analysis.metrics import build_pair_mapping
    from core.config import DATASETS, EVAL_MODES, sort_model_slugs
    from core.io import list_model_slugs, load_base_dataset, run_exists

    if eval_mode not in EVAL_MODES:
        sys.exit(f"Unknown eval mode {eval_mode!r}. Choose from: {list(EVAL_MODES)}")
    pred_key, prob_key = EVAL_MODES[eval_mode]
    dataset_cfg = DATASETS[dataset]

    base_data: list = []
    for sp in splits:
        sp_data = load_base_dataset(dataset_cfg, sp)
        if len(splits) > 1:
            sp_data = [
                {**item, "claim_id_pair": f"{sp}:{item['claim_id_pair']}"}
                if "claim_id_pair" in item else item
                for item in sp_data
            ]
        base_data.extend(sp_data)

    _pairs, valid_ids = build_pair_mapping(base_data)
    originals = {it["claim_id"]: it["claim"] for it in base_data if it.get("claim")}
    id_to_label = {it["claim_id"]: it["label"] for it in base_data if it.get("label")}

    # ---- masks (model-agnostic) ----
    masks: dict[str, dict] = {}
    for cond in conditions:
        pure, flagged, stats = build_condition_mask(
            dataset, splits, cond, mix_seed, originals, valid_ids,
            window=window, direction=direction,
        )
        masks[cond] = {"pure": pure, "flagged": flagged, "stats": stats}
        if verbose:
            s = stats
            print(
                f"  mask[{cond}]: pure={s['n_pure']} flagged={s['n_flagged']} "
                f"({s['flag_rate']:.1%})  tiers={s['tiers']}",
                file=sys.stderr,
            )

    # ---- models ----
    if models:
        slugs = sort_model_slugs(list(models))
    else:
        found: set = set()
        for sp in splits:
            found.update(list_model_slugs(dataset, sp))
        slugs = sort_model_slugs(list(found))
    if not slugs:
        sys.exit(f"No model run directories found under runs/{dataset}/")

    per_model: dict[str, dict] = {}
    pooled: dict[str, dict[str, dict[str, list[float]]]] = {
        cond: {"pure": {"Supported": [], "Refuted": []},
               "flagged": {"Supported": [], "Refuted": []},
               "all": {"Supported": [], "Refuted": []}}
        for cond in conditions
    }

    for slug in slugs:
        if not any(run_exists(dataset, sp, slug, "original") for sp in splits):
            continue
        base_preds = _load_merged(dataset, splits, slug, "original", valid_ids,
                                  pred_key=pred_key, prob_key=prob_key, id_to_label=id_to_label)
        per_model[slug] = {}
        for cond in conditions:
            variant_preds = []
            for variant, _m in _condition_variants(cond, mix_seed):
                vp = _load_merged(dataset, splits, slug, variant, valid_ids,
                                  pred_key=pred_key, prob_key=prob_key, id_to_label=id_to_label)
                if vp:
                    variant_preds.append(vp)
            if not variant_preds:
                continue
            pure_ids = masks[cond]["pure"]
            flagged_ids = masks[cond]["flagged"]
            all_ids = pure_ids | flagged_ids
            d_pure = _label_deltas(base_preds, variant_preds, pure_ids)
            d_flag = _label_deltas(base_preds, variant_preds, flagged_ids)
            d_all = _label_deltas(base_preds, variant_preds, all_ids)
            per_model[slug][cond] = {
                "pure": _summ(d_pure, n_boot=n_boot),
                "flagged": _summ(d_flag, n_boot=n_boot),
                "all": _summ(d_all, n_boot=n_boot),
            }
            for subset, d in (("pure", d_pure), ("flagged", d_flag), ("all", d_all)):
                pooled[cond][subset]["Supported"].extend(d["Supported"])
                pooled[cond][subset]["Refuted"].extend(d["Refuted"])
        if verbose:
            print(f"  [{slug}] done", file=sys.stderr)

    pooled_summ = {
        cond: {subset: _summ(pooled[cond][subset], n_boot=n_boot)
               for subset in ("pure", "flagged", "all")}
        for cond in conditions
    }

    return {
        "meta": {
            "dataset": dataset, "splits": list(splits), "conditions": list(conditions),
            "mix_seed": mix_seed, "eval_mode": eval_mode, "window": window,
            "direction": direction,
            "n_models": len(per_model), "models": list(per_model.keys()),
        },
        "mask_stats": {c: masks[c]["stats"] for c in conditions},
        "pooled": pooled_summ,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return " "
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def print_report(pooled: dict, mask_stats: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    hdr = (f"{'Cond':<6} {'flag%':>6} {'subset':<8} "
           f"{'n_sup':>6} {'n_ref':>6} {'dp_sup':>10} {'dp_ref':>10}")
    print(hdr)
    print("-" * len(hdr))
    for cond, subsets in pooled.items():
        fr = mask_stats.get(cond, {}).get("flag_rate", float("nan"))
        for si, subset in enumerate(("pure", "flagged", "all")):
            s = subsets[subset]
            frs = f"{fr:.0%}" if si == 0 and not np.isnan(fr) else ""
            print(
                f"{cond if si == 0 else '':<6} {frs:>6} {subset:<8} "
                f"{s['n_sup']:>6d} {s['n_ref']:>6d} "
                f"{_f(s['dp_sup'])+_stars(s['p_sup']):>10} "
                f"{_f(s['dp_ref'])+_stars(s['p_ref']):>10}"
            )
        print()


def print_flag_counts(mask_stats: dict, conds: Sequence[str] = ("CT", "C_1")) -> None:
    """Print how many rewrites are classified into the *flagged* category.

    The mask is derived from rewrite text only, so these counts are
    model-agnostic (identical for every inference model / pooled).
    """
    print("\n=== Flagged-rewrite counts (model-agnostic mask) ===")
    for cond in conds:
        s = mask_stats.get(cond)
        if not s:
            continue
        print(
            f"  {cond:<5} flagged = {s['n_flagged']:>4} / {s['n_total']:<4} "
            f"({s['flag_rate']:.1%})   [unflagged = {s['n_pure']}; "
            f"tiers = {s.get('tiers', {})}]"
        )


def _md_cell(dp: float, ci: list) -> str:
    """LaTeX-in-markdown cell: ``$+0.055\\ [+0.037,\\ +0.074]$``."""
    if dp is None or (isinstance(dp, float) and np.isnan(dp)):
        return "n/a"
    lo, hi = (ci or [float("nan"), float("nan")])[:2]
    if np.isnan(lo) or np.isnan(hi):
        return f"${dp:+.3f}$"
    return f"${dp:+.3f}\\ [{lo:+.3f},\\ {hi:+.3f}]$"


def generate_md_table(scope_stats: dict, conds: Sequence[str] = ("CT", "C_1")) -> str:
    """Markdown table: columns Full/Unflagged/Flagged; rows cond x {sup, ref}.

    ``scope_stats`` maps condition -> {"all"|"pure"|"flagged": _summ(...)} (i.e.
    ``result["pooled"]`` or ``result["per_model"][slug]``). Every cell carries the
    signed drift and its 95% bootstrap CI in LaTeX notation.
    """
    col = {"Full": "all", "Unflagged": "pure", "Flagged": "flagged"}
    disp = {"CT": "CT", "C_1": "C1"}
    lines = [
        "| Condition | Full | Unflagged | Flagged |",
        "|---|---|---|---|",
    ]
    for cond in conds:
        subsets = scope_stats.get(cond)
        if not subsets:
            continue
        d = disp.get(cond, cond)
        for metric, dp_key, ci_key, tex in (
            ("sup", "dp_sup", "ci_sup", r"\Delta p_{\text{sup}}"),
            ("ref", "dp_ref", "ci_ref", r"\Delta p_{\text{ref}}"),
        ):
            cells = [
                _md_cell(subsets[col[c]].get(dp_key), subsets[col[c]].get(ci_key))
                for c in ("Full", "Unflagged", "Flagged")
            ]
            lines.append(f"| {d} ${tex}$ | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_table(pooled: dict, mask_stats: dict, meta: dict,
                   *, label: str = "tab:semantic_filter") -> str:
    lo_hi = meta.get("window")
    caption = (
        r"Shift metric ($\overline{\Delta p}$ for Supported / Refuted claims) after "
        r"splitting each condition into a \emph{pure-style} set and a set where the "
        r"rewrite introduced a hedge/approximation adjacent to a precise element "
        r"(number, unit, \%, p-value, comparison, or ranking; adjacency window "
        rf"$={lo_hi}$ tokens). Stars: bootstrap $p<0.05^{{*}}$, $<0.01^{{**}}$, "
        r"$<0.001^{***}$. Persisting effects on the pure set indicate a genuine "
        r"style bias rather than a semantic artifact."
    )
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Cond & Subset & $N$ & $\overline{\Delta p}_{\text{sup}}$ & $\overline{\Delta p}_{\text{ref}}$ \\",
        r"\midrule",
    ]
    for cond, subsets in pooled.items():
        for si, subset in enumerate(("pure", "flagged")):
            s = subsets[subset]
            n = s["n_sup"] + s["n_ref"]
            disp = cond.replace("_", r"\_") if si == 0 else ""
            lines.append(
                f"{disp} & {subset} & {n} & "
                f"${_f(s['dp_sup'])}$ & ${_f(s['dp_ref'])}$ \\\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    from core.config import DATASETS

    p = argparse.ArgumentParser(
        description="Split rewrites into pure-style vs semantically-changed and compare Δp."
    )
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--split", nargs="+", default=["dev", "test"])
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Default: LP FI AS OA CT D_1 C_1")
    p.add_argument("--mix-seed", type=int, default=42, dest="mix_seed")
    p.add_argument("--no-mix", action="store_true", dest="no_mix",
                   help="Use the 5 individual variants per natural group instead of the mix file")
    p.add_argument("--eval-mode", default="consistency", dest="eval_mode",
                   choices=["consistency"])
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--window", type=int, default=2,
                   help="Token adjacency window between hedge and precise element (default: 2)")
    p.add_argument("--direction", default="before", choices=["before", "after", "both"],
                   help="Flag when precise element is 'before' the hedge is placed in "
                        "front of it (default; matches reviewer wording), 'after', or 'both'")
    p.add_argument("--n-boot", type=int, default=1000, dest="n_boot",
                   help="Bootstrap resamples for p-values and CIs (default: 1000, "
                        "matching the paper's bootstrap_mean_p convention)")
    p.add_argument("--out-dir", default=None, dest="out_dir")
    p.add_argument("--per-model", action="store_true", dest="per_model")
    return p


def main(argv: list[str] | None = None) -> None:
    from core.config import CONTROLLED_CONDITIONS, NATURAL_GROUPS

    args = _build_parser().parse_args(argv)
    conditions = args.conditions or (list(NATURAL_GROUPS) + CONTROLLED_CONDITIONS)
    mix_seed = None if args.no_mix else args.mix_seed

    print(f"dataset={args.dataset} split={args.split} mix_seed={mix_seed} "
          f"window={args.window} direction={args.direction} "
          f"eval_mode={args.eval_mode}", file=sys.stderr)

    result = run_analysis(
        args.dataset, args.split, conditions,
        mix_seed=mix_seed, eval_mode=args.eval_mode, models=args.models,
        window=args.window, direction=args.direction, n_boot=args.n_boot, verbose=True,
    )

    if args.per_model:
        for slug, stats in result["per_model"].items():
            print_report(stats, result["mask_stats"], f"per-model: {slug}")

    print_report(result["pooled"], result["mask_stats"],
                 f"pooled across {result['meta']['n_models']} models")

    # Headline numbers, script-generated.
    print_flag_counts(result["mask_stats"], conds=[c for c in ("CT", "C_1") if c in conditions])

    md_conds = [c for c in ("CT", "C_1") if c in conditions]
    models_used = result["meta"]["models"]
    if len(models_used) == 1:
        scope = f"model = {models_used[0]}"
        md_scope = result["pooled"]  # pooled over a single model == that model
    else:
        scope = f"pooled over {result['meta']['n_models']} models"
        md_scope = result["pooled"]
    print(f"\n=== Markdown table ({scope}), 95% CI, B={args.n_boot} ===")
    print(generate_md_table(md_scope, conds=md_conds))

    if args.per_model:
        for slug, stats in result["per_model"].items():
            print(f"\n=== Markdown table (model = {slug}), 95% CI, B={args.n_boot} ===")
            print(generate_md_table(stats, conds=md_conds))

    print("\n=== LaTeX (pooled) ===")
    print(generate_table(result["pooled"], result["mask_stats"], result["meta"]))

    if args.out_dir:
        from pathlib import Path
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "semantic_filter.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "semantic_filter.tex").write_text(
            generate_table(result["pooled"], result["mask_stats"], result["meta"]),
            encoding="utf-8")
        print(f"\nSaved → {out/'semantic_filter.json'} and {out/'semantic_filter.tex'}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
