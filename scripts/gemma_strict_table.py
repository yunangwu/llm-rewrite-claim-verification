#!/usr/bin/env python3
"""
Regenerate the new-rewriter (gemma) shift table with a *strict* injection filter
applied to gemma's C_1 / D_1 only — without touching the shared analyze path.

Rationale
---------
The production pipeline (``metrics.validate_rewrite_claims``) drops only empty /
too-short rewrites. It keeps inject rewrites where a chatty rewriter (gemma-4)
added an off-pool synonym or changed more than one word. This script drops those
non-compliant items *for gemma's result only* (using
``core.analysis.rewrite_quality.strict_inject_clean_ids``) and recomputes the
Δp table, so we can show the conclusion is unchanged under strict one-word-from-
pool compliance. gpt-oss and the natural conditions (CT, OA) are untouched.

Everything here is opt-in and read-only w.r.t. runs/ and rewrites/.

Usage:
  python scripts/gemma_strict_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.analysis.metrics import (
    build_pair_mapping, compute_shifts, load_predictions, validate_rewrite_claims,
)
from core.config import DATASETS, EVAL_MODES
from core.io import load_base_dataset, load_rewrite, run_exists, run_path
from scripts.check_gemma_rewrites import strict_inject_clean_ids

SLUG = "Qwen3-VL-8B-Instruct"
SPLIT = "dev"
CONDS = [("CT-mix-42", "CT"), ("OA-mix-42", "OA"), ("C_1", "C1"), ("D_1", "D1")]
PRED, PROB = EVAL_MODES["consistency"]


def _method(cond: str) -> str:
    return "inject" if cond in ("D_1", "D_2", "C_1", "C_2") else "natural"


def _ci(vec, n=1000, seed=0, a=0.05):
    d = np.asarray(list(vec), float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, size=(n, d.size))].mean(1)
    lo, hi = np.percentile(m, [100 * a / 2, 100 * (1 - a / 2)])
    return float(d.mean()), float(lo), float(hi)


def _cell(t):
    m, lo, hi = t
    if not np.isfinite(m):
        return "n/a"
    return f"${m:+.3f}\\ [{lo:+.3f},\\ {hi:+.3f}]$"


def _shifts(rw_dataset, run_ns, cond, cp, base_claims, vids, patch, *, strict):
    """Per-pair shifts for one condition. If strict, intersect with the
    strict inject-compliance set (inject conditions only)."""
    if not run_exists(run_ns, SPLIT, SLUG, cond):
        return [], 0
    items = load_rewrite(rw_dataset, SPLIT, _method(cond), cond)
    clean, _ = validate_rewrite_claims(items, base_claims, cp)
    n_before = len(clean)
    dropped = 0
    if strict:
        strict_ids = strict_inject_clean_ids(rw_dataset, SPLIT, cond, base_claims=base_claims)
        if strict_ids is not None:
            new_clean = clean & strict_ids
            dropped = n_before - len(new_clean)
            clean = new_clean
    bp = patch(load_predictions(run_path(run_ns, SPLIT, SLUG, "original"), vids,
                                pred_key=PRED, prob_key=PROB))
    vp = patch(load_predictions(run_path(run_ns, SPLIT, SLUG, cond), clean,
                                pred_key=PRED, prob_key=PROB))
    return compute_shifts(bp, vp, cp), dropped


def main() -> int:
    b = load_base_dataset(DATASETS["sciclaimeval_gemma"], SPLIT)
    cp, vids = build_pair_mapping(b)
    base_claims = {it["claim_id"]: (it.get("claim") or "") for it in b}
    lab = {it["claim_id"]: it["label"] for it in b if it.get("label")}

    def patch(p):
        for cid, v in p.items():
            if not v.get("label") and cid in lab:
                v["label"] = lab[cid]
        return p

    gpt, gem_full, gem_strict, drops = {}, {}, {}, {}
    for cond, _ in CONDS:
        gpt[cond], _ = _shifts("sciclaimeval", "sciclaimeval", cond, cp, base_claims, vids, patch, strict=False)
        gem_full[cond], _ = _shifts("sciclaimeval_gemma", "sciclaimeval_gemma", cond, cp, base_claims, vids, patch, strict=False)
        gem_strict[cond], drops[cond] = _shifts("sciclaimeval_gemma", "sciclaimeval_gemma", cond, cp, base_claims, vids, patch, strict=True)

    print("Strict inject drops (gemma only; natural conditions unaffected):")
    for cond, disp in CONDS:
        if _method(cond) == "inject":
            print(f"  {disp}: dropped {drops[cond]} non-compliant pairs "
                  f"({len(gem_full[cond])} -> {len(gem_strict[cond])} pairs)")

    def _row(src, key):
        return " | ".join(_cell(_ci([s[key] for s in src[c]])) for c, _ in CONDS)

    print("\n## Qwen3-VL-8B — gemma strict-filtered vs gpt-oss (dev, 95% CI, B=1000)\n")
    print("| | " + " | ".join(d for _, d in CONDS) + " |")
    print("|---|" + "---|" * len(CONDS))
    print(f"| $\\Delta p_{{\\text{{sup}}}}$ (gpt-oss) | {_row(gpt, 'dp_sup')} |")
    print(f"| $\\Delta p_{{\\text{{ref}}}}$ (gpt-oss) | {_row(gpt, 'dp_ref')} |")
    print(f"| $\\Delta p_{{\\text{{sup}}}}$ (gemma, strict) | {_row(gem_strict, 'dp_sup')} |")
    print(f"| $\\Delta p_{{\\text{{ref}}}}$ (gemma, strict) | {_row(gem_strict, 'dp_ref')} |")

    print("\n## Effect of strict filter on gemma (full -> strict)\n")
    print("| | " + " | ".join(d for _, d in CONDS) + " |")
    print("|---|" + "---|" * len(CONDS))
    for key, tex in (("dp_sup", r"\Delta p_{\text{sup}}"), ("dp_ref", r"\Delta p_{\text{ref}}")):
        cells = []
        for c, _ in CONDS:
            f = _ci([s[key] for s in gem_full[c]])[0]
            s = _ci([s[key] for s in gem_strict[c]])[0]
            cells.append(f"${f:+.3f} \\to {s:+.3f}$")
        print(f"| ${tex}$ | " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
