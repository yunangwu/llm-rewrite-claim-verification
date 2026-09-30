"""
Rewrite-quality analysis: how much do LP / FI / AS (and other conditions)
change the surface form of original claims?

Three complementary metrics
---------------------------
1. Exact-match rate  — fraction of claims identical to the original (lower = more editing).
2. Normalised edit distance (NED)  — character-level Levenshtein / max(|a|, |b|).
   Range [0, 1]; 0 = identical, 1 = completely different.
3. BERTScore F1 (roberta-large)  — semantic similarity that ignores surface variation.
   Range [0, 1]; values near 1 mean the meaning was preserved.

Usage (standalone)
------------------
  python -m core.analysis.rewrite_quality \
      --dataset sciclaimeval --split dev \
      --conditions LP FI AS CT OA D_1 C_1
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

# BERTScore is an optional heavy dependency; import lazily.
_bert_score = None


def _lazy_bert_score():
    global _bert_score
    if _bert_score is None:
        import bert_score as _bs  # type: ignore
        _bert_score = _bs
    return _bert_score


# ---------------------------------------------------------------------------
# Low-level metrics (pure Python, no external deps beyond stdlib/numpy)
# ---------------------------------------------------------------------------


def normalised_edit_distance(a: str, b: str) -> float:
    """Character-level Levenshtein distance normalised by max length."""
    if a == b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    # difflib.SequenceMatcher gives edit operations; derive Levenshtein from
    # the "distance" implied by the opcodes.
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = sm.get_opcodes()
    dist = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        elif tag == "replace":
            dist += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            dist += i2 - i1
        elif tag == "insert":
            dist += j2 - j1
    return dist / max_len


def word_overlap_f1(a: str, b: str) -> float:
    """Unigram token overlap F1 (case-insensitive)."""
    ta = a.lower().split()
    tb = b.lower().split()
    if not ta or not tb:
        return 0.0
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    prec = common / len(tb)
    rec  = common / len(ta)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_original(dataset: str, split: str) -> dict[str, str]:
    """Return {claim_id: original_claim} from the base dataset."""
    from core.config import DATASETS
    from core.io import load_base_dataset

    items = load_base_dataset(DATASETS[dataset], split)
    return {x["claim_id"]: x["claim"] for x in items}


def _load_rewrite(dataset: str, split: str, condition: str) -> list[dict]:
    """Load rewrite items for one condition (auto-detect method)."""
    from core.io import load_rewrite

    inject_conds = {"D_1", "D_2", "C_1", "C_2"}
    method = "inject" if condition in inject_conds else "natural"
    return load_rewrite(dataset, split, method, condition)


# ---------------------------------------------------------------------------
# Per-condition statistics
# ---------------------------------------------------------------------------


def compute_condition_stats(
    originals: dict[str, str],
    rewrites: list[dict],
) -> dict:
    """
    Compute surface-change statistics for one condition.

    Returns
    -------
    dict with keys:
        n, n_exact, exact_rate,
        ned_mean, ned_std, ned_median,
        overlap_f1_mean, overlap_f1_std,
        pairs  (list of {claim_id, orig, rewrite, ned, overlap_f1})
    """
    pairs = []
    for item in rewrites:
        cid  = item["claim_id"]
        orig = originals.get(cid)
        if orig is None:
            continue
        rewr = item["claim"]
        ned  = normalised_edit_distance(orig, rewr)
        f1   = word_overlap_f1(orig, rewr)
        pairs.append({"claim_id": cid, "orig": orig, "rewrite": rewr,
                       "ned": ned, "overlap_f1": f1})

    if not pairs:
        return {}

    neds  = np.array([p["ned"]        for p in pairs])
    f1s   = np.array([p["overlap_f1"] for p in pairs])
    exact = np.array([p["ned"] == 0.0 for p in pairs])

    return {
        "n":              len(pairs),
        "n_exact":        int(exact.sum()),
        "exact_rate":     float(exact.mean()),
        "ned_mean":       float(neds.mean()),
        "ned_std":        float(neds.std()),
        "ned_median":     float(np.median(neds)),
        "ned_q25":        float(np.percentile(neds, 25)),
        "ned_q75":        float(np.percentile(neds, 75)),
        "overlap_f1_mean": float(f1s.mean()),
        "overlap_f1_std":  float(f1s.std()),
        "pairs":          pairs,
    }


def compute_bert_scores(
    pairs: list[dict],
    batch_size: int = 64,
    model_type: str = "roberta-large",
    device: str | None = None,
    verbose: bool = False,
) -> list[float]:
    """
    Run BERTScore (F1) for each (orig, rewrite) pair.
    Returns a list of floats aligned with ``pairs``.
    """
    bs = _lazy_bert_score()
    cands = [p["rewrite"] for p in pairs]
    refs  = [p["orig"]    for p in pairs]

    P, R, F = bs.score(
        cands, refs,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
        lang="en",
    )
    return F.tolist()


# ---------------------------------------------------------------------------
# Multi-condition driver
# ---------------------------------------------------------------------------


def _expand_groups(conditions: Sequence[str]) -> list[str]:
    """
    Expand group aliases (LP, FI, AS, CT, OA) to all 5 numbered variants,
    pass through variant names (LP-1, D_1, …) unchanged.
    """
    from core.config import NATURAL_GROUPS

    group_map = {g: vs for g, vs in NATURAL_GROUPS.items()}
    out = []
    for c in conditions:
        if c in group_map:
            out.extend(group_map[c])
        else:
            out.append(c)
    return out


def run_analysis(
    dataset: str,
    split: str,
    conditions: Sequence[str],
    *,
    use_bert: bool = True,
    bert_model: str = "roberta-large",
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run full quality analysis for each condition.

    Returns
    -------
    dict mapping condition → stats dict (see ``compute_condition_stats``).
    If ``use_bert``, each stats dict additionally has ``bert_f1_mean`` and
    ``bert_f1_std``.
    """
    originals = _load_original(dataset, split)
    expanded  = _expand_groups(conditions)

    results: dict[str, dict] = {}
    for cond in expanded:
        if verbose:
            print(f"  [{cond}] loading rewrites …", file=sys.stderr)
        try:
            items = _load_rewrite(dataset, split, cond)
        except FileNotFoundError:
            if verbose:
                print(f"  [{cond}] NOT FOUND — skipping", file=sys.stderr)
            continue

        stats = compute_condition_stats(originals, items)
        if not stats:
            continue

        if use_bert:
            if verbose:
                print(f"  [{cond}] computing BERTScore ({len(stats['pairs'])} pairs) …",
                      file=sys.stderr)
            bf1 = compute_bert_scores(stats["pairs"], model_type=bert_model,
                                       device=device, verbose=False)
            arr = np.array(bf1)
            stats["bert_f1_mean"] = float(arr.mean())
            stats["bert_f1_std"]  = float(arr.std())
            for p, s in zip(stats["pairs"], bf1):
                p["bert_f1"] = s

        results[cond] = stats

    return results


# ---------------------------------------------------------------------------
# Pretty printer / report
# ---------------------------------------------------------------------------


def print_report(results: dict[str, dict], use_bert: bool = True) -> None:
    has_bert = use_bert and any("bert_f1_mean" in v for v in results.values())

    header = f"{'Condition':<12}  {'N':>5}  {'Exact%':>7}  "
    header += f"{'NED mean':>9}  {'NED med':>8}  {'Overlap-F1':>10}"
    if has_bert:
        header += f"  {'BERTScore-F1':>12}"
    print(header)
    print("-" * len(header))

    for cond, s in results.items():
        row = (
            f"{cond:<12}  {s['n']:>5}  "
            f"{s['exact_rate']*100:>6.1f}%  "
            f"{s['ned_mean']:>9.4f}  "
            f"{s['ned_median']:>8.4f}  "
            f"{s['overlap_f1_mean']:>10.4f}"
        )
        if has_bert:
            row += f"  {s['bert_f1_mean']:>12.4f}"
        print(row)


def aggregate_by_group(
    results: dict[str, dict],
    groups: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """
    Average statistics across variants within each natural group.
    E.g. LP-1 … LP-5 → LP.
    """
    if groups is None:
        from core.config import NATURAL_GROUPS
        groups = NATURAL_GROUPS

    agg: dict[str, dict] = {}
    for grp, variants in groups.items():
        matched = {v: results[v] for v in variants if v in results}
        if not matched:
            continue

        keys_to_avg = ["exact_rate", "ned_mean", "ned_median",
                        "overlap_f1_mean", "bert_f1_mean"]
        agg[grp] = {}
        for k in keys_to_avg:
            vals = [s[k] for s in matched.values() if k in s]
            if vals:
                agg[grp][k] = float(np.mean(vals))

        total_n = sum(s["n"] for s in matched.values())
        total_exact = sum(s["n_exact"] for s in matched.values())
        agg[grp]["n"] = total_n
        agg[grp]["n_exact"] = total_exact
        agg[grp]["variants"] = list(matched.keys())

    # Pass through controlled conditions unchanged
    for cond, stats in results.items():
        if not any(cond in vs for vs in groups.values()):
            agg[cond] = stats

    return agg


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def generate_quality_table(
    agg: dict[str, dict],
    use_bert: bool = True,
    caption: str = "Rewrite surface-change statistics per condition.",
    label: str = "tab:rewrite_quality",
) -> str:
    """
    Emit a LaTeX table with per-condition quality metrics.
    """
    has_bert = use_bert and any("bert_f1_mean" in v for v in agg.values())

    cols = "lrrrrr"
    heads = r"Condition & $N$ & Exact\,\% & NED $\bar{x}$ & NED $\tilde{x}$ & Overlap-F1"
    if has_bert:
        cols += "r"
        heads += r" & BERTScore-F1"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        heads + r" \\",
        r"\midrule",
    ]

    for cond, s in agg.items():
        if not s:
            continue
        row = (
            f"{cond.replace('_', r'\_')} & "
            f"{s.get('n', 0)} & "
            f"{s.get('exact_rate', float('nan'))*100:.1f}\\% & "
            f"{s.get('ned_mean', float('nan')):.4f} & "
            f"{s.get('ned_median', float('nan')):.4f} & "
            f"{s.get('overlap_f1_mean', float('nan')):.4f}"
        )
        if has_bert:
            row += f" & {s.get('bert_f1_mean', float('nan')):.4f}"
        lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse surface changes introduced by text rewriting."
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset key (e.g. sciclaimeval)")
    p.add_argument("--split", default="dev",
                   help="Split to analyse (default: dev)")
    p.add_argument("--conditions", nargs="+",
                   default=["LP", "FI", "AS", "CT", "OA", "D_1", "C_1"],
                   help="Condition groups or variant names to include")
    p.add_argument("--no-bert", action="store_true",
                   help="Skip BERTScore (much faster)")
    p.add_argument("--bert-model", default="roberta-large",
                   help="HuggingFace model for BERTScore (default: roberta-large)")
    p.add_argument("--device", default=None,
                   help="Torch device (default: auto)")
    p.add_argument("--out-dir", default=None,
                   help="Directory to save JSON + TeX outputs (default: print only)")
    p.add_argument("--save-pairs", action="store_true",
                   help="Include per-claim pairs in saved JSON")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    print(f"Dataset: {args.dataset}  split: {args.split}", file=sys.stderr)
    print(f"Conditions: {args.conditions}", file=sys.stderr)
    print(f"BERTScore: {'OFF' if args.no_bert else args.bert_model}", file=sys.stderr)
    print("", file=sys.stderr)

    results = run_analysis(
        args.dataset, args.split, args.conditions,
        use_bert=not args.no_bert,
        bert_model=args.bert_model,
        device=args.device,
        verbose=True,
    )

    agg = aggregate_by_group(results)

    print("\n=== Per-variant results ===")
    print_report(results, use_bert=not args.no_bert)

    print("\n=== Aggregated by group ===")
    print_report(agg, use_bert=not args.no_bert)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Optionally strip heavyweight pairs list before saving
        save_data = {}
        for k, v in results.items():
            entry = {ek: ev for ek, ev in v.items() if ek != "pairs" or args.save_pairs}
            save_data[k] = entry

        (out / "rewrite_quality.json").write_text(
            json.dumps({"by_variant": save_data,
                        "by_group":   {k: {ek: ev for ek, ev in v.items()
                                           if ek != "pairs"}
                                       for k, v in agg.items()}},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tex = generate_quality_table(agg, use_bert=not args.no_bert)
        (out / "rewrite_quality.tex").write_text(tex, encoding="utf-8")
        print(f"\nSaved → {out/'rewrite_quality.json'}  and  {out/'rewrite_quality.tex'}")

    print("\n=== LaTeX table (aggregated) ===")
    print(generate_quality_table(agg, use_bert=not args.no_bert))


if __name__ == "__main__":
    main()
