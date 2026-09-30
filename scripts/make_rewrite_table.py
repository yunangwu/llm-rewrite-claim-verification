#!/usr/bin/env python3
"""
Generate a LaTeX rewrite-example table for one claim from SciClaimEval.

Word-level diffs are highlighted with \\hlc{green!20}{…} for natural rewrites
and \\hlc{blue!20}{…} for controlled injections.

Usage
-----
  # random claim from dev that has all 5 natural conditions:
  python scripts/make_rewrite_table.py

  # specific claim:
  python scripts/make_rewrite_table.py --claim-id val_fig_0042

  # write to file:
  python scripts/make_rewrite_table.py --out results/rewrite_example.tex

  # test split, different random seed:
  python scripts/make_rewrite_table.py --split test --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Conditions shown in the table, in display order.
NATURAL_CONDITIONS: list[tuple[str, str]] = [
    ("Language Polishing (LP)", "LP-1"),
    ("Fluency Improvement (FI)", "FI-1"),
    ("Academic Style Alignment (AS)", "AS-1"),
    ("Outcome-Aware (OA)", "OA-1"),
    ("Cautious Tone (CT)", "CT-1"),
]
CONTROLLED_CONDITIONS: list[tuple[str, str]] = [
    ("Conservative (C1)", "C_1"),
    ("Decorative (D1)", "D_1"),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _base_path(split: str) -> Path:
    fname = "dev_task1_release.json" if split == "dev" else "test_task1.json"
    return REPO_ROOT / "data" / "SciClaimEval" / fname


def _rewrite_path(split: str, method: str, condition: str) -> Path:
    return REPO_ROOT / "rewrites" / "sciclaimeval" / split / method / f"{condition}.json"


def _build_lookup(split: str, method: str, condition: str) -> dict[str, str]:
    """Return {claim_id: rewritten_claim} for one condition file."""
    path = _rewrite_path(split, method, condition)
    if not path.exists():
        return {}
    return {item["claim_id"]: item["claim"] for item in _load_json(path)}


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

_LATEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\textasciicircum{}"),
]


def _escape(text: str) -> str:
    for ch, repl in _LATEX_ESCAPES:
        text = text.replace(ch, repl)
    return text


# ---------------------------------------------------------------------------
# Word-level diff with highlights
# ---------------------------------------------------------------------------

def highlight_diff(original: str, rewritten: str, color: str) -> str:
    """
    Return `rewritten` with changed/inserted token spans wrapped in a single
    soul \\hlg{…} or \\hlb{…} call per contiguous changed run.

    Merging the whole span into one call lets soul break the highlighted text
    across lines (unlike \\colorbox which creates a non-breaking box per token).
    Deleted tokens (present in original but absent from rewrite) are omitted.
    """
    hl_cmd = r"\hlg" if color.startswith("green") else r"\hlb"
    orig_toks = original.split()
    rew_toks  = rewritten.split()
    matcher   = SequenceMatcher(None, orig_toks, rew_toks, autojunk=False)
    parts: list[str] = []
    for op, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.extend(_escape(t) for t in rew_toks[j1:j2])
        elif op in ("replace", "insert"):
            merged = " ".join(_escape(t) for t in rew_toks[j1:j2])
            parts.append(rf"{hl_cmd}{{{merged}}}")
        # "delete" → tokens removed from original; omit from output
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Table generator
# ---------------------------------------------------------------------------

def generate_table(
    claim_id: str,
    split: str,
    nat_lookups:  dict[str, dict[str, str]],
    ctrl_lookups: dict[str, dict[str, str]],
    base_lookup:  dict[str, dict],
) -> str:
    item = base_lookup.get(claim_id)
    if item is None:
        sys.exit(f"claim_id {claim_id!r} not found in {split} split.")
    original = item["claim"]

    lines: list[str] = []

    # ---- preamble reminder ----
    lines += [
        r"% Add to preamble:",
        r"% \usepackage{soul}",
        r"% \usepackage[table]{xcolor}",
        r"% \usepackage{tabularx}",
        r"% \newcommand{\hlg}[1]{{\sethlcolor{green!20}\hl{#1}}}",
        r"% \newcommand{\hlb}[1]{{\sethlcolor{blue!20}\hl{#1}}}",
        "",
    ]

    # ---- table ----
    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\linewidth}{lX}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{Claim} \\",
        r"\midrule",
        f"Original & {_escape(original)} \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Natural Rewriting}} \\",
        r"\midrule",
    ]

    for label, cond in NATURAL_CONDITIONS:
        rewritten = nat_lookups.get(cond, {}).get(claim_id)
        if rewritten is None:
            cell = r"\textit{(not available)}"
        else:
            cell = highlight_diff(original, rewritten, "green!20")
        lines.append(f"{label} & {cell} \\\\")

    lines += [
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Controlled Injection}} \\",
        r"\midrule",
    ]

    for label, cond in CONTROLLED_CONDITIONS:
        rewritten = ctrl_lookups.get(cond, {}).get(claim_id)
        if rewritten is None:
            cell = r"\textit{(not available)}"
        else:
            cell = highlight_diff(original, rewritten, "blue!20")
        lines.append(f"{label} & {cell} \\\\")

    escaped_id = claim_id.replace("_", r"\_")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        (
            r"\caption{Rewrite examples for claim \texttt{"
            + escaped_id
            + r"} from SciClaimEval \citep{ho2026sciclaimevalcrossmodalclaimverification}. "
            r"Natural rewriting: LP (Language Polishing), FI (Fluency Improvement), "
            r"AS (Academic Style Alignment), OA (Outcome-Aware), CT (Cautious Tone). "
            r"Controlled injection: C1 (conservative, 1 word), D1 (decorative, 1 word).}"
        ),
        r"\label{tab:rewrite-example}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX rewrite-example table for one SciClaimEval claim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--claim-id", default=None, dest="claim_id",
        help="Specific claim_id to use (default: random sample from split).",
    )
    parser.add_argument(
        "--split", default="dev", choices=["dev", "test"],
        help="Dataset split to draw from.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write output to this .tex file instead of stdout.",
    )
    args = parser.parse_args()

    # ---- load all rewrite lookups ----
    nat_lookups: dict[str, dict[str, str]] = {
        cond: _build_lookup(args.split, "natural", cond)
        for _, cond in NATURAL_CONDITIONS
    }
    ctrl_lookups: dict[str, dict[str, str]] = {
        cond: _build_lookup(args.split, "inject", cond)
        for _, cond in CONTROLLED_CONDITIONS
    }

    # ---- load base dataset ----
    base_items = _load_json(_base_path(args.split))
    base_lookup = {item["claim_id"]: item for item in base_items}

    # ---- resolve claim_id ----
    if args.claim_id is None:
        # Pick a claim that has all natural + both controlled rewrites available.
        candidates = [
            cid for cid in base_lookup
            if all(cid in lut for lut in nat_lookups.values())
            and all(cid in lut for lut in ctrl_lookups.values())
        ]
        if not candidates:
            sys.exit("No claims found with all rewrite conditions available.")
        random.seed(args.seed)
        args.claim_id = random.choice(candidates)
        print(f"Sampled claim_id: {args.claim_id}", file=sys.stderr)

    # ---- generate ----
    tex = generate_table(
        args.claim_id, args.split,
        nat_lookups, ctrl_lookups, base_lookup,
    )

    if args.out:
        Path(args.out).write_text(tex, encoding="utf-8")
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
