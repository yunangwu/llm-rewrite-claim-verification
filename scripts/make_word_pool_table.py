"""
Generate the LaTeX word-pool table from the live POOLS dict in core/rewrite/inject.py.
Usage:
    python scripts/make_word_pool_table.py [--out PATH]
Output defaults to results/word_pool_table.tex.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rewrite.inject import POOLS


def _wrap_words(words: list[str], source_line_width: int = 68) -> str:
    """
    Join words into a single comma-separated string, with soft line breaks
    (newlines only, no LaTeX \\) so the source file stays readable.
    The X column in tabularx wraps the text automatically in LaTeX.
    """
    parts = [w + "," for w in words[:-1]] + [words[-1]]
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in parts:
        if current and current_len + 1 + len(part) > source_line_width:
            lines.append("   " + " ".join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += (1 if current_len else 0) + len(part)
    if current:
        lines.append("   " + " ".join(current))
    return "\n".join(lines)


def generate_table() -> str:
    c_words = POOLS["C"]
    d_words = POOLS["D"]
    n_c = len(c_words)
    n_d = len(d_words)

    c_text = _wrap_words(c_words)
    d_text = _wrap_words(d_words)

    # Rough multirow height estimates (lines of text)
    c_rows = max(1, (n_c * 11) // 70 + 1)
    d_rows = max(1, (n_d * 11) // 70 + 1)

    table = rf"""\begin{{table*}}[h]
\centering
\begin{{tabularx}}{{\textwidth}}{{l X}}
\toprule
\textbf{{Category}} & \textbf{{Words}} \\
\midrule
\multirow{{{c_rows}}}{{*}}{{\shortstack[l]{{Conservative (C)}}}}
 & {c_text} \\
\midrule
\multirow{{{d_rows}}}{{*}}{{\shortstack[l]{{Decorative (D)}}}}
 & {d_text} \\
\bottomrule
\end{{tabularx}}
\caption{{Word pools used in inject rewriting. These words are frequently
injected by LLMs during rewriting and serve as recognizable markers of
LLM-generated text. Decorative words are sourced from
\citet{{doi:10.1126/sciadv.adt3813}} and \citet{{Matsui2025-nj}};
conservative words are drawn from the same sources and supplemented with
LLM-assisted hedging terms. Category C (Conservative) contains {n_c} words;
Category D (Decorative) contains {n_d} words.}}
\label{{tab:word-pools}}
\end{{table*}}
"""
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate word-pool LaTeX table.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: results/word_pool_table.tex)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "word_pool_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tex = generate_table()
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  C pool: {len(POOLS['C'])} words")
    print(f"  D pool: {len(POOLS['D'])} words")


if __name__ == "__main__":
    main()
