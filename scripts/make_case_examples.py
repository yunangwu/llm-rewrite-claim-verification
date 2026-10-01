#!/usr/bin/env python3
"""
LaTeX table of four illustrative cases for Qwen3-VL-8B-Instruct.

  1. High shift under CT    val_fig_0285   hedged true claim becomes accepted
  2. High shift under OA    val_fig_0052   boosted true claim is rejected
  3. No shift under LP      val_tab_0177   polishing leaves p unchanged
  4. No shift despite C1    val_tab_0029   one inserted hedge, p unchanged

All cases come from the val (dev) split.

Picked by hand from the ranked candidates. Every case has an unflagged rewrite
whose factual content matches the original, a readable evidence image, and 10
valid self-consistency samples. Cases 3 and 4 also have base p away from 0 and
1, so the absence of a shift is not a ceiling effect. p is p(Supported).

Writes <out-dir>/case_examples.tex and copies the evidence images next to it,
so the directory can be uploaded to Overleaf as is. Image paths in the .tex are
<out-dir name>/<file>, i.e. relative to the Overleaf project root.

Usage:
  python scripts/make_case_examples.py [--out-dir results/case_examples]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.build_annotation_sheets import _TYPOGRAPHY, diff_runs
from core.config import DATASETS
from core.io import load_base_dataset

MODEL = "Qwen3-VL-8B-Instruct"
IMAGE_HEIGHT = "3.2cm"
CASES = [
    ("high shift", "val_fig_0285", "CT-mix-42", "CT"),
    ("high shift", "val_fig_0052", "OA-mix-42", "OA"),
    ("no shift", "val_tab_0177", "LP-mix-42", "LP"),
    ("no shift despite hedging", "val_tab_0029", "C_1", "C1"),
]
_LATEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
          "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


def tex(text: str) -> str:
    for src, dst in (("\u2014", "---"), ("\u2013", "--"), ("\u201c", "``"), ("\u201d", "''")):
        text = text.replace(src, dst)
    text = text.translate(_TYPOGRAPHY)
    return "".join(_LATEX.get(ch, ch) for ch in text)


def load_run(split: str, condition: str) -> dict[str, dict]:
    path = ROOT / "runs" / "sciclaimeval" / split / MODEL / f"{condition}.json"
    return {r["claim_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "case_examples")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASETS["sciclaimeval"]
    base = {}
    for split in ("dev", "test"):
        for item in load_base_dataset(cfg, split):
            base[item["claim_id"]] = {**item, "split": split}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}lX@{}}",
        r"\toprule",
    ]
    for i, (title, claim_id, condition, label) in enumerate(CASES, 1):
        item = base[claim_id]
        orig = load_run(item["split"], "original")[claim_id]
        rew = load_run(item["split"], condition)[claim_id]
        p0 = float(orig["p_supported_consistency"])
        p1 = float(rew["p_supported_consistency"])
        flipped = orig["pred_label_consistency"] != rew["pred_label_consistency"]
        dp = round(p1 - p0, 1)
        dp_text = f"{dp:+.1f}" if dp else "0.0"
        original = item["claim"].strip()
        rewritten = rew["claim"].strip()
        marked = "".join(
            rf"\textbf{{{tex(seg)}}}" if is_changed else tex(seg)
            for seg, is_changed in diff_runs(original, rewritten)
        )

        src = ROOT / cfg["data_dir"] / item["evi_path"]
        image = f"{claim_id}{src.suffix}"
        shutil.copyfile(src, args.out_dir / image)

        if i > 1:
            lines.append(r"\midrule")
        lines += [
            rf"\multicolumn{{2}}{{@{{}}p{{\textwidth}}@{{}}}}{{\textbf{{Case {i}: {tex(title)}}} \quad "
            rf"\texttt{{{tex(claim_id)}}} \quad Condition: {label} \quad "
            rf"Gold: {item['label']} \quad "
            rf"$p$: {p0:.1f} $\rightarrow$ {p1:.1f} ($\Delta p = {dp_text}$) \quad "
            rf"Flipped: {'yes' if flipped else 'no'}}} \\",
            rf"Original & {tex(original)} \\",
            rf"Rewritten & {marked} \\",
            rf"Evidence & \hspace*{{\fill}}\raisebox{{-0.9\height}}{{\includegraphics[width=\linewidth,"
            rf"height={IMAGE_HEIGHT},keepaspectratio]{{{args.out_dir.name}/{image}}}}}\hspace*{{\fill}} \\",
        ]
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Example cases for Qwen3-VL-8B-Instruct. $p$ is the probability of "
        r"\emph{Supported} estimated from 10 sampled answers, before and after rewriting. "
        r"Words changed by the rewrite are in bold.}",
        r"\label{tab:case-examples}",
        r"\end{table*}",
    ]
    out = args.out_dir / "case_examples.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
