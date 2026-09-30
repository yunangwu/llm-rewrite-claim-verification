"""
Confidence-window drift analysis (rebuttal experiment).

Question addressed
------------------
"Does restricting to *unclear* items (base p(supported) in [0.3, 0.7]) produce
larger drift than the full set?"

For every model x condition it recomputes the per-claim drift
    Δp = p_supported(rewrite) − p_supported(original)
(averaged across variants, exactly as the main ``analyze`` pipeline does) and
splits claims into an *in-window* group (lo <= base p <= hi) and the rest.
It reports, per condition, both the **magnitude** |Δp| and the **signed** Δp for
the in-window subset next to the full-set value, plus the ratio and a one-sided
bootstrap p-value for "in-window |Δp| > out-of-window |Δp|".

Interpretation caveat (baked into the output)
--------------------------------------------
Δp is a *bounded* quantity: an item at base p=0.95 can only rise +0.05 but fall
−0.95, whereas an item at p=0.5 can move ±0.5. So near-boundary items show larger
|Δp| almost mechanically (ceiling/floor + regression to the mean). The defensible
reading is therefore **within-window, across conditions**: compare a style
condition (e.g. CT hedging) against a surface-only control (e.g. LP grammar) in
the *same* ``|Δp|_win`` column — both share identical headroom, so any gap is a
genuine style effect rather than an artifact.

Usage (standalone — reads existing runs/, writes nothing unless --out-dir)
--------------------------------------------------------------------------
  # Reproduce the submission setting (dev+test pooled, mix-42, consistency):
  python -m core.analysis.drift_window --dataset sciclaimeval --split dev test

  # Custom window and a single split:
  python -m core.analysis.drift_window --dataset sciclaimeval --split dev \
      --window 0.25 0.75

  # Individual variants instead of mix files, and save artifacts:
  python -m core.analysis.drift_window --dataset sciclaimeval --split dev test \
      --no-mix --out-dir results/sciclaimeval/dev+test/consistency/drift_window
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Condition resolution (mirrors run.py analyze conventions)
# ---------------------------------------------------------------------------


def _condition_variants(cond: str, mix_seed: int | None) -> list[str]:
    """Resolve a condition name to the variant file(s) that back it."""
    from core.config import NATURAL_GROUPS

    if cond in NATURAL_GROUPS:
        if mix_seed is not None:
            return [f"{cond}-mix-{mix_seed}"]
        return list(NATURAL_GROUPS[cond])
    return [cond]


def _method_for(variant: str) -> str:
    return "inject" if variant in ("D_1", "D_2", "C_1", "C_2") else "natural"


# ---------------------------------------------------------------------------
# Prediction loading (pooled across splits, like _analyze)
# ---------------------------------------------------------------------------


def _load_merged(
    dataset: str,
    splits: Sequence[str],
    slug: str,
    variant: str,
    valid_ids: set,
    *,
    pred_key: str,
    prob_key: str,
    id_to_label: dict[str, str],
) -> dict:
    """Load + merge predictions for one model/condition across all splits."""
    from core.analysis.metrics import load_predictions
    from core.io import run_exists, run_path

    merged: dict = {}
    for sp in splits:
        if run_exists(dataset, sp, slug, variant):
            preds = load_predictions(
                run_path(dataset, sp, slug, variant), valid_ids,
                pred_key=pred_key, prob_key=prob_key,
            )
            merged.update(preds)
    for cid, pred in merged.items():
        if not pred.get("label") and cid in id_to_label:
            pred["label"] = id_to_label[cid]
    return merged


def _claim_deltas(
    base_preds: dict,
    variant_preds_list: list[dict],
    valid_ids: set,
) -> list[tuple[float, float, str]]:
    """
    Per-claim (base_p, mean Δp across variants, label) for one condition.

    Mirrors ``metrics.compute_claim_deltas`` but keeps the base probability and
    label alongside each delta so we can bin by confidence.
    """
    base_p = {
        cid: pred["p_supported"]
        for cid, pred in base_preds.items()
        if cid in valid_ids and pred["p_supported"] is not None
    }
    per_claim: dict[str, list[float]] = defaultdict(list)
    for preds in variant_preds_list:
        for cid, pred in preds.items():
            if cid in base_p and pred["p_supported"] is not None:
                per_claim[cid].append(pred["p_supported"] - base_p[cid])

    rows: list[tuple[float, float, str]] = []
    for cid, dps in per_claim.items():
        if dps:
            rows.append((base_p[cid], float(np.mean(dps)), base_preds[cid]["label"]))
    return rows


# ---------------------------------------------------------------------------
# Window statistics
# ---------------------------------------------------------------------------


def _boot_one_sided_p(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> float:
    """
    One-sided bootstrap p-value that mean(a) > mean(b).

    Resamples each group independently ``n_boot`` times and returns the fraction
    of resamples in which mean(a) − mean(b) <= 0 (small p ⇒ a robustly larger).
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return float(np.mean(diffs <= 0.0))


def _group_stat(dp: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {"n": 0, "abs_dp": float("nan"), "signed_dp": float("nan")}
    sel = dp[mask]
    return {
        "n": int(mask.sum()),
        "abs_dp": float(np.mean(np.abs(sel))),
        "signed_dp": float(np.mean(sel)),
    }


def window_stats(
    rows: list[tuple[float, float, str]],
    lo: float,
    hi: float,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Split per-claim (base_p, Δp, label) rows into in-/out-of-window groups."""
    if not rows:
        return {}
    p = np.array([r[0] for r in rows])
    dp = np.array([r[1] for r in rows])
    lab = np.array([r[2] for r in rows])

    in_mask = (p >= lo) & (p <= hi)
    out_mask = ~in_mask
    all_mask = np.ones(len(p), dtype=bool)

    win = _group_stat(dp, in_mask)
    out = _group_stat(dp, out_mask)
    alls = _group_stat(dp, all_mask)

    # Signed drift within the window, per true-label group (direction matters).
    win_signed_by_label: dict[str, float] = {}
    for label in ("Supported", "Refuted"):
        m = in_mask & (lab == label)
        win_signed_by_label[label] = float(np.mean(dp[m])) if m.sum() else float("nan")

    ratio = win["abs_dp"] / alls["abs_dp"] if alls["abs_dp"] else float("nan")
    p_win_gt_out = _boot_one_sided_p(
        np.abs(dp[in_mask]), np.abs(dp[out_mask]), n_boot, seed
    )

    return {
        "win": win,
        "out": out,
        "all": alls,
        "abs_ratio_win_over_all": ratio,
        "p_win_gt_out": p_win_gt_out,
        "win_signed_by_label": win_signed_by_label,
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
    lo: float,
    hi: float,
    n_boot: int = 2000,
    verbose: bool = True,
) -> dict:
    """
    Returns
    -------
    {
      "meta":     {...},
      "pooled":   {cond: window_stats},        # all models pooled by claim
      "per_model":{slug: {cond: window_stats}},
    }
    """
    from core.analysis.metrics import build_pair_mapping
    from core.config import DATASETS, EVAL_MODES, sort_model_slugs
    from core.io import list_model_slugs, load_base_dataset, run_exists

    if eval_mode not in EVAL_MODES:
        sys.exit(f"Unknown eval mode {eval_mode!r}. Choose from: {list(EVAL_MODES)}")
    pred_key, prob_key = EVAL_MODES[eval_mode]

    dataset_cfg = DATASETS[dataset]

    # Pool base data across splits; prefix pair ids so they don't collide.
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

    _complete_pairs, valid_ids = build_pair_mapping(base_data)
    id_to_label = {it["claim_id"]: it["label"] for it in base_data if it.get("label")}

    if models:
        slugs = sort_model_slugs(list(models))
    else:
        found: set = set()
        for sp in splits:
            found.update(list_model_slugs(dataset, sp))
        slugs = sort_model_slugs(list(found))
    if not slugs:
        sys.exit(f"No model run directories found under runs/{dataset}/{{{','.join(splits)}}}/")

    per_model: dict[str, dict] = {}
    pooled_rows: dict[str, list] = defaultdict(list)
    base_cov: dict[str, dict] = {}

    for slug in slugs:
        if not any(run_exists(dataset, sp, slug, "original") for sp in splits):
            if verbose:
                print(f"  [{slug}] no original.json — skipping", file=sys.stderr)
            continue
        base_preds = _load_merged(
            dataset, splits, slug, "original", valid_ids,
            pred_key=pred_key, prob_key=prob_key, id_to_label=id_to_label,
        )
        # Base in-window coverage: fraction of analysed claims whose *original*
        # p(supported) falls in [lo, hi]. Condition-independent (depends only on
        # the base run), so it defines the "unclear" slice for this model.
        base_pv = [
            p["p_supported"] for cid, p in base_preds.items()
            if cid in valid_ids and p["p_supported"] is not None
        ]
        base_cov[slug] = {
            "n_all": len(base_pv),
            "n_win": sum(1 for v in base_pv if lo <= v <= hi),
        }
        per_model[slug] = {}
        for cond in conditions:
            variants = _condition_variants(cond, mix_seed)
            variant_preds = []
            for v in variants:
                vp = _load_merged(
                    dataset, splits, slug, v, valid_ids,
                    pred_key=pred_key, prob_key=prob_key, id_to_label=id_to_label,
                )
                if vp:
                    variant_preds.append(vp)
            if not variant_preds:
                continue
            rows = _claim_deltas(base_preds, variant_preds, valid_ids)
            if not rows:
                continue
            per_model[slug][cond] = window_stats(rows, lo, hi, n_boot=n_boot)
            pooled_rows[cond].extend(rows)
        if verbose:
            print(f"  [{slug}] done ({len(per_model[slug])} conditions)", file=sys.stderr)

    pooled = {
        cond: window_stats(rows, lo, hi, n_boot=n_boot)
        for cond, rows in pooled_rows.items()
    }

    # Pooled base coverage over models that contributed at least one condition.
    contributing = [s for s in per_model if per_model[s]]
    n_win_total = sum(base_cov[s]["n_win"] for s in contributing if s in base_cov)
    n_all_total = sum(base_cov[s]["n_all"] for s in contributing if s in base_cov)
    coverage = {
        "per_model": {s: base_cov[s] for s in contributing if s in base_cov},
        "n_win": n_win_total,
        "n_all": n_all_total,
        "pct": (100.0 * n_win_total / n_all_total) if n_all_total else float("nan"),
    }

    return {
        "meta": {
            "dataset": dataset,
            "splits": list(splits),
            "conditions": list(conditions),
            "mix_seed": mix_seed,
            "eval_mode": eval_mode,
            "window": [lo, hi],
            "n_models": len(per_model),
            "models": list(per_model.keys()),
        },
        "coverage": coverage,
        "pooled": pooled,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 4) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def print_report(stats_by_cond: dict, title: str) -> None:
    header = (
        f"{'Cond':<7} {'N_win':>6} {'N_all':>6} "
        f"{'|Δp|_win':>9} {'|Δp|_all':>9} {'ratio':>6} "
        f"{'Δp_win':>8} {'Δp_all':>8} {'p(win>out)':>11}"
    )
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for cond, s in stats_by_cond.items():
        if not s:
            continue
        print(
            f"{cond:<7} {s['win']['n']:>6d} {s['all']['n']:>6d} "
            f"{_fmt(s['win']['abs_dp']):>9} {_fmt(s['all']['abs_dp']):>9} "
            f"{_fmt(s['abs_ratio_win_over_all'], 2):>6} "
            f"{_fmt(s['win']['signed_dp']):>8} {_fmt(s['all']['signed_dp']):>8} "
            f"{_fmt(s['p_win_gt_out'], 3):>11}"
        )


def print_central_summary(
    result: dict,
    lo: float,
    hi: float,
    *,
    ref: str = "LP",
    test_conds: Sequence[str] = ("CT", "C_1"),
) -> None:
    """
    Print the headline numbers for the [lo, hi] "unclear items" experiment:

      1. % of data in the window p in [lo, hi] (pooled over all models);
      2. pooled in-window signed drift (Δp_win) for the test conditions + ref;
      3. per-model check that each test condition beats the reference
         (in-window Δp_win), and how many models pass ``all`` of them.
    """
    cov = result.get("coverage", {})
    pooled = result.get("pooled", {})
    per_model = result.get("per_model", {})
    n_models = result["meta"]["n_models"]

    def win_dp(stats: dict, cond: str) -> float:
        s = stats.get(cond)
        if not s:
            return float("nan")
        return s.get("win", {}).get("signed_dp", float("nan"))

    print(f"\n=== Central experiment: unclear items p in [{lo:g}, {hi:g}] ===")

    # (1) coverage
    print(
        f"[1] In-window coverage (pooled over {len(cov.get('per_model', {}))} "
        f"contributing models): {cov.get('n_win', 0)} / {cov.get('n_all', 0)} "
        f"= {cov.get('pct', float('nan')):.1f}%"
    )

    # (2) pooled in-window signed drift for ref + test conditions
    print("[2] Pooled in-window signed drift (Δp_win):")
    for cond in [ref, *test_conds]:
        print(f"      {cond:<5} = {_fmt(win_dp(pooled, cond), 4)}")

    # (3) per-model test: every test condition > ref (in-window Δp_win)
    print(f"[3] Per-model test — each of {list(test_conds)} > {ref} (Δp_win):")
    hdr = f"      {'model':<28} {ref:>8} " + " ".join(f"{c:>8}" for c in test_conds) + "   pass"
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    n_pass_all = 0
    n_pass_each = {c: 0 for c in test_conds}
    n_eval = 0
    for slug, stats in per_model.items():
        if not stats:
            continue
        ref_v = win_dp(stats, ref)
        if ref_v != ref_v:  # ref missing for this model
            continue
        n_eval += 1
        checks = {}
        for c in test_conds:
            v = win_dp(stats, c)
            ok = (v == v) and (v > ref_v)
            checks[c] = ok
            if ok:
                n_pass_each[c] += 1
        all_ok = all(checks.values())
        if all_ok:
            n_pass_all += 1
        cells = " ".join(f"{_fmt(win_dp(stats, c), 3):>8}" for c in test_conds)
        marks = "".join("Y" if checks[c] else "n" for c in test_conds)
        print(f"      {slug:<28} {_fmt(ref_v, 3):>8} {cells}   {marks}")
    conds_join = " AND ".join(f"{c}>{ref}" for c in test_conds)
    print(f"      => {n_pass_all}/{n_eval} models satisfy ({conds_join})")
    for c in test_conds:
        print(f"         {c}>{ref}: {n_pass_each[c]}/{n_eval}")


def generate_table(
    pooled: dict,
    window: tuple[float, float],
    *,
    caption: str | None = None,
    label: str = "tab:drift_window",
) -> str:
    """LaTeX table of pooled (all-models) in-window vs full-set drift."""
    lo, hi = window
    if caption is None:
        caption = (
            f"Drift restricted to uncertain items (base $p_{{\\mathrm{{sup}}}} \\in "
            f"[{lo:g}, {hi:g}]$) versus the full set, pooled across models. "
            f"$\\overline{{|\\Delta p|}}$ is mean drift magnitude; "
            f"$\\overline{{\\Delta p}}$ is signed drift; "
            f"$p$ is a one-sided bootstrap that in-window magnitude exceeds "
            f"out-of-window. Compare conditions \\emph{{within}} the "
            f"$\\overline{{|\\Delta p|}}_{{\\mathrm{{win}}}}$ column to control for "
            f"the bounded-variable (headroom) confound."
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Cond & $N_{\text{win}}$ & $\overline{|\Delta p|}_{\text{win}}$ & "
        r"$\overline{|\Delta p|}_{\text{all}}$ & ratio & "
        r"$\overline{\Delta p}_{\text{win}}$ & $\overline{\Delta p}_{\text{all}}$ \\",
        r"\midrule",
    ]
    for cond, s in pooled.items():
        if not s:
            continue
        disp = cond.replace("_", r"\_")
        lines.append(
            f"{disp} & {s['win']['n']} & "
            f"{_fmt(s['win']['abs_dp'])} & {_fmt(s['all']['abs_dp'])} & "
            f"{_fmt(s['abs_ratio_win_over_all'], 2)} & "
            f"{_fmt(s['win']['signed_dp'], 3)} & {_fmt(s['all']['signed_dp'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    from core.config import DATASETS

    p = argparse.ArgumentParser(
        description="Confidence-window drift analysis (in-window vs full-set Δp)."
    )
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--split", nargs="+", default=["dev", "test"],
                   help="Split(s); pass multiple to pool (default: dev test)")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Conditions (default: LP FI AS OA CT D_1 C_1)")
    p.add_argument("--window", nargs=2, type=float, default=[0.3, 0.7],
                   metavar=("LO", "HI"),
                   help="Base p(supported) window for 'unclear' items (default: 0.3 0.7)")
    p.add_argument("--mix-seed", type=int, default=42, dest="mix_seed",
                   help="Resolve natural groups to their mix file (default: 42)")
    p.add_argument("--no-mix", action="store_true", dest="no_mix",
                   help="Use the 5 individual variants per group instead of the mix file")
    p.add_argument("--eval-mode", default="consistency", dest="eval_mode",
                   choices=["consistency"])
    p.add_argument("--models", nargs="*", default=None,
                   help="Model slugs (default: all found under runs/<dataset>/<split>/)")
    p.add_argument("--n-boot", type=int, default=2000, dest="n_boot",
                   help="Bootstrap resamples for the in>out test (default: 2000)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="If set, write drift_window.{json,tex} here (default: print only)")
    p.add_argument("--per-model", action="store_true", dest="per_model",
                   help="Also print a per-model table to stdout")
    return p


def main(argv: list[str] | None = None) -> None:
    from core.config import CONTROLLED_CONDITIONS, NATURAL_GROUPS

    args = _build_parser().parse_args(argv)
    conditions = args.conditions or (list(NATURAL_GROUPS) + CONTROLLED_CONDITIONS)
    mix_seed = None if args.no_mix else args.mix_seed
    lo, hi = args.window

    print(
        f"dataset={args.dataset} split={args.split} window=[{lo:g},{hi:g}] "
        f"mix_seed={mix_seed} eval_mode={args.eval_mode}",
        file=sys.stderr,
    )

    result = run_analysis(
        args.dataset, args.split, conditions,
        mix_seed=mix_seed, eval_mode=args.eval_mode, models=args.models,
        lo=lo, hi=hi, n_boot=args.n_boot, verbose=True,
    )

    if args.per_model:
        for slug, stats in result["per_model"].items():
            print_report(stats, f"per-model: {slug}")

    print_report(result["pooled"], f"pooled across {result['meta']['n_models']} models")

    print_central_summary(result, lo, hi)

    print("\n=== LaTeX (pooled) ===")
    print(generate_table(result["pooled"], (lo, hi)))

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "drift_window.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out / "drift_window.tex").write_text(
            generate_table(result["pooled"], (lo, hi)), encoding="utf-8"
        )
        print(
            f"\nSaved → {out/'drift_window.json'}  and  {out/'drift_window.tex'}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
