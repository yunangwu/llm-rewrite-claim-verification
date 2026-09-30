"""
Single CLI entrypoint for the core experiment pipeline.

Subcommands
-----------
  rewrite natural   --dataset sciclaimeval --gen-model <model> [--conditions LP-1 LP-2 ...]
  rewrite inject    --dataset sciclaimeval --gen-model <model> [--conditions D_1 C_2 ...]

  infer             --dataset sciclaimeval --model <hf-model-id>
                    [--conditions original LP-1 D_1 ...] [--resume] [--limit N]

  analyze           --dataset sciclaimeval [--eval-mode consistency|direct|self_reported]
                    [--models slug1 slug2 ...] [--out-dir path]

Usage examples
--------------
  python -m core.run rewrite natural --dataset sciclaimeval --gen-model openai/gpt-4o
  python -m core.run rewrite inject  --dataset sciclaimeval --gen-model openai/gpt-4o
  python -m core.run infer --dataset sciclaimeval --model Qwen/Qwen3-VL-32B-Instruct --resume
  python -m core.run analyze --dataset sciclaimeval --eval-mode consistency
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# rewrite subcommands
# ---------------------------------------------------------------------------


def _rewrite_natural(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from core.config import DATASETS, NATURAL_GROUPS, load_rewrite_cfg
    from core.io import load_base_dataset, rewrite_path, save_rewrite
    from core.rewrite.natural import BARE_OUTPUT_SYSTEM_PROMPT, run_natural_rewrite
    from core.rewrite.prompt import PROMPTS

    cfg = load_rewrite_cfg()["natural"]
    dataset_cfg = DATASETS[args.dataset]
    items = load_base_dataset(dataset_cfg, args.split)

    # Opt-in format-only system prompt for chatty rewriters (e.g. gemma-4).
    # Off by default → requests are byte-identical to the original run.
    system_prompt = BARE_OUTPUT_SYSTEM_PROMPT if getattr(args, "bare_output", False) else None
    if system_prompt:
        print("  [bare-output] format-only system prompt ENABLED "
              "(non-default; keep OFF to reproduce the original run)")

    conditions = (
        args.conditions
        if args.conditions
        else [v for variants in NATURAL_GROUPS.values() for v in variants]
    )

    client = OpenAI(base_url=args.base_url, api_key="dummy", timeout=120.0)

    for cond in conditions:
        if cond not in PROMPTS:
            print(f"  [SKIP] No prompt for condition {cond!r}", file=sys.stderr)
            continue
        out_path = rewrite_path(args.dataset, args.split, "natural", cond)
        if args.resume and out_path.exists():
            print(f"  [SKIP] {out_path} already exists (--resume)")
            continue
        print(f"  Rewriting {len(items)} items → {cond}  (split={args.split})")
        data = items[:args.limit] if args.limit else items
        rewritten = run_natural_rewrite(
            data, cond, client, args.gen_model,
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 512),
            workers=cfg.get("workers", 8),
            system_prompt=system_prompt,
        )
        saved = save_rewrite(rewritten, args.dataset, args.split, "natural", cond)
        print(f"  Wrote {saved} ({len(rewritten)} items)")


def _rewrite_inject(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from core.config import CONTROLLED_CONDITIONS, DATASETS, load_rewrite_cfg
    from core.io import load_base_dataset, rewrite_path, save_rewrite
    from core.rewrite.inject import CONDITION_SPECS, run_inject_rewrite

    cfg = load_rewrite_cfg()["inject"]
    dataset_cfg = DATASETS[args.dataset]
    items = load_base_dataset(dataset_cfg, args.split)

    conditions = args.conditions if args.conditions else CONTROLLED_CONDITIONS

    client = OpenAI(base_url=args.base_url, api_key="dummy", timeout=120.0)

    for cond in conditions:
        if cond not in CONDITION_SPECS:
            print(f"  [SKIP] Unknown inject condition {cond!r}", file=sys.stderr)
            continue
        out_path = rewrite_path(args.dataset, args.split, "inject", cond)
        if args.resume and out_path.exists():
            print(f"  [SKIP] {out_path} already exists (--resume)")
            continue
        print(f"  Injecting {len(items)} items → {cond}  (split={args.split})")
        data = items[:args.limit] if args.limit else items
        rewritten = run_inject_rewrite(
            data, cond, client, args.gen_model,
            temperature=cfg.get("temperature", 0.25),
            max_tokens=cfg.get("max_tokens", 512),
            workers=cfg.get("workers", 8),
            max_retries=cfg.get("max_retries", 3),
            min_distance=cfg.get("min_distance", 3),
        )
        saved = save_rewrite(rewritten, args.dataset, args.split, "inject", cond)
        print(f"  Wrote {saved} ({len(rewritten)} items)")


# ---------------------------------------------------------------------------
# rewrite mix subcommand
# ---------------------------------------------------------------------------


def _rewrite_mix(args: argparse.Namespace) -> None:
    from core.config import DATASETS, NATURAL_GROUPS
    from core.io import load_rewrite, rewrite_path, save_rewrite
    from core.rewrite.mix import create_mix_rewrite

    dataset_cfg = DATASETS[args.dataset]  # noqa: F841 — validates dataset key
    groups = args.groups if args.groups else list(NATURAL_GROUPS)

    for group in groups:
        if group not in NATURAL_GROUPS:
            print(f"  [SKIP] Unknown group {group!r}", file=sys.stderr)
            continue

        variants = NATURAL_GROUPS[group]
        condition_name = f"{group}-mix-{args.seed}"
        out_path = rewrite_path(args.dataset, args.split, "natural", condition_name)

        if args.resume and out_path.exists():
            print(f"  [SKIP] {out_path} already exists (--resume)")
            continue

        # Load all variant files, abort the group if any are missing
        variant_items_list: list[list[dict]] = []
        missing = False
        for v in variants:
            vpath = rewrite_path(args.dataset, args.split, "natural", v)
            if not vpath.exists():
                print(
                    f"  [SKIP] {group}: missing rewrite file for {v!r} — "
                    f"run 'rewrite natural' first",
                    file=sys.stderr,
                )
                missing = True
                break
            variant_items_list.append(load_rewrite(args.dataset, args.split, "natural", v))

        if missing:
            continue

        n = len(variant_items_list[0])
        print(f"  Mixing {n} items → {condition_name}  (seed={args.seed})")
        mixed = create_mix_rewrite(variant_items_list, variants, args.seed, group)
        saved = save_rewrite(mixed, args.dataset, args.split, "natural", condition_name)
        print(f"  Wrote {saved} ({len(mixed)} items)")


# ---------------------------------------------------------------------------
# infer subcommand
# ---------------------------------------------------------------------------


def _infer(args: argparse.Namespace) -> None:
    from core.config import (
        CONTROLLED_CONDITIONS, DATASETS, NATURAL_GROUPS, ALL_VARIANTS,
        load_inference_cfg, load_models_cfg, model_slug,
    )
    from core.inference.backend import create_backend
    from core.inference.pipeline import run_dataset
    from core.io import (
        load_base_dataset, load_rewrite, run_exists, save_run, save_run_meta,
    )

    inference_cfg = load_inference_cfg()
    models_cfg    = load_models_cfg()
    dataset_cfg   = DATASETS[args.dataset]
    slug          = model_slug(args.model)
    data_dir      = str(
        (Path(__file__).resolve().parents[1] / dataset_cfg["data_dir"]).resolve()
    )

    # Per-model overrides from models.yaml
    from core.inference.prompt import MAX_TOKENS_COT
    model_cfg      = models_cfg.get(args.model, {})
    concurrency    = model_cfg.get("concurrency", inference_cfg.get("concurrency", 8))
    max_tokens_cot = model_cfg.get("max_tokens_cot", MAX_TOKENS_COT)

    # Resolve conditions to run
    if args.conditions:
        conditions = args.conditions
    elif getattr(args, "mix_seed", None) is not None:
        mix_conds = [f"{g}-mix-{args.mix_seed}" for g in NATURAL_GROUPS]
        conditions = ["original"] + mix_conds + CONTROLLED_CONDITIONS
    else:
        conditions = ["original"] + ALL_VARIANTS

    backend = create_backend(args.model, inference_cfg, models_cfg)

    consistency_n = (
        args.consistency_n
        if args.consistency_n is not None
        else inference_cfg.get("consistency_n", 10)
    )

    meta = {
        "model":            args.model,
        "dataset":          args.dataset,
        "split":            args.split,
        "consistency_n":    consistency_n,
        "consistency_temp": inference_cfg.get("consistency_temp", 1.0),
        "concurrency":      concurrency,
    }
    save_run_meta(meta, args.dataset, args.split, slug)

    for cond in conditions:
        if args.resume and run_exists(args.dataset, args.split, slug, cond):
            print(f"  [SKIP] {args.dataset}/{args.split}/{slug}/{cond}.json exists (--resume)")
            continue

        # Load dataset for this condition
        if cond == "original":
            items = load_base_dataset(dataset_cfg, args.split)
        else:
            method = "inject" if cond in ("D_1", "D_2", "C_1", "C_2") else "natural"
            try:
                items = load_rewrite(args.dataset, args.split, method, cond)
            except FileNotFoundError:
                print(f"  [SKIP] Rewrite file not found for {cond} — run 'rewrite' first")
                continue

        print(f"\n{'='*60}")
        print(f"  model={slug}  dataset={args.dataset}  split={args.split}  condition={cond}  n={len(items)}")
        if args.limit:
            items = items[:args.limit]

        results = asyncio.run(
            run_dataset(
                backend, items, data_dir,
                concurrency=concurrency,
                consistency_n=consistency_n,
                consistency_temp=inference_cfg.get("consistency_temp", 1.0),
                parse_retries=inference_cfg.get("parse_retries", 3),
                store_responses=inference_cfg.get("store_responses", False),
                fmt=dataset_cfg.get("format", "sciclaimeval"),
                limit=None,  # already sliced above
                skip_missing_images=args.skip_missing_images,
                desc=cond,
                max_tokens_cot=max_tokens_cot,
            )
        )
        saved = save_run(results, args.dataset, args.split, slug, cond)
        print(f"  → Saved {saved} ({len(results)} items)")


# ---------------------------------------------------------------------------
# analyze subcommand
# ---------------------------------------------------------------------------


def _analyze(args: argparse.Namespace) -> None:
    from collections import defaultdict

    import numpy as np

    from core.config import (
        CONTROLLED_CONDITIONS, DATASETS, EVAL_MODES, NATURAL_GROUPS,
        ALL_CONDITIONS, DEFAULT_EVAL_MODE, load_inference_cfg,
        sort_model_slugs,
    )
    from core.analysis.compass import (
        plot_compass_overview, plot_compass_per_model, plot_compass_single,
    )
    from core.analysis.boundary import collect_boundary_data, compute_flip_rates, plot_boundary_scatter
    from core.analysis.drift import collect_drift_data, plot_drift
    from core.analysis.metrics import (
        average_shifts, build_pair_mapping, compute_binned_drift,
        compute_claim_deltas, compute_label_shifts, compute_performance,
        compute_quadrant_proportions, compute_shifts, load_predictions,
        summarize_label_shifts, summarize_shifts, validate_rewrite_claims,
    )
    from core.analysis.stats import (
        bh_fdr, bootstrap_mean_p, f1_pvalue, pair_acc_pvalue, shift_pvalue,
    )
    from core.analysis.strip import plot_strip_chart
    from core.analysis.tables import generate_flip_rate_table, generate_performance_table, generate_shift_table
    from core.io import (
        list_model_slugs, load_base_dataset, load_rewrite, load_run, results_dir,
        run_exists, save_figure, save_text,
    )

    tables_only = bool(getattr(args, "tables_only", False))

    # In mix mode, each natural group resolves to a single mix file instead of 5 variants.
    if getattr(args, "mix_seed", None) is not None:
        NATURAL_GROUPS = {g: [f"{g}-mix-{args.mix_seed}"] for g in NATURAL_GROUPS}

    eval_mode = args.eval_mode or DEFAULT_EVAL_MODE
    if eval_mode not in EVAL_MODES:
        sys.exit(f"Unknown eval mode {eval_mode!r}. Choose from: {list(EVAL_MODES)}")
    pred_key, prob_key = EVAL_MODES[eval_mode]

    from core.io import run_path as _rp

    splits      = args.split          # list[str] — nargs="+" on analyze
    split_label = "+".join(splits) if len(splits) > 1 else splits[0]

    dataset_cfg = DATASETS[args.dataset]
    has_pairs   = dataset_cfg.get("has_pairs", True)
    base_data: list = []
    for _sp in splits:
        sp_data = load_base_dataset(dataset_cfg, _sp)
        if len(splits) > 1:
            # Prefix claim_id_pair with the split name so pair IDs from different
            # splits don't collide (e.g. both dev and test use "0001", "0002"…).
            sp_data = [
                {**item, "claim_id_pair": f"{_sp}:{item['claim_id_pair']}"}
                if "claim_id_pair" in item else item
                for item in sp_data
            ]
        base_data.extend(sp_data)
    complete_pairs, valid_ids = build_pair_mapping(base_data)
    # Ground-truth label map — used to patch run files that omit or blank the label field.
    id_to_label: dict[str, str] = {
        item["claim_id"]: item["label"]
        for item in base_data
        if item.get("label")
    }
    if has_pairs:
        print(f"Pairs: {len(complete_pairs)} complete, {len(valid_ids)} valid claim_ids")
    else:
        print(f"Pairless dataset: {len(valid_ids)} items (compass/pair_acc skipped)")

    if args.models:
        slugs = sort_model_slugs(list(args.models))
    else:
        _all_slugs: set[str] = set()
        for _sp in splits:
            _all_slugs.update(list_model_slugs(args.dataset, _sp))
        slugs = sort_model_slugs(list(_all_slugs))
    if not slugs:
        sys.exit(f"No model run directories found under runs/{args.dataset}/{{{split_label}}}/")

    out_root = (
        Path(args.out_dir) if args.out_dir
        else results_dir(args.dataset, split_label, eval_mode)
    )
    print(f"Output → {out_root}  (split={split_label}  eval_mode={eval_mode})")

    all_perf:  dict = {}
    all_shift: dict = {}
    all_raw:   dict = {}  # {slug: {cond: [shift, ...]}}
    all_flip:  dict = {}  # {slug: {cond: {"flip_sup": float, "flip_ref": float}}}
    filter_stats: dict = {}  # {variant_key: {split: stats_dict}} accumulated across models
    strip_rows: list = []
    # Raw (uncorrected) per-cell p-values. Filled in below per (model, cond).
    perf_pvals:  dict[str, dict[str, dict[str, float]]] = {}
    shift_pvals: dict[str, dict[str, dict[str, float]]] = {}
    # Per-cell sample sizes (for caption / debug output).
    perf_nused:  dict[str, dict[str, dict[str, int]]] = {}
    shift_nused: dict[str, dict[str, dict[str, int]]] = {}

    n_slugs = len(slugs)
    for slug_idx, slug in enumerate(slugs, 1):
        print(f"\n{'='*60}\n  [{slug_idx}/{n_slugs}] {slug}")

        def _exists(cond: str) -> bool:
            return any(run_exists(args.dataset, sp, slug, cond) for sp in splits)

        def _load_preds(cond: str) -> dict:
            """Load and merge predictions from all splits for this model/condition."""
            merged: dict = {}
            for sp in splits:
                if run_exists(args.dataset, sp, slug, cond):
                    preds = load_predictions(
                        _rp(args.dataset, sp, slug, cond), valid_ids,
                        pred_key=pred_key, prob_key=prob_key,
                    )
                    merged.update(preds)
            # Patch true labels from base data — run files may store "" when the
            # inference job predated the label field or used rewrite items that
            # didn't carry it through.
            for cid, pred in merged.items():
                if not pred.get("label") and cid in id_to_label:
                    pred["label"] = id_to_label[cid]
            return merged

        if not _exists("original"):
            print(f"  [SKIP] No original.json found")
            continue

        base_preds = _load_preds("original")
        base_claims = {cid: p.get("claim", "") for cid, p in base_preds.items()}

        # ---- Rewrite validation ----
        # For each variant file, check for empty/broken claims and propagate
        # drops to the paired partner so pair integrity is preserved.
        # Results are cached: {variant_key: set of clean claim_ids}.
        _variant_clean_ids: dict[str, set[str]] = {}

        def _validated_ids_for(variant_key: str) -> set[str]:
            """Return the clean valid_ids for one variant (cached), validating ALL splits."""
            if variant_key in _variant_clean_ids:
                return _variant_clean_ids[variant_key]
            inject_keys = {"D_1", "D_2", "C_1", "C_2"}
            method = "inject" if variant_key in inject_keys else "natural"
            # Collect broken IDs from every split and union them so that a broken
            # claim in *any* split is excluded from the pooled analysis.
            all_broken: set[str] = set()
            found_any = False
            for sp in splits:
                try:
                    items = load_rewrite(args.dataset, sp, method, variant_key)
                    clean, vstats = validate_rewrite_claims(
                        items, base_claims, complete_pairs
                    )
                    found_any = True
                    broken_sp = {item["claim_id"] for item in items
                                 if item.get("claim_id") not in clean}
                    all_broken |= broken_sp
                    # Accumulate per-split stats (rewrite files are model-agnostic).
                    key = f"{variant_key}:{sp}"
                    if key not in filter_stats:
                        filter_stats[key] = {
                            "variant":        variant_key,
                            "split":          sp,
                            "n_total":        vstats["n_total"],
                            "n_broken":       vstats["n_broken"],
                            "n_empty":        vstats["n_empty"],
                            "n_too_short":    vstats["n_too_short"],
                            "n_pair_dropped": vstats["n_pair_dropped"],
                        }
                    if vstats["n_broken"] > 0:
                        print(
                            f"  [FILTER] {variant_key} ({sp}): "
                            f"{vstats['n_broken']}/{vstats['n_total']} broken claims removed "
                            f"({vstats['n_empty']} empty, {vstats['n_too_short']} too short, "
                            f"{vstats['n_pair_dropped']} pairs dropped due to partner)",
                            flush=True,
                        )
                except FileNotFoundError:
                    pass
            if not found_any:
                # Rewrite file not found for any split — fall back to global valid_ids
                _variant_clean_ids[variant_key] = valid_ids
                return valid_ids
            _variant_clean_ids[variant_key] = valid_ids - all_broken
            return _variant_clean_ids[variant_key]

        def _load_preds_clean(variant_key: str) -> dict:
            """Load predictions filtered to the validated clean ids for this variant."""
            clean = _validated_ids_for(variant_key)
            merged: dict = {}
            for sp in splits:
                if run_exists(args.dataset, sp, slug, variant_key):
                    preds = load_predictions(
                        _rp(args.dataset, sp, slug, variant_key), clean,
                        pred_key=pred_key, prob_key=prob_key,
                    )
                    merged.update(preds)
            for cid, pred in merged.items():
                if not pred.get("label") and cid in id_to_label:
                    pred["label"] = id_to_label[cid]
            return merged

        # ---- Performance ----
        print(f"  performance + bootstrap …", flush=True)
        perf: dict = {"base": compute_performance(base_preds, complete_pairs)}
        perf_pvals[slug] = {}
        perf_nused[slug] = {}
        for cond in ALL_CONDITIONS:
            variants = NATURAL_GROUPS[cond] if cond in NATURAL_GROUPS else [cond]
            variant_preds = []
            for v in variants:
                vp = _load_preds_clean(v)
                if vp:
                    variant_preds.append(vp)
            if not variant_preds:
                perf[cond] = {"f1": float("nan"), "pair_acc": float("nan")}
                perf_pvals[slug][cond] = {"pair_acc": float("nan"), "f1": float("nan")}
                perf_nused[slug][cond] = {"pair_acc": 0, "f1": 0}
            else:
                all_m = [compute_performance(vp, complete_pairs) for vp in variant_preds]
                perf[cond] = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
                if has_pairs:
                    p_pa, n_pa = pair_acc_pvalue(base_preds, variant_preds, complete_pairs)
                else:
                    p_pa, n_pa = float("nan"), 0
                p_f1, n_f1 = f1_pvalue(base_preds, variant_preds, seed=0)
                perf_pvals[slug][cond] = {"pair_acc": p_pa, "f1": p_f1}
                perf_nused[slug][cond] = {"pair_acc": n_pa, "f1": n_f1}
        all_perf[slug] = perf

        # ---- Shifts ----
        print(f"  shifts + bootstrap …", flush=True)
        shift_summary: dict = {}
        raw_shifts: dict = {}

        if has_pairs:
            base_vals = {
                pid: {
                    "p_sup": base_preds[ids["Supported"]]["p_supported"],
                    "p_ref": base_preds[ids["Refuted"]]["p_supported"],
                }
                for pid, ids in complete_pairs.items()
                if ids["Supported"] in base_preds and ids["Refuted"] in base_preds
            }
            shift_summary["base"] = {
                "p_sup": float(np.mean([v["p_sup"] for v in base_vals.values() if v["p_sup"] is not None])),
                "p_ref": float(np.mean([v["p_ref"] for v in base_vals.values() if v["p_ref"] is not None])),
                "dp_sup": 0.0, "dp_ref": 0.0,
            }
        else:
            sup_p = [p["p_supported"] for p in base_preds.values()
                     if p["label"] == "Supported" and p["p_supported"] is not None]
            ref_p = [p["p_supported"] for p in base_preds.values()
                     if p["label"] == "Refuted"   and p["p_supported"] is not None]
            _mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
            shift_summary["base"] = {
                "p_sup":     _mean(sup_p), "p_ref":     _mean(ref_p),
                "p_sup_new": _mean(sup_p), "p_ref_new": _mean(ref_p),
                "dp_sup": 0.0, "dp_ref": 0.0,
            }

        shift_pvals[slug] = {}
        shift_nused[slug] = {}
        for cond in ALL_CONDITIONS:
            variants = NATURAL_GROUPS[cond] if cond in NATURAL_GROUPS else [cond]
            raw_shifts[cond] = []

            if has_pairs:
                per_variant_shifts = []
                for v in variants:
                    vp = _load_preds_clean(v)
                    if vp:
                        per_variant_shifts.append(compute_shifts(base_preds, vp, complete_pairs))
                if len(per_variant_shifts) > 1:
                    shifts = average_shifts(per_variant_shifts)
                elif len(per_variant_shifts) == 1:
                    shifts = per_variant_shifts[0]
                else:
                    shifts = []
                raw_shifts[cond] = shifts
                shift_summary[cond] = summarize_shifts(shifts)
                p_sup, n_sup = shift_pvalue(shifts, "dp_sup")
                p_ref, n_ref = shift_pvalue(shifts, "dp_ref")
                shift_pvals[slug][cond] = {"dp_sup": p_sup, "dp_ref": p_ref}
                shift_nused[slug][cond] = {"dp_sup": n_sup, "dp_ref": n_ref}
            else:
                per_variant_label_shifts = []
                for v in variants:
                    vp = _load_preds_clean(v)
                    if vp:
                        per_variant_label_shifts.append(compute_label_shifts(base_preds, vp))
                if per_variant_label_shifts:
                    all_s = [summarize_label_shifts(ls) for ls in per_variant_label_shifts]
                    shift_summary[cond] = {k: float(np.mean([s[k] for s in all_s])) for k in all_s[0]}
                    # Pairless: Wilcoxon on the pooled per-item Δp values per label group.
                    # Each (item, variant) contributes one observation. This is a mild n-inflation
                    # vs strict per-item pairing, but pairless datasets don't have a natural pair
                    # structure to collapse over, so it is the cleanest non-arbitrary choice.
                    sup_vec: list[float] = []
                    ref_vec: list[float] = []
                    for ls in per_variant_label_shifts:
                        for s in ls:
                            (sup_vec if s["label"] == "Supported" else ref_vec).append(s["dp"])
                    shift_pvals[slug][cond] = {
                        "dp_sup": bootstrap_mean_p(sup_vec) if sup_vec else float("nan"),
                        "dp_ref": bootstrap_mean_p(ref_vec) if ref_vec else float("nan"),
                    }
                    shift_nused[slug][cond] = {"dp_sup": len(sup_vec), "dp_ref": len(ref_vec)}
                else:
                    shift_summary[cond] = summarize_label_shifts([])
                    shift_pvals[slug][cond] = {"dp_sup": float("nan"), "dp_ref": float("nan")}
                    shift_nused[slug][cond] = {"dp_sup": 0, "dp_ref": 0}

            q = compute_quadrant_proportions(raw_shifts[cond])
            d = shift_summary[cond]
            print(
                f"  {cond:>4s}: Δp_sup={d['dp_sup']:+.4f}  Δp_ref={d['dp_ref']:+.4f}  "
                f"Opt={q['optimistic']:.0%} Con={q['conservative']:.0%} "
                f"Sharp={q['sharper']:.0%} Dull={q['duller']:.0%}"
            )

        all_shift[slug] = shift_summary
        all_raw[slug]   = raw_shifts

        # ---- Compass plots (pair-based only) ----
        if has_pairs and not tables_only:
            print(f"  compass plots …", flush=True)
            compass_dir = out_root / "compass" / slug
            for cond in ALL_CONDITIONS:
                if not raw_shifts[cond]:
                    continue
                fig = plot_compass_single(raw_shifts[cond], slug, cond)
                save_figure(fig, compass_dir / f"compass_{cond}.pdf")
            fig = plot_compass_per_model(raw_shifts, slug)
            save_figure(fig, compass_dir / "compass_all.pdf")
            print(f"  Compass → {compass_dir}")

        # ---- Strip chart data ----
        def _claim_deltas_for(cond_key: str) -> dict:
            variants = NATURAL_GROUPS[cond_key] if cond_key in NATURAL_GROUPS else [cond_key]
            preds_list = []
            for v in variants:
                vp = _load_preds_clean(v)
                if vp:
                    preds_list.append(vp)
            if not preds_list:
                return {"per_claim": [], "mean": 0.0}
            deltas = compute_claim_deltas(base_preds, preds_list, valid_ids)
            return {"per_claim": deltas, "mean": float(np.mean(deltas)) if deltas else 0.0}

        if tables_only:
            continue

        import re as _re
        # Param count for legacy sort metadata; strip chart ranks by pair_acc_base instead.
        nums = _re.findall(r"(\d+)B", slug, _re.IGNORECASE)
        nums_int = sorted([int(n) for n in nums], reverse=True) if nums else []
        size = nums_int[0] if nums_int else 0
        active_m = _re.search(r"[Aa](\d+)[Bb]", slug)
        if nums_int and active_m:
            size_label = f"{nums_int[0]}B-A{active_m.group(1)}B"
        elif nums_int:
            size_label = f"{nums_int[0]}B"
        else:
            size_label = "?"

        strip_rows.append({
            "slug":       slug,
            "size":       size,
            "size_label": size_label,
            "deltas": {
                **{cond: _claim_deltas_for(cond) for cond in list(NATURAL_GROUPS)},
                **{cond: _claim_deltas_for(cond) for cond in CONTROLLED_CONDITIONS},
            },
        })

        # ---- Drift data + plot ----
        print(f"  drift + boundary plots …", flush=True)
        def _preds_list_for(cond_key: str) -> list[dict]:
            variants = NATURAL_GROUPS[cond_key] if cond_key in NATURAL_GROUPS else [cond_key]
            out = []
            for v in variants:
                vp = _load_preds_clean(v)
                if vp:
                    out.append(vp)
            return out

        rewrite_preds_by_cond = {
            cond: _preds_list_for(cond) for cond in ALL_CONDITIONS
            if _preds_list_for(cond)
        }
        drift_data = collect_drift_data(
            base_preds, rewrite_preds_by_cond, complete_pairs, has_pairs=has_pairs,
        )
        fig = plot_drift(drift_data, title=slug)
        save_figure(fig, out_root / "drift" / f"{slug}_drift.pdf", dpi=200)
        print(f"  Drift  → {out_root / 'drift' / f'{slug}_drift.pdf'}")

        boundary_data = collect_boundary_data(
            base_preds, rewrite_preds_by_cond, valid_ids,
        )
        fig = plot_boundary_scatter(boundary_data, title=slug)
        save_figure(fig, out_root / "boundary" / f"{slug}_boundary.pdf", dpi=200)
        print(f"  Boundary → {out_root / 'boundary' / f'{slug}_boundary.pdf'}")

        all_flip[slug] = compute_flip_rates(
            base_preds, rewrite_preds_by_cond, valid_ids,
        )

    # ---- Strip chart (all models) ----
    if strip_rows and not tables_only:
        fig = plot_strip_chart(strip_rows)
        save_figure(fig, out_root / "strip" / "strip_chart.pdf", dpi=300)
        print(f"\nStrip chart → {out_root / 'strip' / 'strip_chart.pdf'}")

        # Qwen-only strip chart with compact XB labels
        import re as _re2
        def _qwen_label(slug: str) -> str:
            m = _re2.search(r"(\d+(?:\.\d+)?[Bb])", slug)
            return m.group(1).upper() if m else slug

        fig_q = plot_strip_chart(
            strip_rows,
            model_filter=lambda s: "Qwen" in s,
            label_fn=_qwen_label,
        )
        save_figure(fig_q, out_root / "strip" / "strip_chart_qwen.pdf", dpi=300)
        print(f"Strip chart (Qwen) → {out_root / 'strip' / 'strip_chart_qwen.pdf'}")

    # ---- Compass overview (pair-based only) ----
    if has_pairs and all_raw and not tables_only:
        fig = plot_compass_overview(all_raw)
        save_figure(fig, out_root / "compass" / "overview.pdf")
        print(f"Compass overview → {out_root / 'compass' / 'overview.pdf'}")

    # ---- Significance: BH-FDR over all populated cells per table ----
    # Performance: pool P-Acc and F1 p-values from every (model, cond) cell.
    completed_slugs = list(all_perf.keys())

    perf_keys, perf_pool = [], []
    for slug in completed_slugs:
        for cond in ALL_CONDITIONS:
            for mk in ("pair_acc", "f1"):
                p = perf_pvals.get(slug, {}).get(cond, {}).get(mk, float("nan"))
                if not np.isnan(p):
                    perf_keys.append((slug, cond, mk))
                    perf_pool.append(p)
    perf_q = bh_fdr(perf_pool)
    perf_stats: dict[str, dict[str, dict[str, float]]] = {}
    for (slug, cond, mk), q in zip(perf_keys, perf_q):
        perf_stats.setdefault(slug, {}).setdefault(cond, {})[mk] = q

    shift_keys, shift_pool = [], []
    for slug in completed_slugs:
        for cond in ALL_CONDITIONS:
            for mk in ("dp_sup", "dp_ref"):
                p = shift_pvals.get(slug, {}).get(cond, {}).get(mk, float("nan"))
                if not np.isnan(p):
                    shift_keys.append((slug, cond, mk))
                    shift_pool.append(p)
    shift_q = bh_fdr(shift_pool)
    shift_stats: dict[str, dict[str, dict[str, float]]] = {}
    for (slug, cond, mk), q in zip(shift_keys, shift_q):
        shift_stats.setdefault(slug, {}).setdefault(cond, {})[mk] = q

    # ---- LaTeX tables (both plain and with significance markers) ----
    tables_dir = out_root / "tables"
    perf_tex_plain  = generate_performance_table(all_perf, completed_slugs)
    perf_tex_stats  = generate_performance_table(all_perf, completed_slugs, stats=perf_stats)
    shift_tex_plain = generate_shift_table(all_shift, completed_slugs)
    shift_tex_stats = generate_shift_table(all_shift, completed_slugs, stats=shift_stats)
    save_text(perf_tex_plain,  tables_dir / "performance.tex")
    save_text(perf_tex_stats,  tables_dir / "performance_stats.tex")
    save_text(shift_tex_plain, tables_dir / "shifts.tex")
    save_text(shift_tex_stats, tables_dir / "shifts_stats.tex")

    if all_flip:
        flip_tex = generate_flip_rate_table(all_flip, completed_slugs)
        save_text(flip_tex, tables_dir / "flip_rates.tex")

    # ---- Raw stats dump for reproducibility ----
    import json as _json
    stats_payload = {
        "method": {
            "pair_acc": "Paired bootstrap (n_iter=1000, seed=0) on per-pair "
                        "(cond_correct − base_correct); cond_correct averaged across variants. "
                        "Two-sided p-value: 2*min(P(mean>=0), P(mean<=0)).",
            "f1":       "Paired bootstrap on resampled claim_ids (n_iter=1000, seed=0). "
                        "F1(cond) - F1(base) computed per resample; two-sided p-value: "
                        "2*min(P(diff>=0), P(diff<=0)).",
            "dp_sup":   "Paired bootstrap (n_iter=1000, seed=0) on per-pair Δp(supported) "
                        "for the Supported claim of each pair.",
            "dp_ref":   "Paired bootstrap (n_iter=1000, seed=0) on per-pair Δp(supported) "
                        "for the Refuted claim of each pair.",
            "multiplicity": "Benjamini-Hochberg FDR over all populated cells per table.",
        },
        "rewrite_filter_stats":      filter_stats,
        "performance_raw_pvalues":   perf_pvals,
        "performance_fdr_qvalues":   perf_stats,
        "performance_n_used":        perf_nused,
        "shifts_raw_pvalues":        shift_pvals,
        "shifts_fdr_qvalues":        shift_stats,
        "shifts_n_used":             shift_nused,
    }
    save_text(_json.dumps(stats_payload, indent=2), tables_dir / "stats.json")
    print(f"Tables → {tables_dir}")
    print(f"  plain : performance.tex,  shifts.tex")
    print(f"  stats : performance_stats.tex,  shifts_stats.tex")
    print(f"  flip  : flip_rates.tex")
    print(f"  raw   : stats.json")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    from core.config import DATASETS
    dataset_choices = list(DATASETS)

    root = argparse.ArgumentParser(
        prog="python -m core.run",
        description="SciClaimEval rewrite experiment pipeline.",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ---- rewrite ----
    rw = sub.add_parser("rewrite", help="Generate rewrite datasets.")
    rw_sub = rw.add_subparsers(dest="method", required=True)

    for method in ("natural", "inject"):
        p = rw_sub.add_parser(method, help=f"{method} rewrite")
        p.add_argument("--dataset", required=True, choices=dataset_choices,
                       help="Dataset to rewrite")
        p.add_argument("--split", default="dev",
                       help="Dataset split (default: dev)")
        p.add_argument("--gen-model", required=True, dest="gen_model",
                       help="LLM used for rewriting (HF ID or name on the server)")
        p.add_argument("--base-url", default="http://localhost:8000/v1", dest="base_url")
        p.add_argument("--conditions", nargs="*", default=None,
                       help="Specific conditions (default: all)")
        p.add_argument("--limit", type=int, default=None,
                       help="Cap items per condition (debug)")
        p.add_argument("--resume", action="store_true",
                       help="Skip conditions whose output file already exists")
        if method == "natural":
            p.add_argument(
                "--bare-output", action="store_true", dest="bare_output",
                help=(
                    "Add a format-only system prompt telling the rewriter to return "
                    "ONLY the rewritten sentence (for chatty models like gemma-4). "
                    "OFF by default; keep OFF to reproduce the original run."
                ),
            )

    mix_p = rw_sub.add_parser(
        "mix",
        help="Create per-group mix files from existing variant rewrites.",
    )
    mix_p.add_argument("--dataset", required=True, choices=dataset_choices,
                       help="Dataset to mix")
    mix_p.add_argument("--split", default="dev",
                       help="Dataset split (default: dev)")
    mix_p.add_argument("--seed", type=int, required=True,
                       help="RNG seed; embedded in output filename (e.g. LP-mix-42.json)")
    mix_p.add_argument("--groups", nargs="*", default=None,
                       help="Natural groups to mix (default: all — LP FI AS OA CT)")
    mix_p.add_argument("--resume", action="store_true",
                       help="Skip groups whose mix file already exists")

    # ---- infer ----
    inf = sub.add_parser("infer", help="Run three-mode VL inference.")
    inf.add_argument("--dataset",  required=True, choices=dataset_choices)
    inf.add_argument("--split", default="dev",
                     help="Dataset split (default: dev)")
    inf.add_argument("--model",    required=True,
                     help="HF model ID (must be in config/models.yaml for vllm mode)")
    inf.add_argument("--conditions", nargs="*", default=None,
                     help="Conditions to run (default: original + all variants)")
    inf.add_argument("--resume", action="store_true",
                     help="Skip conditions whose output file already exists")
    inf.add_argument("--consistency-n", type=int, default=None, dest="consistency_n",
                     help="Override consistency_n from inference.yaml (e.g. 3 for fast runs)")
    inf.add_argument("--limit", type=int, default=None,
                     help="Cap items per condition (debug)")
    inf.add_argument("--skip-missing-images", action="store_true",
                     dest="skip_missing_images")
    inf.add_argument("--mix-seed", type=int, default=None, dest="mix_seed",
                     help=(
                         "Run mix-variant conditions (LP-mix-{seed}, ...) instead of all "
                         "25 individual variants. Ignored if --conditions is also set."
                     ))

    # ---- analyze ----
    ana = sub.add_parser("analyze", help="Compute metrics and generate plots/tables.")
    ana.add_argument("--dataset", required=True, choices=dataset_choices)
    ana.add_argument("--split", nargs="+", default=["dev"],
                     help="Split(s) to analyze; pass multiple to pool across splits "
                          "(default: dev). E.g. --split dev test")
    ana.add_argument("--eval-mode", default=None, dest="eval_mode",
                     choices=["consistency"],
                     help="Which prediction fields to use (default: consistency)")
    ana.add_argument("--models", nargs="*", default=None,
                     help="Model slugs to include (default: all found in runs/<dataset>/<split>/)")
    ana.add_argument("--out-dir", default=None, dest="out_dir",
                     help="Override output directory (default: results/<dataset>/<split>/<eval-mode>/)")
    ana.add_argument("--mix-seed", type=int, default=None, dest="mix_seed",
                     help=(
                         "Resolve each natural group to its mix file (LP-mix-{seed}, ...) "
                         "instead of averaging across 5 variants."
                     ))
    ana.add_argument("--tables-only", action="store_true", dest="tables_only",
                     help=(
                         "Skip compass/strip/drift plot generation; only compute metrics, "
                         "significance tests, and emit LaTeX tables. Use to iterate quickly "
                         "on table formatting."
                     ))

    return root


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.command == "rewrite":
        if args.method == "natural":
            _rewrite_natural(args)
        elif args.method == "inject":
            _rewrite_inject(args)
        else:
            _rewrite_mix(args)
    elif args.command == "infer":
        _infer(args)
    elif args.command == "analyze":
        _analyze(args)


if __name__ == "__main__":
    main()
