#!/usr/bin/env python3
"""
Quality-check the gemma-4 rewrite files (the rebuttal's second rewriter).

For every rewrites/<dataset>/<split>/{natural,inject}/*.json it reports the
failure modes that actually matter for this pipeline:

  natural (CT/OA):
    - empty / whitespace-only claims           (dropped by validate_rewrite_claims)
    - too short (< min_length chars)           (dropped by validate_rewrite_claims)
    - unchanged vs. the original claim         (rewrite did nothing)
    - chatty / meta output: preamble, markdown, option-lists, multi-line
      ("Here are a few options … **Option 1 (Recommended)** …") — the pipeline
      stores `claim` verbatim, so this silently corrupts the condition
    - thinking-mode leakage (<|channel|> etc.)

  inject (C_1/D_1):
    - result must be the original claim + exactly one word from the pool
      (checked with the real inject helpers _find_insert_pos / POOLS)
    - plus the same empty/too-short/leak checks

It prints a per-file summary table, the pipeline-validator drop count (so you
know exactly how many pairs analysis would discard), and a few worst-offender
examples per file. Exit code 0 = clean, 1 = something needs attention.

Usage:
  python scripts/check_gemma_rewrites.py
  python scripts/check_gemma_rewrites.py --dataset sciclaimeval_gemma --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DATASETS, NATURAL_GROUPS, CONTROLLED_CONDITIONS
from core.io import load_base_dataset, load_rewrite, rewrite_path
from core.analysis.metrics import build_pair_mapping, validate_rewrite_claims
from core.rewrite.inject import (
    CONDITION_SPECS, POOLS, _find_insert_pos, _norm, _strip_brackets,
)

# Same markers used by the server smoke test.
THINKING_MARKERS = ["<|channel", "channel|>", "<|thought", "thought|>"]
CHATTY_MARKERS = [
    "**", "```", "Option ", "Recommended", "Here are", "Here's",
    "Depending on", "few ways", "\n>", "\n-", "\n1.", "\n2.",
]

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _mark(n: int) -> str:
    if n == 0:
        return f"{GREEN}0{RESET}"
    return f"{RED}{n}{RESET}"


def _is_chatty(text: str, original: str) -> bool:
    multiline = "\n" in text.strip()
    marker = any(m in text for m in CHATTY_MARKERS)
    over_long = len(text.split()) > 2 * len(original.split()) + 20
    return multiline or marker or over_long


def _check_natural(cond: str, items: list[dict], base: dict[str, str]) -> dict:
    stats = {"n": len(items), "empty": 0, "too_short": 0, "unchanged": 0,
             "chatty": 0, "leak": 0, "no_base": 0}
    examples: list[tuple[str, str, str]] = []  # (reason, original, rewrite)
    for it in items:
        cid = it.get("claim_id")
        claim = (it.get("claim") or "").strip()
        orig = base.get(cid, "")
        if not orig:
            stats["no_base"] += 1
        if not claim:
            stats["empty"] += 1
            if len(examples) < 6:
                examples.append(("empty", orig, claim))
            continue
        if len(claim) < 10:
            stats["too_short"] += 1
            if len(examples) < 6:
                examples.append(("too_short", orig, claim))
        if orig and claim.lower() == orig.strip().lower():
            stats["unchanged"] += 1
            if len(examples) < 6:
                examples.append(("unchanged", orig, claim))
        if any(m in claim for m in THINKING_MARKERS):
            stats["leak"] += 1
            if len(examples) < 6:
                examples.append(("leak", orig, claim))
        if orig and _is_chatty(claim, orig):
            stats["chatty"] += 1
            if len(examples) < 6:
                examples.append(("chatty", orig, claim))
    stats["_examples"] = examples
    return stats


def _norm_tokens(text: str) -> list[str]:
    """Lower-cased word tokens with surrounding punctuation stripped.

    This makes the injection check robust to punctuation shifts (e.g. a period
    moving from "(c))." to "…) broadly.") so we only flag *genuine* problems.
    """
    import string
    out = []
    for tok in _norm(_strip_brackets(text)):
        t = tok.strip(string.punctuation + "“”‘’").lower()
        if t:
            out.append(t)
    return out


def strict_inject_clean_ids(
    dataset: str,
    split: str,
    condition: str,
    base_claims: dict[str, str] | None = None,
) -> set[str] | None:
    """Claim_ids whose inject rewrite is *strictly compliant*: exactly one word
    added, nothing removed/replaced, and the added word is in the condition's
    pool (``core.rewrite.inject.POOLS``).

    Returns ``None`` for non-inject conditions (no strict rule applies), so a
    caller may treat ``None`` as "keep everything". This is the gemma-specific
    filter — the shared analyze path (``metrics.validate_rewrite_claims``) is
    unchanged, so the original gpt-oss results still replicate exactly.
    """
    from collections import Counter

    if condition not in CONDITION_SPECS:
        return None
    if base_claims is None:
        base = load_base_dataset(DATASETS[dataset], split)
        base_claims = {it["claim_id"]: (it.get("claim") or "") for it in base}

    pool = {w.lower() for w in POOLS[CONDITION_SPECS[condition][0]]}
    clean: set[str] = set()
    for it in load_rewrite(dataset, split, "inject", condition):
        cid = it.get("claim_id")
        new = (it.get("claim") or "").strip()
        orig = base_claims.get(cid, "")
        if not new or not orig:
            continue
        added = Counter(_norm_tokens(new)) - Counter(_norm_tokens(orig))
        removed = Counter(_norm_tokens(orig)) - Counter(_norm_tokens(new))
        if sum(added.values()) == 1 and sum(removed.values()) == 0:
            if next(iter(added)) in pool:
                clean.add(cid)
    return clean


def _check_inject(cond: str, items: list[dict], base: dict[str, str]) -> dict:
    from collections import Counter
    pool_key = CONDITION_SPECS[cond][0]
    pool = {w.lower() for w in POOLS[pool_key]}
    # clean      : exactly one pool word added, nothing removed
    # off_pool   : exactly one word added but it's NOT from the pool
    # malformed  : >1 word added, or words removed/replaced (not a clean +1 insert)
    stats = {"n": len(items), "empty": 0, "too_short": 0, "leak": 0,
             "clean": 0, "off_pool": 0, "malformed": 0}
    examples: list[tuple[str, str, str]] = []
    for it in items:
        cid = it.get("claim_id")
        new = (it.get("claim") or "").strip()
        orig = base.get(cid, "")
        if not new:
            stats["empty"] += 1
            if len(examples) < 8:
                examples.append(("empty", orig, new))
            continue
        if len(new) < 10:
            stats["too_short"] += 1
        if any(m in new for m in THINKING_MARKERS):
            stats["leak"] += 1
            if len(examples) < 8:
                examples.append(("leak", orig, new))
        if not orig:
            continue
        oc = Counter(_norm_tokens(orig))
        nc = Counter(_norm_tokens(new))
        added = nc - oc      # words present in new but not (as many times) in orig
        removed = oc - nc     # words dropped/changed from orig
        n_added = sum(added.values())
        n_removed = sum(removed.values())
        if n_added == 1 and n_removed == 0:
            word = next(iter(added))
            if word in pool:
                stats["clean"] += 1
            else:
                stats["off_pool"] += 1
                if len(examples) < 8:
                    examples.append((f"off_pool:{word}", orig, new))
        else:
            stats["malformed"] += 1
            if n_added == 0 and n_removed == 0:
                stats["no_inject"] = stats.get("no_inject", 0) + 1
            else:
                stats["multi_change"] = stats.get("multi_change", 0) + 1
            if len(examples) < 8:
                tag = f"malformed:+{sorted(added.elements())}/-{sorted(removed.elements())}"
                examples.append((tag[:80], orig, new))
    stats["_examples"] = examples
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", default="sciclaimeval_gemma", choices=list(DATASETS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--examples", type=int, default=3,
                    help="Worst-offender examples to print per file")
    args = ap.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    base_items = load_base_dataset(dataset_cfg, args.split)
    base = {it["claim_id"]: (it.get("claim") or "") for it in base_items}
    complete_pairs, _valid = build_pair_mapping(base_items)
    print(f"Base: {len(base_items)} items, {len(complete_pairs)} complete pairs "
          f"({args.dataset}/{args.split})\n")

    natural_conds = [f"{g}-{i}" for g in ("OA", "CT") for i in range(1, 6)]
    problems = 0

    # ---- Natural ----
    print("NATURAL (CT / OA)")
    print(f"  {'cond':<6} {'n':>5} {'empty':>6} {'short':>6} {'unchg':>6} "
          f"{'chatty':>7} {'leak':>5}   validator-drops")
    nat_examples: list = []
    for cond in natural_conds:
        p = rewrite_path(args.dataset, args.split, "natural", cond)
        if not p.exists():
            continue
        items = load_rewrite(args.dataset, args.split, "natural", cond)
        s = _check_natural(cond, items, base)
        clean, vstats = validate_rewrite_claims(items, base, complete_pairs)
        drops = vstats["n_broken"]
        bad = s["empty"] + s["too_short"] + s["chatty"] + s["leak"]
        problems += bad
        print(f"  {cond:<6} {s['n']:>5} {_mark(s['empty']):>15} {_mark(s['too_short']):>15} "
              f"{_mark(s['unchanged']):>15} {_mark(s['chatty']):>16} {_mark(s['leak']):>14}"
              f"   {drops} ({vstats['n_empty']}e/{vstats['n_too_short']}s/"
              f"{vstats['n_pair_dropped']}p)")
        if s["_examples"]:
            nat_examples.append((cond, s["_examples"]))

    # ---- Inject ----
    print("\nINJECT (C_1 / D_1)   — clean = exactly one pool word added, nothing else changed")
    print(f"  {'cond':<6} {'n':>5} {'clean':>6} {'off_pool':>9} {'malform':>8} "
          f"{'empty':>6} {'leak':>5}")
    inj_examples: list = []
    for cond in CONTROLLED_CONDITIONS:
        p = rewrite_path(args.dataset, args.split, "inject", cond)
        if not p.exists():
            continue
        items = load_rewrite(args.dataset, args.split, "inject", cond)
        s = _check_inject(cond, items, base)
        bad = s["empty"] + s["leak"] + s["off_pool"] + s["malformed"]
        problems += bad
        clean_pct = 100.0 * s["clean"] / s["n"] if s["n"] else 0.0
        print(f"  {cond:<6} {s['n']:>5} {s['clean']:>6} {_mark(s['off_pool']):>18} "
              f"{_mark(s['malformed']):>17} {_mark(s['empty']):>15} {_mark(s['leak']):>14}"
              f"   ({clean_pct:.1f}% clean; malformed = "
              f"{s.get('no_inject', 0)} no-inject / {s.get('multi_change', 0)} multi-word)")
        if s["_examples"]:
            inj_examples.append((cond, s["_examples"]))

    # ---- Examples ----
    def _show(title: str, blocks: list) -> None:
        if not blocks:
            return
        print(f"\n{title}")
        for cond, exs in blocks:
            print(f"  [{cond}]")
            for reason, orig, rw in exs[:args.examples]:
                print(f"    ({reason})")
                print(f"      orig: {orig[:160]}{'…' if len(orig) > 160 else ''}")
                print(f"      →     {rw[:220]}{'…' if len(rw) > 220 else ''}")

    _show("NATURAL — flagged examples", nat_examples)
    _show("INJECT — flagged examples", inj_examples)

    print(f"\n{'='*60}")
    if problems == 0:
        print(f"{GREEN}RESULT: PASS{RESET} — no empty/short/chatty/leak/insert issues found.")
        return 0
    print(f"{RED}RESULT: {problems} flagged items{RESET} — see examples above.")
    print("Note: 'unchanged' is informational (some rewrites legitimately match).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
