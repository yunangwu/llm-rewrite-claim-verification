#!/usr/bin/env python3
"""
Smoke-test a running Gemma-4 vLLM server for use as the *rewriter* in this
pipeline (see core/rewrite/natural.py).

It mirrors exactly how `rewrite natural` talks to the server:
  - synchronous openai.OpenAI client
  - single-turn user message = "<instruction> <factual-preservation-suffix>\n\n<claim>"
  - reads resp.choices[0].message.content as the rewritten claim

It covers exactly the conditions used in the rebuttal run — the natural groups
CT and OA, and the inject conditions C_1 / D_1 (the latter through the real
`run_inject_rewrite` pipeline) — and checks what actually matters:
  1. the server is reachable and serving the expected model,
  2. natural rewrites are non-empty and actually differ from the input,
  3. no Gemma-4 *thinking* delimiters leaked into `content` (start the server
     with `--reasoning-parser gemma4` so the thought goes to `reasoning_content`),
  4. natural rewrites are a *bare* rewritten sentence, not a chatty meta-answer
     ("here are a few options … **Option 1 (Recommended)** …") — the pipeline
     stores `content` verbatim as the claim, so chattiness corrupts the data,
  5. inject rewrites are the original claim plus exactly one pool word.

Prerequisite — start the server first, e.g.:
  vllm serve google/gemma-4-31B-it \
    --data-parallel-size 8 --tensor-parallel-size 1 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --reasoning-parser gemma4 --port 8000

Usage:
  python scripts/test_gemma4_vllm.py
  python scripts/test_gemma4_vllm.py --base-url http://localhost:8000/v1 --model google/gemma-4-31B-it
  python scripts/test_gemma4_vllm.py --temperature 0.7        # match rewrite.yaml natural temp

Exit code 0 = all checks passed, 1 = a check failed (usable in CI / job scripts).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so we reuse the real prompts / suffix.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rewrite.natural import FACTUAL_PRESERVATION_SUFFIX
from core.rewrite.prompt import PROMPTS
from core.rewrite.inject import (
    CONDITION_SPECS,
    POOLS,
    _find_insert_pos,
    _norm,
    _strip_brackets,
    run_inject_rewrite,
)

# A few short, self-contained claims to rewrite. Kept generic on purpose so the
# test doesn't depend on the dataset being present.
SAMPLE_CLAIMS: list[str] = [
    "The proposed method outperform all baselines on the benchmark.",
    "Increasing the model size consistently improves accuracy across all tasks.",
    "Our approach reduces inference latency by nearly half compared to prior work.",
]

# Natural conditions actually used in the rebuttal run (CT + OA). OA is prompt-
# shaped like the chatty LP/FI prompts, so it's the real risk to verify here.
TEST_CONDITIONS: list[str] = ["CT-1", "CT-3", "CT-5", "OA-1", "OA-3", "OA-5"]

# Inject conditions actually used in the rebuttal run.
INJECT_CONDITIONS: list[str] = ["C_1", "D_1"]

# Substrings that indicate Gemma-4 thinking-mode text leaked into `content`.
THINKING_MARKERS: list[str] = ["<|channel", "channel|>", "<|thought", "thought|>"]

# Substrings / shapes that indicate a chatty "meta" answer instead of a bare
# rewrite (e.g. "Depending on the desired tone, here are a few ways ...
# **Option 1: (Recommended)** > \"...\""). The pipeline stores the raw content
# as the claim, so any of these means the rewrite file would be corrupted.
CHATTY_MARKERS: list[str] = [
    "**", "```", "Option ", "Recommended", "Here are", "Here's",
    "Depending on", "few ways", "\n>", "\n-", "\n1.", "\n2.", "rewrite",
]


def _fmt(ok: bool) -> str:
    return "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test a Gemma-4 vLLM server for the rewriter path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1", dest="base_url")
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Match config/rewrite.yaml natural temperature.")
    parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run `uv sync` first.", file=sys.stderr)
        return 1

    client = OpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout)

    # ---- Check 1: server reachable + model present ----------------------
    print(f"[1/3] Connecting to {args.base_url} …")
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as exc:  # noqa: BLE001
        print(f"  {_fmt(False)} could not reach the server: {exc}", file=sys.stderr)
        print("        Is `vllm serve` running and healthy on that port?", file=sys.stderr)
        return 1
    print(f"  served models: {served}")
    model_ok = args.model in served
    if not model_ok:
        # vLLM sometimes serves under a resolved path/alias; warn but continue
        # using the requested name (the completions call will confirm).
        print(f"  {_fmt(False)} '{args.model}' not in served list — "
              f"will still attempt requests with that name.")
    else:
        print(f"  {_fmt(True)} '{args.model}' is being served.")

    # ---- Check 2: natural rewrites (CT / OA) ---------------------------
    print(f"\n[2/4] Natural rewrites — {', '.join(TEST_CONDITIONS)} — "
          f"{len(TEST_CONDITIONS) * len(SAMPLE_CLAIMS)} requests (temp={args.temperature}) …\n")

    all_content_ok = True     # non-empty and changed
    all_clean = True          # no thinking-marker leakage
    all_bare_rewrite = True   # a single bare sentence, not a chatty meta-answer
    saw_reasoning_field = False

    for cond in TEST_CONDITIONS:
        instruction = PROMPTS[cond]
        for claim in SAMPLE_CLAIMS:
            full_prompt = f"{instruction} {FACTUAL_PRESERVATION_SUFFIX}\n\n{claim}"
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {_fmt(False)} [{cond}] request failed: {exc}", file=sys.stderr)
                return 1

            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                saw_reasoning_field = True

            non_empty = bool(content)
            changed = content.lower() != claim.lower()
            leaked = [m for m in THINKING_MARKERS if m in content]

            # "Bare rewrite" heuristic: the pipeline stores `content` verbatim as
            # the claim, so it must be a single rewritten sentence — not a chatty
            # meta-answer with preamble / markdown / option lists, and not wildly
            # longer than the input.
            multiline = "\n" in content.strip()
            chatty = [m for m in CHATTY_MARKERS if m in content]
            over_long = len(content.split()) > 2 * len(claim.split()) + 20
            bare_rewrite = not (multiline or chatty or over_long)

            content_ok = non_empty and changed
            clean = not leaked
            all_content_ok = all_content_ok and content_ok
            all_clean = all_clean and clean
            all_bare_rewrite = all_bare_rewrite and bare_rewrite

            status = _fmt(content_ok and clean and bare_rewrite)
            print(f"  {status} [{cond}] {claim}")
            print(f"        → {content[:200]}{'…' if len(content) > 200 else ''}")
            if not non_empty:
                print("        (!) empty content returned")
            if not changed:
                print("        (!) rewrite identical to input")
            if leaked:
                print(f"        (!) thinking markers leaked into content: {leaked} "
                      f"— start server with --reasoning-parser gemma4")
            if not bare_rewrite:
                why = "too long" if over_long and not (multiline or chatty) else "preamble/markdown/multi-line"
                print(f"        (!) not a bare rewrite ({why}) — this whole blob would be "
                      f"stored verbatim as the claim; constrain the model to output only "
                      f"the rewritten text")
            print()

    # ---- Check 3: inject rewrites (C_1 / D_1) --------------------------
    # These go through the *real* inject pipeline (run_inject_rewrite), which has
    # its own "Output ONLY the modified claim" prompt, quote/fence stripping, and
    # a word-count validation that retries on chatty output. We assert the result
    # is the original claim with exactly one pool word inserted.
    print(f"\n[3/4] Inject rewrites — {', '.join(INJECT_CONDITIONS)} — "
          f"through the real inject pipeline …\n")

    all_inject_ok = True
    items = [{"claim_id": f"smoke_{i}", "claim": c} for i, c in enumerate(SAMPLE_CLAIMS)]
    for cond in INJECT_CONDITIONS:
        pool_key = CONDITION_SPECS[cond][0]
        try:
            out = run_inject_rewrite(
                items, cond, client, args.model,
                temperature=0.25, max_tokens=args.max_tokens, workers=4,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {_fmt(False)} [{cond}] inject run failed: {exc}", file=sys.stderr)
            return 1

        for src, res in zip(items, out):
            orig = src["claim"]
            new = res["claim"]
            orig_w = _norm(_strip_brackets(orig))
            new_w = _norm(_strip_brackets(new))
            pos = _find_insert_pos(orig_w, new_w)
            injected = new_w[pos].strip(".,;:").lower() if pos is not None else None
            in_pool = injected in {w.lower() for w in POOLS[pool_key]} if injected else False
            ok = pos is not None and in_pool
            all_inject_ok = all_inject_ok and ok
            print(f"  {_fmt(ok)} [{cond}] {orig}")
            print(f"        → {new}")
            if pos is None:
                print("        (!) result is not the original claim + exactly one word "
                      "(chatty output or malformed injection)")
            elif not in_pool:
                print(f"        (!) inserted word '{injected}' is not from the {pool_key} pool")
            print()

    # ---- Summary --------------------------------------------------------
    print("[4/4] Summary")
    print(f"  server reachable        : {_fmt(True)}")
    print(f"  rewrites non-empty/diff : {_fmt(all_content_ok)}")
    print(f"  no thinking leakage     : {_fmt(all_clean)}")
    print(f"  bare rewrite (no chat)  : {_fmt(all_bare_rewrite)}")
    print(f"  inject one-word valid   : {_fmt(all_inject_ok)}")
    print(f"  reasoning_content field : {'present (thinking routed correctly)' if saw_reasoning_field else 'not present'}")

    passed = all_content_ok and all_clean and all_bare_rewrite and all_inject_ok
    print(f"\nRESULT: {_fmt(passed)}")
    if not passed:
        print("One or more checks failed — see (!) lines above.", file=sys.stderr)
        return 1
    print("Gemma-4 vLLM server looks good for use as the rewriter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
