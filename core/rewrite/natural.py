"""
Natural (stylistic) claim rewrites via LLM instruction prompts.

Prompts are defined in core/rewrite/prompts.py — edit there to add conditions.

The factual-preservation suffix is appended to every prompt automatically.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from core.rewrite.prompt import PROMPTS

FACTUAL_PRESERVATION_SUFFIX = (
    "During rewriting, you must not alter the factual content of the text. "
    "All changes should be purely stylistic and decorative."
)

# Format-only system prompt for chatty instruct models (e.g. gemma-4) that
# answer terse rewrite prompts with "here are a few options …" meta-text.
# It constrains OUTPUT FORMAT only — it never changes the rewrite operation —
# so it is safe to enable uniformly across conditions. It is OPT-IN: the default
# code path sends no system message, keeping requests byte-identical to the
# original (reproducible) run.
BARE_OUTPUT_SYSTEM_PROMPT = (
    "You are a text-rewriting engine. Output ONLY the rewritten text as a single "
    "plain sentence, with no preamble, explanations, alternatives, options, "
    "quotation marks, or markdown formatting."
)


def _rewrite_one(
    client: OpenAI,
    model: str,
    item: dict,
    instruction: str,
    *,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
) -> dict:
    claim = item["claim"]
    full_prompt = f"{instruction} {FACTUAL_PRESERVATION_SUFFIX}\n\n{claim}"
    messages: list[dict] = [{"role": "user", "content": full_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    return {**item, "claim": content.strip() if content else claim}


def run_natural_rewrite(
    items: list[dict],
    condition: str,
    client: OpenAI,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    workers: int = 8,
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Rewrite all items for one condition key (e.g. 'LP-1', 'OA-3').
    Raises KeyError if condition is not in PROMPTS.
    Returns rewritten list in original order.

    system_prompt: optional format-only system message (see
    BARE_OUTPUT_SYSTEM_PROMPT). Default None keeps requests byte-identical to
    the original run.
    """
    instruction = PROMPTS[condition]
    results: list[dict | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _rewrite_one, client, model, item, instruction,
                max_tokens=max_tokens, temperature=temperature,
                system_prompt=system_prompt,
            ): i
            for i, item in enumerate(items)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=condition):
            results[futures[fut]] = fut.result()
    return [r for r in results if r is not None]
