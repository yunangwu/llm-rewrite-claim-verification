"""
LLM word-injection rewrites (decorative / conservative pools).

Two pools × two levels → four conditions: D_1, D_2, C_1, C_2.
Each level inserts exactly one word from the pool; level 2 adds a second
word at least MIN_DISTANCE positions away from the first.
"""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Word pools and operation specs
# ---------------------------------------------------------------------------

POOLS: dict[str, list[str]] = {
    "D": [
        "accentuates", "adept", "advocates", "affirming", "aptly", "attains",
        "augmenting", "avenue", "boast", "bolster", "burgeoning", "capitalizing",
        "catalyze", "combating", "commendable", "compelling", "compellingly",
        "comprehensive", "consolidates", "crafted", "critical", "crucial",
        "culminating", "deeper", "dependability", "dependable", "detrimentally",
        "disrupts", "distinctive", "effectively", "effortlessly", "elevate",
        "emphasises", "emphasising", "emphasize", "empowers", "enduring",
        "enhance", "essential", "excel", "excellently", "exceptional",
        "exceptionally", "exhaustive", "expansive", "formidable", "fortify",
        "foster", "foundational", "fresh", "fundamental", "groundbreaking",
        "illuminate", "imperative", "impressive", "impressively", "ingenious",
        "innovative", "intriguing", "invaluable", "leverage", "lucidly",
        "meticulous", "meticulously", "multifaceted", "notable", "notably",
        "noteworthy", "optimizing", "orchestrating", "outperform", "paving",
        "pinpoint", "pioneering", "pivotal", "poised", "potent", "pressing",
        "promise", "pronounced", "propelling", "remarkable", "renowned",
        "revolutionize", "seamless", "seamlessly", "showcase", "significant",
        "solidify", "strategically", "streamline", "substantiated", "surged",
        "surmount", "surpass", "swift", "swiftly", "thorough", "transcend",
        "transform", "transformative", "uncharted", "uncovering", "underscore",
        "unlocking", "unparalleled", "unraveling", "unveil", "uphold",
        "valuable", "versatile", "versatility", "warranting", "well-rounded",
    ],
    "C": [
        "acknowledges", "allegedly", "apparently", "approximately", "arguably",
        "basically", "broadly", "comparatively", "conceivably", "essentially",
        "generally", "hinting", "inadequately", "largely", "mainly",
        "marginally", "moderately", "mostly", "often", "ostensibly",
        "partially", "perhaps", "plausibly", "possibly", "potential",
        "presumably", "primarily", "purportedly", "relatively", "roughly",
        "seemingly", "slightly", "somewhat", "supposedly", "tentatively",
        "typically", "underexplored", "unexplored", "usually",
    ],
}

POOL_SAMPLE_SIZE = {"D": 30, "C": 15}

# condition_name → (pool_key, level)
CONDITION_SPECS: dict[str, tuple[str, int]] = {
    "D_1": ("D", 1),
    "D_2": ("D", 2),
    "C_1": ("C", 1),
    "C_2": ("C", 2),
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_L1 = """\
You are a linguistic editor. Your task is to inject exactly 1 word from the provided word list into the given claim.

Rules:
- You MUST insert exactly 1 word from the word list below. This is mandatory.
- Do not delete, replace, or reorder any existing words. Every original word must remain.
- The injection should be stylistic only — it must not change the factual content.
- The injected word must be grammatically natural in its position.

Word list:
{word_list}

Original claim:
{claim}

Output ONLY the modified claim with exactly one word inserted. No explanation, no preamble."""

_PROMPT_L2 = """\
You are a linguistic editor. Your task is to inject exactly 1 word from the provided word list into the given claim.

Rules:
- You MUST insert exactly 1 word from the word list below. This is mandatory.
- Do not delete, replace, or reorder any existing words. Every original word must remain.
- The injection should be stylistic only — it must not change the factual content.
- The injected word must be grammatically natural in its position.
- The word in [brackets] was previously injected. You MUST place your new word at least {min_distance} words away from it.

Word list:
{word_list}

Do not use any of these words (already injected): {excluded_words}

Claim (with [brackets] marking the previous injection):
{claim}

Output ONLY the modified claim with exactly one new word inserted. Keep the [brackets] around the previous word. No explanation, no preamble."""

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _norm(text: str) -> list[str]:
    return re.sub(r"\s+", " ", text.strip()).split()


def _find_insert_pos(orig: list[str], mod: list[str]) -> int | None:
    if len(mod) != len(orig) + 1:
        return None
    for skip in range(len(mod)):
        if mod[:skip] + mod[skip + 1:] == orig:
            return skip
    return None


def _find_bracket_pos(words: list[str]) -> int | None:
    for i, w in enumerate(words):
        if re.match(r"\[.+\]", w):
            return i
    return None


def _clean_model_output(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def _strip_brackets(text: str) -> str:
    return re.sub(r"\[(\w+(?:-\w+)*)\]", r"\1", text)


def _mark_word(text: str, word: str) -> str:
    return re.compile(re.escape(word), re.IGNORECASE).sub(f"[{word}]", text, count=1)


def _detect_injected(original: str, modified: str, pool: list[str]) -> str | None:
    ol, ml = original.lower(), modified.lower()
    for w in pool:
        if w.lower() in ml and w.lower() not in ol:
            return w
    return None


# ---------------------------------------------------------------------------
# Core injection logic
# ---------------------------------------------------------------------------


def _inject_once(
    client: OpenAI,
    model: str,
    claim: str,
    pool_key: str,
    pool_words: list[str],
    level: int,
    seed: str,
    *,
    excluded: list[str] | None = None,
    temperature: float = 0.25,
    max_tokens: int = 512,
    max_retries: int = 3,
    min_distance: int = 3,
) -> str:
    clean_claim = _strip_brackets(claim)
    for attempt in range(max_retries):
        rng = random.Random(f"{seed}_{level}_{attempt}")
        sampled = rng.sample(pool_words, min(POOL_SAMPLE_SIZE[pool_key], len(pool_words)))
        if level == 1:
            prompt = _PROMPT_L1.format(word_list=", ".join(sampled), claim=claim)
        else:
            prompt = _PROMPT_L2.format(
                word_list=", ".join(sampled),
                excluded_words=", ".join(excluded or []),
                claim=claim,
                min_distance=min_distance,
            )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = resp.choices[0].message.content or ""
        result = _clean_model_output(raw)
        result_clean = _strip_brackets(result)

        if _find_insert_pos(_norm(clean_claim), _norm(result_clean)) is None:
            if attempt == max_retries - 1:
                print(f"  [WARN] {seed} L{level}: word-count check failed")
            continue

        if level == 2:
            orig_words = _norm(claim)
            mod_words = _norm(result)
            ins = _find_insert_pos(orig_words, mod_words)
            brk = _find_bracket_pos(mod_words)
            if ins is not None and brk is not None and abs(ins - brk) < min_distance:
                if attempt == max_retries - 1:
                    print(f"  [WARN] {seed} L{level}: adjacency check failed")
                continue

        return result

    return result  # last attempt as fallback


def _apply_inject(
    client: OpenAI,
    model: str,
    item: dict,
    condition: str,
    *,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    min_distance: int,
) -> dict:
    pool_key, level = CONDITION_SPECS[condition]
    pool_words = POOLS[pool_key]
    seed = f"{item.get('claim_id', '')}_{pool_key}"
    original = item["claim"]

    kw = dict(temperature=temperature, max_tokens=max_tokens,
               max_retries=max_retries, min_distance=min_distance)

    if level == 1:
        l1_raw = _inject_once(client, model, original, pool_key, pool_words, 1, seed, **kw)
        new_claim = _strip_brackets(l1_raw)
    else:
        l1_raw = _inject_once(client, model, original, pool_key, pool_words, 1, seed, **kw)
        l1_clean = _strip_brackets(l1_raw)
        l1_word = _detect_injected(original, l1_clean, pool_words)
        l2_input = _mark_word(l1_clean, l1_word) if l1_word else l1_clean
        l2_raw = _inject_once(
            client, model, l2_input, pool_key, pool_words, 2, seed,
            excluded=[l1_word] if l1_word else None, **kw,
        )
        new_claim = _strip_brackets(l2_raw)

    return {**item, "claim": new_claim}


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


def run_inject_rewrite(
    items: list[dict],
    condition: str,
    client: OpenAI,
    model: str,
    *,
    temperature: float = 0.25,
    max_tokens: int = 512,
    workers: int = 8,
    max_retries: int = 3,
    min_distance: int = 3,
) -> list[dict]:
    """Apply word injection for one condition to all items. Returns rewritten list."""
    if condition not in CONDITION_SPECS:
        raise ValueError(f"Unknown inject condition {condition!r}. Known: {list(CONDITION_SPECS)}")
    results: list[dict | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _apply_inject, client, model, item, condition,
                temperature=temperature, max_tokens=max_tokens,
                max_retries=max_retries, min_distance=min_distance,
            ): i
            for i, item in enumerate(items)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=condition):
            results[futures[fut]] = fut.result()
    return [r for r in results if r is not None]
