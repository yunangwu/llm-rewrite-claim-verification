"""
Prompt construction and answer parsing for CoT self-consistency inference.

Builds the vision-language message list for each claim item and parses
the model's chain-of-thought answer back into Supported / Refuted.

Format-aware dispatchers (``build_prompt`` / ``parse_answer``) select the
correct builder / parser based on the dataset format string so the pipeline
does not need to know about individual dataset implementations.
"""

from __future__ import annotations

import base64
import functools
import json
import re
from pathlib import Path
from string import Template

MAX_TOKENS_COT = 2048

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_media_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    return "image/png"


def resolve_image_path(data_dir: str, evi_path: str) -> str:
    """Resolve evi_path to an existing image file under data_dir."""
    full = Path(data_dir) / evi_path
    if full.exists():
        return str(full)
    # Fallback: convert table paths to PNG equivalents
    png = str(full).replace("/tables/", "/tables_png/")
    for ext in (".tex", ".html", ".csv"):
        png = png.replace(ext, ".png")
    return png if Path(png).exists() else str(full)


# ---------------------------------------------------------------------------
# Paper context extraction
# ---------------------------------------------------------------------------


def extract_context(paper_path: str, max_chars: int = 30_000) -> str:
    with open(paper_path, encoding="utf-8") as f:
        paper = json.load(f)
    parts: list[str] = []
    if paper.get("title"):
        parts.append(str(paper["title"]))
    if paper.get("abstract"):
        parts.append(str(paper["abstract"]))
    sections = paper.get("full_text") or paper.get("sections", [])
    for sec in sections:
        if sec.get("section_name", "").strip().lower() == "conclusion":
            break
        for para in sec.get("paragraphs", []) or [sec.get("text", "")]:
            if para:
                parts.append(str(para))
    return " ".join(parts)[:max_chars]


# ---------------------------------------------------------------------------
# Shared content builder
# ---------------------------------------------------------------------------


def _build_content(item: dict, data_dir: str) -> tuple[list[dict], list[str]]:
    """Build [image_part], [text_parts] shared by all prompts."""
    image_path = resolve_image_path(data_dir, item["evi_path"])
    if not Path(image_path).exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}  (evi_path={item['evi_path']}, data_dir={data_dir})"
        )
    b64 = encode_image_b64(image_path)
    content: list[dict] = [
        {"type": "image_url",
         "image_url": {"url": f"data:{image_media_type(image_path)};base64,{b64}"}}
    ]
    text_parts: list[str] = []
    if item.get("caption"):
        text_parts.append(f"Image Caption:\n{item['caption']}")
    ctx = item.get("use_context", "no")
    if ctx == "yes" and item.get("context"):
        text_parts.append(f"Context:\n{item['context']}")
    elif ctx == "other sources" and item.get("paper_path"):
        paper_full = str(Path(data_dir) / item["paper_path"])
        if Path(paper_full).exists():
            text_parts.append(f"Context:\n{extract_context(paper_full)}")
    return content, text_parts


# ---------------------------------------------------------------------------
# CoT prompt
# ---------------------------------------------------------------------------


def build_cot_prompt(item: dict, data_dir: str) -> list[dict]:
    content, text_parts = _build_content(item, data_dir)
    evi_type = item.get("evi_type", "")
    evidence_label = "table image" if evi_type == "table" else "image"
    has_caption = bool(item.get("caption"))
    has_context = bool(
        (item.get("use_context") == "yes" and item.get("context"))
        or (
            item.get("use_context") == "other sources"
            and item.get("paper_path")
            and Path(data_dir, item["paper_path"]).exists()
        )
    )
    parts = [evidence_label]
    if has_caption:
        parts.append("image caption")
    if has_context:
        parts.append("context")
    text_parts.append(
        f"Use the provided {', '.join(parts)} to predict the label "
        f"for this claim: '{item['claim']}'; the label can be Supported or Refuted. "
        f"Think step by step before answering. "
        f"Please format your final answer within brackets as follows: <ans> YOUR ANSWER </ans>"
    )
    content.append({"type": "text", "text": "\n".join(text_parts)})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------


def parse_cot_answer(text: str) -> str:
    """Extract Supported / Refuted from a CoT response. Returns '' on failure."""
    for ans in re.findall(r"<ans>\s*(.*?)\s*</ans>", text, re.DOTALL | re.IGNORECASE):
        a = ans.strip()
        if a.lower() not in {"your answer", "the answer", "", "insert answer here"}:
            return a.capitalize()
    m = re.search(r"\[(Supported|Refuted|Neutral)\]", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    m = re.search(
        r"\bthe claim (is|appears to be|can be considered to be)?\s*(supported|refuted)\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(2).capitalize()
    return ""


# ---------------------------------------------------------------------------
# SciVer prompt and parser
# ---------------------------------------------------------------------------

# Prompt templates adapted from SciVer's utils/constant.py.
# Single-image variants only (direct and analytical).
_SCIVER_PROMPT_DIRECT = Template("""\
Claim: $claim
Context: $context
Caption: $caption

Your task is to critically evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Be skeptical and cautious: if there is any inconsistency, missing evidence, or ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying potential contradictions, lack of support, or misleading interpretations. If the claim is unsupported or contradicted by the caption and image, respond with 'no'. Only respond with 'yes' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

_SCIVER_PROMPT_ANALYTICAL = Template("""\
Claim: $claim
Context: $context
Caption: $caption

Your task is to evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Apply any relevant scientific principles, statistical logic, or domain knowledge necessary to link the evidence to the claim. Be balanced: actively look for confirming details as well as inconsistencies, missing evidence, or ambiguities.

Start by explaining your reasoning process step by step—describe what you observe in the image and caption, what background knowledge you use, and how you test whether each key part of the claim is supported. If every essential component of the claim is clearly and completely backed by the caption and image, respond with 'yes'. If any critical point is contradicted, unsupported, or unclear, respond with 'no'.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

_SCIVER_PROMPT_BY_TYPE: dict[str, Template] = {
    "direct":     _SCIVER_PROMPT_DIRECT,
    "analytical": _SCIVER_PROMPT_ANALYTICAL,
}


def build_sciver_prompt(item: dict, data_dir: str) -> list[dict]:
    """Build a SciVer-style vision-language message using the original prompt templates."""
    image_path = resolve_image_path(data_dir, item["evi_path"])
    if not Path(image_path).exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}  (evi_path={item['evi_path']}, data_dir={data_dir})"
        )
    b64 = encode_image_b64(image_path)
    tmpl = _SCIVER_PROMPT_BY_TYPE.get(item.get("claim_type", "analytical"), _SCIVER_PROMPT_ANALYTICAL)
    text = tmpl.substitute(
        claim=item.get("claim", ""),
        context=item.get("context", ""),
        caption=item.get("caption", ""),
    )
    content: list[dict] = [
        {"type": "image_url",
         "image_url": {"url": f"data:{image_media_type(image_path)};base64,{b64}"}},
        {"type": "text", "text": text},
    ]
    return [{"role": "user", "content": content}]


def parse_sciver_answer(text: str) -> str:
    """
    Extract Supported / Refuted from a SciVer-style CoT response.

    The model is instructed to end with ``Answer: yes`` or ``Answer: no``.
    Falls back to scanning the tail of the response for 'yes'/'no'.
    Returns '' if no valid answer is found.
    """
    m = re.search(r"Answer:\s*(yes|no)\b", text, re.IGNORECASE)
    if m:
        return "Supported" if m.group(1).lower() == "yes" else "Refuted"
    # Fallback: look near the end for an explicit yes/no
    tail = text[-300:].lower()
    final_idx = tail.rfind("final answer")
    region = tail[final_idx:] if final_idx >= 0 else tail
    if "yes" in region:
        return "Supported"
    if "no" in region:
        return "Refuted"
    return ""


# ---------------------------------------------------------------------------
# Format-aware dispatchers
# ---------------------------------------------------------------------------


def build_prompt(item: dict, data_dir: str, fmt: str = "sciclaimeval") -> list[dict]:
    """Return the vision-language message list for *item* using *fmt*'s prompt."""
    if fmt == "sciver":
        return build_sciver_prompt(item, data_dir)
    return build_cot_prompt(item, data_dir)


def parse_answer(text: str, fmt: str = "sciclaimeval") -> str:
    """Parse a model response back to 'Supported' / 'Refuted' using *fmt*'s parser."""
    if fmt == "sciver":
        return parse_sciver_answer(text)
    return parse_cot_answer(text)
