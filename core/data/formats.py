"""
Format-specific normalizers: convert raw dataset items to the common internal schema.

Adding a new dataset
--------------------
1. Write a ``_normalize_<format>(item, data_dir)`` function below.
2. Register it in ``_NORMALIZERS``.
3. Set ``"format": "<format>"`` in the dataset's ``DATASETS`` entry in config.py.
4. If normalization requires paper-JSON lookups or other I/O, do it here
   (lazy, on every load_base_dataset call — fast enough for local file I/O).

Common internal schema
----------------------
Required:
    claim_id   str   unique identifier
    claim      str   text to verify / rewrite
    label      str   "Supported" or "Refuted"
    evi_path   str   relative path to evidence image under dataset data_dir

Optional (inference uses sensible defaults when absent):
    evi_type       str   "figure"   evidence type; "table" triggers PNG fallback lookup
    caption        str   ""         image caption
    use_context    str   "no"       "no" | "yes" | "other sources"
    context        str   ""         inline context (used when use_context == "yes")
    paper_path     str   ""         path to paper JSON (use_context == "other sources")

Pair analysis (omit for pairless datasets):
    claim_id_pair  any   pair identifier linking one Supported and one Refuted claim

Normalizers may return ``None`` to drop an item (e.g. unsupported claim types).
load_base_dataset filters out None values after normalization.
"""

from __future__ import annotations

import functools
import json
import os


# ---------------------------------------------------------------------------
# SciClaimEval — identity (already in common schema)
# ---------------------------------------------------------------------------


def _normalize_sciclaimeval(item: dict, data_dir: str) -> dict:
    """SciClaimEval items already use the common internal schema — pass through."""
    return item


# ---------------------------------------------------------------------------
# colortest (mock dataset)
# ---------------------------------------------------------------------------


def _normalize_colortest(item: dict, data_dir: str) -> dict:
    """
    Map colortest raw fields to the common internal schema.

    Raw schema:  id, image, claim, label
    Normalized:  claim_id (← id), evi_path (← image), + defaults.
    claim_id_pair is absent intentionally — colortest is pairless.
    """
    return {
        **item,
        "claim_id":    item["id"],
        "evi_path":    item["image"],
        "evi_type":    "figure",
        "caption":     "",
        "use_context": "no",
    }


# ---------------------------------------------------------------------------
# SciVer
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _load_paper_json(path: str) -> dict:
    """Load and cache a paper JSON by absolute path."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _sciver_abs_paper_path(raw_paper_path: str, data_dir: str) -> str:
    """
    Resolve a SciVer paper path like ``./SciVer/papers/X.json`` to an absolute path.
    SciVer raw paths use ``./SciVer/`` as a prefix relative to the ``data/`` directory,
    so we strip that prefix and join with data_dir (which points at data/SciVer/).
    """
    rel = raw_paper_path
    for prefix in ("./SciVer/", "SciVer/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return os.path.join(data_dir, rel)


def _sciver_caption(paper: dict, item_type: str, item_id: str) -> str:
    """Extract figure/table caption from a loaded SciVer paper dict."""
    item_id = str(item_id)
    if item_type == "chart":
        return paper.get("image_paths", {}).get(item_id, {}).get("caption", "")
    else:
        return paper.get("tables", {}).get(item_id, {}).get("capture", "")


def _sciver_context(paper: dict, section_list: list) -> str:
    """Extract the relevant paper sections as a single context string."""
    top_sections = []
    for sec_id in section_list:
        top = sec_id.split(".")[0]
        if top not in top_sections:
            top_sections.append(top)
    parts: list[str] = []
    for sec in paper.get("sections", []):
        sec_top = sec.get("section_id", "").split(".")[0]
        if sec_top in top_sections:
            name = sec.get("section_name", "")
            text = sec.get("text", "")
            if name or text:
                parts.append(f"{name}:\n{text}" if name else text)
    return "\n".join(parts)


def _normalize_sciver(item: dict, data_dir: str) -> dict | None:
    """
    Normalize a SciVer item to the common internal schema.

    - Skips parallel/sequential (two-image) items — returns None.
    - Maps boolean label (True/False) → "Supported"/"Refuted".
    - Creates a unique claim_id from request_id + label suffix.
    - Extracts caption and context from the paper JSON (lazy, cached per path).
    - Strips the ``./SciVer/`` path prefix so evi_path is relative to data_dir.
    """
    if item.get("claim_type") in ("parallel", "sequential"):
        return None

    label_bool = item.get("label")
    label = "Supported" if label_bool is True else "Refuted"
    suffix = "T" if label_bool is True else "F"
    claim_id = f"{item['request_id']}_{suffix}"

    # Strip ./SciVer/ prefix → path relative to data_dir (data/SciVer/)
    def _strip(raw: str) -> str:
        for prefix in ("./SciVer/", "SciVer/"):
            if raw.startswith(prefix):
                return raw[len(prefix):]
        return raw

    evi_path   = _strip(item.get("image_path", ""))
    paper_rel  = _strip(item.get("paper_path", ""))
    paper_abs  = os.path.join(data_dir, paper_rel)
    paper      = _load_paper_json(paper_abs)

    item_type  = item.get("type", "chart")
    item_id    = str(item.get("item", ""))
    caption    = _sciver_caption(paper, item_type, item_id)
    context    = _sciver_context(paper, item.get("section", []))

    return {
        **item,
        "claim_id":    claim_id,
        "label":       label,
        "evi_path":    evi_path,
        "evi_type":    "figure",
        "caption":     caption,
        "use_context": "yes",
        "context":     context,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_NORMALIZERS: dict[str, callable] = {
    "sciclaimeval": _normalize_sciclaimeval,
    "colortest":    _normalize_colortest,
    "sciver":       _normalize_sciver,
}


def get_normalizer(fmt: str):
    """
    Return the normalizer callable for the given format key.

    Raises ValueError for unknown formats so misconfigured datasets are caught
    immediately on load rather than silently producing wrong field names.
    """
    if fmt not in _NORMALIZERS:
        raise ValueError(
            f"Unknown dataset format {fmt!r}. "
            f"Registered formats: {sorted(_NORMALIZERS)}"
        )
    return _NORMALIZERS[fmt]
