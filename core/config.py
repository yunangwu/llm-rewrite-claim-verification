"""
Central configuration: conditions, eval modes, field keys, and config loading.

Add a new dataset  → add an entry to DATASETS.
Add a new model    → add an entry to config/models.yaml.
Switch eval mode   → pass --eval-mode at CLI or change DEFAULT_EVAL_MODE here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "sciclaimeval": {
        # base files per split (relative to REPO_ROOT)
        "splits": {
            "dev":  "data/SciClaimEval/dev_task1_release.json",
            "test": "data/SciClaimEval/test_task1.json",
        },
        # root for image/paper resolution (evi_path is relative to this)
        "data_dir": "data/SciClaimEval",
        "format": "sciclaimeval",
        # paired datasets support pair_acc, compass scatter, and pair-based shifts
        "has_pairs": True,
    },
    # Alias for the rebuttal's second rewriter (gemma-4). Identical source data
    # to `sciclaimeval`; only the dataset KEY differs, which namespaces the
    # rewrites/<key>/… and runs/<key>/… folders so gemma output never overwrites
    # the original gpt-oss files.
    "sciclaimeval_gemma": {
        "splits": {
            "dev":  "data/SciClaimEval/dev_task1_release.json",
            "test": "data/SciClaimEval/test_task1.json",
        },
        "data_dir": "data/SciClaimEval",
        "format": "sciclaimeval",
        "has_pairs": True,
    },
    "colortest": {
        "splits": {
            "dev": "data/colortest/dev_release.json",
        },
        "data_dir": "data/colortest",
        "format": "colortest",
        # no Supported/Refuted pairing: pair_acc and compass scatter are skipped;
        # F1, strip chart, drift, and shift table use per-item label grouping.
        "has_pairs": False,
    },
    "sciver": {
        "splits": {
            "dev":  "data/SciVer/valset.json",
            "test": "data/SciVer/testset.json",
        },
        # evi_path values in normalized items are relative to this directory
        "data_dir": "data/SciVer",
        "format": "sciver",
        # parallel/sequential (two-image) items are dropped at normalization time;
        # remaining direct/analytical items are pairless — each has a unique claim_id.
        "has_pairs": False,
    },
}

DEFAULT_SPLIT = "dev"

# ---------------------------------------------------------------------------
# Rewrite conditions
# ---------------------------------------------------------------------------

# Natural rewrite groups: each group is averaged over its variants in analysis.
NATURAL_GROUPS: dict[str, list[str]] = {
    "LP": [f"LP-{i}" for i in range(1, 6)],
    "FI": [f"FI-{i}" for i in range(1, 6)],
    "AS": [f"AS-{i}" for i in range(1, 6)],
    "OA": [f"OA-{i}" for i in range(1, 6)],
    "CT": [f"CT-{i}" for i in range(1, 6)],
}

# Controlled (word-injection) conditions: single file each.
CONTROLLED_CONDITIONS: list[str] = ["D_1", "C_1"]

# Flat list used by analysis loops (natural group names + controlled).
ALL_CONDITIONS: list[str] = list(NATURAL_GROUPS) + CONTROLLED_CONDITIONS

# All individual variant keys (used by inference and rewrite CLI).
ALL_VARIANTS: list[str] = (
    [v for variants in NATURAL_GROUPS.values() for v in variants]
    + CONTROLLED_CONDITIONS
)

# ---------------------------------------------------------------------------
# Eval mode: inference now outputs only self-consistency fields.
# The dict is kept for analysis flexibility (e.g. comparing against old runs
# that contain direct/self_reported fields), but "consistency" is the default
# and only actively produced by pipeline.py.
# ---------------------------------------------------------------------------

# (pred_label_key, p_supported_key)
EVAL_MODES: dict[str, tuple[str, str]] = {
    "consistency": ("pred_label_consistency", "p_supported_consistency"),
}

DEFAULT_EVAL_MODE = "consistency"

# ---------------------------------------------------------------------------
# Display / plotting metadata
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "Kimi-VL-A3B-Thinking-2506": "Kimi-VL-A3B",
}

CONDITION_DISPLAY: dict[str, str] = {
    "LP": "LP", "FI": "FI", "AS": "AS", "OA": "OA", "CT": "CT",
    "D_1": "D1", "D_2": "D2", "C_1": "C1", "C_2": "C2",
}

CONDITION_COLORS: dict[str, str] = {
    "LP": "#1f77b4", "FI": "#ff7f0e", "AS": "#2ca02c", "OA": "#d62728",
    "CT": "#8c564b",
    "D_1": "#9467bd", "D_2": "#9467bd", "C_1": "#e377c2", "C_2": "#e377c2",
}

CONDITION_MARKERS: dict[str, str] = {
    "LP": "o", "FI": "o", "AS": "o", "OA": "o", "CT": "o",
    "D_1": "o", "D_2": "*", "C_1": "o", "C_2": "*",
}

CONDITION_MARKER_SIZES: dict[str, int] = {
    "LP": 120, "FI": 120, "AS": 120, "OA": 120, "CT": 120,
    "D_1": 120, "D_2": 200, "C_1": 120, "C_2": 200,
}

# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models_cfg() -> dict[str, dict]:
    return _load_yaml(CONFIG_DIR / "models.yaml")


def load_inference_cfg() -> dict:
    return _load_yaml(CONFIG_DIR / "inference.yaml")


def load_rewrite_cfg() -> dict:
    return _load_yaml(CONFIG_DIR / "rewrite.yaml")


def model_slug(model_name: str) -> str:
    """Last path component of HF model ID, used as directory name."""
    return model_name.split("/")[-1]


def model_family_sort_prefix(slug: str) -> str:
    """
    Rough "family" for ordering tables/plots: characters before the first digit in the slug
    (e.g. ``Qwen3-VL-8B`` → ``qwen``, ``InternVL3-2B`` → ``internvl``, ``gemma-4-26B`` → ``gemma``).
    """
    import re

    m = re.match(r"^([^0-9]+)", slug)
    if not m:
        return slug.lower()
    return m.group(1).strip("-_.").lower()


def model_slug_sort_key(slug: str) -> tuple[str, int, str]:
    """
    Sort key for run-directory slugs: family alphabetically, then billions ascending,
    then slug — keeps lines together (e.g. all Qwen3-VL by size) instead of mixing
    unrelated models purely by parameter count.
    """
    import re

    family = model_family_sort_prefix(slug)
    nums = [int(m) for m in re.findall(r"(\d+)\s*B\b", slug, re.IGNORECASE)]
    billions = max(nums) if nums else 0
    return (family, billions, slug.lower())


def sort_model_slugs(slugs: list[str]) -> list[str]:
    """Return a copy sorted for stable reporting (tables, plots)."""
    return sorted(slugs, key=model_slug_sort_key)
