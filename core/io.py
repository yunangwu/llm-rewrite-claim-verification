"""
File I/O and directory layout helpers.

Layout (all paths relative to REPO_ROOT = 0356_hedge_booster/):
  rewrites/<dataset>/<split>/natural/<condition>.json  — natural rewrite files
  rewrites/<dataset>/<split>/inject/<condition>.json   — injection rewrite files
  runs/<dataset>/<split>/<model-slug>/<condition>.json — inference outputs
  runs/<dataset>/<split>/<model-slug>/run_meta.json    — run parameters (audit only)
  results/<dataset>/<split>/<eval-mode>/compass/       — analysis artifacts
  results/<dataset>/<split>/<eval-mode>/drift/
  results/<dataset>/<split>/<eval-mode>/strip/
  results/<dataset>/<split>/<eval-mode>/tables/
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import REPO_ROOT, sort_model_slugs
from core.data.formats import get_normalizer


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def runs_dir(dataset: str, split: str) -> Path:
    return REPO_ROOT / "runs" / dataset / split


def model_run_dir(dataset: str, split: str, slug: str) -> Path:
    return runs_dir(dataset, split) / slug


def results_dir(dataset: str, split: str, eval_mode: str) -> Path:
    return REPO_ROOT / "results" / dataset / split / eval_mode


def rewrite_dir(dataset: str, split: str, method: str) -> Path:
    """method: 'natural' or 'inject'"""
    return REPO_ROOT / "rewrites" / dataset / split / method


# ---------------------------------------------------------------------------
# Run file paths
# ---------------------------------------------------------------------------


def run_path(dataset: str, split: str, slug: str, condition: str) -> Path:
    return model_run_dir(dataset, split, slug) / f"{condition}.json"


def run_meta_path(dataset: str, split: str, slug: str) -> Path:
    return model_run_dir(dataset, split, slug) / "run_meta.json"


def run_exists(dataset: str, split: str, slug: str, condition: str) -> bool:
    return run_path(dataset, split, slug, condition).exists()


# ---------------------------------------------------------------------------
# Load / save inference runs
# ---------------------------------------------------------------------------


def load_run(dataset: str, split: str, slug: str, condition: str) -> list[dict[str, Any]]:
    path = run_path(dataset, split, slug, condition)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def save_run(results: list[dict], dataset: str, split: str, slug: str, condition: str) -> Path:
    path = run_path(dataset, split, slug, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def save_run_meta(meta: dict, dataset: str, split: str, slug: str) -> None:
    path = run_meta_path(dataset, split, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Dataset / model discovery
# ---------------------------------------------------------------------------


def list_model_slugs(dataset: str, split: str) -> list[str]:
    """Return model slugs under runs/<dataset>/<split>/ sorted by size then name."""
    d = runs_dir(dataset, split)
    if not d.exists():
        return []
    names = [
        p.name for p in d.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    return sort_model_slugs(names)


def load_base_dataset(dataset_cfg: dict, split: str) -> list[dict]:
    """Load the base (original) dataset JSON for the given split."""
    path = REPO_ROOT / dataset_cfg["splits"][split]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    normalize = get_normalizer(dataset_cfg.get("format", "sciclaimeval"))
    data_dir_abs = str((REPO_ROOT / dataset_cfg["data_dir"]).resolve())
    normalized = [normalize(item, data_dir_abs) for item in data]
    return [item for item in normalized if item is not None]


# ---------------------------------------------------------------------------
# Rewrite file paths
# ---------------------------------------------------------------------------


def rewrite_path(dataset: str, split: str, method: str, condition: str) -> Path:
    return rewrite_dir(dataset, split, method) / f"{condition}.json"


def load_rewrite(dataset: str, split: str, method: str, condition: str) -> list[dict]:
    path = rewrite_path(dataset, split, method, condition)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def save_rewrite(items: list[dict], dataset: str, split: str, method: str, condition: str) -> Path:
    path = rewrite_path(dataset, split, method, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Analysis artifact helpers
# ---------------------------------------------------------------------------


def save_figure(fig: Any, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clf()
    import matplotlib.pyplot as plt
    plt.close(fig)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
