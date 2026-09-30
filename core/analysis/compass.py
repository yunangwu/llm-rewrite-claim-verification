"""
Compass scatter plots: Δp(supported on Supported) vs Δp(supported on Refuted).

Functions return matplotlib Figure objects — no file I/O.
Caller (run.py) saves them via io.save_figure().
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.config import (
    ALL_CONDITIONS,
    CONDITION_COLORS,
    CONDITION_DISPLAY,
    CONDITION_MARKERS,
    CONDITION_MARKER_SIZES,
)


# ---------------------------------------------------------------------------
# Shared axes setup
# ---------------------------------------------------------------------------


def _setup_axes(ax: plt.Axes, lim: float | None = None) -> None:
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    if lim is None:
        lim = max(
            abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
            abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]),
            0.05,
        )
    ax.plot([-lim, lim], [-lim, lim], color="#d8d8d8", linewidth=0.9, linestyle=":", zorder=0)
    ax.plot([-lim, lim], [lim, -lim], color="#d8d8d8", linewidth=0.9, linestyle=":", zorder=0)
    off_hi = lim * 0.76   # upper quadrants — more room
    off_lo = lim * 0.62   # lower quadrants — keep above legend / data
    for (x, y, label) in [
        ( off_hi,  off_hi, "Globally\nOptimistic"),
        (-off_lo, -off_lo, "Globally\nConservative"),
        ( off_lo, -off_lo, "Sharper"),
        (-off_hi,  off_hi, "Duller"),
    ]:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=19, color="#aaaaaa", fontstyle="italic")
    ax.set_xlabel(r"$\Delta p(\mathrm{support})$ on Supported evidence", fontsize=18)
    ax.set_ylabel(r"$\Delta p(\mathrm{support})$ on Refuted evidence", fontsize=18)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=15, color="#bbbbbb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_edgecolor("#cccccc")
    ax.spines["left"].set_edgecolor("#cccccc")
    ax.set_facecolor("#f9f9f9")


# ---------------------------------------------------------------------------
# Single condition compass (all points for one model × condition)
# ---------------------------------------------------------------------------


def plot_compass_single(
    shifts: list[dict],
    model_name: str,
    condition_name: str,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if not shifts:
        ax.set_title(f"{model_name} — {condition_name} (no data)")
        return fig
    dp_sups = [s["dp_sup"] for s in shifts]
    dp_refs = [s["dp_ref"] for s in shifts]
    ax.scatter(dp_sups, dp_refs, alpha=0.12, s=15, c="steelblue", edgecolors="none")
    ax.scatter(
        np.mean(dp_sups), np.mean(dp_refs), s=120, c="red",
        edgecolors="black", linewidths=1.5, zorder=5,
        label=f"Mean ({np.mean(dp_sups):+.4f}, {np.mean(dp_refs):+.4f})",
    )
    _setup_axes(ax)
    ax.set_title(f"{model_name} — {condition_name}", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-model compass: all conditions on one plot (mean points only)
# ---------------------------------------------------------------------------


def plot_compass_per_model(
    shifts_by_cond: dict[str, list[dict]],
    model_name: str,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for cond in ALL_CONDITIONS:
        shifts = shifts_by_cond.get(cond, [])
        if not shifts:
            continue
        mean_sup = np.mean([s["dp_sup"] for s in shifts])
        mean_ref = np.mean([s["dp_ref"] for s in shifts])
        ax.scatter(
            mean_sup, mean_ref,
            s=CONDITION_MARKER_SIZES[cond],
            c=CONDITION_COLORS[cond],
            marker=CONDITION_MARKERS[cond],
            edgecolors="black", linewidths=1.2, zorder=5,
            label=CONDITION_DISPLAY[cond],
        )
    _setup_axes(ax)
    ax.set_title(f"{model_name} — All Conditions", fontsize=13)
    ax.legend(fontsize=11, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Overview: grand-mean across all models for each condition
# ---------------------------------------------------------------------------


def plot_compass_overview(all_data: dict[str, dict[str, list[dict]]]) -> Figure:
    """
    all_data: {model_slug: {condition: [shift, ...]}}
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    model_names = list(all_data)

    grand_points: list[tuple[str, float, float]] = []  # (cond, gx, gy) for annotation pass

    for cond in ALL_CONDITIONS:
        color = CONDITION_COLORS[cond]
        per_model: list[tuple[float, float]] = []
        for model in model_names:
            shifts = all_data[model].get(cond, [])
            if not shifts:
                continue
            ms = float(np.mean([s["dp_sup"] for s in shifts]))
            mr = float(np.mean([s["dp_ref"] for s in shifts]))
            per_model.append((ms, mr))
            # Per-model dots: small uniform circles, condition colour, semi-transparent
            ax.scatter(ms, mr, s=18, c=color, marker="o",
                       alpha=0.18, edgecolors="none", linewidths=0, zorder=3)
        if not per_model:
            continue
        gx = float(np.mean([p[0] for p in per_model]))
        gy = float(np.mean([p[1] for p in per_model]))
        grand_points.append((cond, gx, gy))
        # Grand-mean dot: single solid circle with thin white border
        ax.scatter(
            gx, gy,
            s=CONDITION_MARKER_SIZES[cond] + 80,
            c=color, marker=CONDITION_MARKERS[cond],
            edgecolors="white", linewidths=1.5, zorder=6,
            label=f"{CONDITION_DISPLAY[cond]}  ({gx:+.3f}, {gy:+.3f})",
        )

    _setup_axes(ax)

    leg = ax.legend(
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # just below the x-axis label
        ncol=3,
        framealpha=0.97, edgecolor="#d0d0d0",
        borderpad=0.85, handletextpad=0.5, labelspacing=0.45,
        columnspacing=0.9, handlelength=1.0, handleheight=1.2,
    )
    leg.get_frame().set_linewidth(0.8)

    # Extra bottom margin to accommodate the legend below the axes
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.22)
    return fig
