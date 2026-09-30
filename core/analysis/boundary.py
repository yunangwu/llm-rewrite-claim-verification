"""
Boundary scatter: p_base vs p_rewrite with decision-boundary crosshairs.

Each point is one claim. The crosshairs at (0.5, 0.5) divide the plot into
four quadrants:

  top-right   (p_base > 0.5, p_rewrite > 0.5)  → Supported stays Supported  ✓
  bottom-left (p_base < 0.5, p_rewrite < 0.5)  → Refuted stays Refuted      ✓
  bottom-right (p_base > 0.5, p_rewrite < 0.5) → Supported flips to Refuted  ✗
  top-left    (p_base < 0.5, p_rewrite > 0.5)  → Refuted flips to Supported  ✗

Points moving along the diagonal = no change in confidence.
Points departing from the diagonal but staying inside a "safe" quadrant =
  shifted confidence but accuracy preserved (the key story of this experiment).

Returns matplotlib Figure — no file I/O.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from core.config import ALL_CONDITIONS, CONDITION_DISPLAY


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_boundary_data(
    base_preds: dict,
    rewrite_preds_by_condition: dict[str, list[dict]],
    valid_ids: set,
) -> dict[str, dict]:
    """
    Build per-condition scatter data consumed by plot_boundary_scatter().

    rewrite_preds_by_condition: {condition: list[preds_dict]}
        For natural groups, pass all per-variant preds; p_rewrite is averaged
        across variants per claim to mirror the analysis convention.
        For controlled conditions, pass a single-element list.

    Returns:
        {condition: {
            "sup": {"p_base": [...], "p_rewrite": [...]},
            "ref": {"p_base": [...], "p_rewrite": [...]},
        }}
    """
    result: dict[str, dict] = {}

    for cond, preds_list in rewrite_preds_by_condition.items():
        if not preds_list:
            continue

        sup_base: list[float] = []
        sup_rew:  list[float] = []
        ref_base: list[float] = []
        ref_rew:  list[float] = []

        # Collect per-claim average p_rewrite across variants
        for cid in sorted(valid_ids):
            base = base_preds.get(cid)
            if base is None or base["p_supported"] is None:
                continue
            p_base = base["p_supported"]
            label  = base["label"]

            per_variant_p = [
                preds[cid]["p_supported"]
                for preds in preds_list
                if cid in preds and preds[cid]["p_supported"] is not None
            ]
            if not per_variant_p:
                continue
            p_rew = float(np.mean(per_variant_p))

            if label == "Supported":
                sup_base.append(p_base)
                sup_rew.append(p_rew)
            elif label == "Refuted":
                ref_base.append(p_base)
                ref_rew.append(p_rew)

        result[cond] = {
            "sup": {"p_base": sup_base, "p_rewrite": sup_rew},
            "ref": {"p_base": ref_base, "p_rewrite": ref_rew},
        }

    return result


# ---------------------------------------------------------------------------
# Flip-rate computation
# ---------------------------------------------------------------------------


def compute_flip_rates(
    base_preds: dict,
    rewrite_preds_by_condition: dict[str, list[dict]],
    valid_ids: set,
) -> dict[str, dict]:
    """
    Per-condition fraction of claims whose binary label (≥ 0.5) flips after
    rewriting, reported separately for Supported and Refuted claims.

    For natural conditions (multiple variants), p_rewrite is averaged across
    variants per claim before applying the threshold — consistent with how
    the shift and boundary-scatter analyses work.

    Returns:
        {condition: {"flip_sup": float, "flip_ref": float}}
        Values are in [0, 1] (fractions, not percentages).
        NaN when no valid claims of that label are found.
    """
    result: dict[str, dict] = {}

    for cond, preds_list in rewrite_preds_by_condition.items():
        if not preds_list:
            result[cond] = {"flip_sup": float("nan"), "flip_ref": float("nan")}
            continue

        sup_flips, sup_total = 0, 0
        ref_flips, ref_total = 0, 0

        for cid in valid_ids:
            base = base_preds.get(cid)
            if base is None or base["p_supported"] is None:
                continue
            p_base = base["p_supported"]
            label  = base["label"]

            per_variant_p = [
                preds[cid]["p_supported"]
                for preds in preds_list
                if cid in preds and preds[cid]["p_supported"] is not None
            ]
            if not per_variant_p:
                continue
            p_rew = float(np.mean(per_variant_p))

            flipped = (p_base >= 0.5) != (p_rew >= 0.5)
            if label == "Supported":
                sup_total += 1
                sup_flips += int(flipped)
            elif label == "Refuted":
                ref_total += 1
                ref_flips += int(flipped)

        result[cond] = {
            "flip_sup": sup_flips / sup_total if sup_total else float("nan"),
            "flip_ref": ref_flips / ref_total if ref_total else float("nan"),
        }

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _boundary_axes(ax: plt.Axes) -> None:
    """Draw the diagonal, boundary crosshairs, and quadrant labels."""
    ax.plot([0, 1], [0, 1], color="#d0d0d0", linewidth=0.8, linestyle="--",
            zorder=1, label="_nolegend_")
    ax.axvline(0.5, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=2)
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _flip_rate(p_base: list[float], p_rew: list[float]) -> float:
    """Fraction of claims that cross the 0.5 boundary after rewriting."""
    if not p_base:
        return float("nan")
    flips = sum(
        (b >= 0.5) != (r >= 0.5)
        for b, r in zip(p_base, p_rew)
    )
    return flips / len(p_base)


def plot_boundary_scatter(
    boundary_data: dict[str, dict],
    title: str,
    *,
    alpha: float = 0.25,
    point_size: float = 6.0,
) -> Figure:
    """
    boundary_data: output of collect_boundary_data().
    title:         suptitle string (typically the model slug).

    Subplot grid follows ALL_CONDITIONS order (natural + controlled).
    Each panel shows Supported (blue) and Refuted (red) claims as a scatter
    of (p_base, p_rewrite) with boundary crosshairs at 0.5 and the identity
    diagonal. A flip-rate annotation shows the fraction of claims whose binary
    label changed after rewriting.
    """
    n_cond = len(ALL_CONDITIONS)
    ncols  = min(3, max(1, n_cond))
    nrows  = (n_cond + ncols - 1) // ncols
    fig_w  = 3.6 * ncols
    fig_h  = 3.4 * nrows + 0.9

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    SUP_COLOR = "#4878A8"
    REF_COLOR = "#C75A5A"

    for idx, cond in enumerate(ALL_CONDITIONS):
        ax   = axes_flat[idx]
        disp = CONDITION_DISPLAY.get(cond, cond)
        _boundary_axes(ax)

        if cond not in boundary_data:
            ax.set_title(disp, fontsize=11, pad=4)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#aaaaaa")
            continue

        d = boundary_data[cond]
        sup_b, sup_r = d["sup"]["p_base"], d["sup"]["p_rewrite"]
        ref_b, ref_r = d["ref"]["p_base"], d["ref"]["p_rewrite"]

        # Scatter — draw refuted first so supported is on top
        if ref_b:
            ax.scatter(ref_b, ref_r, s=point_size, c=REF_COLOR,
                       alpha=alpha, edgecolors="none", zorder=3)
        if sup_b:
            ax.scatter(sup_b, sup_r, s=point_size, c=SUP_COLOR,
                       alpha=alpha, edgecolors="none", zorder=4)

        # Flip rate annotation (combined across both label groups)
        all_b = sup_b + ref_b
        all_r = sup_r + ref_r
        fr = _flip_rate(all_b, all_r)
        if not np.isnan(fr):
            txt = ax.text(
                0.97, 0.03,
                f"flip {fr:.1%}",
                transform=ax.transAxes,
                fontsize=8, color="#555555",
                ha="right", va="bottom",
            )
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        ax.set_title(disp, fontsize=11, pad=4)

        # Tick / spine cosmetics
        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.tick_params(axis="x", length=3, width=0.4, labelsize=9)
        else:
            ax.tick_params(axis="x", length=0, labelbottom=False)
        if col == 0:
            ax.tick_params(axis="y", length=3, width=0.4, labelsize=9)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    # Hide unused panels
    for j in range(n_cond, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared axis labels
    fig.text(0.5,  0.00, r"Original $p(\mathrm{supported})$",
             ha="center", fontsize=11)
    fig.text(0.01, 0.5,  r"Rewritten $p(\mathrm{supported})$",
             va="center", rotation="vertical", fontsize=11)

    # Legend (first panel)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SUP_COLOR,
               markersize=6, label="Supported"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=REF_COLOR,
               markersize=6, label="Refuted"),
    ]
    axes_flat[0].legend(handles=legend_handles, fontsize=9,
                        loc="upper left", frameon=False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout(rect=[0.04, 0.03, 1.0, 1.0])
    return fig
