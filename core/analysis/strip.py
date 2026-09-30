"""
Strip chart: per-claim Δp(supported) distribution across models.

Each panel = one rewrite condition family (LP, FI, AS, OA, CT, D, C).
Row order matches tables: ``model_slug_sort_key`` (family, then size, then name).

Returns a matplotlib Figure — no file I/O.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.config import model_slug_sort_key


_PANEL_ORDER = ["LP", "FI", "AS", "OA", "CT", "D", "C"]
_PANEL_TITLES = {
    "LP": "Language Polishing",
    "FI": "Fluency Improvement",
    "AS": "Academic Style",
    "OA": "Outcome-Aware",
    "CT": "Cautious Tone",
    "D":  "Decorative (ctrl)",
    "C":  "Conservative (ctrl)",
}
_PANEL_COLORS = {
    "LP": "#4878A8", "FI": "#5BA08C", "AS": "#8E6BAD", "OA": "#C75A5A",
    "CT": "#8c564b",
    "D":  "#D4863A", "C":  "#3A7D44",
}


def plot_strip_chart(
    model_strip_data: list[dict],
    *,
    model_filter: "callable[[str], bool] | None" = None,
    label_fn: "callable[[str], str] | None" = None,
) -> Figure:
    """
    model_strip_data: list of dicts, one per model:
        {
          "slug":    str,
          "deltas":  {
              "LP": {"per_claim": [float], "mean": float},
              "FI": ...,
              "AS": ...,
              "OA": ...,
              "CT": ...,
              "D_1": {"per_claim": [float], "mean": float},
              "D_2": {"per_claim": [float], "mean": float},
              "C_1": {"per_claim": [float], "mean": float},
              "C_2": {"per_claim": [float], "mean": float},
          }
        }
    """
    if not model_strip_data:
        fig, _ = plt.subplots(1, 1)
        return fig

    data = model_strip_data
    if model_filter is not None:
        data = [r for r in data if model_filter(r.get("slug", ""))]

    ordered = sorted(
        data,
        key=lambda r: model_slug_sort_key(r.get("slug", "")),
    )
    y_pos = np.arange(len(ordered))
    if label_fn is not None:
        y_labels = [label_fn(row.get("slug", "?")) for row in ordered]
    else:
        y_labels = [row.get("slug", "?") for row in ordered]

    # Compute shared x-axis limits
    all_means = []
    for row in ordered:
        for key, cdata in row["deltas"].items():
            all_means.append(abs(cdata["mean"]))
    tick_max = max(np.ceil(max(all_means, default=0.02) * 1.18 * 100) / 100, 0.02)
    ticks    = np.linspace(-tick_max, tick_max, 5)

    n_models = len(ordered)
    fig_h = max(7.0, n_models * 0.72 + 2.5)
    fig_w = 4.5
    fig, axes = plt.subplots(
        nrows=len(_PANEL_ORDER), ncols=1,
        figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"hspace": 0.28},
    )
    longest = max(len(lbl) for lbl in y_labels) if y_labels else 8
    left_pad = min(0.52, max(0.30, 0.085 + longest * 0.009))
    fig.subplots_adjust(left=left_pad, right=0.98, top=0.88, bottom=0.10)

    for ax_i, (ax, panel) in enumerate(zip(axes, _PANEL_ORDER)):
        color = _PANEL_COLORS[panel]

        for i, y in enumerate(y_pos):
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#fafafa", zorder=0)
            ax.axhline(y, color="#ececec", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#7f7f7f", linewidth=0.55, zorder=1)

        for y, row in zip(y_pos, ordered):
            deltas = row["deltas"]
            if panel in ("LP", "FI", "AS", "OA", "CT"):
                if panel not in deltas:
                    continue
                cd = deltas[panel]
                pts = cd["per_claim"]
                if not pts:
                    continue  # no real data — skip to avoid phantom dot at x=0
                ax.scatter(cd["mean"], y, s=32, color=color,
                           edgecolors="white", linewidths=0.6, zorder=5)
            else:
                keys = ["D_1", "D_2"] if panel == "D" else ["C_1", "C_2"]
                matching = [(k, deltas[k]) for k in keys if k in deltas and deltas[k]["per_claim"]]
                if not matching:
                    continue
                offsets = np.linspace(-0.15, 0.15, len(matching)) if len(matching) > 1 else [0.0]
                for off, (key, cd) in zip(offsets, matching):
                    pts = cd["per_claim"]
                    is_l2 = key.endswith("2")
                    marker = "*" if is_l2 else "o"
                    ms = 52 if is_l2 else 30
                    ax.scatter(cd["mean"], y + off, s=ms, marker=marker,
                               color=color, edgecolors="white", linewidths=0.6, zorder=6)

        ax.set_ylim(len(ordered) - 0.5, -0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(_PANEL_TITLES[panel], loc="right", fontsize=11, pad=3, color=color)
        ax.grid(axis="x", alpha=0.14, linewidth=0.3)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=9, length=2, width=0.4)
        ax.set_xticks(ticks)
        ax.set_xlim(-tick_max, tick_max)

    axes[-1].set_xlabel(
        r"$\longleftarrow$ conservative   "
        r"$\Delta p(\mathrm{supported})$"
        r"   optimistic $\longrightarrow$",
        fontsize=10,
    )
    return fig
