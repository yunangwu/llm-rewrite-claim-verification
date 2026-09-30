"""
Drift-vs-confidence scatter: Δp(supported) as a function of original p(supported).

Shows how confident (high p) vs uncertain (p≈0.5) claims respond to rewrites.
Returns matplotlib Figure — no file I/O.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.config import ALL_CONDITIONS, CONDITION_DISPLAY
from core.analysis.metrics import compute_binned_drift

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _HAS_LOWESS = True
except ImportError:
    _HAS_LOWESS = False


def plot_drift(
    drift_data: dict[str, dict],
    title: str,
) -> Figure:
    """
    drift_data: {condition: {
        "sup_bins": (centers, means, counts),
        "ref_bins": (centers, means, counts),
        "sup_raw":  (p_orig_list, dp_list),   # optional, for density histogram
        "ref_raw":  (p_orig_list, dp_list),
    }}
    Subplot grid follows ALL_CONDITIONS order (natural + controlled; currently 9 panels).
    """
    n_cond = len(ALL_CONDITIONS)
    ncols = min(3, max(1, n_cond))
    nrows = (n_cond + ncols - 1) // ncols
    fig_w = 4.0 * ncols
    fig_h = 3.2 * nrows + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, cond in enumerate(ALL_CONDITIONS):
        ax = axes_flat[idx]
        disp = CONDITION_DISPLAY.get(cond, cond)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if cond not in drift_data:
            ax.set_title(disp, fontsize=12, pad=4)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        else:
            d = drift_data[cond]
            ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=2)

            sup_centers, sup_means, _ = d["sup_bins"]
            ref_centers, ref_means, _ = d["ref_bins"]

            def _plot_line(centers, means, color, label=None):
                if len(centers) <= 1:
                    return
                if _HAS_LOWESS:
                    smooth = _lowess(means, centers, frac=0.35, return_sorted=False)
                else:
                    smooth = means
                ax.plot(centers, smooth, color=color, linewidth=1.6, alpha=0.95,
                        label=label, zorder=4)

            _plot_line(sup_centers, sup_means, "#4878A8", label="Supported" if idx == 0 else None)
            _plot_line(ref_centers, ref_means, "#C75A5A", label="Refuted"   if idx == 0 else None)

            # density histogram at bottom
            all_p = []
            if "sup_raw" in d and d["sup_raw"][0]:
                all_p.extend(d["sup_raw"][0])
            if "ref_raw" in d and d["ref_raw"][0]:
                all_p.extend(d["ref_raw"][0])
            if all_p:
                counts_h, edges_h = np.histogram(all_p, bins=25, range=(0, 1))
                counts_norm = counts_h / counts_h.max() * 0.10
                bw = edges_h[1] - edges_h[0]
                for k in range(len(counts_h)):
                    ax.bar(edges_h[k], counts_norm[k], width=bw, bottom=-0.22,
                           color="#b0b0b0", alpha=0.7, edgecolor="none", zorder=0, align="edge")

            ax.set_title(disp, fontsize=12, pad=4)
            ax.set_ylim(-0.22, 0.22)
            ax.set_xlim(0, 1)

        row = idx // ncols
        col = idx % ncols
        if row == nrows - 1:
            ax.tick_params(axis="x", length=3, width=0.4, labelsize=10)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_linewidth(0.4)
            ax.spines["bottom"].set_color("#999999")
        else:
            ax.tick_params(axis="x", length=0, labelbottom=False)

        if col == 0:
            ax.tick_params(axis="y", length=2, width=0.4, labelsize=10)
            ax.set_ylabel(r"$\Delta p(\mathrm{supported})$", fontsize=12)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)

        ax.yaxis.grid(True, alpha=0.15, linewidth=0.3)
        ax.xaxis.grid(False)

    for j in range(len(ALL_CONDITIONS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.text(0.5, -0.01, r"Original $p(\mathrm{supported})$", ha="center", fontsize=12)
    axes_flat[0].legend(fontsize=11, loc="best", frameon=False)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def collect_drift_data(
    base_preds: dict,
    rewrite_preds_by_condition: dict[str, list[dict]],
    complete_pairs: dict,
    num_bins: int = 30,
    *,
    has_pairs: bool = True,
) -> dict[str, dict]:
    """
    Build drift_data dict consumed by plot_drift().

    rewrite_preds_by_condition: {condition: list_of_preds_dicts}
        For natural groups, pass a list of per-variant preds averaged together.
        For controlled, pass a single-element list.

    has_pairs: when False, uses per-item label grouping instead of pair-based
        accumulation.  The output format is identical, so plot_drift() is
        unaffected.  This path is the strict generalization — on a fully paired
        dataset both paths produce the same numbers (mean commutes with subtract).
    """
    if not has_pairs:
        return _collect_drift_label(base_preds, rewrite_preds_by_condition, num_bins)

    result: dict[str, dict] = {}
    for cond, preds_list in rewrite_preds_by_condition.items():
        sup_accum: dict = {}  # pair_id -> {"p_orig": float, "dps": [float]}
        ref_accum: dict = {}
        for preds in preds_list:
            for pid, ids in complete_pairs.items():
                sup_id, ref_id = ids["Supported"], ids["Refuted"]
                if not all(cid in base_preds and cid in preds for cid in (sup_id, ref_id)):
                    continue
                p_sup_base = base_preds[sup_id]["p_supported"]
                p_ref_base = base_preds[ref_id]["p_supported"]
                p_sup_new  = preds[sup_id]["p_supported"]
                p_ref_new  = preds[ref_id]["p_supported"]
                if any(v is None for v in (p_sup_base, p_ref_base, p_sup_new, p_ref_new)):
                    continue
                if pid not in sup_accum:
                    sup_accum[pid] = {"p_orig": p_sup_base, "dps": []}
                    ref_accum[pid] = {"p_orig": p_ref_base, "dps": []}
                sup_accum[pid]["dps"].append(p_sup_new - p_sup_base)
                ref_accum[pid]["dps"].append(p_ref_new - p_ref_base)

        sup_p  = [v["p_orig"]              for v in sup_accum.values() if v["dps"]]
        sup_dp = [float(np.mean(v["dps"])) for v in sup_accum.values() if v["dps"]]
        ref_p  = [v["p_orig"]              for v in ref_accum.values() if v["dps"]]
        ref_dp = [float(np.mean(v["dps"])) for v in ref_accum.values() if v["dps"]]

        result[cond] = {
            "sup_bins": compute_binned_drift(sup_p, sup_dp, num_bins),
            "ref_bins": compute_binned_drift(ref_p, ref_dp, num_bins),
            "sup_raw":  (sup_p, sup_dp),
            "ref_raw":  (ref_p, ref_dp),
        }
    return result


def _collect_drift_label(
    base_preds: dict,
    rewrite_preds_by_condition: dict[str, list[dict]],
    num_bins: int,
) -> dict[str, dict]:
    """
    Pairless drift accumulation: group items by label instead of by pair_id.
    Produces the same output structure as the pair-based path.
    """
    result: dict[str, dict] = {}
    for cond, preds_list in rewrite_preds_by_condition.items():
        sup_accum: dict = {}  # cid -> {"p_orig": float, "dps": [float]}
        ref_accum: dict = {}
        for preds in preds_list:
            for cid, base in base_preds.items():
                if cid not in preds:
                    continue
                p_orig = base["p_supported"]
                p_new  = preds[cid]["p_supported"]
                if p_orig is None or p_new is None:
                    continue
                label = base["label"]
                if label == "Supported":
                    accum = sup_accum
                elif label == "Refuted":
                    accum = ref_accum
                else:
                    continue
                if cid not in accum:
                    accum[cid] = {"p_orig": p_orig, "dps": []}
                accum[cid]["dps"].append(p_new - p_orig)

        sup_p  = [v["p_orig"]              for v in sup_accum.values() if v["dps"]]
        sup_dp = [float(np.mean(v["dps"])) for v in sup_accum.values() if v["dps"]]
        ref_p  = [v["p_orig"]              for v in ref_accum.values() if v["dps"]]
        ref_dp = [float(np.mean(v["dps"])) for v in ref_accum.values() if v["dps"]]

        result[cond] = {
            "sup_bins": compute_binned_drift(sup_p, sup_dp, num_bins),
            "ref_bins": compute_binned_drift(ref_p, ref_dp, num_bins),
            "sup_raw":  (sup_p, sup_dp),
            "ref_raw":  (ref_p, ref_dp),
        }
    return result
