"""
LaTeX table generation for performance and shift results.

Returns strings — no file I/O.
"""

from __future__ import annotations

import numpy as np

from core.config import ALL_CONDITIONS, CONDITION_DISPLAY, NATURAL_GROUPS, CONTROLLED_CONDITIONS, MODEL_DISPLAY_NAMES

# Table column headers — use the same compact labels as the compass/legend.
_TABLE_DISPLAY: dict[str, str] = dict(CONDITION_DISPLAY)


def _natural_and_controlled_order() -> tuple[list[str], list[str]]:
    """Partition ALL_CONDITIONS into natural groups vs controlled (stable vs config)."""
    controlled_set = set(CONTROLLED_CONDITIONS)
    natural = [c for c in ALL_CONDITIONS if c in NATURAL_GROUPS]
    controlled = [c for c in ALL_CONDITIONS if c in controlled_set]
    return natural, controlled


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(val: float | None, fmt: str = ".3f") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:{fmt}}"


def _signed_cell_highlight(delta: float, inner: str, max_val: float = 0.10) -> str:
    """
    Gradient cell fill based on sign and magnitude of a delta value.

    Intensity scales linearly from 0 (Δ=0) to 40 (|Δ| >= max_val).
    Default max_val=0.10 suits Δp shifts; pass a different value for other metrics.
    """
    eps = 1e-9
    if abs(delta) <= eps:
        return inner
    intensity = min(int(abs(delta) / max_val * 40), 40)
    intensity = max(intensity, 5)  # floor so even tiny effects get a hint of colour
    color = "green" if delta > eps else "red"
    return rf"\cellcolor{{{color}!{intensity}}}" + inner


def _stars(q: float | None) -> str:
    """Format a q-value as a LaTeX superscript star marker."""
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    if q < 0.001:
        return r"\textsuperscript{***}"
    if q < 0.01:
        return r"\textsuperscript{**}"
    if q < 0.05:
        return r"\textsuperscript{*}"
    return ""


def _vs_base_cell_highlight(
    val: float, base: float | None, inner: str, max_val: float = 0.05
) -> str:
    """
    Gradient cell fill vs base value.

    Intensity scales linearly from 0 (no change) to 40 (|Δ| >= max_val).
    Default max_val=0.05 suits F1 / P-Acc differences.
    """
    if base is None or (isinstance(base, float) and np.isnan(base)):
        return inner
    if np.isnan(val):
        return inner
    delta = val - base
    eps = 1e-9
    if abs(delta) <= eps:
        return inner
    intensity = min(int(abs(delta) / max_val * 40), 40)
    intensity = max(intensity, 5)
    color = "green" if delta > eps else "red"
    return rf"\cellcolor{{{color}!{intensity}}}" + inner


def _format_condition_performance_cells(
    raw_vals: list[float | None],
    base_raw: float | None,
    fmt: str = ".3f",
    qvals: list[float | None] | None = None,
) -> list[str]:
    """
    Per-condition cells: vs-base background tint only.
    Requires \\usepackage[table]{xcolor} (provides \\cellcolor).

    If ``qvals`` is given (one BH-FDR q-value per condition), star markers are
    appended inside the colored cell.
    """
    cells: list[str] = []
    for i, v in enumerate(raw_vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            cells.append("—")
            continue
        inner = _fmt(v, fmt)
        if qvals is not None:
            inner = inner + _stars(qvals[i])
        inner = _vs_base_cell_highlight(v, base_raw, inner)
        cells.append(inner)

    return cells


def _escape(s: str) -> str:
    return MODEL_DISPLAY_NAMES.get(s, s).replace("_", r"\_")


# ---------------------------------------------------------------------------
# Performance table
# ---------------------------------------------------------------------------


def generate_performance_table(
    all_results: dict[str, dict],
    model_names: list[str],
    stats: dict[str, dict[str, dict[str, float]]] | None = None,
) -> str:
    """
    all_results: {model: {
        "base": {"f1": float, "pair_acc": float},
        "LP":   {"f1": float, "pair_acc": float},
        ...   (one key per ALL_CONDITIONS entry)
    }}

    stats (optional): {model: {cond: {"pair_acc": q_value, "f1": q_value}}}
        BH-FDR q-values per cell. When provided, star markers are appended
        to each condition cell.

    Condition cells worse than Base get a light red fill; better than Base get light green.
    Use \\usepackage[table]{xcolor} for \\cellcolor.
    """
    natural, controlled = _natural_and_controlled_order()
    n_nat, n_ctrl = len(natural), len(controlled)
    cond_headers = " & ".join(_TABLE_DISPLAY[c] for c in ALL_CONDITIONS)
    col_spec_perf = "l l c " + ("c" * (n_nat + n_ctrl))
    nat_end = 3 + n_nat
    ctrl_start = 4 + n_nat
    ctrl_end = 3 + n_nat + n_ctrl
    lines = [
        "\\begin{tabular}{" + col_spec_perf + "}",
        r"\toprule",
        rf" & & & \multicolumn{{{n_nat}}}{{c}}{{\textbf{{Natural Rewriting}}}} & \multicolumn{{{n_ctrl}}}{{c}}{{\textbf{{Controlled Rewriting}}}} \\",
        rf"\cmidrule(lr){{4-{nat_end}}} \cmidrule(lr){{{ctrl_start}-{ctrl_end}}}",
        rf"\textbf{{Model}} & & \textbf{{Base}} & {cond_headers} \\",
        r"\midrule",
    ]
    metric_rows = [("pair_acc", "P-Acc"), ("f1", "F1")]
    for m_idx, model in enumerate(model_names):
        res = all_results[model]
        n_rows = len(metric_rows)
        for j, (mk, ml) in enumerate(metric_rows):
            prefix = (
                r"\multirow{" + str(n_rows) + r"}{*}{" + _escape(model) + "}"
                if j == 0 else ""
            )
            base_raw = res["base"].get(mk)
            base_val = _fmt(base_raw)
            raw_vals = [res.get(cond, {}).get(mk) for cond in ALL_CONDITIONS]
            if stats is not None:
                model_stats = stats.get(model, {})
                qvals = [model_stats.get(cond, {}).get(mk) for cond in ALL_CONDITIONS]
            else:
                qvals = None
            cond_vals = _format_condition_performance_cells(raw_vals, base_raw, qvals=qvals)
            row = f" {prefix} & {ml} & {base_val}"
            for v in cond_vals:
                row += f" & {v}"
            row += r" \\"
            lines.append(row)
        if m_idx < len(model_names) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shift table
# ---------------------------------------------------------------------------


def generate_shift_table(
    all_shift_results: dict[str, dict],
    model_names: list[str],
    stats: dict[str, dict[str, dict[str, float]]] | None = None,
) -> str:
    """
    all_shift_results: {model: {
        "base": {"p_sup": float, "p_ref": float, ...},
        "LP":   {"p_sup": float, "p_ref": float, "dp_sup": float, "dp_ref": float},
        ...
    }}

    stats (optional): {model: {cond: {"dp_sup": q_value, "dp_ref": q_value}}}
        BH-FDR q-values per cell from Wilcoxon signed-rank on per-pair Δ.
        When provided, star markers are appended to each delta cell.
    """
    natural, controlled = _natural_and_controlled_order()
    n_nat, n_ctrl = len(natural), len(controlled)
    cond_headers = " & ".join(_TABLE_DISPLAY[c] for c in ALL_CONDITIONS)
    col_spec_shift = "l l c l " + ("c" * (n_nat + n_ctrl))
    nat_end = 4 + n_nat
    ctrl_start = 5 + n_nat
    ctrl_end = 4 + n_nat + n_ctrl
    lines = [
        "\\begin{tabular}{" + col_spec_shift + "}",
        r"\toprule",
        rf" & & & & \multicolumn{{{n_nat}}}{{c}}{{\textbf{{Natural Rewriting}}}} & \multicolumn{{{n_ctrl}}}{{c}}{{\textbf{{Controlled Rewriting}}}} \\",
        rf"\cmidrule(lr){{5-{nat_end}}} \cmidrule(lr){{{ctrl_start}-{ctrl_end}}}",
        rf"\textbf{{Model}} & & \textbf{{Base}} & & {cond_headers} \\",
        r"\midrule",
    ]
    sub_rows = [
        ("p_sup", "dp_sup", r"$p_{\text{sup}}$",  r"$\Delta p_{\text{sup}}$"),
        ("p_ref", "dp_ref", r"$p_{\text{ref}}$",  r"$\Delta p_{\text{ref}}$"),
    ]
    for m_idx, model in enumerate(model_names):
        res = all_shift_results[model]
        n_rows = len(sub_rows)
        for j, (base_key, delta_key, base_label, delta_label) in enumerate(sub_rows):
            prefix = (
                r"\multirow{" + str(n_rows) + r"}{*}{" + _escape(model) + "}"
                if j == 0 else ""
            )
            base_val = _fmt(res["base"].get(base_key))
            cond_vals = []
            model_stats = stats.get(model, {}) if stats is not None else {}
            for cond in ALL_CONDITIONS:
                dv = res.get(cond, {}).get(delta_key)
                if dv is None or np.isnan(dv):
                    cond_vals.append("—")
                else:
                    inner = f"{dv:+.3f}"
                    if stats is not None:
                        q = model_stats.get(cond, {}).get(delta_key)
                        inner = inner + _stars(q)
                    cond_vals.append(_signed_cell_highlight(dv, inner))
            row = f" {prefix} & {base_label} & {base_val} & {delta_label}"
            for v in cond_vals:
                row += f" & {v}"
            row += r" \\"
            lines.append(row)
        if m_idx < len(model_names) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flip-rate table
# ---------------------------------------------------------------------------


def generate_flip_rate_table(
    all_flip_results: dict[str, dict[str, dict]],
    model_names: list[str],
) -> str:
    """
    all_flip_results: {model: {condition: {"flip_sup": float, "flip_ref": float}}}

    Produces a table with:
      - rows: model × {flip_sup (%), flip_ref (%)}
      - columns: one per ALL_CONDITIONS (natural groups + controlled)
      - cells: percentage formatted to one decimal, coloured by magnitude
               (deeper red = more flipping; white at 0 %, full red at >= 10 %)

    No 'Base' column -- the flip rate is inherently a comparison to base.
    """
    natural, controlled = _natural_and_controlled_order()
    n_nat, n_ctrl = len(natural), len(controlled)
    cond_headers = " & ".join(_TABLE_DISPLAY[c] for c in ALL_CONDITIONS)
    col_spec   = "l l " + ("c" * (n_nat + n_ctrl))
    nat_end    = 2 + n_nat
    ctrl_start = 3 + n_nat
    ctrl_end   = 2 + n_nat + n_ctrl

    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        rf" & & \multicolumn{{{n_nat}}}{{c}}{{\textbf{{Natural Rewriting}}}} & \multicolumn{{{n_ctrl}}}{{c}}{{\textbf{{Controlled Rewriting}}}} \\",
        rf"\cmidrule(lr){{3-{nat_end}}} \cmidrule(lr){{{ctrl_start}-{ctrl_end}}}",
        rf"\textbf{{Model}} & & {cond_headers} \\",
        r"\midrule",
    ]

    sub_rows = [
        ("flip_sup", r"$\uparrow$Supported"),
        ("flip_ref", r"$\downarrow$Refuted"),
    ]

    def _flip_cell(val: float | None) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "---"
        pct = val * 100.0
        # Colour intensity scales linearly: 0 % -> white, >= 20 % -> red!40
        intensity = min(int(round(pct / 20.0 * 40)), 40)
        inner = f"{pct:.1f}\\%"
        if intensity > 0:
            return rf"\cellcolor{{red!{intensity}}}{inner}"
        return inner

    for m_idx, model in enumerate(model_names):
        res = all_flip_results.get(model, {})
        n_rows = len(sub_rows)
        for j, (key, label) in enumerate(sub_rows):
            prefix = (
                r"\multirow{" + str(n_rows) + r"}{*}{" + _escape(model) + "}"
                if j == 0 else ""
            )
            cells = [_flip_cell(res.get(cond, {}).get(key)) for cond in ALL_CONDITIONS]
            row = f" {prefix} & {label}"
            for c in cells:
                row += f" & {c}"
            row += r" \\"
            lines.append(row)
        if m_idx < len(model_names) - 1:
            lines.append(r"\addlinespace[2pt]")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
