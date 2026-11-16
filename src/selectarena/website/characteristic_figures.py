"""Interactive figures of :mod:`selectarena.evaluation.characteristics` for the leaderboard Space.

Self-contained HTML pages with an inline SVG, like the other figures (:mod:`selectarena.website.figures`):
a tooltip per cell or point, text and axes that follow the page's light or dark theme, and SVG / PNG /
CSV downloads in the light paper look.
"""

from __future__ import annotations

import html
import math

import pandas as pd

from selectarena.website.figures import _EXTRA_CSS, ORANGE, _figure_page, _tip
from selectarena.website.html import _FIGURE_CSS

BLUE = "#0072B2"
_FONT = 12
# Theme colours in SVG presentation attributes do not resolve var(); these classes do.
_CSS = ".sa-hollow { fill:var(--bg); stroke:var(--fg); } .sa-ci-none { stroke:var(--muted); }"


def _svg_open(width: float, height: float, label: str) -> str:
    # At most the natural size, centred: a narrow figure (e.g. a heat map of three groups) is not
    # blown up to the width of its frame.
    return (
        f"<svg id='sa-figure' viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='{html.escape(label, True)}' "
        f"style='max-width:{width * 1.15:.0f}px;margin:0 auto'>"
        f"<style>{_FIGURE_CSS}{_EXTRA_CSS}{_CSS}</style>"
    )


def method_order(table: pd.DataFrame) -> list[str]:
    """Methods by their overall improvement over Random (group ``All`` of
    :func:`~selectarena.evaluation.characteristics.improvement_by_grouping`), best first."""
    overall = table[table["group"] == "All"].sort_values("improvement", ascending=False)
    return overall["method"].tolist()


# --------------------------------------------------------------------------------------------------
# Improvement over Random per group (heat map)


def _blues(t: float) -> str:
    """Colour of the Blues colormap at ``t`` in [0, 1] (0 light, 1 dark)."""
    import matplotlib

    r, g, b, _ = matplotlib.colormaps["Blues"](0.12 + 0.83 * min(max(t, 0.0), 1.0))
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def improvement_heatmap_svg(table: pd.DataFrame, grouping_label: str) -> str:
    """Methods (rows, by overall improvement) by groups (columns, ``All`` first): the improvement over
    Random (%), darker is better; the best method of every group is outlined."""
    methods = method_order(table)
    groups = list(dict.fromkeys(table["group"]))
    if not methods or len(groups) < 2:
        return "<p>Not enough results for this figure.</p>"
    values = table.set_index(["group", "method"])["improvement"]
    counts = table.groupby("group", sort=False)["n_datasets"].first()
    # The colour scale starts at Random (0): worse methods get the lightest colour, so a strongly
    # negative method does not squeeze the colours of all the others together.
    vmin, vmax = 0.0, max(float(values.max()), 1.0)
    span = (vmax - vmin) or 1.0

    label_w, col_w, gap, row_h = 128, 92, 14, 25
    top, legend_h = 58, 64
    width = label_w + len(groups) * col_w + gap + 16
    height = top + len(methods) * row_h + legend_h

    def col_x(i: int) -> float:
        return label_w + i * col_w + (gap if i > 0 else 0)

    parts = [_svg_open(width, height, f"Improvement over Random of each method per {grouping_label.lower()}")]
    for i, group in enumerate(groups):
        cx = col_x(i) + col_w / 2
        weight = "700" if i == 0 else "600"
        parts.append(
            f"<text class='sa-text' x='{cx:.1f}' y='{top - 26}' text-anchor='middle' font-size='{_FONT}' "
            f"font-weight='{weight}'>{html.escape(str(group))}</text>"
        )
        parts.append(
            f"<text class='sa-muted' x='{cx:.1f}' y='{top - 10}' text-anchor='middle' font-size='{_FONT - 1}'>"
            f"{int(counts[group])} datasets</text>"
        )
    for j, method in enumerate(methods):
        y = top + j * row_h
        parts.append(
            f"<text class='sa-text' x='{label_w - 120:.1f}' y='{y + row_h / 2 + 4:.1f}' font-size='{_FONT}'>"
            f"{html.escape(method)}</text>"
        )
    for i, group in enumerate(groups):
        column = values.loc[group] if group in values.index.get_level_values(0) else pd.Series(dtype=float)
        best = column.max() if len(column) else None
        ranked = column.rank(method="min", ascending=False)
        for j, method in enumerate(methods):
            x, y = col_x(i), top + j * row_h
            if method not in column.index or pd.isna(column[method]):
                parts.append(
                    f"<text class='sa-muted' x='{x + col_w / 2:.1f}' y='{y + row_h / 2 + 4:.1f}' "
                    f"text-anchor='middle' font-size='{_FONT}'>–</text>"
                )
                continue
            value = float(column[method])
            t = max(value - vmin, 0.0) / span
            fill = _blues(t)
            ink = "#ffffff" if t > 0.55 else "#1f2328"
            is_best = best is not None and value == best
            parts.append(
                f"<rect x='{x + 1:.1f}' y='{y + 1:.1f}' width='{col_w - 2}' height='{row_h - 2}' rx='3' fill='{fill}'/>"
            )
            if is_best:
                parts.append(
                    f"<rect class='sa-ink' x='{x + 1.5:.1f}' y='{y + 1.5:.1f}' width='{col_w - 3}' height='{row_h - 3}' "
                    "rx='3' fill='none' stroke-width='2'/>"
                )
            weight = " font-weight='700'" if is_best else ""
            parts.append(
                f"<text x='{x + col_w / 2:.1f}' y='{y + row_h / 2 + 4:.1f}' text-anchor='middle' font-size='{_FONT}' "
                f"fill='{ink}'{weight}>{value:.1f}</text>"
            )
            place = int(ranked[method])
            lines = [
                f"{method} · {group}",
                f"Improvement over Random {value:+.1f}%: place {place} of {len(column)}"
                + (" (best)" if is_best else ""),
                f"{int(counts[group])} datasets",
            ]
            parts.append(
                f"<rect class='sa-hit' x='{x:.1f}' y='{y:.1f}' width='{col_w}' height='{row_h}' data-tip='{_tip(lines)}'/>"
            )
    # Colour scale.
    ly = top + len(methods) * row_h + 38
    bar_x, bar_w = label_w, min(300, width - label_w - 20)
    steps = 24
    for k in range(steps):
        t = 1 - k / (steps - 1)
        parts.append(
            f"<rect x='{bar_x + k * bar_w / steps:.1f}' y='{ly - 10}' width='{bar_w / steps + 0.5:.1f}' height='10' "
            f"fill='{_blues(t)}'/>"
        )
    parts.append(
        f"<text class='sa-muted' x='{bar_x:.1f}' y='{ly + 13}' font-size='{_FONT - 1}'>{vmax:.1f} (better)</text>"
    )
    parts.append(
        f"<text class='sa-muted' x='{bar_x + bar_w:.1f}' y='{ly + 13}' text-anchor='end' font-size='{_FONT - 1}'>"
        f"≤ 0 (Random or worse)</text>"
    )
    parts.append(
        f"<text class='sa-text' x='{bar_x:.1f}' y='{ly - 16}' font-size='{_FONT - 1}'>Improvement over Random (%)</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def improvement_heatmap_html(table: pd.DataFrame, grouping_label: str, *, title: str, name: str) -> str:
    csv = table.assign(improvement=table["improvement"].round(3))
    return _figure_page(
        improvement_heatmap_svg(table, grouping_label),
        csv,
        name=name,
        hint="Hover or tap a cell for its values.",
        title=title,
    )


# --------------------------------------------------------------------------------------------------
# Rank correlation with the size of the datasets


def rank_sensitivity_svg(
    sens: pd.DataFrame,
    methods: list[str],
    *,
    better: str = "ranks better",
    worse: str = "ranks worse",
    direction: str = "better",
    label: str = "Rank correlation of each method with the size of the datasets",
) -> str:
    """One panel per size characteristic: each method's rank correlation (dot) and 95% bootstrap
    interval (line); positive means the method ranks better (``better``: e.g. is more stable) on
    larger values."""
    panels = list(dict.fromkeys(sens["characteristic"]))
    methods = [m for m in methods if m in set(sens["method"])] or sorted(sens["method"].unique())
    if not panels or not methods:
        return "<p>Not enough results for this figure.</p>"
    label_w, panel_w, gap, row_h = 128, 210, 22, 25
    top, bottom_h = 46, 80
    width = label_w + len(panels) * (panel_w + gap)
    height = top + len(methods) * row_h + bottom_h
    plot_h = len(methods) * row_h
    data = sens.set_index(["characteristic", "method"])

    parts = [_svg_open(width, height, label)]
    for j, method in enumerate(methods):
        y = top + j * row_h + row_h / 2
        parts.append(
            f"<text class='sa-text' x='{label_w - 120:.1f}' y='{y + 4:.1f}' font-size='{_FONT}'>{html.escape(method)}</text>"
        )
    for p, panel in enumerate(panels):
        x0 = label_w + p * (panel_w + gap) + 8
        x1 = x0 + panel_w - 16

        def X(v: float, x0: float = x0, x1: float = x1) -> float:
            return x0 + (v + 1) / 2 * (x1 - x0)

        parts.append(
            f"<text class='sa-text' x='{(x0 + x1) / 2:.1f}' y='{top - 18}' text-anchor='middle' font-size='{_FONT}' "
            f"font-weight='600'>{html.escape(panel)}</text>"
        )
        for t in (-1, -0.5, 0, 0.5, 1):
            cls = "sa-ink" if t == 0 else "sa-grid"
            opacity = " stroke-opacity='0.5'" if t == 0 else ""
            parts.append(
                f"<line class='{cls}' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 4}' y2='{top + plot_h:.1f}'{opacity}/>"
            )
            parts.append(
                f"<text class='sa-muted' x='{X(t):.1f}' y='{top + plot_h + 16:.1f}' text-anchor='middle' "
                f"font-size='{_FONT - 1}'>{t:g}</text>"
            )
        noun = panel.lower()
        parts.append(
            f"<text class='sa-muted' x='{(x0 + x1) / 2:.1f}' y='{top + plot_h + 33:.1f}' text-anchor='middle' "
            f"font-size='{_FONT - 1}'>← {direction} with fewer · more →</text>"
        )
        for j, method in enumerate(methods):
            y = top + j * row_h + row_h / 2
            if (panel, method) not in data.index:
                continue
            row = data.loc[(panel, method)]
            rho, lo, hi = float(row["rho"]), float(row["ci_low"]), float(row["ci_high"])
            if not math.isfinite(rho):
                continue
            if lo > 0:
                color, verdict = BLUE, f"{better} with more {noun}"
            elif hi < 0:
                color, verdict = ORANGE, f"{worse} with more {noun}"
            else:
                color, verdict = None, "no clear trend (the interval includes 0)"
            if math.isfinite(lo) and math.isfinite(hi):
                stroke = f" stroke='{color}'" if color else " class='sa-ci-none'"
                parts.append(
                    f"<line x1='{X(lo):.1f}' x2='{X(hi):.1f}' y1='{y:.1f}' y2='{y:.1f}'{stroke} "
                    "stroke-width='2' stroke-linecap='round'/>"
                )
            if color:
                parts.append(f"<circle cx='{X(rho):.1f}' cy='{y:.1f}' r='4.5' fill='{color}'/>")
            else:
                parts.append(f"<circle class='sa-hollow' cx='{X(rho):.1f}' cy='{y:.1f}' r='4' stroke-width='1.5'/>")
            lines = [
                f"{method} · {panel}",
                f"Rank correlation {rho:+.2f} (95% CI {lo:+.2f} to {hi:+.2f})",
                verdict[0].upper() + verdict[1:],
                f"{int(row['n_datasets'])} datasets",
            ]
            parts.append(
                f"<rect class='sa-hit' x='{x0 - 6:.1f}' y='{y - row_h / 2:.1f}' width='{x1 - x0 + 12:.1f}' "
                f"height='{row_h}' data-tip='{_tip(lines)}'/>"
            )
    ly = top + plot_h + 62
    x = label_w
    for color, text in ((BLUE, "Positive correlation"), (ORANGE, "Negative correlation"), (None, "No clear trend")):
        if color:
            parts.append(f"<circle cx='{x + 5:.1f}' cy='{ly - 4:.1f}' r='4.5' fill='{color}'/>")
        else:
            parts.append(f"<circle class='sa-hollow' cx='{x + 5:.1f}' cy='{ly - 4:.1f}' r='4' stroke-width='1.5'/>")
        parts.append(f"<text class='sa-text' x='{x + 14:.1f}' y='{ly}' font-size='{_FONT}'>{text}</text>")
        x += 30 + 7.2 * len(text)
    parts.append("</svg>")
    return "".join(parts)


def rank_sensitivity_html(
    sens: pd.DataFrame,
    methods: list[str],
    *,
    title: str,
    name: str = "selectarena_rank_by_size",
    better: str = "ranks better",
    worse: str = "ranks worse",
    direction: str = "better",
    label: str = "Rank correlation of each method with the size of the datasets",
) -> str:
    csv = sens.assign(**{c: sens[c].round(3) for c in ("rho", "ci_low", "ci_high")})
    return _figure_page(
        rank_sensitivity_svg(sens, methods, better=better, worse=worse, direction=direction, label=label),
        csv,
        name=name,
        hint="Hover or tap a method for its values.",
        title=title,
    )


# --------------------------------------------------------------------------------------------------
# Selection time against the number of features


# --------------------------------------------------------------------------------------------------
# Improvement over Random per downstream model

#: Colour and marker per downstream model (Okabe-Ito colours, which stay apart for colour-blind readers).
MODEL_STYLES = [
    ("#0072B2", "circle"),
    ("#E69F00", "square"),
    ("#009E73", "diamond"),
    ("#CC79A7", "triangle"),
    ("#56B4E9", "down"),
    ("#D55E00", "circle"),
    ("#999999", "square"),
    ("#882255", "diamond"),
    ("#44AA99", "triangle"),
    ("#117733", "down"),
    ("#DDAA33", "circle"),
]


def _marker(shape: str, x: float, y: float, color: str, r: float = 5.0) -> str:
    if shape == "square":
        return f"<rect x='{x - r * 0.85:.1f}' y='{y - r * 0.85:.1f}' width='{r * 1.7:.1f}' height='{r * 1.7:.1f}' fill='{color}'/>"
    if shape == "diamond":
        pts = f"{x:.1f},{y - r * 1.2:.1f} {x + r * 1.2:.1f},{y:.1f} {x:.1f},{y + r * 1.2:.1f} {x - r * 1.2:.1f},{y:.1f}"
        return f"<polygon points='{pts}' fill='{color}'/>"
    if shape in ("triangle", "down"):
        s = -1 if shape == "triangle" else 1
        pts = f"{x:.1f},{y + s * r * 1.15:.1f} {x + r * 1.1:.1f},{y - s * r * 0.75:.1f} {x - r * 1.1:.1f},{y - s * r * 0.75:.1f}"
        return f"<polygon points='{pts}' fill='{color}'/>"
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='{color}'/>"


def _nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    step = 10 ** math.floor(math.log10(max(hi - lo, 1e-9) / n))
    for factor in (1, 2, 5, 10):
        if (hi - lo) / (step * factor) <= n:
            step *= factor
            break
    first = math.floor(lo / step) * step
    return [first + i * step for i in range(math.ceil((hi - first) / step) + 1)]


def _fmt(value: float, unit: str) -> str:
    return f"{value:+.1f}%" if unit == "%" else f"{value:.3f}"


def group_improvement_svg(
    table: pd.DataFrame,
    methods: list[str],
    agreement: float | None = None,
    *,
    all_label: str = "All models",
    agreement_label: str = "the downstream models",
    value: str = "improvement",
    axis_label: str = "Improvement over Random (%)",
    unit: str = "%",
) -> str:
    """Improvement over Random per method (rows) with one marker per group (downstream model or task
    type, ``group`` of :func:`~selectarena.evaluation.characteristics.improvement_by`) and a bar for
    all groups together: how much each group gains from each selector, and whether the groups agree."""
    models = [m for m in dict.fromkeys(table["group"]) if m != "All"]
    methods = [m for m in methods if m in set(table["method"])] or sorted(table["method"].unique())
    if len(models) < 1 or not methods:
        return "<p>Not enough results for this figure.</p>"
    values = table.set_index(["group", "method"])[value]
    lo, hi = float(values.min()), float(values.max())
    ticks = _nice_ticks(min(lo, 0), max(hi, 0))
    label_w, plot_w, row_h, top = 128, 560, 25, 22
    width = label_w + plot_w + 30
    # Legend items (the bar for all groups, then a marker per group), wrapped onto several lines.
    items, x, y_line = [], float(label_w), 0
    for name in [all_label, *models]:
        item_w = 30 + 6.8 * len(name)
        if x + item_w > width - 10 and x > label_w:
            x, y_line = float(label_w), y_line + 1
        items.append((name, x, y_line))
        x += item_w
    legend_h = 70 + 20 * y_line + (22 if agreement is not None else 0)
    height = top + len(methods) * row_h + legend_h
    plot_h = len(methods) * row_h
    x0, x1 = label_w + 10, label_w + plot_w

    def X(v: float) -> float:
        return x0 + (v - ticks[0]) / (ticks[-1] - ticks[0]) * (x1 - x0)

    parts = [_svg_open(width, height, f"{axis_label} of each method, per group of {agreement_label}")]
    for t in ticks:
        cls = "sa-ink" if t == 0 else "sa-grid"
        opacity = " stroke-opacity='0.5'" if t == 0 else ""
        parts.append(f"<line class='{cls}' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top}' y2='{top + plot_h}'{opacity}/>")
        parts.append(
            f"<text class='sa-muted' x='{X(t):.1f}' y='{top + plot_h + 16}' text-anchor='middle' "
            f"font-size='{_FONT - 1}'>{t:g}</text>"
        )
    parts.append(
        f"<text class='sa-text' x='{(x0 + x1) / 2:.1f}' y='{top + plot_h + 34}' text-anchor='middle' "
        f"font-size='{_FONT}'>{axis_label}</text>"
    )
    styles = {m: MODEL_STYLES[i % len(MODEL_STYLES)] for i, m in enumerate(models)}
    for j, method in enumerate(methods):
        y = top + j * row_h + row_h / 2
        parts.append(
            f"<text class='sa-text' x='{label_w - 120:.1f}' y='{y + 4:.1f}' font-size='{_FONT}'>{html.escape(method)}</text>"
        )
        row = {m: values.get((m, method)) for m in models}
        row = {m: float(v) for m, v in row.items() if v is not None and pd.notna(v)}
        if row:
            parts.append(
                f"<line class='sa-ci-none' x1='{X(min(row.values())):.1f}' x2='{X(max(row.values())):.1f}' "
                f"y1='{y:.1f}' y2='{y:.1f}' stroke-width='1.5'/>"
            )
        overall = values.get(("All", method))
        if overall is not None and pd.notna(overall):
            parts.append(
                f"<line class='sa-ink' x1='{X(float(overall)):.1f}' x2='{X(float(overall)):.1f}' "
                f"y1='{y - 8:.1f}' y2='{y + 8:.1f}' stroke-width='2.5'/>"
            )
        for model, value in row.items():
            color, shape = styles[model]
            parts.append(_marker(shape, X(value), y, color, 4.5))
        lines = [method]
        if overall is not None and pd.notna(overall):
            lines.append(f"{all_label}: {_fmt(float(overall), unit)}")
        lines += [f"{m}: {_fmt(v, unit)}" for m, v in sorted(row.items(), key=lambda kv: -kv[1])]
        parts.append(
            f"<rect class='sa-hit' x='{x0 - 6:.1f}' y='{y - row_h / 2:.1f}' width='{x1 - x0 + 12:.1f}' "
            f"height='{row_h}' data-tip='{_tip(lines)}'/>"
        )
    ly = top + plot_h + 60
    for name, x, line in items:
        y = ly + 20 * line
        if name == all_label:
            parts.append(
                f"<line class='sa-ink' x1='{x + 4:.1f}' x2='{x + 4:.1f}' y1='{y - 12}' y2='{y + 2}' stroke-width='2.5'/>"
            )
            parts.append(
                f"<text class='sa-text' x='{x + 12:.1f}' y='{y}' font-size='{_FONT}'>{html.escape(name)}</text>"
            )
            continue
        color, shape = styles[name]
        parts.append(_marker(shape, x + 5, y - 4, color, 4.5))
        parts.append(f"<text class='sa-text' x='{x + 14:.1f}' y='{y}' font-size='{_FONT}'>{html.escape(name)}</text>")
    if agreement is not None and math.isfinite(agreement):
        parts.append(
            f"<text class='sa-muted' x='{label_w}' y='{ly + 20 * items[-1][2] + 24}' font-size='{_FONT - 1}'>"
            f"Agreement of the method rankings between {agreement_label}: mean Spearman ρ = {agreement:.2f}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def group_improvement_html(
    table: pd.DataFrame,
    methods: list[str],
    agreement: float | None,
    *,
    title: str,
    name: str,
    all_label: str = "All models",
    agreement_label: str = "the downstream models",
    value: str = "improvement",
    axis_label: str = "Improvement over Random (%)",
    unit: str = "%",
) -> str:
    csv = table.assign(**{value: table[value].round(4)})
    return _figure_page(
        group_improvement_svg(
            table,
            methods,
            agreement,
            all_label=all_label,
            agreement_label=agreement_label,
            value=value,
            axis_label=axis_label,
            unit=unit,
        ),
        csv,
        name=name,
        hint="Hover or tap a method for its values.",
        title=title,
    )


# --------------------------------------------------------------------------------------------------
# Elo with and without the datasets where a method ran out of time (the paper's Figure D.3 design)

#: Bars of Figure D.3: the default Elo in blue, the ablated one in light orange (orange at 30%).
ELO_PAIR_COLORS = (BLUE, "#F2CFB2")


def elo_pair_svg(default: pd.DataFrame, ablated: pd.DataFrame, *, labels: tuple[str, str]) -> str:
    """Two Elo bars per method, as in the paper's Figure D.3: the ablated Elo (light orange, left)
    next to the default one (blue, right), with 95% intervals and each bar's rank written on it;
    methods ordered by the default Elo, worst first. ``default`` and ``ablated`` are
    :func:`~selectarena.evaluation.elo.compute_elo` tables; ``labels`` name them (default first)."""
    methods = [m for m in default.sort_values("elo")["method"] if m in set(ablated["method"])]
    if len(methods) < 2:
        return "<p>Not enough results for this figure.</p>"
    tables = {"ablated": ablated.set_index("method"), "default": default.set_index("method")}
    ranks = {k: t.loc[methods, "elo"].rank(ascending=False, method="min") for k, t in tables.items()}
    low = min(float(t.loc[methods, "elo_ci_low"].min()) for t in tables.values())
    high = max(float(t.loc[methods, "elo_ci_high"].max()) for t in tables.values())
    y_min = math.floor(low / 50) * 50 - 25
    ticks = [t for t in _nice_ticks(y_min, high, 8) if t >= y_min]
    y_max = ticks[-1] if ticks[-1] > high else ticks[-1] + (ticks[1] - ticks[0])

    slot, bar = 66, 28
    left, right, top, bottom = 62, 16, 40, 92
    plot_w, plot_h = slot * len(methods), 300
    width, height = left + plot_w + right, top + plot_h + bottom

    def Y(v: float) -> float:
        return top + plot_h * (1 - (v - y_min) / (y_max - y_min))

    parts = [_svg_open(width, height, f"Elo of each method: {labels[0]} and {labels[1]}")]
    for t in ticks:
        if y_min <= t <= y_max:
            parts.append(f"<line class='sa-grid' x1='{left}' x2='{left + plot_w}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>")
            parts.append(
                f"<text class='sa-muted' x='{left - 6}' y='{Y(t) + 4:.1f}' text-anchor='end' font-size='{_FONT - 1}'>{t:g}</text>"
            )
    parts.append(
        f"<text class='sa-text' transform='translate(16 {top + plot_h / 2:.0f}) rotate(-90)' text-anchor='middle' "
        f"font-size='{_FONT}'>Elo</text>"
    )
    base = Y(y_min)
    for i, method in enumerate(methods):
        cx = left + slot * i + slot / 2
        lines = [method]
        for key, dx, color, ink, label in (
            ("ablated", -bar / 2 - 1, ELO_PAIR_COLORS[1], "#1f2328", labels[1]),
            ("default", bar / 2 + 1, ELO_PAIR_COLORS[0], "#ffffff", labels[0]),
        ):
            row = tables[key].loc[method]
            elo, lo, hi = float(row["elo"]), float(row["elo_ci_low"]), float(row["elo_ci_high"])
            x = cx + dx - bar / 2
            parts.append(
                f"<rect x='{x:.1f}' y='{Y(elo):.1f}' width='{bar}' height='{max(base - Y(elo), 0):.1f}' fill='{color}'/>"
            )
            xm = cx + dx
            parts.append(
                f"<line class='sa-ink' x1='{xm:.1f}' x2='{xm:.1f}' y1='{Y(hi):.1f}' y2='{Y(lo):.1f}' stroke-width='1.2'/>"
            )
            for v in (lo, hi):
                parts.append(
                    f"<line class='sa-ink' x1='{xm - 4:.1f}' x2='{xm + 4:.1f}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke-width='1.2'/>"
                )
            rank = int(ranks[key][method])
            parts.append(
                f"<text x='{xm:.1f}' y='{base - 8:.1f}' text-anchor='middle' font-size='{_FONT - 1}' "
                f"font-weight='700' fill='{ink}'>{rank}</text>"
            )
            lines.append(f"{label}: Elo {elo:.0f} ({lo:.0f} to {hi:.0f}), rank {rank}")
        parts.append(
            f"<text class='sa-text' text-anchor='end' font-size='{_FONT}' "
            f"transform='translate({cx + 6:.1f} {base + 14:.1f}) rotate(-35)'>{html.escape(method)}</text>"
        )
        parts.append(
            f"<rect class='sa-hit' x='{cx - slot / 2:.1f}' y='{top}' width='{slot}' height='{plot_h}' data-tip='{_tip(lines)}'/>"
        )
    parts.append(
        f"<line class='sa-ink' x1='{left}' x2='{left + plot_w}' y1='{base:.1f}' y2='{base:.1f}' stroke-opacity='0.5'/>"
    )
    # Legend on top, default first (as in the paper).
    x = left + plot_w / 2 - 6.5 * (len(labels[0]) + len(labels[1])) / 2 - 30
    for color, label in ((ELO_PAIR_COLORS[0], labels[0]), (ELO_PAIR_COLORS[1], labels[1])):
        parts.append(f"<rect x='{x:.1f}' y='{top - 30}' width='14' height='14' fill='{color}'/>")
        parts.append(
            f"<text class='sa-text' x='{x + 20:.1f}' y='{top - 18}' font-size='{_FONT}'>{html.escape(label)}</text>"
        )
        x += 20 + 6.8 * len(label) + 24
    parts.append("</svg>")
    return "".join(parts)


def elo_pair_html(
    default: pd.DataFrame, ablated: pd.DataFrame, *, labels: tuple[str, str], title: str, name: str
) -> str:
    csv = pd.concat([default.assign(results=labels[0]), ablated.assign(results=labels[1])], ignore_index=True)
    return _figure_page(
        elo_pair_svg(default, ablated, labels=labels),
        csv[["results", "method", "elo", "elo_ci_low", "elo_ci_high"]].round(1),
        name=name,
        hint="Hover or tap a method for its values.",
        title=title,
    )
