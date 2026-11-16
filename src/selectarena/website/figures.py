"""Interactive versions of the paper's Pareto and lasagna figures for the leaderboard Space.

Like the Elo figure (:func:`selectarena.website.html.elo_explorer_html`), each page is one
self-contained HTML file with an inline SVG: a tooltip per point or cell, and colours of text, axes
and grid that follow the page's light or dark theme. The SVG / PNG downloads always use the light
colours (the paper look). The layout follows the matplotlib figures of
:mod:`selectarena.evaluation.pareto` and :mod:`selectarena.evaluation.lasagna`; the Pareto front
even reuses their label positions (``adjustText``).
"""

from __future__ import annotations

import base64
import html
import io
import math

import numpy as np
import pandas as pd

from selectarena.evaluation import lasagna
from selectarena.evaluation.pareto import HIDDEN_IN_PLOT, pareto_figure, pareto_front
from selectarena.website.html import _FIGURE_CSS, _FIGURE_JS, _TIP_CSS, _csv_block, _page, figure_toolbar

ORANGE = "#D55E00"
GREEN = "#009E73"
#: SVG units per point of the matplotlib figure: fonts and lines keep the paper's proportions.
_SCALE = 1.5

# Extra classes on top of the Elo figure's (text, grid and ink follow the theme).
_EXTRA_CSS = """
.sa-dim { fill:var(--muted); }
.sa-frame-line { stroke:var(--fg); stroke-opacity:0.3; fill:none; }
.sa-point { stroke:var(--fg); }
.sa-legend { fill:var(--bg); stroke:var(--line); }
.sa-heat { image-rendering:pixelated; image-rendering:crisp-edges; }
.sa-halo { paint-order:stroke; stroke:var(--bg); stroke-width:3px; stroke-opacity:0.7; }
"""


def _tip(lines: list[str]) -> str:
    return html.escape("\n".join(lines), quote=True).replace("\n", "&#10;")


def _svg_open(width: float, height: float, label: str) -> str:
    return (
        f"<svg id='sa-figure' viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='{html.escape(label, True)}'>"
        f"<style>{_FIGURE_CSS}{_EXTRA_CSS}</style>"
    )


def _figure_page(svg: str, csv: pd.DataFrame, *, name: str, hint: str, title: str) -> str:
    return _page(title, figure_toolbar(name, hint) + svg + _csv_block(csv), _TIP_CSS, _FIGURE_JS)


# --------------------------------------------------------------------------------------------------
# Pareto front


def _pareto_layout(summary: pd.DataFrame, hidden: frozenset[str]) -> dict:
    """Axes limits, ticks, plot area and label positions of the paper's matplotlib Pareto figure,
    so the interactive version places everything (labels included) the same way."""
    import matplotlib.pyplot as plt

    fig = pareto_figure(summary, hidden)
    fig.canvas.draw()
    ax = fig.axes[0]
    methods = set(summary["method"])
    labels = {t.get_text(): t.get_position() for t in ax.texts if t.get_text() in methods}
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    yticks = [t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]]
    box = ax.get_position()
    width, height = fig.get_size_inches() * 72
    plt.close(fig)
    return {
        "xlim": xlim,
        "ylim": ylim,
        "xticks": xticks,
        "yticks": yticks,
        "labels": labels,
        "size": (width * _SCALE, height * _SCALE),
        "area": (box.x0, box.y0, box.x1, box.y1),
    }


def _decade_label(value: float) -> str:
    exponent = round(math.log10(value))
    return f"10<tspan dy='-7' font-size='10'>{exponent}</tspan>"


def pareto_svg(summary: pd.DataFrame, hidden: frozenset[str] = HIDDEN_IN_PLOT) -> str:
    """The Pareto front of :func:`~selectarena.evaluation.pareto.improvement_and_runtime`'s summary."""
    shown = summary[~summary["method"].isin(hidden) & (summary["time_per_1k"] > 0)].reset_index(drop=True)
    if len(shown) < 2:
        return "<p>Not enough methods for a Pareto front.</p>"
    layout = _pareto_layout(summary, hidden)
    width, height = layout["size"]
    x0f, y0f, x1f, y1f = layout["area"]
    left, right = x0f * width, x1f * width
    top, bottom = (1 - y1f) * height, (1 - y0f) * height
    (xa, xb), (ya, yb) = layout["xlim"], layout["ylim"]

    def X(v: float) -> float:
        return left + (math.log10(v) - math.log10(xa)) / (math.log10(xb) - math.log10(xa)) * (right - left)

    def Y(v: float) -> float:
        return bottom - (v - ya) / (yb - ya) * (bottom - top)

    s = _SCALE
    parts = [_svg_open(width, height, "Improvement over Random against feature selection time per 1k features")]
    # Grid, ticks and frame.
    for t in layout["xticks"]:
        parts.append(f"<line class='sa-grid' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top:.1f}' y2='{bottom:.1f}'/>")
        parts.append(
            f"<text class='sa-text' x='{X(t):.1f}' y='{bottom + 15 * s:.1f}' text-anchor='middle' "
            f"font-size='{10 * s:.0f}'>{_decade_label(t)}</text>"
        )
    for t in layout["yticks"]:
        parts.append(f"<line class='sa-grid' x1='{left:.1f}' x2='{right:.1f}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>")
        parts.append(
            f"<text class='sa-text' x='{left - 6 * s:.1f}' y='{Y(t) + 4 * s:.1f}' text-anchor='end' "
            f"font-size='{10 * s:.0f}'>{t:g}</text>"
        )
    parts.append(
        f"<rect class='sa-frame-line' x='{left:.1f}' y='{top:.1f}' width='{right - left:.1f}' height='{bottom - top:.1f}'/>"
    )
    parts.append(
        f"<text class='sa-text' x='{(left + right) / 2:.1f}' y='{height - 8 * s:.1f}' text-anchor='middle' "
        f"font-size='{10 * s:.0f}'>Feature Selection Time (s) per 1k Features</text>"
    )
    parts.append(
        f"<text class='sa-text' transform='translate({14 * s:.1f} {(top + bottom) / 2:.1f}) rotate(-90)' "
        f"text-anchor='middle' font-size='{10 * s:.0f}'>Improvement over Random (%)</text>"
    )

    on_front = pareto_front(shown)
    front = shown[on_front].sort_values("time_per_1k")
    # "optimal" arrow towards the top-left corner, as in the paper.
    ox, oy = left + 0.02 * (right - left), top + 0.11 * (bottom - top)
    parts.append(
        f"<defs><marker id='sa-arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
        f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10' fill='none' stroke='{GREEN}' stroke-width='1.6'/></marker></defs>"
        f"<line x1='{ox + 18 * s:.1f}' y1='{oy - 2 * s:.1f}' x2='{left + 2:.1f}' y2='{top + 2:.1f}' stroke='{GREEN}' "
        f"stroke-width='{1.5 * s:.1f}' marker-end='url(#sa-arrow)'/>"
        f"<text x='{ox:.1f}' y='{oy + 14 * s:.1f}' fill='{GREEN}' font-size='{12 * s:.0f}' font-weight='700'>optimal</text>"
    )
    # Front line, then the points.
    path = " ".join(f"{X(r.time_per_1k):.1f},{Y(r.improvement):.1f}" for r in front.itertuples())
    parts.append(f"<polyline points='{path}' fill='none' stroke='{ORANGE}' stroke-width='{2 * s:.1f}'/>")
    for i, row in shown.iterrows():
        cx, cy = X(row.time_per_1k), Y(row.improvement)
        if on_front[i]:
            parts.append(
                f"<circle class='sa-point' cx='{cx:.1f}' cy='{cy:.1f}' r='{4.5 * s:.1f}' fill='{ORANGE}' stroke-width='1'/>"
            )
        else:
            parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{3.9 * s:.1f}' fill='#d3d3d3' fill-opacity='0.8' "
                "stroke='#808080' stroke-width='1'/>"
            )
    # Labels at the paper's (adjustText) positions, with a leader line when moved away.
    for i, row in shown.iterrows():
        cx, cy = X(row.time_per_1k), Y(row.improvement)
        lx, ly = layout["labels"].get(row.method, (row.time_per_1k, row.improvement))
        tx, ty = X(lx), Y(ly)
        if math.hypot(tx - cx, ty - cy) > 14 * s:
            parts.append(
                f"<line class='sa-ink' x1='{cx:.1f}' y1='{cy:.1f}' x2='{tx:.1f}' y2='{ty:.1f}' stroke-width='1' opacity='0.6'/>"
            )
        weight, cls = ("700", "sa-text sa-halo") if on_front[i] else ("400", "sa-dim sa-halo")
        parts.append(
            f"<text class='{cls}' x='{tx:.1f}' y='{ty:.1f}' text-anchor='middle' dominant-baseline='central' "
            f"font-size='{12 * s:.0f}' font-weight='{weight}'>{html.escape(row.method)}</text>"
        )
    # Legend (bottom left, as in the paper's Figure 4): the slow search methods sit bottom right.
    lw, lh = 150 * s, 40 * s
    lx0, ly0 = left + 8 * s, bottom - lh - 8 * s
    parts.append(f"<rect class='sa-legend' x='{lx0:.1f}' y='{ly0:.1f}' width='{lw:.1f}' height='{lh:.1f}' rx='4'/>")
    for k, (fill, stroke, label) in enumerate(
        [("#d3d3d3", "#808080", "Dominated Methods"), (ORANGE, None, "Pareto Front")]
    ):
        yy = ly0 + (12 + 16 * k) * s
        stroke_attr = "class='sa-point'" if stroke is None else f"stroke='{stroke}'"
        parts.append(f"<circle {stroke_attr} cx='{lx0 + 14 * s:.1f}' cy='{yy:.1f}' r='{4 * s:.1f}' fill='{fill}'/>")
        parts.append(
            f"<text class='sa-text' x='{lx0 + 26 * s:.1f}' y='{yy + 4 * s:.1f}' font-size='{10 * s:.0f}'>{label}</text>"
        )
    # Hover targets last, so they sit on top. Points closer than a hover target list each other:
    # otherwise a point hidden under another could not be hovered.
    hit_r = 7 * s
    xy = {i: (X(row.time_per_1k), Y(row.improvement)) for i, row in shown.iterrows()}
    for i, row in shown.iterrows():
        tip = [
            f"{row.method}" + (" (on the Pareto front)" if on_front[i] else ""),
            f"Improvement over Random: {row.improvement:.1f}%",
            f"Selection time per 1k features: {row.time_per_1k:.3g} s",
            f"Datasets: {int(row.n_datasets)}",
        ]
        for j, other in shown.iterrows():
            if j != i and math.dist(xy[i], xy[j]) < 2 * hit_r:
                tip.append(f"Next to {other.method}: {other.improvement:.1f}%, {other.time_per_1k:.3g} s")
        text = _tip(tip)
        parts.append(
            f"<circle class='sa-hit' cx='{X(row.time_per_1k):.1f}' cy='{Y(row.improvement):.1f}' r='{hit_r:.1f}' "
            f"data-tip='{text}' aria-label='{text}'/>"
        )
    parts.append("</svg>")
    return "".join(parts)


def pareto_explorer_html(summary: pd.DataFrame, *, title: str) -> str:
    """``pareto.html``: the interactive Pareto front with SVG / PNG / CSV export."""
    return _figure_page(
        pareto_svg(summary),
        summary,
        name="selectarena_pareto",
        hint="Hover or tap a method for its values.",
        title=title,
    )


# --------------------------------------------------------------------------------------------------
# Lasagna plots


def _png_data_uri(rgba: np.ndarray) -> str:
    import matplotlib.image

    buffer = io.BytesIO()
    matplotlib.image.imsave(buffer, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _hex(rgba) -> str:
    return "#" + "".join(f"{round(float(c) * 255):02x}" for c in rgba[:3])


def lasagna_svg(
    matrix: pd.DataFrame,
    metric: str,
    variant: str = "plain",
    *,
    datasets: dict[float, list[str]] | None = None,
    smoothing: float = lasagna.SMOOTHING,
) -> str:
    """The lasagna plot of ``matrix`` (methods x EPV, see :func:`~selectarena.evaluation.lasagna.epv_matrix`).

    ``datasets`` maps an EPV value to the names of its datasets, for the tooltips.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("Blues")
    s = _SCALE
    methods = list(matrix.index)
    n = len(methods)
    log_epv = np.log(matrix.columns.to_numpy(dtype=float))
    edges = lasagna._bin_edges(log_epv)
    x_lo, x_hi = edges[0], edges[-1]

    label_w = max(len(m) for m in methods) * 6.2 * s + 14 * s
    left, right_pad = label_w, (140 if variant.startswith("binary_") else 110) * s
    row_h = 11.5 * s
    top, plot_h = 14 * s, n * row_h
    width = 576 * s
    right = width - right_pad
    bottom = top + plot_h
    height = bottom + 40 * s

    def X(log_value: float) -> float:  # high EPV (easy settings) on the left, as in the paper
        return left + (x_hi - log_value) / (x_hi - x_lo) * (right - left)

    def Y(row: float) -> float:
        return top + row * row_h

    title = f"{metric.capitalize()} per method and events per variable"
    parts = [_svg_open(width, height, title)]
    rule = variant.split("_", 1)[1] if "_" in variant else None
    mask = {"top_3": lasagna.top_k_mask, "thr_003": lasagna.threshold_mask}[rule](matrix) if rule else None

    if variant.startswith("binary_"):
        for r in range(n):
            for c in range(len(log_epv)):
                color = _hex(cmap(1.0 if mask.iloc[r, c] else 0.0))
                parts.append(
                    f"<rect x='{X(edges[c + 1]):.2f}' y='{Y(r):.2f}' width='{X(edges[c]) - X(edges[c + 1]) + 0.3:.2f}' "
                    f"height='{row_h + 0.3:.2f}' fill='{color}' shape-rendering='crispEdges'/>"
                )
    else:
        _, _, Z = lasagna.smoothed_grid(matrix, smoothing)
        rgba = cmap(np.clip(np.nan_to_num(Z, nan=0.0), 0, 1))
        rgba[np.isnan(Z)] = 0
        image = _png_data_uri(rgba[:, ::-1])  # flipped: high EPV on the left
        parts.append(
            f"<image class='sa-heat' x='{left:.2f}' y='{top:.2f}' width='{right - left:.2f}' height='{plot_h:.2f}' "
            f"preserveAspectRatio='none' href='{image}'/>"
        )
        if mask is not None:  # white outlines around each run of selected cells
            for r in range(n):
                values = mask.iloc[r].to_numpy(dtype=int)
                start = None
                for c, v in enumerate(values):
                    if v == 1 and start is None:
                        start = c
                    if start is not None and (v == 0 or c == len(values) - 1):
                        end = c if v == 0 else c + 1
                        x_a, x_b = X(edges[end]), X(edges[start])
                        parts.append(
                            f"<rect x='{x_a:.2f}' y='{Y(r):.2f}' width='{x_b - x_a:.2f}' height='{row_h:.2f}' "
                            "fill='none' stroke='#ffffff' stroke-width='1.5'/>"
                        )
                        start = None
    parts.append(
        f"<rect class='sa-frame-line' x='{left:.1f}' y='{top:.1f}' width='{right - left:.1f}' height='{plot_h:.1f}' "
        "stroke-opacity='0.8'/>"
    )
    # Method labels and EPV ticks.
    for r, method in enumerate(methods):
        parts.append(
            f"<text class='sa-text' x='{left - 6 * s:.1f}' y='{Y(r + 0.5):.1f}' text-anchor='end' dominant-baseline='central' "
            f"font-size='{9.5 * s:.1f}'>{html.escape(str(method))}</text>"
        )
    step = max(1, len(log_epv) // 5)
    ticks = lasagna._spread_ticks(np.unique(np.concatenate(([log_epv[0]], log_epv[::step], [log_epv[-1]]))))
    for t in ticks:
        parts.append(
            f"<line class='sa-ink' x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{bottom:.1f}' y2='{bottom + 4 * s:.1f}' stroke-width='1'/>"
        )
        parts.append(
            f"<text class='sa-text' x='{X(t):.1f}' y='{bottom + 14 * s:.1f}' text-anchor='middle' "
            f"font-size='{9 * s:.1f}'>{math.exp(t):.1f}</text>"
        )
    parts.append(
        f"<text class='sa-text' x='{(left + right) / 2:.1f}' y='{bottom + 30 * s:.1f}' text-anchor='middle' "
        f"font-size='{10 * s:.0f}'>{lasagna.X_LABEL}</text>"
    )

    # Colour bar (heat maps) or legend (binary maps), right of the plot.
    bx = right + 26 * s
    top_label, bottom_label, _ = lasagna._COLORBAR_LABELS[metric]
    if variant.startswith("binary_"):
        name = "Top-3" if rule == "top_3" else "Threshold"
        for k, (value, label) in enumerate([(1.0, f"Within {name}"), (0.0, f"Outside {name}")]):
            yy = top + plot_h / 2 - 14 * s + k * 18 * s
            parts.append(
                f"<rect x='{bx - 12 * s:.1f}' y='{yy:.1f}' width='{12 * s:.1f}' height='{10 * s:.1f}' "
                f"fill='{_hex(cmap(value))}' class='sa-point' stroke-width='0.6'/>"
            )
            parts.append(
                f"<text class='sa-text' x='{bx + 4 * s:.1f}' y='{yy + 8.5 * s:.1f}' font-size='{9 * s:.1f}'>{label}</text>"
            )
    else:
        bar_h = plot_h * 0.6
        by = top + (plot_h - bar_h) / 2
        stops = "".join(f"<stop offset='{t:.2f}' stop-color='{_hex(cmap(1 - t))}'/>" for t in np.linspace(0, 1, 11))
        parts.append(f"<defs><linearGradient id='sa-cbar' x1='0' y1='0' x2='0' y2='1'>{stops}</linearGradient></defs>")
        parts.append(
            f"<rect x='{bx:.1f}' y='{by:.1f}' width='{9 * s:.1f}' height='{bar_h:.1f}' fill='url(#sa-cbar)' "
            "class='sa-point' stroke-width='0.6'/>"
        )
        for t in np.linspace(0, 1, 6):
            parts.append(
                f"<text class='sa-text' x='{bx + 12 * s:.1f}' y='{by + bar_h * (1 - t) + 3.5 * s:.1f}' "
                f"font-size='{8.5 * s:.1f}'>{t:.1f}</text>"
            )
        parts.append(
            f"<text class='sa-text' x='{bx + 4.5 * s:.1f}' y='{by - 7 * s:.1f}' text-anchor='middle' "
            f"font-size='{9 * s:.1f}'>{top_label}</text>"
        )
        parts.append(
            f"<text class='sa-text' x='{bx + 4.5 * s:.1f}' y='{by + bar_h + 13 * s:.1f}' text-anchor='middle' "
            f"font-size='{9 * s:.1f}'>{bottom_label}</text>"
        )

    # Hover targets: one cell per method and EPV, with the unsmoothed value.
    top3 = lasagna.top_k_mask(matrix)
    near = lasagna.threshold_mask(matrix)
    for r, method in enumerate(methods):
        for c, epv in enumerate(matrix.columns):
            value = matrix.iloc[r, c]
            if pd.isna(value):
                continue
            tip = [
                str(method),
                f"{metric.capitalize()}: {value:.3f} (mean over budgets)",
                f"Events per variable: {epv:.3g}",
            ]
            names = (datasets or {}).get(float(epv))
            if names:
                tip.append("Dataset" + ("s" if len(names) > 1 else "") + ": " + ", ".join(names))
            notes = [
                label
                for flag, label in ((top3.iloc[r, c], "top-3"), (near.iloc[r, c], "within 0.03 of the best"))
                if flag
            ]
            if notes:
                tip.append("At this EPV: " + ", ".join(notes))
            text = _tip(tip)
            x_a, x_b = X(edges[c + 1]), X(edges[c])
            parts.append(
                f"<rect class='sa-hit' x='{x_a:.2f}' y='{Y(r):.2f}' width='{x_b - x_a:.2f}' height='{row_h:.2f}' "
                f"data-tip='{text}' aria-label='{text}'/>"
            )
    parts.append("</svg>")
    return "".join(parts)


def lasagna_explorer_html(
    matrix: pd.DataFrame,
    long: pd.DataFrame,
    metric: str,
    variant: str,
    *,
    datasets: dict[float, list[str]] | None = None,
    title: str,
) -> str:
    """``<variant>.html``: an interactive lasagna plot with SVG / PNG / CSV (``long``: per_epv.csv) export."""
    svg = lasagna_svg(matrix, metric, variant, datasets=datasets)
    return _figure_page(
        svg, long, name=f"selectarena_{metric}_{variant}", hint="Hover or tap a cell for its value.", title=title
    )
