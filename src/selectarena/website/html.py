"""Self-contained HTML artifacts for the leaderboard Space.

Like TabArena's interactive explorers, each artifact is a single HTML file with inline CSS and
JavaScript and no external requests, which the Space embeds in an iframe. Everything is text, so
the Space repo needs no Git LFS:

* :func:`leaderboard_table_html`: a sortable table (``leaderboard_table.html``,
  ``per_dataset_table.html``);
* :func:`elo_explorer_html`: the paper's main figure as SVG (``elo_explorer.html``): Elo per method
  with its 95% interval, the two bars of each method shaded by stability and validity, a tooltip
  per method and SVG / PNG / CSV export.

Every page posts ``{saFrameHeight: <px>}`` to its parent window so the Space can size the iframe.

The generated JavaScript avoids ``${`` and ``{{`` so the pages can also be embedded through
template engines (e.g. Gradio's ``gr.HTML``) that treat them as placeholders.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

_BASE_CSS = """
:root { --fg:#1f2328; --muted:#59636e; --bg:#ffffff; --line:#d1d9e0; --soft:#f6f8fa; --accent:#0072B2; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --line:#3d444d; --soft:#151b23; --accent:#4aa3df; }
}
html, body { margin:0; padding:0; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.toolbar { display:flex; gap:6px; justify-content:flex-end; align-items:center; padding:4px 2px 8px; }
.toolbar .hint { margin-right:auto; color:var(--muted); font-size:12px; }
.toolbar button { font:inherit; font-size:12px; color:var(--fg); background:var(--soft);
  border:1px solid var(--line); border-radius:6px; padding:3px 10px; cursor:pointer; }
.toolbar button:hover { border-color:var(--accent); }
"""

_DOWNLOAD_JS = """
function saDownload(name, blob) {
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
function saCsv(name) {
  saDownload(name, new Blob([document.getElementById('sa-csv').textContent], {type: 'text/csv'}));
}
// Report the page height to an embedding page (the Space resizes its iframe to it).
(function () {
  function post() {
    try {
      var height = Math.ceil(document.body.getBoundingClientRect().height) + 4;
      if (window.parent !== window) window.parent.postMessage({saFrameHeight: height}, '*');
    } catch (e) {}
  }
  window.addEventListener('load', post);
  window.addEventListener('resize', post);
  if (window.ResizeObserver) new ResizeObserver(post).observe(document.body);
})();
"""


def _page(title: str, body: str, css: str = "", js: str = "") -> str:
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_BASE_CSS}{css}</style></head>"
        f"<body>{body}<script>{_DOWNLOAD_JS}{js}</script></body></html>\n"
    )


def _csv_block(df: pd.DataFrame) -> str:
    return f"<script type='text/plain' id='sa-csv'>{html.escape(df.to_csv(index=False))}</script>"


def _is_missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _fmt(value, digits: int | None) -> str:
    if _is_missing(value):
        return "–"
    if digits is None:
        return html.escape(str(value))
    return f"{float(value):,.{digits}f}"


# --------------------------------------------------------------------------------------------------
# Tables

_TABLE_CSS = """
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { position:sticky; top:0; background:var(--soft); font-weight:600; text-align:left;
  cursor:pointer; user-select:none; font-size:13px; }
th.num, td.num { text-align:right; }
th .arrow { color:var(--muted); font-weight:400; }
th[aria-sort] { color:var(--accent); }
tbody tr:hover { background:var(--soft); }
tbody tr:last-child td { border-bottom:none; }
td.method { font-weight:600; }
td .sub { color:var(--muted); font-size:12px; font-weight:400; margin-left:6px; }
td.na { color:var(--muted); }
.pill { display:inline-block; padding:0 8px; border:1px solid var(--line); border-radius:999px;
  font-size:12px; color:var(--muted); }
"""

_TABLE_JS = """
(function () {
  var table = document.getElementById('sa-table');
  var headers = table.querySelectorAll('th');
  headers.forEach(function (th, col) {
    th.addEventListener('click', function () {
      var asc = th.getAttribute('aria-sort') !== 'ascending';
      headers.forEach(function (h) { h.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
      var numeric = th.classList.contains('num');
      var rows = Array.prototype.slice.call(table.tBodies[0].rows);
      rows.sort(function (a, b) {
        var x = a.cells[col].getAttribute('data-sort'), y = b.cells[col].getAttribute('data-sort');
        if (numeric) {
          x = x === '' ? NaN : parseFloat(x); y = y === '' ? NaN : parseFloat(y);
          if (isNaN(x)) return 1;
          if (isNaN(y)) return -1;
          return asc ? x - y : y - x;
        }
        return asc ? x.localeCompare(y) : y.localeCompare(x);
      });
      rows.forEach(function (r) { table.tBodies[0].appendChild(r); });
    });
  });
})();
"""


def table_html(
    df: pd.DataFrame,
    columns: list[dict],
    *,
    title: str,
    csv: pd.DataFrame | None = None,
    csv_name: str = "selectarena.csv",
    hint: str = "Click a column to sort.",
) -> str:
    """A sortable table.

    Each column spec has ``key`` (column of ``df``), ``label``, optional ``digits`` (numeric
    column, ``None`` for text), ``help`` (header tooltip), ``cls`` (cell class) and ``render``
    (``row -> inner HTML``, overriding the default cell text).
    """
    head = []
    for c in columns:
        numeric = c.get("digits") is not None or c.get("numeric", False)
        tip = f" title='{html.escape(c['help'], quote=True)}'" if c.get("help") else ""
        head.append(f"<th class='{'num' if numeric else ''}'{tip}>{html.escape(c['label'])}</th>")
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            value = row.get(c["key"])
            numeric = c.get("digits") is not None or c.get("numeric", False)
            sort = "" if _is_missing(value) else html.escape(str(value), quote=True)
            inner = c["render"](row) if "render" in c else _fmt(value, c.get("digits"))
            classes = " ".join(
                x for x in ["num" if numeric else "", c.get("cls", ""), "na" if inner == "–" else ""] if x
            )
            cells.append(f"<td class='{classes}' data-sort='{sort}'>{inner}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    toolbar = (
        f"<div class='toolbar'><span class='hint'>{html.escape(hint)}</span>"
        f"<button onclick=\"saCsv('{html.escape(csv_name, quote=True)}')\">CSV</button></div>"
    )
    content = (
        f"{toolbar}<div class='wrap'><table id='sa-table'><thead><tr>{''.join(head)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>{_csv_block(df if csv is None else csv)}"
    )
    return _page(title, content, _TABLE_CSS, _TABLE_JS)


def _elo_cell(row) -> str:
    if _is_missing(row["Elo"]):
        return "–"
    ci = ""
    if not _is_missing(row["Elo CI low"]) and not _is_missing(row["Elo CI high"]):
        ci = f"<span class='sub'>+{row['Elo CI high'] - row['Elo']:.0f} / −{row['Elo'] - row['Elo CI low']:.0f}</span>"
    return f"{row['Elo']:.0f}{ci}"


LEADERBOARD_TABLE_COLUMNS = [
    {"key": "Rank", "label": "#", "digits": 0},
    {"key": "Method", "label": "Method", "cls": "method"},
    {
        "key": "TypeName",
        "label": "Type",
        "render": lambda r: (
            f"<span class='pill'>{html.escape(str(r['Type']))} {html.escape(str(r['TypeName']))}</span>"
        ),
        "help": "Filter: scores features from the data alone. Wrapper: searches feature subsets by refitting "
        "a proxy model. Embedded: reads the selection off a fitted model.",
    },
    {
        "key": "Elo",
        "label": "Elo ↑",
        "digits": 0,
        "render": _elo_cell,
        "help": "Bradley-Terry Elo from pairwise matches within each dataset, budget and downstream model. "
        "Random = 1000. +/− give the 95% bootstrap interval.",
    },
    {"key": "Avg. rank", "label": "Avg. rank ↓", "digits": 2, "help": "Mean rank of the downstream error."},
    {
        "key": "Win rate (%)",
        "label": "Win rate ↑",
        "digits": 1,
        "help": "Share of pairwise comparisons won, in %. Ties count half.",
    },
    {
        "key": "Improvability (%)",
        "label": "Improvability ↓",
        "digits": 1,
        "help": "Mean relative gap (error − best error) / error to the best method, in %.",
    },
    {
        "key": "Error vs. all features",
        "label": "Error vs. all ↓",
        "digits": 3,
        "help": "Mean ratio of the error to the same model trained on all features. Below 1: the selection helped.",
    },
    {
        "key": "Stability",
        "label": "Stability ↑",
        "digits": 3,
        "help": "Agreement of the selected features across bootstrap resamples (Nogueira et al., 2018).",
    },
    {
        "key": "Validity",
        "label": "Validity ↑",
        "digits": 3,
        "help": "Share of the budget filled with original features when as many noise features are added.",
    },
    {
        "key": "Selection time (s)",
        "label": "Time (s) ↓",
        "digits": 1,
        "help": "Median wall-clock time of the selection.",
    },
    {
        "key": "Fallback rate (%)",
        "label": "Fallback (%)",
        "digits": 1,
        "help": "Share of runs where the method ran out of time and random features filled the budget.",
    },
    {"key": "#Datasets", "label": "#Datasets", "digits": 0},
]


def leaderboard_table_html(leaderboard: pd.DataFrame, *, title: str) -> str:
    """``leaderboard_table.html`` of one subset, from its ``website_leaderboard.csv`` table."""
    return table_html(
        leaderboard,
        LEADERBOARD_TABLE_COLUMNS,
        title=title,
        csv_name="selectarena_leaderboard.csv",
        hint="Click a column to sort. Hover a header for its definition.",
    )


def per_dataset_table_html(per_dataset: pd.DataFrame, *, title: str) -> str:
    """``per_dataset_table.html``: one row per dataset, the mean rank of each method."""
    columns = [{"key": "dataset", "label": "Dataset", "cls": "method"}]
    columns += [
        {"key": c, "label": c, "digits": 2} for c in per_dataset.columns if c not in ("dataset", "problem_type")
    ]
    if "problem_type" in per_dataset:
        columns.insert(1, {"key": "problem_type", "label": "Problem type"})
    return table_html(
        per_dataset,
        columns,
        title=title,
        csv_name="selectarena_per_dataset_ranks.csv",
        hint="Mean rank of each method per dataset (lower is better). Click a column to sort.",
    )


def _budget_label(index: int, total_budget: int | None) -> str:
    label = f"Budget {index + 1}"
    if total_budget and index == 0:
        return f"{label} (smallest)"
    if total_budget and index == total_budget - 1:
        return f"{label} (largest)"
    return label


def extra_table_html(table: pd.DataFrame, metric: str, *, title: str, total_budget: int | None = None) -> str:
    """``stability_table.html`` / ``validity_table.html``: the mean ``metric`` per method, overall
    and per budget (and per noise level for validity)."""
    budget_columns = sorted(
        (c for c in table.columns if c.startswith(f"{metric}_budget_")), key=lambda c: int(c.rsplit("_", 1)[1])
    )
    columns = []
    if "noise" in table:
        columns.append(
            {"key": "noise", "label": "Noise level", "digits": 1, "help": "Noise features per original feature."}
        )
    columns += [
        {"key": "method", "label": "Method", "cls": "method"},
        {"key": metric, "label": f"{metric.capitalize()} ↑", "digits": 3, "help": "Mean over datasets and budgets."},
    ]
    columns += [
        {"key": c, "label": _budget_label(int(c.rsplit("_", 1)[1]), total_budget), "digits": 3} for c in budget_columns
    ]
    return table_html(table, columns, title=title, csv_name=f"selectarena_{metric}.csv")


# --------------------------------------------------------------------------------------------------
# Main figure

STABILITY_COLOR = "#0072B2"
VALIDITY_COLOR = "#D55E00"

_FIGURE_CSS = """
svg { display:block; width:100%; height:auto; }
.sa-text { fill:var(--fg); }
.sa-muted { fill:var(--muted); }
.sa-grid { stroke:var(--line); }
.sa-ink { stroke:var(--fg); }
.sa-hit { fill:transparent; }
.sa-hit:hover { fill:var(--fg); fill-opacity:0.06; }
"""

_FIGURE_JS = """
function saSvgString() {
  var svg = document.getElementById('sa-figure').cloneNode(true);
  var style = getComputedStyle(document.documentElement);
  var vars = ['--fg', '--muted', '--line', '--bg'];
  var css = svg.querySelector('style').textContent;
  vars.forEach(function (v) { css = css.split('var(' + v + ')').join(style.getPropertyValue(v).trim()); });
  svg.querySelector('style').textContent = css;
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  return new XMLSerializer().serializeToString(svg);
}
function saSvg() {
  saDownload('selectarena_elo.svg', new Blob([saSvgString()], {type: 'image/svg+xml'}));
}
function saPng() {
  var svg = document.getElementById('sa-figure');
  var box = svg.viewBox.baseVal, scale = 3;
  var img = new Image();
  img.onload = function () {
    var canvas = document.createElement('canvas');
    canvas.width = box.width * scale; canvas.height = box.height * scale;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = getComputedStyle(document.body).backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(function (blob) { saDownload('selectarena_elo.png', blob); });
  };
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(saSvgString());
}
"""


def _shade(color: str, value) -> str:
    """``color`` mixed with white: white at 0, ``color`` at 1 (the paper's colormaps)."""
    t = float(np.clip(value, 0, 1))
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (c - 255) * t):02x}" for c in rgb)


def _nice_step(span: float) -> int:
    for step in (25, 50, 100, 200, 250, 500):
        if span / step <= 8:
            return step
    return 1000


def elo_figure_svg(leaderboard: pd.DataFrame) -> str:
    """The main figure as inline SVG, methods from worst to best Elo (port of ``elo_v4.plot``)."""
    df = leaderboard.dropna(subset=["Elo"]).sort_values("Elo").reset_index(drop=True)
    if df.empty:
        return "<p>No Elo ratings in this subset.</p>"
    lo = df["Elo CI low"].fillna(df["Elo"]).to_numpy(float)
    hi = df["Elo CI high"].fillna(df["Elo"]).to_numpy(float)
    pt = df["Elo"].to_numpy(float)

    n = len(df)
    slot = 64
    left, right, top, bottom = 64, 132, 18, 112
    plot_w = slot * n
    width, height = left + plot_w + right, 440
    plot_h = height - top - bottom
    y_min = math.floor(lo.min() / 50) * 50 - 25
    step = _nice_step(max(hi.max(), pt.max()) - y_min)
    y_max = math.ceil(max(hi.max(), pt.max()) / step) * step
    if y_max <= hi.max():
        y_max += step

    def y(v: float) -> float:
        return top + plot_h * (1 - (v - y_min) / (y_max - y_min))

    parts = [
        f"<svg id='sa-figure' viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Elo rating of each feature selection method, shaded by stability and validity'>",
        f"<style>{_FIGURE_CSS}</style>",
        "<defs><pattern id='sa-na' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
        "<rect width='6' height='6' fill='#e5e7eb'/><line x1='0' y1='0' x2='0' y2='6' stroke='#9ca3af' stroke-width='2'/>"
        "</pattern>"
        f"<linearGradient id='sa-g-stab' x1='0' y1='1' x2='0' y2='0'><stop offset='0' stop-color='#ffffff'/>"
        f"<stop offset='1' stop-color='{STABILITY_COLOR}'/></linearGradient>"
        f"<linearGradient id='sa-g-val' x1='0' y1='1' x2='0' y2='0'><stop offset='0' stop-color='#ffffff'/>"
        f"<stop offset='1' stop-color='{VALIDITY_COLOR}'/></linearGradient></defs>",
    ]
    # Grid and y axis.
    first_tick = math.ceil(y_min / step) * step
    for v in range(int(first_tick), int(y_max) + 1, step):
        parts.append(
            f"<line class='sa-grid' x1='{left}' x2='{left + plot_w}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke-width='1'/>"
        )
        parts.append(
            f"<text class='sa-muted' x='{left - 8}' y='{y(v) + 4:.1f}' text-anchor='end' font-size='12'>{v}</text>"
        )
    parts.append(
        f"<text class='sa-text' transform='translate(16 {top + plot_h / 2:.0f}) rotate(-90)' "
        "text-anchor='middle' font-size='13'>Elo</text>"
    )
    if y_min < 1000 < y_max:
        parts.append(
            f"<line class='sa-ink' x1='{left}' x2='{left + plot_w}' y1='{y(1000):.1f}' y2='{y(1000):.1f}' "
            "stroke-width='1' stroke-dasharray='4 4' opacity='0.5'/>"
        )

    half = slot * 0.7 / 2
    base = y(y_min)
    for i, row in df.iterrows():
        cx = left + slot * (i + 0.5)
        top_y = y(pt[i])
        bars = [(cx - half, row.get("Stability"), STABILITY_COLOR), (cx, row.get("Validity"), VALIDITY_COLOR)]
        for x0, value, color in bars:
            fill = "url(#sa-na)" if _is_missing(value) else _shade(color, value)
            parts.append(
                f"<rect x='{x0:.1f}' y='{top_y:.1f}' width='{half:.1f}' height='{base - top_y:.1f}' "
                f"fill='{fill}' stroke='#000' stroke-width='0.8' fill-opacity='0.9'/>"
            )
        # 95% interval.
        parts.append(
            f"<g class='sa-ink' stroke-width='1.5'><line x1='{cx:.1f}' x2='{cx:.1f}' y1='{y(hi[i]):.1f}' y2='{y(lo[i]):.1f}'/>"
            f"<line x1='{cx - 5:.1f}' x2='{cx + 5:.1f}' y1='{y(hi[i]):.1f}' y2='{y(hi[i]):.1f}'/>"
            f"<line x1='{cx - 5:.1f}' x2='{cx + 5:.1f}' y1='{y(lo[i]):.1f}' y2='{y(lo[i]):.1f}'/></g>"
        )
        label = html.escape(str(row["Method"]))
        parts.append(
            f"<text class='sa-text' font-size='12' text-anchor='end' "
            f"transform='translate({cx + 4:.1f} {base + 14:.1f}) rotate(-35)'>{label}</text>"
        )
        tip = [
            f"{row['Method']} ({row.get('TypeName', '')})",
            f"Elo {pt[i]:.0f}  [95% CI {lo[i]:.0f}–{hi[i]:.0f}]",
            "Stability " + ("n/a" if _is_missing(row.get("Stability")) else f"{row['Stability']:.3f}"),
            "Validity " + ("n/a" if _is_missing(row.get("Validity")) else f"{row['Validity']:.3f}"),
        ]
        parts.append(
            f"<rect class='sa-hit' x='{cx - slot / 2:.1f}' y='{top}' width='{slot}' height='{plot_h + bottom - 8}'>"
            f"<title>{html.escape(chr(10).join(tip))}</title></rect>"
        )
    parts.append(
        f"<line class='sa-ink' x1='{left}' x2='{left + plot_w}' y1='{base:.1f}' y2='{base:.1f}' stroke-width='1' opacity='0.4'/>"
    )

    # Colorbars (stability top, validity below), as in the paper.
    bar_x = left + plot_w + 46
    bar_h = (plot_h - 96) / 2
    for k, (gradient, top_label, bottom_label) in enumerate(
        [("sa-g-stab", "stable", "unstable"), ("sa-g-val", "valid", "invalid")]
    ):
        y0 = top + 22 + k * (bar_h + 60)
        parts.append(
            f"<rect x='{bar_x}' y='{y0:.1f}' width='12' height='{bar_h:.1f}' fill='url(#{gradient})' "
            "stroke='#000' stroke-width='0.6'/>"
        )
        parts.append(
            f"<text class='sa-text' x='{bar_x + 6}' y='{y0 - 10:.1f}' text-anchor='middle' font-size='12'>{top_label}</text>"
        )
        parts.append(
            f"<text class='sa-text' x='{bar_x + 6}' y='{y0 + bar_h + 18:.1f}' text-anchor='middle' "
            f"font-size='12'>{bottom_label}</text>"
        )
        for t in (0.0, 0.5, 1.0):
            ty = y0 + bar_h * (1 - t)
            parts.append(f"<text class='sa-muted' x='{bar_x + 18}' y='{ty + 4:.1f}' font-size='11'>{t:.1f}</text>")
    na_y = top + 22 + 2 * (bar_h + 60) - 22
    parts.append(
        f"<rect x='{bar_x}' y='{na_y:.1f}' width='12' height='12' fill='url(#sa-na)' stroke='#000' stroke-width='0.6'/>"
        f"<text class='sa-muted' x='{bar_x + 18}' y='{na_y + 10:.1f}' font-size='11'>no result</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def elo_explorer_html(leaderboard: pd.DataFrame, *, title: str) -> str:
    """``elo_explorer.html``: the main figure of one subset with SVG / PNG / CSV export."""
    toolbar = (
        "<div class='toolbar'><span class='hint'>Hover a method for its values.</span>"
        "<button onclick='saSvg()'>SVG</button><button onclick='saPng()'>PNG</button>"
        "<button onclick=\"saCsv('selectarena_elo.csv')\">CSV</button></div>"
    )
    columns = ["Method", "TypeName", "Elo", "Elo CI low", "Elo CI high", "Stability", "Validity"]
    csv = leaderboard[[c for c in columns if c in leaderboard]]
    return _page(title, toolbar + elo_figure_svg(leaderboard) + _csv_block(csv), "", _FIGURE_JS)
