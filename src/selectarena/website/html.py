"""Self-contained HTML artifacts for the leaderboard Space.

Like TabArena's interactive explorers, each artifact is a single HTML file with inline CSS and
JavaScript and no external requests, which the Space embeds in an iframe. Everything is text, so
the Space repo needs no Git LFS:

* :func:`leaderboard_table_html` and :func:`per_model_table_html`: sortable tables
  (``leaderboard_table.html``, ``per_model_table.html``);
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

_DARK_TOKENS = "--fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --line:#3d444d; --soft:#151b23; --accent:#4aa3df;"
# Dark mode follows the system unless the Space's theme button set data-theme on the page.
_BASE_CSS = (
    """
:root { --fg:#1f2328; --muted:#59636e; --bg:#ffffff; --line:#d1d9e0; --soft:#f6f8fa; --accent:#0072B2; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { """
    + _DARK_TOKENS
    + """ }
}
:root[data-theme="dark"] { """
    + _DARK_TOKENS
    + """ }
html, body { margin:0; padding:0; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.toolbar { display:flex; gap:6px; justify-content:flex-end; align-items:center; padding:4px 2px 8px; }
.toolbar .hint { margin-right:auto; color:var(--muted); font-size:12px; }
.toolbar button { font:inherit; font-size:12px; color:var(--fg); background:var(--soft);
  border:1px solid var(--line); border-radius:6px; padding:3px 10px; cursor:pointer; }
.toolbar button:hover { border-color:var(--accent); }
"""
)

_DOWNLOAD_JS = """
// Theme: the Space sets data-theme on this page when its theme button was used (same origin), or
// sends {saTheme: 'light' | 'dark' | null}; without either, the page follows the system setting.
(function () {
  function apply(theme) {
    if (theme === 'light' || theme === 'dark') document.documentElement.setAttribute('data-theme', theme);
    else document.documentElement.removeAttribute('data-theme');
  }
  try { apply(window.parent.document.documentElement.getAttribute('data-theme')); } catch (e) {}
  window.addEventListener('message', function (event) {
    if (event.data && 'saTheme' in event.data) apply(event.data.saTheme);
  });
})();
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
  font-size:12px; color:var(--fg); }
.pill.cat-search { border-color:var(--accent); }
.pill.cat-baseline { color:var(--muted); }
.medal { display:inline-block; width:1.4em; text-align:center; font-size:15px; line-height:1; }
td.shaded { font-weight:600; }
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
    column, ``None`` for text), ``help`` (header tooltip), ``cls`` (cell class), ``render``
    (``row -> inner HTML``, overriding the default cell text) and ``style`` (``row -> inline CSS``
    of the cell, e.g. a background colour).
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
            style = c["style"](row) if "style" in c else ""
            style = f" style='{html.escape(style, quote=True)}'" if style else ""
            cells.append(f"<td class='{classes}' data-sort='{sort}'{style}>{inner}</td>")
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


MEDALS = {1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}
"""Gold, silver and bronze for the three best Elo ratings."""
_MEDAL_NAMES = {1: "gold", 2: "silver", 3: "bronze"}


def _medal(rank) -> str:
    if _is_missing(rank) or int(rank) not in MEDALS:
        return "<span class='medal'></span>"
    r = int(rank)
    return f"<span class='medal' role='img' aria-label='{_MEDAL_NAMES[r]} medal'>{MEDALS[r]}</span>"


def _rank_cell(row) -> str:
    return f"{_medal(row['Rank'])}{_fmt(row['Rank'], 0)}"


def _shade_style(color: str):
    """Cell background in ``color``, stronger for higher values (0 to 1), as the bars of the Elo
    figure. Semi-transparent, so it works on light and dark backgrounds."""
    rgb = ",".join(str(int(color[i : i + 2], 16)) for i in (1, 3, 5))

    def style(row, key: str) -> str:
        value = row.get(key)
        if _is_missing(value):
            return ""
        return f"background:rgba({rgb},{0.08 + 0.67 * float(np.clip(value, 0, 1)):.2f})"

    return style


def _elo_cell(row) -> str:
    if _is_missing(row["Elo"]):
        return "–"
    ci = ""
    if not _is_missing(row["Elo CI low"]) and not _is_missing(row["Elo CI high"]):
        ci = f"<span class='sub'>+{row['Elo CI high'] - row['Elo']:.0f} / −{row['Elo'] - row['Elo CI low']:.0f}</span>"
    return f"{row['Elo']:.0f}{ci}"


LEADERBOARD_TABLE_COLUMNS = [
    {"key": "Rank", "label": "#", "digits": 0, "render": _rank_cell, "help": "Rank by Elo."},
    {"key": "Method", "label": "Method", "cls": "method"},
    {
        "key": "Subgroup",
        "label": "Type",
        "render": lambda r: (
            f"<span class='pill cat-{html.escape(str(r['Category']).lower(), quote=True)}'>"
            f"{html.escape(str(r['Subgroup']))}</span>"
            + (
                ""
                if r["Category"] in ("Baseline", "Other")
                else f"<span class='sub'>{html.escape(str(r['Category']).lower())}</span>"
            )
        ),
        "help": "Taxonomy of the paper (Appendix A). Search methods evaluate many feature subsets with the model "
        "(forward, backward, leave-one-out); non-search methods use a proxy instead (correlation, distance, "
        "information theory, tree, regularization).",
    },
    {
        "key": "Elo",
        "label": "Elo ↑",
        "digits": 0,
        "render": _elo_cell,
        "help": "Bradley-Terry Elo from pairwise matches within each dataset, budget and downstream model. "
        "Random = 1000. +/− give the 95% bootstrap interval.",
    },
    {
        "key": "Improvement over Random (%)",
        "label": "Improvement over Random ↑",
        "digits": 1,
        "help": "Mean reduction of the downstream error compared with random selection on the same dataset, "
        "in % (as in the Pareto figure; Random is 0).",
    },
    {
        "key": "Stability",
        "label": "Stability ↑",
        "digits": 3,
        "cls": "shaded",
        "style": lambda r: _shade_style(STABILITY_COLOR)(r, "Stability"),
        "help": "Agreement of the selected features across bootstrap resamples (Nogueira et al., 2018).",
    },
    {
        "key": "Validity",
        "label": "Validity ↑",
        "digits": 3,
        "cls": "shaded",
        "style": lambda r: _shade_style(VALIDITY_COLOR)(r, "Validity"),
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
]


#: Columns of the table's CSV download: what the table shows (its Type column is Category and
#: Subgroup, its Elo cell includes the interval). ``website_leaderboard.csv`` keeps every column.
LEADERBOARD_DOWNLOAD_COLUMNS = [
    "Rank",
    "Method",
    "Category",
    "Subgroup",
    "Elo",
    "Elo CI low",
    "Elo CI high",
    "Improvement over Random (%)",
    "Stability",
    "Validity",
    "Selection time (s)",
    "Fallback rate (%)",
]


def leaderboard_table_html(leaderboard: pd.DataFrame, *, title: str) -> str:
    """``leaderboard_table.html`` of one subset, from its ``website_leaderboard.csv`` table."""
    return table_html(
        leaderboard,
        LEADERBOARD_TABLE_COLUMNS,
        csv=leaderboard[[c for c in LEADERBOARD_DOWNLOAD_COLUMNS if c in leaderboard]],
        title=title,
        csv_name="selectarena_leaderboard.csv",
        hint="Click a column to sort. Hover a header for its definition. Stability and validity are shaded "
        "as in the Elo figure: darker is better.",
    )


def per_model_table(leaderboards: dict[str, pd.DataFrame], overall: pd.DataFrame) -> pd.DataFrame:
    """One row per method: its Elo over all downstream models and per model, with its rank per model.

    ``leaderboards`` maps a model name to the ``website_leaderboard.csv`` table of that model
    (same task type and budget); ``overall`` is the table over all models.
    """
    out = overall[["Rank", "Method", "Category", "Subgroup", "Elo"]].copy()
    for model, board in leaderboards.items():
        values = board.set_index("Method")
        out[f"Elo {model}"] = out["Method"].map(values["Elo"])
        out[f"Rank {model}"] = out["Method"].map(values["Rank"]).astype("Int64")
    return out


def per_model_table_html(table: pd.DataFrame, models: list[str], *, title: str) -> str:
    """``per_model_table.html``: the Elo of each method per downstream model, medals per model."""

    def model_cell(model: str):
        def render(row) -> str:
            if _is_missing(row[f"Elo {model}"]):
                return "–"
            return f"{_medal(row[f'Rank {model}'])}{row[f'Elo {model}']:.0f}"

        return render

    columns = [
        {"key": "Rank", "label": "#", "digits": 0, "render": _rank_cell, "help": "Rank by Elo over all models."},
        {"key": "Method", "label": "Method", "cls": "method"},
        {
            "key": "Elo",
            "label": "All models ↑",
            "digits": 0,
            "render": lambda r: "–" if _is_missing(r["Elo"]) else f"{r['Elo']:.0f}",
            "help": "Elo over all downstream models (as in the main leaderboard).",
        },
    ]
    columns += [
        {
            "key": f"Elo {m}",
            "label": f"{m} ↑",
            "digits": 0,
            "render": model_cell(m),
            "help": f"Elo when the selected features are used to train {m}; medals for the three best.",
        }
        for m in models
    ]
    return table_html(
        table,
        columns,
        title=title,
        csv_name="selectarena_per_model.csv",
        hint="Elo per downstream model; medals mark the three best methods of each model. Click a column to sort.",
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
.sa-hit { fill:transparent; cursor:default; }
.sa-hit:hover, .sa-hit.sa-on { fill:var(--fg); fill-opacity:0.06; }
"""

_TIP_CSS = """
#sa-tip { position:fixed; z-index:10; pointer-events:none; display:none; max-width:280px;
  padding:6px 9px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--fg);
  font-size:12px; line-height:1.45; white-space:pre-line; box-shadow:0 2px 8px rgba(0,0,0,0.15); }
#sa-tip b { font-weight:600; }
"""

_FIGURE_JS = """
// Instant tooltip: a native SVG <title> only shows after a browser delay of 1-3 s.
(function () {
  var tip = document.createElement('div');
  tip.id = 'sa-tip';
  document.body.appendChild(tip);
  var current = null;
  function show(hit, x, y) {
    if (current !== hit) {
      if (current) current.classList.remove('sa-on');
      current = hit;
      hit.classList.add('sa-on');
      var lines = hit.getAttribute('data-tip').split('\\n');
      tip.textContent = '';
      var head = document.createElement('b');
      head.textContent = lines[0];
      tip.appendChild(head);
      tip.appendChild(document.createTextNode('\\n' + lines.slice(1).join('\\n')));
      tip.style.display = 'block';
    }
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 4) left = Math.max(4, x - w - 14);
    if (top + h > window.innerHeight - 4) top = Math.max(4, y - h - 14);
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hide() {
    if (current) current.classList.remove('sa-on');
    current = null;
    tip.style.display = 'none';
  }
  document.addEventListener('pointermove', function (e) {
    var hit = e.target.closest ? e.target.closest('.sa-hit') : null;
    if (hit) show(hit, e.clientX, e.clientY); else if (e.pointerType === 'mouse') hide();
  });
  document.addEventListener('pointerdown', function (e) {  // touch: tap a bar to show, elsewhere to hide
    var hit = e.target.closest ? e.target.closest('.sa-hit') : null;
    if (hit) show(hit, e.clientX, e.clientY); else hide();
  });
  document.addEventListener('mouseleave', hide);
})();
// Downloads always use the light colours (the paper look), whatever the page's theme.
var SA_LIGHT = {'--fg': '#1f2328', '--muted': '#59636e', '--line': '#d1d9e0', '--bg': '#ffffff',
  '--soft': '#f6f8fa', '--accent': '#0072B2'};
function saSvgString() {
  var svg = document.getElementById('sa-figure').cloneNode(true);
  var style = svg.querySelector('style');
  var css = style.textContent;
  Object.keys(SA_LIGHT).forEach(function (v) { css = css.split('var(' + v + ')').join(SA_LIGHT[v]); });
  style.textContent = css;
  svg.querySelectorAll('.sa-hit').forEach(function (hit) { hit.remove(); });
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  svg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  return new XMLSerializer().serializeToString(svg);
}
function saSvg(name) {
  saDownload(name + '.svg', new Blob([saSvgString()], {type: 'image/svg+xml'}));
}
function saPng(name) {
  var svg = document.getElementById('sa-figure');
  var box = svg.viewBox.baseVal, scale = 3;
  var img = new Image();
  img.onload = function () {
    var canvas = document.createElement('canvas');
    canvas.width = box.width * scale; canvas.height = box.height * scale;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = SA_LIGHT['--bg'];
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(function (blob) { saDownload(name + '.png', blob); });
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
            f"{row['Method']} ({row.get('Subgroup', '')}, {str(row.get('Category', '')).lower()})",
            f"Elo {pt[i]:.0f}  [95% CI {lo[i]:.0f}–{hi[i]:.0f}]",
            "Stability " + ("n/a" if _is_missing(row.get("Stability")) else f"{row['Stability']:.3f}"),
            "Validity " + ("n/a" if _is_missing(row.get("Validity")) else f"{row['Validity']:.3f}"),
        ]
        text = html.escape(chr(10).join(tip), quote=True).replace(chr(10), "&#10;")
        parts.append(
            f"<rect class='sa-hit' x='{cx - slot / 2:.1f}' y='{top}' width='{slot}' height='{plot_h + bottom - 8}' "
            f"data-tip='{text}' aria-label='{text}'/>"
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


def figure_toolbar(name: str, hint: str) -> str:
    """Hint plus SVG / PNG / CSV buttons of a figure page; the files are ``<name>.svg`` etc."""
    return (
        f"<div class='toolbar'><span class='hint'>{html.escape(hint)}</span>"
        f"<button onclick=\"saSvg('{name}')\">SVG</button><button onclick=\"saPng('{name}')\">PNG</button>"
        f"<button onclick=\"saCsv('{name}.csv')\">CSV</button></div>"
    )


def elo_explorer_html(leaderboard: pd.DataFrame, *, title: str) -> str:
    """``elo_explorer.html``: the main figure of one subset with SVG / PNG / CSV export."""
    toolbar = figure_toolbar("selectarena_elo", "Hover or tap a method for its values.")
    columns = ["Method", "Category", "Subgroup", "Elo", "Elo CI low", "Elo CI high", "Stability", "Validity"]
    csv = leaderboard[[c for c in columns if c in leaderboard]]
    return _page(title, toolbar + elo_figure_svg(leaderboard) + _csv_block(csv), _TIP_CSS, _FIGURE_JS)


# --------------------------------------------------------------------------------------------------
# Results explorer

_EXPLORER_CSS = """
.filters { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:2px 2px 10px; }
.filters input, .filters select { font:inherit; font-size:13px; color:var(--fg); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
.filters input { min-width:200px; }
.filters label { font-size:12px; color:var(--muted); display:flex; gap:6px; align-items:center; }
.pager { display:flex; gap:8px; align-items:center; justify-content:flex-end; padding:8px 2px; font-size:13px;
  color:var(--muted); }
.pager button { font:inherit; font-size:12px; color:var(--fg); background:var(--soft);
  border:1px solid var(--line); border-radius:6px; padding:3px 10px; cursor:pointer; }
.pager button:disabled { opacity:0.4; cursor:default; }
"""

_EXPLORER_JS = """
(function () {
  var data = JSON.parse(document.getElementById('sa-data').textContent);
  var cols = data.columns, rows = data.rows, PAGE = 50;
  var state = { page: 0, sort: null, asc: true };
  var filters = {};
  var table = document.getElementById('sa-table');

  function cell(row, key) { return row[cols.indexOf(key)]; }
  function fmt(value, digits) {
    if (value === null || value === undefined || value === '') return '–';
    if (digits === null || digits === undefined) return String(value);
    return Number(value).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  }
  function matches(row) {
    var text = (filters.dataset || '').toLowerCase();
    if (text && String(cell(row, 'dataset')).toLowerCase().indexOf(text) < 0) return false;
    for (var key in filters) {
      if (key === 'dataset' || !filters[key]) continue;
      if (String(cell(row, key)) !== filters[key]) return false;
    }
    return true;
  }
  function current() {
    var out = rows.filter(matches);
    if (state.sort !== null) {
      var i = state.sort, dir = state.asc ? 1 : -1;
      out.sort(function (a, b) {
        var x = a[i], y = b[i];
        if (x === null) return 1;
        if (y === null) return -1;
        if (typeof x === 'number' && typeof y === 'number') return (x - y) * dir;
        return String(x).localeCompare(String(y)) * dir;
      });
    }
    return out;
  }
  function render() {
    var list = current();
    var pages = Math.max(1, Math.ceil(list.length / PAGE));
    if (state.page >= pages) state.page = pages - 1;
    var body = table.tBodies[0];
    body.textContent = '';
    list.slice(state.page * PAGE, (state.page + 1) * PAGE).forEach(function (row) {
      var tr = document.createElement('tr');
      data.spec.forEach(function (spec) {
        var td = document.createElement('td');
        var value = cell(row, spec.key);
        td.textContent = fmt(value, spec.digits);
        if (spec.digits !== null && spec.digits !== undefined) td.className = 'num';
        if (td.textContent === '–') td.className += ' na';
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    document.getElementById('sa-count').textContent = list.length.toLocaleString('en-US') + ' rows · page ' +
      (state.page + 1) + ' of ' + pages;
    document.getElementById('sa-prev').disabled = state.page === 0;
    document.getElementById('sa-next').disabled = state.page >= pages - 1;
  }
  data.spec.forEach(function (spec, k) {
    var th = table.tHead.rows[0].cells[k];
    th.addEventListener('click', function () {
      var i = cols.indexOf(spec.key);
      state.asc = state.sort === i ? !state.asc : true;
      state.sort = i;
      Array.prototype.forEach.call(table.tHead.rows[0].cells, function (h) { h.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', state.asc ? 'ascending' : 'descending');
      render();
    });
  });
  document.querySelectorAll('[data-filter]').forEach(function (input) {
    input.addEventListener('input', function () {
      filters[input.getAttribute('data-filter')] = input.value;
      state.page = 0;
      render();
    });
  });
  document.getElementById('sa-prev').addEventListener('click', function () { state.page -= 1; render(); });
  document.getElementById('sa-next').addEventListener('click', function () { state.page += 1; render(); });
  window.saExplorerCsv = function () {
    function quote(v) {
      if (v === null || v === undefined) return '';
      var s = String(v);
      return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    }
    var lines = [cols.join(',')].concat(current().map(function (r) { return r.map(quote).join(','); }));
    saDownload('selectarena_results.csv', new Blob([lines.join('\\n') + '\\n'], {type: 'text/csv'}));
  };
  render();
})();
"""

#: Columns of the results explorer: ``key`` (column of the results table), ``label``, ``digits``
#: (``None`` for text), ``help`` (header tooltip), ``filter`` (a select box over its values).
RESULTS_EXPLORER_COLUMNS = [
    {"key": "dataset", "label": "Dataset", "digits": None},
    {"key": "task_type", "label": "Task", "digits": None, "filter": True},
    {"key": "method", "label": "Method", "digits": None, "filter": True},
    {"key": "model", "label": "Model", "digits": None, "filter": True},
    {"key": "budget", "label": "Budget", "digits": None, "filter": True, "help": "1 is the smallest budget"},
    {"key": "max_features", "label": "#Features", "digits": 0, "help": "Number of features kept"},
    {"key": "metric", "label": "Metric", "digits": None},
    {"key": "error", "label": "Error", "digits": 4, "help": "Mean test error over the splits (lower is better)"},
    {"key": "error_std", "label": "Error std.", "digits": 4},
    {"key": "n_splits", "label": "Splits", "digits": 0},
    {"key": "fs_time_s", "label": "Selection time (s)", "digits": 2, "help": "Mean wall-clock time of the selection"},
    {
        "key": "fallback_pct",
        "label": "Fallback (%)",
        "digits": 0,
        "help": "Share of splits whose budget was filled at random",
    },
]


def results_explorer_html(results: pd.DataFrame, *, title: str) -> str:
    """``results_explorer.html``: the complete results (per dataset, method, model and budget), with filters,
    sorting, pages of 50 rows and CSV export of the filtered rows. The data is embedded as JSON."""
    import json

    spec = [c for c in RESULTS_EXPLORER_COLUMNS if c["key"] in results]
    columns = [c["key"] for c in spec]
    frame = results[columns].astype(object).where(results[columns].notna(), None)
    payload = {
        "columns": columns,
        "spec": [{"key": c["key"], "digits": c["digits"]} for c in spec],
        "rows": frame.to_numpy().tolist(),
    }
    data = json.dumps(payload, separators=(",", ":"), default=float).replace("</", "<\\/")

    controls = [
        "<label>Dataset <input type='search' data-filter='dataset' placeholder='e.g. credit' aria-label='Filter datasets'></label>"
    ]
    for c in spec:
        if not c.get("filter"):
            continue
        values = sorted({str(v) for v in results[c["key"]].dropna()}, key=lambda v: (len(v) > 3, v))
        options = "".join(f"<option value='{html.escape(v, quote=True)}'>{html.escape(v)}</option>" for v in values)
        controls.append(
            f"<label>{html.escape(c['label'])} <select data-filter='{c['key']}'>"
            f"<option value=''>All</option>{options}</select></label>"
        )
    head = "".join(
        f"<th class='{'num' if c['digits'] is not None else ''}'"
        + (f" title='{html.escape(c['help'], quote=True)}'" if c.get("help") else "")
        + f">{html.escape(c['label'])}</th>"
        for c in spec
    )
    body = (
        f"<div class='filters'>{''.join(controls)}</div>"
        "<div class='toolbar'><span class='hint'>Click a column to sort; the CSV holds the filtered rows.</span>"
        "<button onclick='saExplorerCsv()'>CSV</button></div>"
        f"<div class='wrap'><table id='sa-table'><thead><tr>{head}</tr></thead><tbody></tbody></table></div>"
        "<div class='pager'><span id='sa-count'></span><button id='sa-prev'>Previous</button>"
        "<button id='sa-next'>Next</button></div>"
        f"<script type='application/json' id='sa-data'>{data}</script>"
    )
    return _page(title, body, _TABLE_CSS + _EXPLORER_CSS, _EXPLORER_JS)
