"""Lasagna plots of stability and validity: one row per method, events per variable on the x axis.

The same figures as in the paper. Each column is one events-per-variable (EPV) value, i.e. one dataset:
the samples of its smallest class (all samples for regression) per original feature. A cell is the
mean stability or validity of the method over the budgets of the datasets with that EPV. Rows are
sorted by their mean, best on top; high EPV (easy settings) is on the left.

Variants (names as in the paper's file names):

* ``plain``: the heat map, interpolated over EPV and smoothed with a Gaussian filter;
* ``overlay_top_3``: the heat map with white outlines around the three best methods per EPV;
* ``overlay_thr_003``: the heat map with outlines around the methods within 0.03 of the best;
* ``binary_top_3`` / ``binary_thr_003``: unsmoothed 0/1 maps of the same selections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

TOP_K = 3
THRESHOLD = 0.03
SMOOTHING = 10
X_LABEL = "Events per Variable"
VARIANTS = ("plain", "overlay_top_3", "binary_thr_003")
"""The variants of the paper's main stability and validity figures."""
VARIANT_LABELS = {
    "plain": "Heat map",
    "overlay_top_3": "Top-3 outlined",
    "overlay_thr_003": "Within 0.03 of the best outlined",
    "binary_top_3": "Top-3 only",
    "binary_thr_003": "Within 0.03 of the best",
}

# Colour-bar end labels and the x offset of the bottom label, per metric.
_COLORBAR_LABELS = {
    "stability": ("stable", "unstable", 0.55),
    "validity": ("valid", "invalid", 0.65),
}


def per_epv(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean ``value`` per (method, epv) from rows with ``method``, ``epv`` and ``value`` columns."""
    rows = rows.dropna(subset=["method", "epv", value])
    rows = rows[np.isfinite(rows["epv"]) & (rows["epv"] > 0)]
    return rows.groupby(["method", "epv"], as_index=False)[value].mean()


def epv_matrix(long: pd.DataFrame, value: str) -> pd.DataFrame:
    """Long (method, epv, value) data -> methods x EPV matrix, best mean on top."""
    matrix = long.pivot(index="method", columns="epv", values=value)
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]


def top_k_mask(matrix: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """1 where a method is among the k best at that EPV."""
    mask = pd.DataFrame(0, index=matrix.index, columns=matrix.columns)
    for epv in matrix.columns:
        column = matrix[epv].dropna()
        mask.loc[column.nlargest(min(k, len(column))).index, epv] = 1
    return mask


def threshold_mask(matrix: pd.DataFrame, tolerance: float = THRESHOLD) -> pd.DataFrame:
    """1 where a method is within ``tolerance`` of the best method at that EPV."""
    return matrix.ge(matrix.max(axis=0) - tolerance, axis=1).astype(int)


_MASKS = {"top_3": top_k_mask, "thr_003": threshold_mask}
_LEGEND_LABELS = {"top_3": "Top-3", "thr_003": "Threshold"}


def lasagna_figure(matrix: pd.DataFrame, metric: str, variant: str = "plain", smoothing: float = SMOOTHING) -> Figure:
    """The lasagna plot of ``matrix`` (see :func:`epv_matrix`) for ``metric`` ("stability" or "validity")."""
    if variant == "plain":
        return _heatmap(matrix, metric, smoothing)
    if variant.startswith("overlay_"):
        return _heatmap(matrix, metric, smoothing, highlight=_MASKS[variant.removeprefix("overlay_")](matrix))
    if variant.startswith("binary_"):
        return _binary_map(matrix, variant.removeprefix("binary_"), metric)
    raise ValueError(f"unknown lasagna variant {variant!r}")


# --- Drawing ------------------------------------------------------------------------------------


def _cmap():
    import matplotlib.pyplot as plt

    return plt.get_cmap("Blues")


def _figure_size(matrix: pd.DataFrame) -> tuple[float, float]:
    # The paper's 8 x 3.5 inches fit its 15 methods; grow with more rows.
    return 8, max(3.5, 0.22 * len(matrix.index) + 0.6)


def _bin_edges(x: np.ndarray) -> np.ndarray:
    """Cell edges halfway between neighbouring x values."""
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def _spread_ticks(ticks: np.ndarray, min_gap: float = 0.08) -> np.ndarray:
    """Drop ticks closer than ``min_gap`` (share of the axis range) to a kept neighbour, so their
    labels do not overlap; both ends are kept."""
    if len(ticks) < 3:
        return ticks
    gap = min_gap * (ticks[-1] - ticks[0])
    kept = [ticks[0]]
    for t in ticks[1:-1]:
        if t - kept[-1] >= gap and ticks[-1] - t >= gap:
            kept.append(t)
    kept.append(ticks[-1])
    return np.array(kept)


def _format_axes(ax, matrix: pd.DataFrame, log_epv: np.ndarray) -> None:
    """EPV ticks (about 5, always including both ends), method labels, best method on top,
    easy (high EPV) settings on the left."""
    step = max(1, len(log_epv) // 5)
    ticks = _spread_ticks(np.unique(np.concatenate(([log_epv[0]], log_epv[::step], [log_epv[-1]]))))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{np.exp(t):.1f}" for t in ticks], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(X_LABEL)
    ax.invert_yaxis()
    ax.invert_xaxis()


def smoothed_grid(matrix: pd.DataFrame, smoothing: float = SMOOTHING) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The heat map's values: ``(log_epv, dense_edges, Z)`` with ``Z`` of shape (methods, 1000) on
    the log-EPV grid ``dense_edges``, each method interpolated over the EPVs where it has a value
    and smoothed with a Gaussian filter."""
    from scipy.ndimage import gaussian_filter1d

    log_epv = np.log(matrix.columns.to_numpy(dtype=float))
    edges = _bin_edges(log_epv)
    dense_edges = np.linspace(edges[0], edges[-1], 1001)
    dense = 0.5 * (dense_edges[:-1] + dense_edges[1:])
    Z = np.vstack(
        [
            np.interp(dense, log_epv[row.notna().to_numpy()], row.dropna().to_numpy())
            if row.notna().any()
            else np.full_like(dense, np.nan)
            for _, row in matrix.iterrows()
        ]
    )
    if smoothing:
        Z = gaussian_filter1d(Z, sigma=smoothing, axis=1, mode="nearest")
    return log_epv, dense_edges, Z


def _heatmap(matrix: pd.DataFrame, metric: str, smoothing: float, highlight: pd.DataFrame | None = None) -> Figure:
    """Interpolated and Gaussian-smoothed heat map, optionally with white outlines around the
    cells where ``highlight`` is 1."""
    import matplotlib.pyplot as plt

    log_epv, dense_edges, Z = smoothed_grid(matrix, smoothing)
    edges = _bin_edges(log_epv)
    y_edges = np.arange(len(matrix.index) + 1) - 0.5

    fig, ax = plt.subplots(figsize=_figure_size(matrix))
    image = ax.pcolormesh(dense_edges, y_edges, Z, cmap=_cmap(), vmin=0, vmax=1, shading="flat", rasterized=True)
    _format_axes(ax, matrix, log_epv)
    if highlight is not None:
        _outline_cells(ax, highlight, edges)

    top, bottom, bottom_x = _COLORBAR_LABELS[metric]
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04, shrink=0.6)
    cbar.ax.set_title(top, pad=12, fontsize=10)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(length=0, labelsize=10)
    cbar.ax.text(bottom_x, -0.1, bottom, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=10)
    fig.tight_layout()
    return fig


def _binary_map(matrix: pd.DataFrame, rule: str, metric: str) -> Figure:
    """Unsmoothed 0/1 map of the methods selected by ``rule`` ('top_3' or 'thr_003')."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cmap = _cmap()
    mask = _MASKS[rule](matrix)
    log_epv = np.log(matrix.columns.to_numpy(dtype=float))
    y_edges = np.arange(len(matrix.index) + 1) - 0.5

    fig, ax = plt.subplots(figsize=_figure_size(matrix))
    ax.pcolormesh(
        _bin_edges(log_epv), y_edges, mask.to_numpy(), cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=True
    )
    _format_axes(ax, matrix, log_epv)

    name = _LEGEND_LABELS[rule]
    legend = [
        Patch(facecolor=cmap(1.0), edgecolor="black", label=f"Within {name}"),
        Patch(facecolor=cmap(0.0), edgecolor="black", label=f"Outside {name}"),
    ]
    if rule == "thr_003":
        anchor = (1.005, 0.2)
    else:
        anchor = (0.15, 0.11) if metric == "validity" else (0.15, 0.0)
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=anchor)
    fig.tight_layout()
    return fig


def _outline_cells(ax, mask: pd.DataFrame, x_edges: np.ndarray) -> None:
    """Draw one white rectangle around each horizontal run of 1s in ``mask``."""
    from matplotlib.patches import Rectangle

    for row_idx, (_, row) in enumerate(mask.iterrows()):
        values = row.to_numpy(dtype=int)
        start = None
        for col, v in enumerate(values):
            if v == 1 and start is None:
                start = col
            if start is not None and (v == 0 or col == len(values) - 1):
                end = col if v == 0 else col + 1
                ax.add_patch(
                    Rectangle(
                        (x_edges[start], row_idx - 0.5),
                        x_edges[end] - x_edges[start],
                        1.0,
                        fill=False,
                        edgecolor="white",
                        linewidth=1,
                        linestyle="-",
                    )
                )
                start = None
