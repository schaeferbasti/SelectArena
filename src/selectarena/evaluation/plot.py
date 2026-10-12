"""The SelectArena main figure: Elo per method, bars shaded by stability and validity.

Port of ``plot`` in ``performance/baseline/elo_v4.py`` (paper Figure 1). Methods are ordered
from worst to best Elo. Each method gets two bars of the same height: the left one shaded by
stability (blue), the right one by validity (orange), darker is better. The error bar is the
95% bootstrap interval of the Elo rating.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Shade used when a method has no stability / validity result (the script's default).
MISSING_SCORE = 0.5


def plot_main_figure(main: pd.DataFrame, out_paths: list[str | Path]) -> None:
    """Draw the figure from a table with ``method``, ``elo``, ``elo_ci_low``, ``elo_ci_high``,
    ``stability`` and ``validity`` columns and save it to every path in ``out_paths``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    df = main.sort_values("elo").reset_index(drop=True)
    pt = df["elo"].to_numpy(float)
    lo = df["elo_ci_low"].to_numpy(float)
    hi = df["elo_ci_high"].to_numpy(float)
    stability = df["stability"].fillna(MISSING_SCORE) if "stability" in df else pd.Series(MISSING_SCORE, df.index)
    validity = df["validity"].fillna(MISSING_SCORE) if "validity" in df else pd.Series(MISSING_SCORE, df.index)

    fig, ax = plt.subplots(figsize=(11, 3))
    x = np.arange(len(df))
    half_width = 0.7 / 2
    base_bottom = np.floor(lo.min() / 50) * 50 - 25

    cmap_stability = mcolors.LinearSegmentedColormap.from_list("stability_cmap", ["#ffffff", "#0072B2"])
    cmap_validity = mcolors.LinearSegmentedColormap.from_list("validity_cmap", ["#ffffff", "#D55E00"])
    bar_kwargs = {"bottom": base_bottom, "width": half_width, "edgecolor": "black", "linewidth": 0.8, "alpha": 0.9}
    ax.bar(
        x - half_width / 2,
        pt - base_bottom,
        color=[cmap_stability(s) for s in stability],
        label="Stability",
        **bar_kwargs,
    )
    ax.bar(
        x + half_width / 2, pt - base_bottom, color=[cmap_validity(v) for v in validity], label="Validity", **bar_kwargs
    )
    ax.errorbar(x, pt, yerr=[pt - lo, hi - pt], fmt="none", ecolor="black", capsize=3, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    ax.set_ylabel("Elo")
    ax.set_ylim(bottom=base_bottom)
    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(0.3)

    for cmap, rect, top, bottom in [
        (cmap_stability, [0.92, 0.0, 0.01, 0.3], "stable", "unstable"),
        (cmap_validity, [0.92, 0.55, 0.01, 0.3], "valid", "invalid"),
    ]:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=fig.add_axes(rect))
        cbar.ax.set_title(top, pad=12, fontsize=10)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.tick_params(length=0, labelsize=10)
        cbar.ax.text(0.5, -0.1, bottom, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=10)
    plt.subplots_adjust(right=0.88)

    for path in out_paths:
        fig.savefig(path, dpi=300 if str(path).endswith(".pdf") else 150, bbox_inches="tight")
    plt.close(fig)
