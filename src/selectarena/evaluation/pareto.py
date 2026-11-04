"""Predictive performance vs. selection time: the Pareto front of the feature selection methods.

Ported from the paper's plotting code (``Plotting_FS/scripts/pareto.py``) so the leaderboard shows
the same figure as the paper:

* improvement over Random per dataset: ``(error_random - error_method) / error_random * 100``;
* selection time per 1k features: ``selection time / number of original features * 1000``.

Both are averaged over budgets, downstream models and splits per dataset, then over datasets. A
method is on the Pareto front when no other method is at least as fast and at least as good, and
strictly better in one of the two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

RANDOM = "Random"
HIDDEN_IN_PLOT = frozenset({"LaplacianScore"})
"""Left out of the figure only, as in the paper: far below Random, it would squash the y axis."""

_ORANGE = "#D55E00"
_GREEN = "#009E73"


def _improvement_per_dataset(per_dataset: pd.DataFrame) -> pd.DataFrame:
    """Add ``improvement`` (%) to per-(dataset, method) mean errors; datasets without Random are dropped."""
    random_error = per_dataset[per_dataset["method"] == RANDOM].set_index("dataset")["metric_error"]
    per_dataset = per_dataset.assign(random_error=per_dataset["dataset"].map(random_error))
    per_dataset = per_dataset[per_dataset["random_error"] > 0]
    return per_dataset.assign(
        improvement=(per_dataset["random_error"] - per_dataset["metric_error"]) / per_dataset["random_error"] * 100
    )


def improvement_over_random(df: pd.DataFrame) -> pd.Series:
    """Mean improvement over Random (%) of each method, as in the Pareto figure.

    ``df`` holds one row per result with ``dataset``, ``method`` and ``metric_error`` (runs without
    feature selection have no ``method`` and are ignored). The improvement on a dataset is
    ``(error_random - error_method) / error_random * 100`` of the errors averaged over budgets,
    downstream models and splits; the result is its mean over datasets (Random: 0).
    """
    df = df[df["method"].notna()]
    per_dataset = df.groupby(["dataset", "method"], as_index=False)["metric_error"].mean()
    return _improvement_per_dataset(per_dataset).groupby("method")["improvement"].mean()


def improvement_and_runtime(df: pd.DataFrame, num_features: Mapping[str, float]) -> pd.DataFrame:
    """Mean improvement over Random (%) and mean selection time per 1k features of each method.

    ``df`` holds one row per result with ``dataset``, ``method``, ``metric_error`` and
    ``feature_selection_fit_time`` (runs without feature selection have no ``method`` and are
    ignored); ``num_features`` maps ``dataset`` to its number of original features. Datasets
    without a Random result or a feature count are left out.

    Returns ``method``, ``improvement``, ``time_per_1k``, ``n_datasets`` and ``pareto_optimal``.
    """
    df = df[df["method"].notna()].copy()
    df["num_features"] = df["dataset"].map(num_features)
    df = df[df["num_features"] > 0]
    df["time_per_1k"] = df["feature_selection_fit_time"] / df["num_features"] * 1000
    per_dataset = df.groupby(["dataset", "method"], as_index=False)[["metric_error", "time_per_1k"]].mean()
    per_dataset = _improvement_per_dataset(per_dataset)
    summary = (
        per_dataset.groupby("method")
        .agg(
            improvement=("improvement", "mean"),
            time_per_1k=("time_per_1k", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    summary["pareto_optimal"] = pareto_front(summary)
    return summary.sort_values("improvement", ascending=False).reset_index(drop=True)


def pareto_front(summary: pd.DataFrame) -> pd.Series:
    """True for methods not dominated in (lower ``time_per_1k``, higher ``improvement``)."""
    t = summary["time_per_1k"].to_numpy()
    s = summary["improvement"].to_numpy()
    optimal = [
        not any(t[j] <= t[i] and s[j] >= s[i] and (t[j] < t[i] or s[j] > s[i]) for j in range(len(t)) if j != i)
        for i in range(len(t))
    ]
    return pd.Series(optimal, index=summary.index)


def pareto_figure(summary: pd.DataFrame, hidden: frozenset[str] = HIDDEN_IN_PLOT) -> Figure:
    """The paper's Pareto figure of :func:`improvement_and_runtime`'s ``summary``.

    The front is recomputed without the ``hidden`` methods, as in the paper. Labels are placed
    with ``adjustText`` when it is installed.
    """
    import matplotlib.pyplot as plt

    summary = summary[~summary["method"].isin(hidden)].reset_index(drop=True)
    summary = summary[summary["time_per_1k"] > 0].reset_index(drop=True)  # log x axis
    on_front = pareto_front(summary)
    front = summary[on_front].sort_values("time_per_1k", ascending=False)
    dominated = summary[~on_front]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xscale("log")
    ax.scatter(
        dominated["time_per_1k"],
        dominated["improvement"],
        s=60,
        color="lightgray",
        edgecolor="gray",
        alpha=0.8,
        label="Dominated Methods",
        zorder=1,
    )
    ax.scatter(
        front["time_per_1k"],
        front["improvement"],
        s=80,
        color=_ORANGE,
        edgecolors="black",
        label="Pareto Front",
        zorder=3,
    )
    ax.plot(front["time_per_1k"], front["improvement"], color=_ORANGE, linewidth=2, zorder=2)
    ax.annotate(
        "optimal",
        xy=(0.0, 1.0),
        xytext=(0.02, 0.89),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": _GREEN, "lw": 1.5},
        fontsize=12,
        fontweight="bold",
        color=_GREEN,
        ha="left",
        va="top",
    )
    texts = [
        ax.text(
            row.time_per_1k,
            row.improvement,
            row.method,
            fontsize=12,
            ha="center",
            va="center",
            **({"weight": "bold", "color": "black"} if on_front[i] else {"color": "dimgray"}),
        )
        for i, row in summary.iterrows()
    ]
    try:
        from adjustText import adjust_text
    except ImportError:  # labels stay centred on their points
        for text in texts:
            text.set_va("bottom")
    else:
        adjust_text(
            texts,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1, "alpha": 0.7},
            expand=(1.5, 1.5),
            min_arrow_len=9,
            zorder=3,
        )
    ax.set_xlabel("Feature Selection Time (s) per 1k Features")
    ax.set_ylabel("Improvement over Random (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for side in ("left", "bottom", "right", "top"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(0.3)
    fig.tight_layout()
    return fig
