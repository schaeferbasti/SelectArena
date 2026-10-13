"""Turn leaderboard tables into the website's ``website_leaderboard.csv`` format.

The equivalent of TabArena's ``tabarena.website.website_format``: it fixes the column names and
order the Space shows, and adds the method family (``Type``/``TypeName``). A method whose family is
not registered in :data:`selectarena.methods.registry.METHOD_FAMILIES` shows up as "Other"; the
``update-leaderboard`` skill checks for that before generating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from selectarena.methods.registry import METHOD_FAMILIES, PAPER_NAMES

FAMILY_EMOJI = {"baseline": "⚪", "filter": "🔍", "wrapper": "🔁", "embedded": "🌳", "other": "❓"}
FAMILY_NAME = {
    "baseline": "Baseline",
    "filter": "Filter",
    "wrapper": "Wrapper",
    "embedded": "Embedded",
    "other": "Other",
}
_PAPER_NAME_TO_CLASS = {name: cls for cls, name in PAPER_NAMES.items()}

#: Columns of ``website_leaderboard.csv``, in order. The Space formats them by these names.
LEADERBOARD_COLUMNS = [
    "Rank",
    "Type",
    "TypeName",
    "Method",
    "Elo",
    "Elo CI low",
    "Elo CI high",
    "Avg. rank",
    "Win rate (%)",
    "Improvability (%)",
    "Error vs. all features",
    "Stability",
    "Validity",
    "Selection time (s)",
    "Fallback rate (%)",
    "#Datasets",
    "#Comparisons",
]


def method_family(method: str) -> str:
    """Family key of a leaderboard method label (``MI`` or ``MI [nofallback]``), ``other`` if unknown."""
    base = method.split(" [")[0]
    return METHOD_FAMILIES.get(_PAPER_NAME_TO_CLASS.get(base, base), "other")


def unclassified_methods(methods: list[str]) -> list[str]:
    """Methods that would show up as "Other" on the leaderboard."""
    return sorted({m for m in methods if method_family(m) == "other"})


def format_leaderboard(
    elo: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    stability: pd.Series | None = None,
    validity: pd.Series | None = None,
    n_datasets: pd.Series | None = None,
) -> pd.DataFrame:
    """The website table of one subset, best Elo first.

    Args:
        elo: ``method``, ``elo``, ``elo_ci_low``, ``elo_ci_high`` (:func:`~selectarena.evaluation.elo.compute_elo`).
        summary: :func:`~selectarena.evaluation.leaderboard.aggregate` of the subset's scores.
        stability, validity, n_datasets: values per method (index), if available.
    """
    df = elo.merge(summary, on="method", how="outer")
    empty = pd.Series(dtype=float)
    df["stability"] = df["method"].map(stability if stability is not None else empty)
    df["validity"] = df["method"].map(validity if validity is not None else empty)
    df["n_datasets"] = df["method"].map(n_datasets if n_datasets is not None else empty)
    df = df.sort_values(["elo", "avg_rank"], ascending=[False, True], na_position="last").reset_index(drop=True)
    family = df["method"].map(method_family)
    out = pd.DataFrame(
        {
            "Rank": np.arange(1, len(df) + 1),
            "Type": family.map(FAMILY_EMOJI),
            "TypeName": family.map(FAMILY_NAME),
            "Method": df["method"],
            "Elo": df["elo"].round(0),
            "Elo CI low": df["elo_ci_low"].round(0),
            "Elo CI high": df["elo_ci_high"].round(0),
            "Avg. rank": df["avg_rank"].round(2),
            "Win rate (%)": (100 * df["win_rate"]).round(1),
            "Improvability (%)": (100 * df["improvability"]).round(1),
            "Error vs. all features": df["rel_error_vs_all_features"].round(3),
            "Stability": df["stability"].round(3),
            "Validity": df["validity"].round(3),
            "Selection time (s)": df["median_fs_time_s"].round(1),
            "Fallback rate (%)": (100 * df["fallback_rate"].astype(float)).round(1),
            "#Datasets": df["n_datasets"].astype("Int64"),
            "#Comparisons": df["n_comparisons"].astype("Int64"),
        }
    )
    return out[LEADERBOARD_COLUMNS]
