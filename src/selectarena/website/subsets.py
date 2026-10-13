"""The leaderboard's subsets: one leaderboard per task type, downstream model and budget.

Every subset is a folder ``tasks_<t>/models_<m>/budgets_<b>/`` of the website artifacts, the same
idea as TabArena's ``splits_*/tasks_*/datasets_*`` folders. The Space reads the folder of the
selection the viewer makes, so the folder names are the contract between the generator here and
the Space's ``data_loading.Subset``: change them in both places.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

ALL = "all"

#: Task-type axis: folder segment -> problem types it covers (``None``: every task).
TASK_TYPES: dict[str, set[str] | None] = {
    ALL: None,
    "classification": {"binary", "multiclass"},
    "binary": {"binary"},
    "multiclass": {"multiclass"},
    "regression": {"regression"},
}


def model_key(model: str) -> str:
    """Folder-safe key of a downstream model name (``TabICLv2`` -> ``tabiclv2``)."""
    return re.sub(r"[^a-z0-9]+", "-", str(model).lower()).strip("-")


@dataclass(frozen=True)
class Subset:
    """One cell of the leaderboard: a task type, a downstream model and a budget index."""

    tasks: str = ALL
    model: str = ALL
    """``all`` or a :func:`model_key`."""
    budget: str = ALL
    """``all`` or a budget index as a string."""

    @property
    def rel_path(self) -> str:
        return f"tasks_{self.tasks}/models_{self.model}/budgets_{self.budget}"

    @property
    def is_default(self) -> bool:
        return self.tasks == ALL and self.model == ALL and self.budget == ALL

    def filter(self, df: pd.DataFrame, *, problem_types: pd.Series | None = None) -> pd.DataFrame:
        """Rows of ``df`` inside this subset.

        ``df`` may lack the ``model`` or ``budget_index`` column (e.g. stability results, which do
        not depend on the downstream model); that axis is then not filtered. ``problem_types``
        maps ``dataset`` to its problem type for tables without a ``problem_type`` column.
        """
        mask = pd.Series(True, index=df.index)
        allowed = TASK_TYPES[self.tasks]
        if allowed is not None:
            types = df["problem_type"] if "problem_type" in df else df["dataset"].map(problem_types)
            mask &= types.isin(allowed)
        if self.model != ALL and "model" in df:
            mask &= df["model"].map(model_key) == self.model
        if self.budget != ALL and "budget_index" in df:
            mask &= df["budget_index"].astype("Int64").astype(str) == self.budget
        return df[mask.fillna(False)]


def enumerate_subsets(scores: pd.DataFrame) -> list[Subset]:
    """Every subset with results, the default (all, all, all) first."""
    task_types = [
        t for t, allowed in TASK_TYPES.items() if allowed is None or scores["problem_type"].isin(allowed).any()
    ]
    models = [ALL, *sorted({model_key(m) for m in scores["model"].dropna()})]
    budgets = [ALL, *[str(b) for b in sorted(scores["budget_index"].dropna().astype(int).unique())]]
    return [Subset(t, m, b) for t in task_types for m in models for b in budgets]
