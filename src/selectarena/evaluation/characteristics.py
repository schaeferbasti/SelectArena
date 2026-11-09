"""Where and why feature selection methods win: ranks by dataset and model characteristics.

The leaderboard ranks methods over all datasets. This module breaks the ranks down:

* :func:`improvement_by_grouping`: the improvement over Random of every method per group of datasets
  (task type, number of samples, number of features, samples per feature), per downstream model or
  per budget;
* :func:`rank_sensitivity`: how a method's rank changes with the size of a dataset, as the rank
  correlation between the method's mean rank on a dataset and the dataset's number of samples,
  number of features or samples per feature, with a bootstrap confidence interval over datasets;
* :func:`improvement_by`: the improvement over Random of every method per downstream model or task
  type.

Improvements over Random are those of the leaderboard (:func:`selectarena.evaluation.pareto.improvement_over_random`);
ranks those of :func:`selectarena.evaluation.leaderboard.per_group_scores` (1 = lowest error among
the methods on one dataset split, downstream model and budget). Means are taken per dataset first,
so every dataset counts once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from selectarena.evaluation.leaderboard import events_per_variable


@dataclass(frozen=True)
class Grouping:
    """One way to split the results into groups: ``column`` of the scores (after
    :func:`add_characteristics`), and bins for a numeric characteristic."""

    key: str
    label: str
    column: str
    bins: tuple[float, ...] | None = None
    bin_labels: tuple[str, ...] | None = None


_INF = float("inf")
GROUPINGS: tuple[Grouping, ...] = (
    Grouping("task_type", "Task type", "problem_type"),
    Grouping("samples", "Samples", "num_samples", (0, 1_000, 10_000, _INF), ("< 1k", "1k–10k", "≥ 10k")),
    Grouping("features", "Features", "num_features", (0, 20, 100, _INF), ("< 20", "20–99", "≥ 100")),
    Grouping("epv", "Samples per feature", "epv", (0, 10, 100, _INF), ("< 10", "10–99", "≥ 100")),
    Grouping("model", "Downstream model", "model"),
    Grouping("budget", "Budget", "budget_index"),
)
#: Groupings that need the size of the datasets (from the stability or validity results).
SIZE_GROUPINGS = frozenset({"samples", "features", "epv"})
#: Characteristics of :func:`rank_sensitivity`: column -> label.
SIZE_CHARACTERISTICS = {"num_samples": "Samples", "num_features": "Features", "epv": "Samples per feature"}
TASK_TYPE_LABELS = {"binary": "Binary", "multiclass": "Multiclass", "regression": "Regression"}


def dataset_characteristics(rows: list[pd.DataFrame | None], problem_types: pd.Series) -> pd.DataFrame:
    """Samples, features and samples per feature (EPV) of every dataset, from the rows of the
    stability or validity results (``dataset``, ``num_samples``, ``num_features``,
    ``min_samples_per_class``). The first frame wins for a dataset in several (pass the validity
    rows first: the per-class counts of the stability results are unreliable).
    """
    columns = ["dataset", "num_samples", "num_features", "min_samples_per_class"]
    frames = [r[[c for c in columns if c in r]] for r in rows if r is not None and not r.empty]
    if not frames:
        return pd.DataFrame(columns=["num_samples", "num_features", "epv"], index=pd.Index([], name="dataset"))
    chars = pd.concat(frames).dropna(subset=["num_samples", "num_features"]).groupby("dataset").first()
    if "min_samples_per_class" not in chars:
        chars["min_samples_per_class"] = np.nan
    chars["epv"] = events_per_variable(chars.reset_index(), problem_types).to_numpy()
    return chars[["num_samples", "num_features", "epv"]].astype(float)


def add_characteristics(scores: pd.DataFrame, chars: pd.DataFrame) -> pd.DataFrame:
    """``scores`` (by ``dataset_key``) with the columns of :func:`dataset_characteristics`."""
    return scores.join(chars, on="dataset_key")


def _group_values(scores: pd.DataFrame, grouping: Grouping) -> pd.Series:
    values = scores[grouping.column]
    if grouping.bins is not None:
        return pd.cut(values, list(grouping.bins), labels=list(grouping.bin_labels), right=False)
    if grouping.key == "task_type":
        order = [k for k in TASK_TYPE_LABELS if k in set(values)]
        return pd.Categorical(values.map(TASK_TYPE_LABELS), [TASK_TYPE_LABELS[k] for k in order], ordered=True)
    if grouping.key == "budget":
        indices = sorted(int(v) for v in values.dropna().unique())
        labels = {i: budget_label(i, indices) for i in indices}
        return pd.Categorical(values.map(lambda v: labels.get(int(v)) if pd.notna(v) else None), list(labels.values()))
    return pd.Categorical(values, sorted(values.dropna().unique()))


def budget_label(index: int, indices: list[int]) -> str:
    """``1 (smallest)``, ``2``, ..., ``5 (largest)``: the label of budget ``index`` among ``indices``."""
    label = str(index + 1)
    if index == indices[0]:
        return f"{label} (smallest)"
    if index == indices[-1]:
        return f"{label} (largest)"
    return label


def available_groupings(scores: pd.DataFrame) -> list[Grouping]:
    """The groupings ``scores`` can be split by: at least two groups, and the dataset sizes known."""
    out = []
    for grouping in GROUPINGS:
        if grouping.column not in scores or scores[grouping.column].isna().all():
            continue
        if pd.Series(_group_values(scores, grouping)).nunique() >= 2:
            out.append(grouping)
    return out


def improvement_by_grouping(results: pd.DataFrame, grouping: Grouping) -> pd.DataFrame:
    """Improvement over Random (%, as on the leaderboard) of every method per group, and over all
    results (group ``All``).

    ``results`` holds one row per result (``dataset_key``, ``method``, ``metric_error`` and the
    grouping's column, after :func:`add_characteristics`). Returns one row per group and method:
    ``group`` (ordered as the groups, ``All`` first), ``method``, ``improvement``, ``n_datasets``
    (of the group); best method first within a group.
    """
    from selectarena.evaluation.pareto import improvement_over_random

    data = results.assign(group=_group_values(results, grouping), dataset=results["dataset_key"])
    data = data[data["method"].notna()]
    groups = (
        list(data["group"].cat.categories) if hasattr(data["group"], "cat") else sorted(data["group"].dropna().unique())
    )
    frames = []
    for group in ["All", *groups]:
        rows = data if group == "All" else data[data["group"] == group]
        values = improvement_over_random(rows)
        if len(values):
            frames.append(
                pd.DataFrame(
                    {
                        "group": str(group),
                        "method": values.index,
                        "improvement": values.to_numpy(),
                        "n_datasets": rows["dataset_key"].nunique(),
                    }
                ).sort_values("improvement", ascending=False)
            )
    if not frames:
        return pd.DataFrame(columns=["group", "method", "improvement", "n_datasets"])
    return pd.concat(frames, ignore_index=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def rank_sensitivity(
    scores: pd.DataFrame, chars: pd.DataFrame, *, n_bootstrap: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """How each method's rank changes with the size of the datasets.

    For every method and size characteristic (:data:`SIZE_CHARACTERISTICS`): the Spearman
    correlation between the dataset's characteristic and the method's *negated* mean rank on the
    dataset (positive: the method ranks better on larger values), with a 95% percentile bootstrap
    interval over datasets.
    """
    per_dataset = -scores.groupby(["dataset_key", "method"])["rank"].mean().unstack("method")
    return _size_correlation(per_dataset, chars, n_bootstrap, seed)


def _size_correlation(per_dataset: pd.DataFrame, chars: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Spearman correlation (and 95% bootstrap interval over datasets) between every column of
    ``per_dataset`` (datasets x methods) and every size characteristic of the datasets."""
    rng = np.random.default_rng(seed)
    rows = []
    for column, label in SIZE_CHARACTERISTICS.items():
        if column not in chars:
            continue
        x_all = chars[column].reindex(per_dataset.index)
        for method in per_dataset.columns:
            ok = x_all.notna() & per_dataset[method].notna()
            x, y = x_all[ok].to_numpy(float), per_dataset.loc[ok, method].to_numpy(float)
            if len(x) < 5:
                continue
            rho = _spearman(x, y)
            boot = [_spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_bootstrap))]
            boot = np.array([b for b in boot if np.isfinite(b)])
            lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
            rows.append(
                {
                    "characteristic": label,
                    "method": method,
                    "rho": rho,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "n_datasets": int(ok.sum()),
                }
            )
    return pd.DataFrame(rows, columns=["characteristic", "method", "rho", "ci_low", "ci_high", "n_datasets"])


def improvement_by(df: pd.DataFrame, column: str, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Improvement over Random (%, as on the leaderboard) of every method per value of ``column``
    (``model``: per downstream model; ``problem_type``: per task type; ``budget_index``: per feature
    budget) and over all of them (group
    ``All``): ``group``, ``method``, ``improvement``. ``df`` holds the results with a ``dataset``
    column; ``labels`` names and orders the groups (others are left out)."""
    from selectarena.evaluation.pareto import improvement_over_random

    present = set(df[column].dropna())
    groups = [g for g in labels if g in present] if labels else sorted(present)
    frames = []
    for group in ["All", *groups]:
        rows = df if group == "All" else df[df[column] == group]
        values = improvement_over_random(rows)
        if len(values):
            name = group if group == "All" or not labels else labels[group]
            frames.append(pd.DataFrame({"group": name, "method": values.index, "improvement": values.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["group", "method", "improvement"])
    return pd.concat(frames, ignore_index=True)


def ranking_agreement(table: pd.DataFrame, value: str = "improvement") -> float:
    """Mean Spearman correlation between the method rankings of every pair of groups (``table`` from
    :func:`improvement_by`); 1 means every group ranks the methods the same."""
    wide = table[table["group"] != "All"].pivot_table(index="method", columns="group", values=value)
    models = list(wide.columns)
    pairs = [
        wide[[a, b]].dropna().corr(method="spearman").iloc[0, 1]
        for i, a in enumerate(models)
        for b in models[i + 1 :]
        if wide[[a, b]].dropna().shape[0] >= 3
    ]
    return float(np.nanmean(pairs)) if pairs else float("nan")
