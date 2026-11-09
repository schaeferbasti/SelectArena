"""Where and why feature selection methods win: ranks by dataset and model characteristics.

The leaderboard ranks methods over all datasets. This module breaks the ranks down:

* :func:`improvement_by_grouping`: the improvement over Random of every method per group of datasets
  (task type, number of samples, number of features, samples per feature), per downstream model or
  per budget;
* :func:`rank_sensitivity`: how a method's rank changes with the size of a dataset, as the rank
  correlation between the method's mean rank on a dataset and the dataset's number of samples,
  number of features or samples per feature, with a bootstrap confidence interval over datasets;
* :func:`improvement_by`: the improvement over Random of every method per downstream model, task type
  or domain (:func:`domain_labels`);
* :func:`measure_by` and :func:`measure_sensitivity`: the same per domain and dataset size for the
  stability and validity of the selections;
* :func:`selection_time`: the feature selection time of every method against the number of
  features, with the time limit; search methods reach it on wide datasets.

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
        labels = {i: _budget_label(i, indices) for i in indices}
        return pd.Categorical(values.map(lambda v: labels.get(int(v)) if pd.notna(v) else None), list(labels.values()))
    return pd.Categorical(values, sorted(values.dropna().unique()))


def _budget_label(index: int, indices: list[int]) -> str:
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


def measure_sensitivity(
    rows: pd.DataFrame, value: str, chars: pd.DataFrame, *, n_bootstrap: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """As :func:`rank_sensitivity`, for a per-dataset measure such as ``stability`` or ``validity``:
    the Spearman correlation between the dataset's size and the method's mean ``value`` on the
    dataset (``rows``: ``dataset``, ``method``, ``value``; positive: higher values on larger datasets)."""
    per_dataset = rows.groupby(["dataset", "method"])[value].mean().unstack("method")
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


def measure_by(rows: pd.DataFrame, value: str, groups: pd.Series) -> pd.DataFrame:
    """Mean ``value`` (e.g. ``stability``) of every method per group of datasets and over all
    datasets (group ``All``), each dataset counted once: ``group``, ``method``, ``value``.
    ``rows``: ``dataset``, ``method``, ``value``; ``groups``: dataset -> group, a categorical whose
    categories give the order (as :func:`domain_labels`)."""
    per_dataset = rows.groupby(["dataset", "method"], as_index=False)[value].mean()
    per_dataset["group"] = per_dataset["dataset"].map(groups.astype(str))
    order = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(groups.dropna().unique())
    frames = [per_dataset.groupby("method", as_index=False)[value].mean().assign(group="All")]
    for group in order:
        part = per_dataset[per_dataset["group"] == group]
        if len(part):
            frames.append(part.groupby("method", as_index=False)[value].mean().assign(group=str(group)))
    return pd.concat(frames, ignore_index=True)[["group", "method", value]]


def selection_time(df: pd.DataFrame, chars: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Median feature selection time of every method per dataset (over splits, budgets and downstream
    models), with the dataset's number of features; and the selection time limit (``None`` if unknown).
    """
    rows = df[df["method"].notna() & df["feature_selection_fit_time"].notna()]
    table = (
        rows.groupby(["method", "dataset_key"])["feature_selection_fit_time"]
        .median()
        .rename("time_s")
        .reset_index()
        .join(chars["num_features"], on="dataset_key")
        .dropna(subset=["num_features"])
        .rename(columns={"dataset_key": "dataset"})
    )
    limit = None
    if "feature_selection_time_limit" in rows and rows["feature_selection_time_limit"].notna().any():
        limit = float(rows["feature_selection_time_limit"].max())
    return table[["method", "dataset", "num_features", "time_s"]], limit


def binned_time(table: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Median selection time per method in ``n_bins`` bins of the number of features that hold about
    as many datasets each (bins of equal width would hold only a few datasets at the wide end):
    ``method``, ``features_low``, ``features_high``, ``features_mid`` (median number of features of
    the bin's datasets), ``time_s`` (median), ``n_datasets``."""
    columns = ["method", "features_low", "features_high", "features_mid", "time_s", "n_datasets"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    datasets = table.groupby("dataset")["num_features"].first().sort_values()
    bins = pd.Series(np.minimum(np.arange(len(datasets)) * n_bins // len(datasets), n_bins - 1), index=datasets.index)
    data = table.assign(bin=table["dataset"].map(bins))
    out = (
        data.groupby(["method", "bin"])
        .agg(time_s=("time_s", "median"), n_datasets=("dataset", "nunique"))
        .reset_index()
    )
    edges = datasets.groupby(bins).agg(["min", "max", "median"])
    out["features_low"] = out["bin"].map(edges["min"])
    out["features_high"] = out["bin"].map(edges["max"])
    out["features_mid"] = out["bin"].map(edges["median"])
    return out[columns]


def improvement_by(df: pd.DataFrame, column: str, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Improvement over Random (%, as on the leaderboard) of every method per value of ``column``
    (``model``: per downstream model; ``problem_type``: per task type) and over all of them (group
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


def domain_labels(datasets, domains: dict[str, str] | None = None, *, min_datasets: int = 1) -> pd.Series:
    """The domain of every dataset in ``datasets``, as shown on the leaderboard: capitalised, with
    the number of datasets (e.g. ``Financial (26)``); domains with fewer than ``min_datasets``
    datasets together as ``Other``. Datasets without a domain are left out. The categories are
    ordered by size, ``Other`` last."""
    if domains is None:
        from selectarena.datasets import DATASET_DOMAINS as domains
    names = pd.Series(list(dict.fromkeys(datasets)), dtype=object)
    raw = names.map(lambda d: domains.get(d)).dropna()
    counts = raw.value_counts()
    small = set(counts[counts < min_datasets].index)
    merged = raw.map(lambda d: "Other" if d in small else d[:1].upper() + d[1:])
    sizes = merged.value_counts()
    order = sorted(sizes.index, key=lambda d: (d == "Other", -sizes[d], d))
    labels = {d: f"{d} ({sizes[d]})" for d in order}
    out = merged.map(labels)
    out.index = names[raw.index].to_numpy()
    return pd.Series(pd.Categorical(out, categories=list(labels.values()), ordered=True), index=out.index)


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
