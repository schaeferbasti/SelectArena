"""Leaderboard tables from SelectArena results.

Feature selection methods compete within a *comparison group*: one dataset split, one downstream
model configuration and one budget. Within each group every method is ranked by the downstream
``metric_error``; the leaderboard aggregates over groups:

* ``elo``: Bradley-Terry Elo with Random at 1000 and a 95% cluster-bootstrap interval
  (:mod:`selectarena.evaluation.elo`, the paper's main metric);
* ``avg_rank``: mean rank in the groups (1 is best, ties share the mean rank);
* ``win_rate``: share of pairwise comparisons won against the other methods (ties count half);
* ``improvability``: mean of ``(error - best error) / error``, how far a method is from the best
  method of the group (0 is best);
* ``rel_error_vs_all_features``: mean ratio of the error to the error of the same model trained
  on all features (pipeline ``default``), when that run exists (below 1 means the selection helped);
* ``median_fs_time_s`` and ``fallback_rate`` (share of runs where the random fallback filled the
  budget).

The selection-only benchmark gives ``stability`` (Nogueira et al., 2018, over bootstrap
resamples) and ``validity`` (share of selected features that are original, not injected noise).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import combinations

import numpy as np
import pandas as pd

from selectarena.datasets import dataset_name
from selectarena.evaluation.elo import compute_elo
from selectarena.methods.registry import LEGACY_NAMES, PAPER_NAMES

_CONFIG_NAME = re.compile(r"^(?P<model>.+?)_(?P<config>[cr]\d+)(?:_(?P<pipeline>.+))?_BAG_L1$")
GROUP_COLUMNS = ["dataset", "split_idx", "model", "config", "budget_index"]
#: Noise level whose validity shades the main figure (the paper's Figure 1 values match 100% noise).
VALIDITY_NOISE_MAIN_FIGURE = 1.0


def parse_method_name(name: str) -> dict:
    """Split a result's method name, e.g. ``LightGBM_c1_FSBench__MIFeatureSelector__5__0__lgbm__3600_BAG_L1``.

    A name without a pipeline (``LightGBM_c1_BAG_L1``, how the fork named runs on all features) is
    the ``default`` pipeline.
    """
    match = _CONFIG_NAME.match(name)
    if match is None:
        return {"model": name, "config": None, "pipeline": None}
    parsed = match.groupdict()
    parsed["pipeline"] = parsed["pipeline"] or "default"
    return parsed


def method_class(pipeline: str | None) -> str | None:
    """Current class name of an FSBench pipeline's method (fork-era names are translated)."""
    if not isinstance(pipeline, str) or not pipeline.startswith("FSBench__"):
        return None
    name = pipeline.split("__")[1]
    return LEGACY_NAMES.get(name, name)


def method_label(pipeline: str) -> str | None:
    """Display name of an FSBench pipeline's method plus any ablation tokens, ``None`` for ``default``."""
    cls = method_class(pipeline)
    if cls is None:
        return None
    parts = pipeline.split("__")
    label = PAPER_NAMES.get(cls, cls)
    ablations = [p for p in parts[6:] if p]
    return f"{label} [{', '.join(ablations)}]" if ablations else label


def prepare_results(results: pd.DataFrame, *, task_names: Mapping[int, str] | None = None) -> pd.DataFrame:
    """Add ``model``, ``config``, ``pipeline``, ``method``, ``budget_index`` and ``dataset`` columns.

    ``task_names`` maps task ids to dataset names beyond the containers listed in
    :mod:`selectarena.datasets` (see :func:`~selectarena.datasets.task_names_from_ids`).
    """
    df = results.copy()
    parsed = pd.DataFrame([parse_method_name(n) for n in df["experiment_method_name_string"]], index=df.index)
    df = df.drop(columns=[c for c in parsed.columns if c in df.columns]).join(parsed)
    df["method"] = df["pipeline"].map(method_label)
    df["method_class"] = df["pipeline"].map(method_class)
    if "feature_selection_method" in df.columns:
        # Results whose name carries no FSBench pipeline: take the method from the result itself.
        recorded = df["feature_selection_method"].map(
            lambda v: LEGACY_NAMES.get(v, v) if isinstance(v, str) and v else None
        )
        missing = df["method"].isna() & recorded.notna()
        df.loc[missing, "method_class"] = recorded[missing]
        df.loc[missing, "method"] = recorded[missing].map(lambda c: PAPER_NAMES.get(c, c))
    # Results identify the task by name, container path or TabArena task id; use the dataset name.
    df["dataset"] = (df["name"] if "name" in df.columns else df["dataset"]).map(
        lambda task: dataset_name(task, task_names)
    )
    df["budget_index"] = df["feature_selection_budget_index"].astype("Int64")
    df["used_fallback"] = df.get("random_feature_names", pd.Series(index=df.index, dtype=object)).map(
        lambda v: bool(_as_list(v))
    )
    return df


#: ``np.float64(0.5)``-style reprs (numpy >= 2) and ``nan`` / ``inf``, which ``ast.literal_eval`` rejects.
_NUMPY_SCALAR = re.compile(r"np\.\w+\(([^()]*)\)")
_NON_LITERAL = re.compile(r"\b(?:nan|inf)\b")


def _as_list(value) -> list:
    """A list stored in a CSV cell (its Python repr), ``[]`` if it is missing or unreadable."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        cleaned = _NON_LITERAL.sub("None", _NUMPY_SCALAR.sub(r"\1", value))
        for text in (value, cleaned):
            try:
                return list(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                continue
    return []


def feature_names(value) -> list:
    """Selected features as names. Current runs store names; runs of the fork before commit
    ``0dcac02a`` stored the selector's ``{"feature": name, "value": score}`` entries."""
    return [item.get("feature") if isinstance(item, dict) else item for item in _as_list(value)]


def _pairwise_wins(group: pd.DataFrame) -> pd.Series:
    errors = group.set_index("method")["metric_error"]
    wins = pd.Series(0.0, index=errors.index)
    for a, b in combinations(errors.index, 2):
        if errors[a] < errors[b]:
            wins[a] += 1
        elif errors[b] < errors[a]:
            wins[b] += 1
        else:
            wins[a] += 0.5
            wins[b] += 0.5
    return wins / max(len(errors) - 1, 1)


def per_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (comparison group, method) with rank, win rate, improvability and error ratio."""
    fs = df[df["method"].notna()].copy()
    # A method can appear twice in a group only if a pipeline was rerun; keep the mean error.
    fs = fs.groupby([*GROUP_COLUMNS, "problem_type", "method"], as_index=False, dropna=False).agg(
        metric_error=("metric_error", "mean"),
        fs_time_s=("feature_selection_fit_time", "mean"),
        used_fallback=("used_fallback", "max"),
    )
    keys = GROUP_COLUMNS
    fs["rank"] = fs.groupby(keys, dropna=False)["metric_error"].rank(method="average")
    best = fs.groupby(keys, dropna=False)["metric_error"].transform("min")
    fs["improvability"] = np.where(fs["metric_error"] > 0, (fs["metric_error"] - best) / fs["metric_error"], 0.0)
    fs["win_rate"] = (
        fs.groupby(keys, dropna=False, group_keys=False)
        .apply(lambda g: pd.Series(_pairwise_wins(g).values, index=g.index), include_groups=False)
        .astype(float)
    )

    baseline = (
        df[df["pipeline"] == "default"].groupby(["dataset", "split_idx", "model", "config"])["metric_error"].mean()
    )
    if len(baseline):
        fs = fs.join(baseline.rename("error_all_features"), on=["dataset", "split_idx", "model", "config"])
        with np.errstate(divide="ignore", invalid="ignore"):
            fs["rel_error_vs_all_features"] = fs["metric_error"] / fs["error_all_features"]
        fs.loc[~np.isfinite(fs["rel_error_vs_all_features"]), "rel_error_vs_all_features"] = np.nan
    else:
        fs["rel_error_vs_all_features"] = np.nan
    return fs


def aggregate(scores: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Leaderboard over the per-group scores, optionally split by extra columns (e.g. ``budget_index``)."""
    by = by or []
    table = (
        scores.groupby([*by, "method"], dropna=False)
        .agg(
            avg_rank=("rank", "mean"),
            win_rate=("win_rate", "mean"),
            improvability=("improvability", "mean"),
            rel_error_vs_all_features=("rel_error_vs_all_features", "mean"),
            median_fs_time_s=("fs_time_s", "median"),
            fallback_rate=("used_fallback", "mean"),
            n_comparisons=("rank", "size"),
        )
        .reset_index()
        .sort_values([*by, "avg_rank"])
    )
    return table.round(4)


def nogueira_stability(selections: list[list[str]], features: list[str]) -> float:
    """Stability of feature selections over repeats (Nogueira, Sechidis & Brown, JMLR 2018)."""
    if len(selections) < 2 or not features:
        return float("nan")
    index = {f: i for i, f in enumerate(features)}
    Z = np.zeros((len(selections), len(features)))
    for m, selected in enumerate(selections):
        for f in selected:
            if f in index:
                Z[m, index[f]] = 1
    M, d = Z.shape
    p = Z.mean(axis=0)
    kbar = p.sum()
    denom = (kbar / d) * (1 - kbar / d)
    if denom == 0:
        return float("nan")
    return float(1 - (M / (M - 1)) * np.mean(p * (1 - p)) / denom)


def _budget_index(method_name: str) -> int | None:
    """The budget index of an ``FSBench__*`` pipeline name."""
    parts = method_name.split("__")
    return int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None


def task_dataset(task_id: str) -> str:
    """Dataset name of a TabArena ``task_id_str`` (``UserTask|<id>|<dataset>/<uuid>[|...]``)."""
    return dataset_name(task_id)


#: One extra-benchmark run: older runners wrote every row twice, so rows are deduplicated on these.
_EXTRA_RUN_KEYS = ["method", "data_foundry_task_id", "mode", "mode_kwargs", "repeat"]


def _drop_duplicate_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=[c for c in _EXTRA_RUN_KEYS if c in df.columns], keep="last")


def stability_per_task(stability: pd.DataFrame) -> pd.DataFrame:
    """Nogueira stability of each (method, task) over its bootstrap repeats."""
    rows = []
    stability = _drop_duplicate_runs(stability)
    for (method_name, task), group in stability.groupby(["method", "data_foundry_task_id"]):
        # A missing selection is a failed run, not an empty selection.
        selections = [feature_names(s) for s in group["selected_features"].dropna()]
        features = feature_names(group["original_features"].iloc[0])
        rows.append(
            {
                "method": method_label(method_name),
                "budget_index": _budget_index(method_name),
                "task": task,
                "dataset": task_dataset(task),
                "stability": nogueira_stability(selections, features),
            }
        )
    return pd.DataFrame(rows, columns=["method", "budget_index", "task", "dataset", "stability"])


def validity_per_run(validity: pd.DataFrame) -> pd.DataFrame:
    """Validity (share of the budget filled with original features) of each validity run."""
    df = _drop_duplicate_runs(validity).copy()
    df["noise"] = df["mode_kwargs"].map(lambda v: ast.literal_eval(v).get("noise") if isinstance(v, str) else None)
    df["budget_index"] = df["method"].map(_budget_index)
    df["dataset"] = df["data_foundry_task_id"].map(task_dataset)

    def _validity(row) -> float:
        if not isinstance(row["selected_features"], (str, list)):
            return np.nan  # failed run
        selected = feature_names(row["selected_features"])
        original = set(feature_names(row["original_features"]))
        return len([f for f in selected if f in original]) / row["max_features"] if row["max_features"] else np.nan

    df["validity"] = df.apply(_validity, axis=1)
    df["method"] = df["method"].map(method_label)
    return df[["method", "budget_index", "noise", "dataset", "validity"]]


def stability_table(stability: pd.DataFrame) -> pd.DataFrame:
    """Mean stability per method (and budget) from the stability benchmark's results."""
    if stability.empty:
        return pd.DataFrame()
    return _summarize(stability_per_task(stability), "stability")


def validity_table(validity: pd.DataFrame) -> pd.DataFrame:
    """Mean validity (share of selected features that are original) per method, budget and noise level."""
    if validity.empty:
        return pd.DataFrame()
    return _summarize(validity_per_run(validity), "validity", extra=["noise"])


def _summarize(df: pd.DataFrame, column: str, extra: list[str] | None = None) -> pd.DataFrame:
    by = [*(extra or [])]
    overall = df.groupby([*by, "method"])[column].mean().rename(column).reset_index()
    per_budget = (
        df.pivot_table(index=[*by, "method"], columns="budget_index", values=column, aggfunc="mean")
        .add_prefix(f"{column}_budget_")
        .reset_index()
    )
    out = overall.merge(per_budget, on=[*by, "method"]).sort_values([*by, column], ascending=[True] * len(by) + [False])
    return out.round(4)


def main_figure_table(
    elo: pd.DataFrame,
    stability: pd.DataFrame | None,
    validity: pd.DataFrame | None,
    *,
    validity_noise: float | None = VALIDITY_NOISE_MAIN_FIGURE,
) -> pd.DataFrame:
    """Elo with its interval plus mean stability and validity per method (the main figure's data)."""
    main = elo.copy()
    if stability is not None and not stability.empty:
        main = main.merge(stability[["method", "stability"]], on="method", how="left")
    if validity is not None and not validity.empty:
        v = validity if validity_noise is None else validity[validity["noise"] == validity_noise]
        main = main.merge(v.groupby("method", as_index=False)["validity"].mean(), on="method", how="left")
    return main


def build_leaderboard(
    results: pd.DataFrame,
    *,
    stability: pd.DataFrame | None = None,
    validity: pd.DataFrame | None = None,
    benchmark_name: str = "SelectArena",
    n_bootstrap: int = 1000,
    validity_noise: float | None = VALIDITY_NOISE_MAIN_FIGURE,
) -> dict[str, pd.DataFrame | dict]:
    """All tables the leaderboard Space shows, keyed by file stem (plus ``meta``).

    ``main_figure`` holds the Elo ratings with stability and validity (validity at
    ``validity_noise``, or averaged over noise levels when ``None``).
    """
    df = prepare_results(results)
    scores = per_group_scores(df)
    elo = compute_elo(scores, n_bootstrap=n_bootstrap)
    overall = elo.merge(aggregate(scores), on="method", how="outer").sort_values("elo", ascending=False)
    tables: dict[str, pd.DataFrame | dict] = {
        "overall": overall,
        "by_budget": aggregate(scores, ["budget_index"]),
        "by_model": aggregate(scores, ["model"]),
        "by_problem_type": aggregate(scores, ["problem_type"]),
        "per_dataset": scores.pivot_table(index="method", columns="dataset", values="rank", aggfunc="mean")
        .round(2)
        .reset_index(),
    }
    if stability is not None and not stability.empty:
        tables["stability"] = stability_table(stability)
    if validity is not None and not validity.empty:
        tables["validity"] = validity_table(validity)
    tables["main_figure"] = main_figure_table(
        elo, tables.get("stability"), tables.get("validity"), validity_noise=validity_noise
    )
    tables["meta"] = {
        "benchmark_name": benchmark_name,
        "updated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "n_results": len(df),
        "n_datasets": int(df["dataset"].nunique()),
        "n_methods": int(scores["method"].nunique()),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "elo_bootstrap_samples": n_bootstrap,
        "validity_noise_main_figure": validity_noise,
    }
    return tables
