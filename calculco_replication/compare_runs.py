"""Compare the results of the same jobs run twice, e.g. with tabarena_fork_fs and with SelectArena.

    python compare_runs.py --fork <fork results.csv> --selectarena <SelectArena results.csv> --out report/
    python compare_runs.py --fork <run A> --selectarena <run B> --labels freiburg,calculco

``--fork`` is the reference run, ``--selectarena`` the run checked against it; ``--labels`` names
them in the report and in the column names of the CSVs (default ``fork,selectarena``).

Run it with SelectArena's environment. Runs are matched on dataset, split, model, config and
pipeline (``FSBench__<method>__5__<budget index>__lgbm__3600``, or ``default`` for all features).
For every matched run it compares the budget (``max_features``), the selected features, the
downstream test error, the random fallback and the selection time; then it compares the two
leaderboards (average rank and win rate per method).

The ``default`` runs (LightGBM on all features) contain no feature selection: their error
difference is what the library versions alone change, and it sets the tolerance for the rest.

Writes ``runs.csv`` (one row per run), ``methods.csv`` (one row per method), ``leaderboard.csv``
and ``report.md`` to ``--out`` and prints the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from selectarena.datasets import dataset_name
from selectarena.evaluation.leaderboard import (
    _as_list,
    aggregate,
    feature_names,
    per_group_scores,
    prepare_results,
)

KEY = ["dataset", "split_idx", "model", "config", "pipeline"]
VALUES = ["max_features", "selected", "n_random", "metric_error", "fs_time_s"]


def load(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The prepared results table and one row per run with the compared values."""
    df = prepare_results(pd.read_csv(path))
    # Both versions derive the task id from the container path; the task names differ (Task-<id> / slug).
    df["dataset"] = df["tid"].map(lambda t: dataset_name(int(t)))
    df["selected"] = df["selected_features"].map(lambda v: frozenset(feature_names(v)))
    df["n_random"] = df["random_feature_names"].map(lambda v: len(_as_list(v)))
    df["fs_time_s"] = df["feature_selection_fit_time"]
    runs = df.groupby(KEY, dropna=False).agg(
        method=("method", "first"),
        budget_index=("budget_index", "first"),
        problem_type=("problem_type", "first"),
        metric=("metric", "first"),
        n_results=("metric_error", "size"),
        **{v: (v, "first") for v in VALUES if v != "metric_error"},
        metric_error=("metric_error", "mean"),
    )
    return df, runs.reset_index()


def jaccard(a: frozenset, b: frozenset) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare_runs(fork: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = fork.merge(
        new, on=KEY, how="outer", suffixes=("_fork", "_sa"), indicator=True
    )
    for column in ("method", "budget_index", "problem_type", "metric"):
        merged[column] = merged[f"{column}_fork"].combine_first(merged[f"{column}_sa"])
    merged["method"] = merged["method"].fillna("All features (default)")
    both = merged["_merge"] == "both"
    fs = both & (merged["pipeline"] != "default")
    merged["same_budget"] = np.where(
        fs, merged["max_features_fork"] == merged["max_features_sa"], np.nan
    )
    merged["same_selection"] = np.where(
        fs, merged["selected_fork"] == merged["selected_sa"], np.nan
    )
    merged["jaccard"] = [
        jaccard(a, b) if ok else np.nan
        for a, b, ok in zip(merged["selected_fork"], merged["selected_sa"], fs)
    ]
    merged["error_diff"] = merged["metric_error_sa"] - merged["metric_error_fork"]
    scale = (
        merged[["metric_error_fork", "metric_error_sa"]]
        .abs()
        .max(axis=1)
        .replace(0, np.nan)
    )
    merged["rel_error_diff"] = (
        (merged["error_diff"].abs() / scale).fillna(0.0).where(both)
    )
    merged["missing_in"] = merged["_merge"].map(
        {"left_only": "selectarena", "right_only": "fork", "both": ""}
    )
    for side in ("fork", "sa"):
        merged[f"selected_{side}"] = merged[f"selected_{side}"].map(
            lambda s: sorted(s) if isinstance(s, frozenset) else None
        )
    columns = [
        *KEY[:-1],
        "method",
        "budget_index",
        "pipeline",
        "problem_type",
        "metric",
        "missing_in",
        "max_features_fork",
        "max_features_sa",
        "same_budget",
        "same_selection",
        "jaccard",
        "metric_error_fork",
        "metric_error_sa",
        "error_diff",
        "rel_error_diff",
        "n_random_fork",
        "n_random_sa",
        "fs_time_s_fork",
        "fs_time_s_sa",
        "selected_fork",
        "selected_sa",
    ]
    return (
        merged[columns]
        .sort_values(["method", "dataset", "split_idx", "budget_index"])
        .reset_index(drop=True)
    )


def summarize_methods(runs: pd.DataFrame) -> pd.DataFrame:
    matched = runs[runs["missing_in"] == ""]
    table = matched.groupby("method").agg(
        runs=("pipeline", "size"),
        same_budget=("same_budget", "mean"),
        identical_selection=("same_selection", "mean"),
        mean_jaccard=("jaccard", "mean"),
        median_rel_error_diff=("rel_error_diff", "median"),
        max_rel_error_diff=("rel_error_diff", "max"),
        fallback_runs_fork=("n_random_fork", lambda s: int((s > 0).sum())),
        fallback_runs_sa=("n_random_sa", lambda s: int((s > 0).sum())),
        median_fs_time_s_fork=("fs_time_s_fork", "median"),
        median_fs_time_s_sa=("fs_time_s_sa", "median"),
    )
    missing = (
        runs[runs["missing_in"] != ""]
        .groupby(["method", "missing_in"])
        .size()
        .unstack(fill_value=0)
    )
    for side in ("fork", "selectarena"):
        table[f"missing_in_{side}"] = missing.get(side, 0)
    return table.fillna(
        {"missing_in_fork": 0, "missing_in_selectarena": 0}
    ).reset_index()


def flag(methods: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Mark the methods whose runs differ by more than library versions explain."""
    reasons = []
    for row in methods.itertuples():
        why = []
        if row.missing_in_fork or row.missing_in_selectarena:
            why.append("missing runs")
        if row.method != "All features (default)":
            if row.same_budget < 1:
                why.append("different budget")
            if row.mean_jaccard < 0.9 or row.identical_selection < 0.5:
                why.append(f"selections differ (mean Jaccard {row.mean_jaccard:.2f})")
            if row.fallback_runs_fork != row.fallback_runs_sa:
                why.append("fallback differs")
        if row.median_rel_error_diff > tolerance:
            why.append(f"error differs by {100 * row.median_rel_error_diff:.1f}%")
        reasons.append("; ".join(why))
    methods = methods.copy()
    methods["check"] = reasons
    return methods


def compare_leaderboards(
    fork_df: pd.DataFrame, new_df: pd.DataFrame, runs: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Average rank and win rate per method on each side, over the runs both sides have."""
    matched = runs.loc[runs["missing_in"] == "", KEY]
    tables = []
    for side, df in (("fork", fork_df), ("sa", new_df)):
        df = df.merge(matched, on=KEY)
        table = aggregate(per_group_scores(df))[
            ["method", "avg_rank", "win_rate"]
        ].set_index("method")
        tables.append(table.add_suffix(f"_{side}"))
    board = tables[0].join(tables[1], how="outer")
    board["rank_diff"] = board["avg_rank_sa"] - board["avg_rank_fork"]
    rho = board[["avg_rank_fork", "avg_rank_sa"]].corr(method="spearman").iloc[0, 1]
    return board.sort_values("avg_rank_fork").reset_index(), float(rho)


def pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}%"


def num(value: float, digits: int) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def report(
    runs: pd.DataFrame,
    methods: pd.DataFrame,
    board: pd.DataFrame,
    rho: float,
    tolerance: float,
    labels: tuple[str, str] = ("fork", "selectarena"),
) -> str:
    a, b = labels
    matched = runs[runs["missing_in"] == ""]
    fs = matched[matched["pipeline"] != "default"]
    lines = [
        f"# {a} vs. {b}",
        "",
        (
            f"- Matched runs: {len(matched)} ({len(fs)} with feature selection); "
            f"missing in {b}: {(runs['missing_in'] == 'selectarena').sum()}, "
            f"missing in {a}: {(runs['missing_in'] == 'fork').sum()}."
        ),
        f"- Same budget (max_features): {pct(fs['same_budget'].mean())} of the runs.",
        (
            f"- Identical selected features: {pct(fs['same_selection'].mean())} of the runs "
            f"(mean Jaccard {fs['jaccard'].mean():.3f})."
        ),
        f"- Error tolerance (from LightGBM on all features): {pct(tolerance)} relative difference.",
        f"- Leaderboard: Spearman correlation of the average ranks {rho:.3f}.",
        "",
        "## Per method",
        "",
        (
            "| Method | Runs | Identical selection | Mean Jaccard | Median / max rel. error diff | "
            f"Fallback runs {a} / {b} | Median FS time {a} / {b} (s) | Check |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in methods.itertuples():
        lines.append(
            f"| {r.method} | {r.runs} | {pct(r.identical_selection)} | {num(r.mean_jaccard, 3)} | "
            f"{pct(r.median_rel_error_diff)} / {pct(r.max_rel_error_diff)} | "
            f"{r.fallback_runs_fork} / {r.fallback_runs_sa} | "
            f"{num(r.median_fs_time_s_fork, 1)} / {num(r.median_fs_time_s_sa, 1)} | {r.check or 'ok'} |"
        )
    lines += [
        "",
        "## Leaderboard (all datasets, splits, budgets)",
        "",
        f"| Method | Avg. rank {a} | Avg. rank {b} | Win rate {a} | Win rate {b} |",
        "|---|---|---|---|---|",
    ]
    for r in board.itertuples():
        lines.append(
            f"| {r.method} | {r.avg_rank_fork:.2f} | {r.avg_rank_sa:.2f} | {r.win_rate_fork:.3f} | {r.win_rate_sa:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fork", required=True, help="results.csv of the fork run (fork_side.py eval)."
    )
    parser.add_argument(
        "--selectarena",
        required=True,
        help="results.csv of the SelectArena run (run_benchmark.py eval).",
    )
    parser.add_argument("--out", default="report")
    parser.add_argument(
        "--labels", default="fork,selectarena", help="Names of the two runs."
    )
    parser.add_argument(
        "--min-tolerance",
        type=float,
        default=0.01,
        help="Smallest relative error difference that is flagged (default 1%%).",
    )
    args = parser.parse_args()

    fork_df, fork_runs = load(args.fork)
    new_df, new_runs = load(args.selectarena)
    runs = compare_runs(fork_runs, new_runs)
    methods = summarize_methods(runs)
    default = runs[(runs["pipeline"] == "default") & (runs["missing_in"] == "")][
        "rel_error_diff"
    ]
    tolerance = (
        max(args.min_tolerance, 2 * float(default.median()))
        if len(default)
        else args.min_tolerance
    )
    methods = flag(methods, tolerance)
    board, rho = compare_leaderboards(fork_df, new_df, runs)

    labels = tuple(args.labels.split(","))
    if len(labels) != 2:
        raise SystemExit("--labels takes two names, e.g. freiburg,calculco")

    def named(table: pd.DataFrame) -> pd.DataFrame:
        table = table.rename(
            columns=lambda c: (
                c.replace("_fork", f"_{labels[0]}")
                .replace("_selectarena", f"_{labels[1]}")
                .replace("_sa", f"_{labels[1]}")
            )
        )
        if "missing_in" in table:
            table["missing_in"] = table["missing_in"].map(
                {"fork": labels[0], "selectarena": labels[1], "": ""}
            )
        return table

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    named(runs).to_csv(out / "runs.csv", index=False)
    named(methods).to_csv(out / "methods.csv", index=False)
    named(board).to_csv(out / "leaderboard.csv", index=False)
    text = report(runs, methods, board, rho, tolerance, labels)
    (out / "report.md").write_text(text + "\n")
    print(text)
    print(f"\nDetails: {out / 'runs.csv'} (one row per run, with both selections)")


if __name__ == "__main__":
    main()
