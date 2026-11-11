"""Generate the leaderboard website artifacts from SelectArena results.

The SelectArena counterpart of TabArena's ``scripts/run_generate_website_artifacts.py`` pipeline:

* ``raw_website_artifacts/``: the paper figure (``main_figure.png`` / ``.pdf``) and the full
  per-comparison scores, for paper use; not published.
* ``clean_website_artifacts/website_data/``: what the leaderboard Space serves as its ``data/``
  folder, text only:

  - ``meta.json``: counts, axis values and the list of published subsets;
  - ``subsets/tasks_<t>/models_<m>/budgets_<b>/``: per subset ``website_leaderboard.csv``,
    ``leaderboard_table.html``, ``elo_explorer.html`` and, when the number of features of the
    datasets is known (from the stability or validity results), the Pareto front of improvement
    over Random vs. selection time, ``pareto.html`` (interactive) and ``pareto.csv`` (see
    :mod:`selectarena.website.subsets` and :mod:`selectarena.evaluation.pareto`); the all-models
    folder of each task type and budget also holds ``per_model.csv`` and ``per_model_table.html``,
    the Elo of each method per downstream model (``"per_model": true`` in ``meta.json``);
  - ``lasagna/stability/tasks_<t>/`` and ``lasagna/validity/noise_<level>/tasks_<t>/``: the
    paper's lasagna plots per task type, one interactive ``<variant>.html`` per variant plus ``per_epv.csv``
    (see :mod:`selectarena.evaluation.lasagna`);
  - ``results.csv`` and ``results_explorer.html``: the complete results, per dataset, method, downstream
    model and budget, averaged over splits;
  - ``per_dataset.csv``: mean rank per dataset and method (data only; the site does not show it);
  - ``analysis/``: where and why methods win (see :mod:`selectarena.evaluation.characteristics`):
    ``heatmap_<grouping>.html`` (improvement over Random per task type, dataset size, downstream
    model and budget, data in ``heatmap.csv``), ``rank_by_size.html`` / ``.csv`` (rank correlation with the
    dataset size), ``improvement_by_model.html``, ``improvement_by_task_type.html`` and
    ``.csv`` (improvement over Random per downstream model and task type), ``elo_no_timeouts.html``
    and ``leaderboard_no_timeouts.csv`` (the Elo on the datasets where no method ran out of time next
    to the default Elo, as in the paper's Figure D.3, and the leaderboard of those datasets);
  - ``stability.csv`` and ``stability_table.html``; ``validity.csv`` and one
    ``validity_noise_<level>_table.html`` per noise level (listed in ``meta.json``), if available.

* ``clean_website_artifacts.zip``: the same folder zipped.

Both artifact folders are deleted and rewritten on every run, so no subset that disappeared can
survive as a leftover.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from selectarena.datasets import dataset_name, task_names_from_ids
from selectarena.evaluation import characteristics, lasagna
from selectarena.evaluation.elo import compute_elo
from selectarena.evaluation.leaderboard import (
    VALIDITY_NOISE_MAIN_FIGURE,
    aggregate,
    events_per_variable,
    main_figure_table,
    per_group_scores,
    prepare_results,
    stability_per_task,
    stability_table,
    validity_per_run,
    validity_table,
)
from selectarena.evaluation.pareto import improvement_and_runtime, improvement_over_random
from selectarena.methods.registry import NAME_TO_MODULE_MAP
from selectarena.website.characteristic_figures import (
    elo_pair_html,
    group_improvement_html,
    improvement_heatmap_html,
    method_order,
    rank_sensitivity_html,
)
from selectarena.website.figures import lasagna_explorer_html, pareto_explorer_html
from selectarena.website.html import (
    elo_explorer_html,
    extra_table_html,
    leaderboard_table_html,
    per_model_table,
    per_model_table_html,
    results_explorer_html,
)
from selectarena.website.subsets import ALL, TASK_TYPES, Subset, enumerate_subsets, model_key
from selectarena.website.website_format import format_leaderboard, unclassified_methods

RAW_DIR = "raw_website_artifacts"
CLEAN_DIR = "clean_website_artifacts"
WEBSITE_DATA = "website_data"
SUBSET_FILES = ("website_leaderboard.csv", "leaderboard_table.html", "elo_explorer.html")


@dataclass
class WebsiteArtifacts:
    raw_dir: Path
    website_data: Path
    zip_path: Path | None
    subsets: list[dict]
    unclassified_methods: list[str]


def _dataset_key(name) -> str:
    """Datasets are matched across result files by name (task ids and ``<dataset>/<uuid>`` paths are mapped)."""
    return dataset_name(name)


def _total_budget(df: pd.DataFrame) -> int | None:
    totals = df["pipeline"].dropna().map(lambda p: p.split("__")[2] if p.startswith("FSBench__") else None).dropna()
    return int(totals.astype(int).mode().iloc[0]) if len(totals) else None


def _version() -> str:
    try:
        return metadata.version("selectarena")
    except metadata.PackageNotFoundError:
        return "dev"


def _noise_label(noise: float) -> str:
    return f"{float(noise):g}"


def _lasagna_plots(
    rows: pd.DataFrame, value: str, folder: Path, problem_types: pd.Series, log: Callable[[str], None]
) -> list[str]:
    """Write one folder per task type with ``per_epv.csv`` and one interactive page per variant
    (``<variant>.html``); return the task types."""
    written = []
    for tasks, allowed in TASK_TYPES.items():
        sub = rows if allowed is None else rows[rows["dataset"].map(problem_types).isin(allowed)]
        long = lasagna.per_epv(sub, value)
        if long["method"].nunique() < 2 or long["epv"].nunique() < 2:
            continue
        target = folder / f"tasks_{tasks}"
        target.mkdir(parents=True, exist_ok=True)
        long.round(4).to_csv(target / "per_epv.csv", index=False)
        matrix = lasagna.epv_matrix(long, value)
        named = sub.dropna(subset=["epv"])
        datasets = {float(epv): sorted(group["dataset"].unique()) for epv, group in named.groupby("epv")}
        for variant in lasagna.VARIANTS:
            page = lasagna_explorer_html(
                matrix,
                long.round(4),
                value,
                variant,
                datasets=datasets,
                title=f"SelectArena: {value} ({folder.name.replace('_', ' ')}, {tasks} tasks)",
            )
            (target / f"{variant}.html").write_text(page)
        written.append(tasks)
    log(f"{value} lasagna plots ({folder.name}): {', '.join(written) or 'none'}")
    return written


def _analysis_figures(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    chars: pd.DataFrame,
    folder: Path,
    n_bootstrap: int,
    log: Callable[[str], None],
) -> dict:
    """The figures of the Performance tab's "where methods win" section, in ``folder``; returns their
    entry of ``meta.json``."""
    folder.mkdir(parents=True)
    with_chars = characteristics.add_characteristics(df[df["method"].notna()], chars)
    groupings = characteristics.available_groupings(with_chars)
    if not chars.empty and len(chars) < scores["dataset_key"].nunique():
        log(
            f"Dataset sizes known for {len(chars)} of {scores['dataset_key'].nunique()} datasets (from the extra results)."
        )
    tables, order = [], None
    for grouping in groupings:
        table = characteristics.improvement_by_grouping(with_chars, grouping)
        order = order or method_order(table)
        tables.append(table.assign(grouping=grouping.label))
        (folder / f"heatmap_{grouping.key}.html").write_text(
            improvement_heatmap_html(
                table,
                grouping.label,
                title=f"SelectArena: improvement over Random by {grouping.label.lower()}",
                name=f"selectarena_improvement_by_{grouping.key}_groups",
            )
        )
    if tables:
        pd.concat(tables)[["grouping", "group", "method", "improvement", "n_datasets"]].round(4).to_csv(
            folder / "heatmap.csv", index=False
        )
    # Improvement over Random per downstream model and per task type.
    per_group = {}
    named = df.assign(dataset=df["dataset_key"])
    for key, column, labels, all_label, between in (
        ("model", "model", None, "All models", "the downstream models"),
        ("task_type", "problem_type", characteristics.TASK_TYPE_LABELS, "All task types", "the task types"),
    ):
        per_group[key] = False
        if not order or named[column].nunique() < 2:
            continue
        table = characteristics.improvement_by(named, column, labels)
        if table.empty:
            continue
        table.rename(columns={"group": key}).round(4).to_csv(folder / f"improvement_by_{key}.csv", index=False)
        (folder / f"improvement_by_{key}.html").write_text(
            group_improvement_html(
                table,
                order,
                characteristics.ranking_agreement(table),
                title=f"SelectArena: improvement over Random per {all_label.removeprefix('All ').rstrip('s')}",
                name=f"selectarena_improvement_by_{key}",
                all_label=all_label,
                agreement_label=between,
            )
        )
        per_group[key] = True
    has_size = False
    if not chars.empty and order:
        sens = characteristics.rank_sensitivity(scores, chars, n_bootstrap=n_bootstrap)
        if not sens.empty:
            sens.round(4).to_csv(folder / "rank_by_size.csv", index=False)
            (folder / "rank_by_size.html").write_text(
                rank_sensitivity_html(sens, order, title="SelectArena: rank by dataset size")
            )
            has_size = True
    log(
        f"Analysis figures: {len(groupings)} groupings, per model {per_group['model']}, per task type "
        f"{per_group['task_type']}, rank by size {has_size}."
    )
    return {
        "path": "analysis",
        "groupings": [{"key": g.key, "label": g.label} for g in groupings],
        "improvement_by_model": per_group["model"],
        "improvement_by_task_type": per_group["task_type"],
        "rank_by_size": has_size,
    }


def _results_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset, method, downstream model and budget, averaged over the splits."""
    rows = df.assign(
        method=df["method"].fillna("All features"),
        fs_time_s=df["feature_selection_fit_time"],
        fallback=df["used_fallback"].astype(float),
        # Not written by every TabArena version (nor by the tests' minimal results).
        max_features=df["max_features"] if "max_features" in df else np.nan,
        metric=df["metric"] if "metric" in df else "",
    )
    rows["budget"] = rows["budget_index"].map(lambda b: "–" if pd.isna(b) else str(int(b) + 1))
    keys = ["dataset_key", "problem_type", "method", "model", "budget"]
    table = (
        rows.groupby(keys, dropna=False)
        .agg(
            max_features=("max_features", "mean"),
            metric=("metric", "first"),
            error=("metric_error", "mean"),
            error_std=("metric_error", "std"),
            n_splits=("metric_error", "size"),
            fs_time_s=("fs_time_s", "mean"),
            fallback_pct=("fallback", "mean"),
        )
        .reset_index()
        .rename(columns={"dataset_key": "dataset", "problem_type": "task_type"})
    )
    table["fallback_pct"] = (table["fallback_pct"] * 100).where(table["method"] != "All features")
    table = table.sort_values(["dataset", "model", "budget", "error"]).reset_index(drop=True)
    for column, digits in (("error", 6), ("error_std", 6), ("fs_time_s", 3), ("fallback_pct", 1), ("max_features", 1)):
        table[column] = table[column].round(digits)
    return table


def _subset_title(subset: Subset, model_names: dict[str, str]) -> str:
    parts = [
        "all tasks" if subset.tasks == ALL else subset.tasks,
        "all models" if subset.model == ALL else model_names.get(subset.model, subset.model),
        "all budgets" if subset.budget == ALL else f"budget {int(subset.budget) + 1}",
    ]
    return "SelectArena leaderboard: " + ", ".join(parts)


def generate_website_artifacts(
    results: pd.DataFrame,
    *,
    out_dir: str | Path,
    stability: pd.DataFrame | None = None,
    validity: pd.DataFrame | None = None,
    benchmark_name: str = "SelectArena",
    n_bootstrap: int = 1000,
    validity_noise: float | None = VALIDITY_NOISE_MAIN_FIGURE,
    paper_figures: bool = True,
    zip_clean: bool = True,
    include_ablations: bool = False,
    include_unregistered: bool = False,
    log: Callable[[str], None] = print,
) -> WebsiteArtifacts:
    """Write the raw and clean website artifacts for ``results`` (a run's ``results.csv``).

    Args:
        stability, validity: ``stability_results.csv`` / ``validity_results.csv`` of the extra benchmark.
        n_bootstrap: bootstrap resamples per Elo interval (1000 as in the paper; 1 for a quick check).
        validity_noise: noise level whose validity the leaderboard shows (``None``: mean over levels).
        include_ablations: keep ablation runs (``MI [nofallback]``, ``MI [permenc1]``) as methods of
            their own; by default only the main runs are ranked.
        include_unregistered: keep methods that are not in the registry (e.g. a method removed
            from it); by default they are left out.
    """
    out_dir = Path(out_dir)
    raw_dir, clean_dir = out_dir / RAW_DIR, out_dir / CLEAN_DIR
    website_data = clean_dir / WEBSITE_DATA
    for path in (raw_dir, clean_dir, out_dir / f"{CLEAN_DIR}.zip"):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    raw_dir.mkdir(parents=True)
    website_data.mkdir(parents=True)

    # The extra results name every task (UserTask|<id>|<dataset>/<uuid>): use them to name task ids
    # whose containers are not listed in selectarena.datasets.
    task_names = {}
    for extra in (stability, validity):
        if extra is not None and "data_foundry_task_id" in extra:
            task_names.update(task_names_from_ids(extra["data_foundry_task_id"].dropna().unique()))
    df = prepare_results(results, task_names=task_names)
    if not include_unregistered:
        unregistered = df["method_class"].notna() & ~df["method_class"].isin(NAME_TO_MODULE_MAP)
        if unregistered.any():
            log(
                f"Leaving out {int(unregistered.sum())} results of methods not in the registry "
                f"({sorted(df.loc[unregistered, 'method_class'].unique())})."
            )
            df = df[~unregistered]
    if not include_ablations:
        ablations = df["method"].fillna("").str.contains(" [", regex=False)
        if ablations.any():
            log(
                f"Leaving out {int(ablations.sum())} ablation results ({sorted(df.loc[ablations, 'method'].unique())})."
            )
            df = df[~ablations]
    scores = per_group_scores(df)
    scores["dataset_key"] = scores["dataset"].map(_dataset_key)
    unnamed = sorted({d for d in scores["dataset_key"] if re.fullmatch(r"(?:Task-)?\d+", str(d))})
    if unnamed:
        log(
            f"WARNING: {len(unnamed)} task ids could not be mapped to dataset names (e.g. {unnamed[:3]}); "
            "their containers are not in selectarena.datasets."
        )
    scores.to_csv(raw_dir / "per_group_scores.csv", index=False)
    unclassified = unclassified_methods(scores["method"].dropna().unique().tolist())
    if unclassified:
        log(f"WARNING: no subgroup for {unclassified}; add them to METHOD_SUBGROUPS (they show as 'Other').")

    df["dataset_key"] = df["dataset"].map(_dataset_key)
    problem_types = df.groupby("dataset_key")["problem_type"].first()
    stab_rows = stability_per_task(stability) if stability is not None and not stability.empty else None
    val_all = validity_per_run(validity) if validity is not None and not validity.empty else None
    ranked = set(scores["method"].dropna())
    stab_rows = stab_rows[stab_rows["method"].isin(ranked)].copy() if stab_rows is not None else None
    val_all = val_all[val_all["method"].isin(ranked)].copy() if val_all is not None else None
    for rows in (stab_rows, val_all):
        if rows is not None:
            rows["dataset"] = rows["dataset"].map(_dataset_key)
    val_rows = val_all
    if val_rows is not None and validity_noise is not None:
        val_rows = val_rows[val_rows["noise"] == validity_noise]

    # Number of original features per dataset (for the Pareto front's time per 1k features).
    feature_counts = [r[["dataset", "num_features"]] for r in (val_all, stab_rows) if r is not None]
    num_features = (
        pd.concat(feature_counts).dropna().groupby("dataset")["num_features"].max() if feature_counts else pd.Series()
    )
    if num_features.empty:
        log("No stability or validity results: the number of features is unknown, so no Pareto fronts.")

    model_names = {model_key(m): m for m in sorted(df["model"].dropna().unique())}
    published: list[dict] = []
    leaderboards: dict[tuple[str, str, str], pd.DataFrame] = {}
    default_elo = None
    candidates = enumerate_subsets(scores)
    for k, subset in enumerate(candidates, start=1):
        sub = subset.filter(scores)
        if sub["method"].nunique() < 2:
            continue
        elo = compute_elo(sub, n_bootstrap=n_bootstrap)
        if subset.is_default:
            default_elo = elo

        def _mean(rows: pd.DataFrame | None, column: str, subset: Subset = subset) -> pd.Series | None:
            if rows is None:
                return None
            return subset.filter(rows, problem_types=problem_types).groupby("method")[column].mean()

        leaderboard = format_leaderboard(
            elo,
            aggregate(sub),
            stability=_mean(stab_rows, "stability"),
            validity=_mean(val_rows, "validity"),
            n_datasets=sub.groupby("method")["dataset_key"].nunique(),
            improvement=improvement_over_random(subset.filter(df).assign(dataset=lambda d: d["dataset_key"])),
        )
        target = website_data / "subsets" / subset.rel_path
        target.mkdir(parents=True)
        title = _subset_title(subset, model_names)
        leaderboard.to_csv(target / "website_leaderboard.csv", index=False)
        leaderboards[(subset.tasks, subset.model, subset.budget)] = leaderboard
        (target / "leaderboard_table.html").write_text(leaderboard_table_html(leaderboard, title=title))
        (target / "elo_explorer.html").write_text(elo_explorer_html(leaderboard, title=title))
        has_pareto = False
        if not num_features.empty:
            front = improvement_and_runtime(subset.filter(df).assign(dataset=lambda d: d["dataset_key"]), num_features)
            if len(front) >= 2 and "Random" in set(front["method"]):
                front.round(4).to_csv(target / "pareto.csv", index=False)
                (target / "pareto.html").write_text(pareto_explorer_html(front.round(4), title=title))
                has_pareto = True
        published.append(
            {
                "tasks": subset.tasks,
                "model": subset.model,
                "budget": subset.budget,
                "path": f"subsets/{subset.rel_path}",
                "n_datasets": int(sub["dataset_key"].nunique()),
                "n_methods": int(sub["method"].nunique()),
                "n_comparisons": len(sub),
                "pareto": has_pareto,
            }
        )
        log(f"[{k}/{len(candidates)}] {subset.rel_path}: {sub['method'].nunique()} methods")

    # Per downstream model view: the Elo of every method per model, in the all-models folder of
    # each task type and budget.
    for info in published:
        if info["model"] != ALL:
            continue
        per_model = {
            name: leaderboards[(info["tasks"], key, info["budget"])]
            for key, name in model_names.items()
            if (info["tasks"], key, info["budget"]) in leaderboards
        }
        info["per_model"] = len(per_model) >= 2
        if info["per_model"]:
            table = per_model_table(per_model, leaderboards[(info["tasks"], ALL, info["budget"])])
            target = website_data / info["path"]
            table.to_csv(target / "per_model.csv", index=False)
            title = _subset_title(Subset(info["tasks"], ALL, info["budget"]), model_names) + " · per model"
            (target / "per_model_table.html").write_text(per_model_table_html(table, list(per_model), title=title))

    per_dataset = (
        scores.pivot_table(index="dataset_key", columns="method", values="rank", aggfunc="mean")
        .round(2)
        .reset_index()
        .rename(columns={"dataset_key": "dataset"})
    )
    per_dataset.insert(1, "problem_type", per_dataset["dataset"].map(problem_types))
    per_dataset.to_csv(website_data / "per_dataset.csv", index=False)

    # Lasagna plots. The stability results store unreliable per-class counts; as in the paper, take
    # them from the validity results of the same dataset.
    lasagna_meta: dict = {"variants": list(lasagna.VARIANTS), "variant_labels": lasagna.VARIANT_LABELS}
    if stab_rows is not None:
        rows = stab_rows.copy()
        if val_all is not None:
            per_class = val_all.groupby("dataset")["min_samples_per_class"].first()
            rows["min_samples_per_class"] = rows["dataset"].map(per_class).fillna(rows["min_samples_per_class"])
        rows["epv"] = events_per_variable(rows, problem_types)
        task_types = _lasagna_plots(rows, "stability", website_data / "lasagna" / "stability", problem_types, log)
        if task_types:
            lasagna_meta["stability"] = {"path": "lasagna/stability", "task_types": task_types}
    if val_all is not None:
        val_all["epv"] = events_per_variable(val_all, problem_types)
        levels = []
        for noise, rows in val_all.dropna(subset=["noise"]).groupby("noise"):
            folder = website_data / "lasagna" / "validity" / f"noise_{_noise_label(noise)}"
            task_types = _lasagna_plots(rows, "validity", folder, problem_types, log)
            if task_types:
                levels.append(
                    {
                        "noise": float(noise),
                        "path": f"lasagna/validity/noise_{_noise_label(noise)}",
                        "task_types": task_types,
                    }
                )
        if levels:
            lasagna_meta["validity"] = levels

    chars = characteristics.dataset_characteristics([val_all, stab_rows], problem_types)
    analysis_meta = _analysis_figures(df, scores, chars, website_data / "analysis", n_bootstrap, log)
    # Elo on the datasets where no method ran out of time, next to the default Elo (the paper's
    # Figure D.3 design), for the methods with complete results only (preliminary ones are left out).
    complete = characteristics.complete_methods(scores)
    complete_scores = scores[scores["method"].isin(complete)]
    left_out = sorted(set(scores["method"].dropna()) - set(complete))
    if left_out:
        log(f"Time-out ablation without the methods with incomplete results: {left_out}.")
    no_timeouts = characteristics.datasets_without_timeouts(complete_scores)
    analysis_meta["elo_without_timeouts"] = bool(no_timeouts) and default_elo is not None
    if no_timeouts and default_elo is not None:
        sub = complete_scores[complete_scores["dataset_key"].isin(no_timeouts)]
        reference_elo = compute_elo(complete_scores, n_bootstrap=n_bootstrap) if left_out else default_elo

        def _mean_no_timeouts(rows: pd.DataFrame | None, column: str) -> pd.Series | None:
            if rows is None:
                return None
            rows = rows[rows["dataset"].isin(no_timeouts) & rows["method"].isin(complete)]
            return rows.groupby("method")[column].mean()

        ablated_elo = compute_elo(sub, n_bootstrap=n_bootstrap)
        leaderboard = format_leaderboard(
            ablated_elo,
            aggregate(sub),
            stability=_mean_no_timeouts(stab_rows, "stability"),
            validity=_mean_no_timeouts(val_rows, "validity"),
            n_datasets=sub.groupby("method")["dataset_key"].nunique(),
            improvement=improvement_over_random(
                df[df["dataset_key"].isin(no_timeouts)].assign(dataset=lambda d: d["dataset_key"])
            ),
        )
        leaderboard.to_csv(website_data / "analysis" / "leaderboard_no_timeouts.csv", index=False)
        all_datasets = complete_scores["dataset_key"].nunique()
        (website_data / "analysis" / "elo_no_timeouts.html").write_text(
            elo_pair_html(
                reference_elo,
                ablated_elo,
                labels=(
                    f"with random fallback ({all_datasets} datasets)",
                    f"no time-outs ({len(no_timeouts)} datasets)",
                ),
                title=f"SelectArena: Elo on the {len(no_timeouts)} datasets where no method timed out",
                name="selectarena_elo_without_timeouts",
            )
        )
        log(f"Leaderboard without time-outs: {len(no_timeouts)} datasets.")

    total_budget = _total_budget(df)
    results_table = _results_table(df)
    results_table.to_csv(website_data / "results.csv", index=False)
    (website_data / "results_explorer.html").write_text(
        results_explorer_html(results_table, title="SelectArena: complete results")
    )

    stab_table = stability_table(stability) if stab_rows is not None else None
    val_table = validity_table(validity) if validity is not None and not validity.empty else None
    stab_table = stab_table[stab_table["method"].isin(ranked)] if stab_table is not None else None
    val_table = val_table[val_table["method"].isin(ranked)] if val_table is not None else None
    if stab_table is not None:
        stab_table.to_csv(website_data / "stability.csv", index=False)
        (website_data / "stability_table.html").write_text(
            extra_table_html(stab_table, "stability", title="SelectArena: stability", total_budget=total_budget)
        )
    validity_tables = []
    if val_table is not None:
        val_table.to_csv(website_data / "validity.csv", index=False)
        # One table per noise level (the noise column would be constant within each).
        for noise in sorted(val_table["noise"].dropna().unique()):
            name = f"validity_noise_{float(noise):g}_table.html"
            table = (
                val_table[val_table["noise"] == noise].drop(columns="noise").sort_values("validity", ascending=False)
            )
            (website_data / name).write_text(
                extra_table_html(
                    table, "validity", title=f"SelectArena: validity at noise {noise:g}", total_budget=total_budget
                )
            )
            validity_tables.append({"noise": float(noise), "path": name})

    meta = {
        "benchmark_name": benchmark_name,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "selectarena_version": _version(),
        "n_results": len(df),
        "n_datasets": int(scores["dataset_key"].nunique()),
        "n_methods": int(scores["method"].nunique()),
        "methods": sorted(scores["method"].dropna().unique().tolist()),
        "models": model_names,
        "budgets": sorted(int(b) for b in scores["budget_index"].dropna().unique()),
        "total_budget": total_budget,
        "task_types": sorted({s["tasks"] for s in published}),
        "elo_bootstrap_rounds": n_bootstrap,
        "validity_noise": validity_noise,
        "has_stability": stab_table is not None,
        "has_validity": val_table is not None,
        "validity_tables": validity_tables,
        "lasagna": lasagna_meta,
        "analysis": analysis_meta,
        "has_results_explorer": True,
        "subsets": published,
    }
    (website_data / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if paper_figures and default_elo is not None:
        from selectarena.evaluation.plot import plot_main_figure

        main = main_figure_table(default_elo, stab_table, val_table, validity_noise=validity_noise)
        main.to_csv(raw_dir / "main_figure.csv", index=False)
        plot_main_figure(main, [raw_dir / "main_figure.png", raw_dir / "main_figure.pdf"])
    if stab_table is not None:
        stab_table.to_csv(raw_dir / "stability.csv", index=False)
    if val_table is not None:
        val_table.to_csv(raw_dir / "validity.csv", index=False)

    zip_path = None
    if zip_clean:
        zip_path = Path(shutil.make_archive(str(out_dir / CLEAN_DIR), "zip", root_dir=clean_dir))
    return WebsiteArtifacts(raw_dir, website_data, zip_path, published, unclassified)
