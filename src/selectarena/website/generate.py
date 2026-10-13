"""Generate the leaderboard website artifacts from SelectArena results.

The SelectArena counterpart of TabArena's ``scripts/run_generate_website_artifacts.py`` pipeline:

* ``raw_website_artifacts/``: the paper figure (``main_figure.png`` / ``.pdf``) and the full
  per-comparison scores, for paper use; not published.
* ``clean_website_artifacts/website_data/``: what the leaderboard Space serves as its ``data/``
  folder, text only:

  - ``meta.json``: counts, axis values and the list of published subsets;
  - ``subsets/tasks_<t>/models_<m>/budgets_<b>/``: per subset ``website_leaderboard.csv``,
    ``leaderboard_table.html`` and ``elo_explorer.html`` (see :mod:`selectarena.website.subsets`);
  - ``per_dataset.csv`` and ``per_dataset_table.html``: mean rank per dataset and method;
  - ``stability.csv`` and ``validity.csv``: the full stability / validity tables, if available.

* ``clean_website_artifacts.zip``: the same folder zipped.

Both artifact folders are deleted and rewritten on every run, so no subset that disappeared can
survive as a leftover.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from selectarena.evaluation.elo import compute_elo
from selectarena.evaluation.leaderboard import (
    VALIDITY_NOISE_MAIN_FIGURE,
    aggregate,
    main_figure_table,
    per_group_scores,
    prepare_results,
    stability_per_task,
    stability_table,
    validity_per_run,
    validity_table,
)
from selectarena.methods.registry import NAME_TO_MODULE_MAP
from selectarena.website.html import elo_explorer_html, leaderboard_table_html, per_dataset_table_html
from selectarena.website.subsets import ALL, Subset, enumerate_subsets, model_key
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
    """Datasets are matched across results by name (``anneal`` and ``anneal/<uuid>`` are the same)."""
    return str(name).split("/")[0]


def _total_budget(df: pd.DataFrame) -> int | None:
    totals = df["pipeline"].dropna().map(lambda p: p.split("__")[2] if p.startswith("FSBench__") else None).dropna()
    return int(totals.astype(int).mode().iloc[0]) if len(totals) else None


def _version() -> str:
    try:
        return metadata.version("selectarena")
    except metadata.PackageNotFoundError:
        return "dev"


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
        include_unregistered: keep methods that are not in the registry (e.g. retired fork-era
            methods such as ``AccuracyLinearFeatureSelector``); by default they are left out.
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

    df = prepare_results(results)
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
    scores.to_csv(raw_dir / "per_group_scores.csv", index=False)
    unclassified = unclassified_methods(scores["method"].dropna().unique().tolist())
    if unclassified:
        log(f"WARNING: no family for {unclassified}; add them to METHOD_FAMILIES (they show as 'Other').")

    problem_types = df.assign(key=df["dataset"].map(_dataset_key)).groupby("key")["problem_type"].first()
    stab_rows = stability_per_task(stability) if stability is not None and not stability.empty else None
    val_rows = validity_per_run(validity) if validity is not None and not validity.empty else None
    if val_rows is not None and validity_noise is not None:
        val_rows = val_rows[val_rows["noise"] == validity_noise]
    ranked = set(scores["method"].dropna())
    stab_rows = stab_rows[stab_rows["method"].isin(ranked)].copy() if stab_rows is not None else None
    val_rows = val_rows[val_rows["method"].isin(ranked)].copy() if val_rows is not None else None
    for rows in (stab_rows, val_rows):
        if rows is not None:
            rows["dataset"] = rows["dataset"].map(_dataset_key)

    model_names = {model_key(m): m for m in sorted(df["model"].dropna().unique())}
    published: list[dict] = []
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
        )
        target = website_data / "subsets" / subset.rel_path
        target.mkdir(parents=True)
        title = _subset_title(subset, model_names)
        leaderboard.to_csv(target / "website_leaderboard.csv", index=False)
        (target / "leaderboard_table.html").write_text(leaderboard_table_html(leaderboard, title=title))
        (target / "elo_explorer.html").write_text(elo_explorer_html(leaderboard, title=title))
        published.append(
            {
                "tasks": subset.tasks,
                "model": subset.model,
                "budget": subset.budget,
                "path": f"subsets/{subset.rel_path}",
                "n_datasets": int(sub["dataset_key"].nunique()),
                "n_methods": int(sub["method"].nunique()),
                "n_comparisons": len(sub),
            }
        )
        log(f"[{k}/{len(candidates)}] {subset.rel_path}: {sub['method'].nunique()} methods")

    per_dataset = (
        scores.pivot_table(index="dataset_key", columns="method", values="rank", aggfunc="mean")
        .round(2)
        .reset_index()
        .rename(columns={"dataset_key": "dataset"})
    )
    per_dataset.insert(1, "problem_type", per_dataset["dataset"].map(problem_types))
    per_dataset.to_csv(website_data / "per_dataset.csv", index=False)
    (website_data / "per_dataset_table.html").write_text(
        per_dataset_table_html(per_dataset, title="SelectArena: mean rank per dataset")
    )

    stab_table = stability_table(stability) if stab_rows is not None else None
    val_table = validity_table(validity) if validity is not None and not validity.empty else None
    stab_table = stab_table[stab_table["method"].isin(ranked)] if stab_table is not None else None
    val_table = val_table[val_table["method"].isin(ranked)] if val_table is not None else None
    if stab_table is not None:
        stab_table.to_csv(website_data / "stability.csv", index=False)
    if val_table is not None:
        val_table.to_csv(website_data / "validity.csv", index=False)

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
        "total_budget": _total_budget(df),
        "task_types": sorted({s["tasks"] for s in published}),
        "elo_bootstrap_rounds": n_bootstrap,
        "validity_noise": validity_noise,
        "has_stability": stab_table is not None,
        "has_validity": val_table is not None,
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
