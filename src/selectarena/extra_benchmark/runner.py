"""Stability and validity benchmark for feature selectors (no downstream model is fitted).

* ``stability``: run a selector on bootstrap resamples of a task's data and record the selected
  features per resample (``STABILITY_REPEATS`` resamples).
* ``validity``: add synthetic noise features (``VALIDITY_NOISE_LEVELS``, as a fraction of the
  number of original features) and record which features the selector keeps.

One job is one ``(mode, FSBench pipeline, task)``. A job appends its results to one CSV in
``results_dir`` and, when rerun, only computes the repeats or noise levels missing from it.

Usage::

    python -m selectarena.extra_benchmark.runner --mode validity \\
        --method_name FSBench__RandomFeatureSelector__5__0__lgbm__3600 \\
        --task_id "UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792" \\
        --results_dir extra_benchmark_results
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from selectarena.benchmark.fs_config import selector_and_config_from_string
from selectarena.extra_benchmark.stability import get_dataset_for_stability
from selectarena.extra_benchmark.validity import get_dataset_for_validity

logger = logging.getLogger(__name__)

MODES = ("stability", "validity")
STABILITY_REPEATS = 25
VALIDITY_NOISE_LEVELS = (0.5, 0.75, 1.0)


@dataclass(frozen=True, kw_only=True)
class ExtraBenchmarkJob:
    mode: str
    method_name: str
    """An ``FSBench__*`` pipeline name (see :class:`~selectarena.benchmark.fs_config.FSBenchConfig`)."""
    task_id: str
    """A TabArena ``task_id_str`` (``UserTask|<id>|<name>[|<cache path>]``)."""
    results_dir: str
    repeat: int = 0
    noise: float | None = None
    noise_type: str = "gaussian"
    ignore_cache: bool = False


@dataclass
class FeatureSelectionResult:
    method: str
    data_foundry_task_id: str
    mode: str
    mode_kwargs: dict[str, Any]
    repeat: int
    original_features: list[str]
    max_features: int
    selected_features: list[Any] | None
    num_classes: int
    num_samples: int
    min_samples_per_class: int
    elapsed_time_fs: float


def get_cache_path(job: ExtraBenchmarkJob) -> Path:
    """The CSV a job appends its results to."""
    task_name = re.sub(r"_+", "_", re.sub(r"[|: /\[\]\(\)]", "_", job.task_id))
    safe_method = job.method_name.replace("/", "_").replace(" ", "_")
    return Path(job.results_dir) / f"{job.mode}_{safe_method}_{task_name}.csv"


def _augment_dataset(
    *, mode: str, X: pd.DataFrame, y: pd.Series, rng: np.random.Generator, **kwargs
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    original_features = list(X.columns)
    if mode == "validity":
        X = get_dataset_for_validity(X=X, rng=rng, **kwargs)
    elif mode == "stability":
        X, y = get_dataset_for_stability(X=X, y=y, rng=rng, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    X = X[rng.permutation(X.columns)]
    return X, y, original_features


def run_benchmark(job: ExtraBenchmarkJob) -> FeatureSelectionResult:
    """Run the selector once (one stability repeat or one validity noise level)."""
    from tabarena.benchmark.task import UserTask

    rng = np.random.default_rng(42 + job.repeat)
    task = UserTask.from_task_id_str(job.task_id).load()
    X, y = task.get_X_y()

    feature_selector, config = selector_and_config_from_string(preprocessing_name=job.method_name)
    kwargs: dict[str, Any] = {}
    if job.mode == "validity":
        kwargs = {"noise": job.noise, "noise_type": job.noise_type}
    X, y, original_features = _augment_dataset(mode=job.mode, X=X, y=y, rng=rng, **kwargs)

    start_time = time.monotonic()
    feature_selector.fit_transform(
        X=X,
        y=y,
        time_limit=config.fs_time,
        eval_metric=task.eval_metric,
        problem_type=task.problem_type,
    )
    elapsed_time = time.monotonic() - start_time

    selected = feature_selector._selected_features
    return FeatureSelectionResult(
        method=job.method_name,
        data_foundry_task_id=job.task_id,
        mode=job.mode,
        mode_kwargs=kwargs,
        repeat=job.repeat,
        original_features=original_features,
        max_features=feature_selector.max_features,
        selected_features=[entry["feature"] for entry in selected] if selected is not None else None,
        num_classes=int(y.nunique()),
        num_samples=int(X.shape[0]),
        min_samples_per_class=int(y.value_counts().min()),
        elapsed_time_fs=elapsed_time,
    )


def save_results(job: ExtraBenchmarkJob, results: list[FeatureSelectionResult]) -> Path:
    """Append ``results`` to the job's CSV (writing the header for a new file)."""
    cache_path = get_cache_path(job)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not cache_path.exists() or cache_path.stat().st_size == 0
    pd.DataFrame([r.__dict__ for r in results]).to_csv(
        cache_path, index=False, mode="w" if new_file else "a", header=new_file
    )
    return cache_path


def _completed(job: ExtraBenchmarkJob) -> tuple[set[int], set[float]]:
    """Stability repeats and validity noise levels already in the job's CSV."""
    cache_path = get_cache_path(job)
    if job.ignore_cache or not cache_path.exists() or cache_path.stat().st_size == 0:
        return set(), set()
    try:
        df = pd.read_csv(cache_path)
    except (pd.errors.ParserError, OSError) as e:
        logger.warning("Could not parse existing results %s: %s", cache_path, e)
        return set(), set()
    repeats: set[int] = set()
    noises: set[float] = set()
    if job.mode == "stability" and "repeat" in df.columns:
        repeats = {int(r) for r in df["repeat"].dropna().unique()}
    elif job.mode == "validity" and "mode_kwargs" in df.columns:
        for value in df["mode_kwargs"].dropna():
            try:
                noise = ast.literal_eval(value).get("noise")
            except (ValueError, SyntaxError):
                continue
            if noise is not None:
                noises.add(float(noise))
    return repeats, noises


def run_extra_benchmark_job(
    base_job: ExtraBenchmarkJob,
    *,
    stability_repeats: int = STABILITY_REPEATS,
    validity_noise_levels: tuple[float, ...] = VALIDITY_NOISE_LEVELS,
) -> Path | None:
    """Run every missing stability repeat / validity noise level of ``base_job``; return its CSV path."""
    done_repeats, done_noises = _completed(base_job)
    if base_job.mode == "stability":
        todo = [replace(base_job, repeat=rep) for rep in range(stability_repeats) if rep not in done_repeats]
    elif base_job.mode == "validity":
        todo = [replace(base_job, noise=noise) for noise in validity_noise_levels if noise not in done_noises]
    else:
        raise ValueError(f"Unknown mode: {base_job.mode}")

    if not todo:
        logger.info("Nothing left to run for %s.", get_cache_path(base_job))
        return None
    for job in todo:
        logger.info("Running %s (repeat=%s, noise=%s)", job.method_name, job.repeat, job.noise)
        save_results(base_job, [run_benchmark(job)])
    return get_cache_path(base_job)


def collect_results(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Concatenate the per-job CSVs in ``results_dir`` into one frame per mode."""
    results_dir = Path(results_dir)
    out = {}
    for mode in MODES:
        frames = [pd.read_csv(f).assign(source_file=f.name) for f in sorted(results_dir.glob(f"{mode}_*.csv"))]
        out[mode] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SelectArena stability/validity job.")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--method_name", required=True, help="FSBench__* pipeline name.")
    parser.add_argument("--task_id", required=True, help="TabArena task_id_str of the task.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--noise_type", default="gaussian", choices=["gaussian", "uniform"])
    parser.add_argument("--openml_cache", default=None, help="OpenML cache root holding the UserTasks.")
    parser.add_argument("--ignore_cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if args.openml_cache is not None:
        from tabarena.caching import CacheConfig

        CacheConfig(openml=args.openml_cache).apply()
    run_extra_benchmark_job(
        ExtraBenchmarkJob(
            mode=args.mode,
            method_name=args.method_name,
            task_id=args.task_id,
            results_dir=args.results_dir,
            noise_type=args.noise_type,
            ignore_cache=args.ignore_cache,
        )
    )


if __name__ == "__main__":
    main()
