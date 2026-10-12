"""Run a small SelectArena benchmark locally, through TabArena's experiment runner.

Builds a toy classification task, runs LightGBM with two feature selectors at every budget
(plus LightGBM without feature selection), and prints the resulting results table, including
what each selector kept. The same experiments run on SLURM via ``scripts/run_benchmark.py``.

    python examples/run_local.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.contexts import AbstractArenaContext

from selectarena.benchmark import (
    SELECTARENA_VALIDATION_PROTOCOL,
    SelectArenaExperimentBundle,
    get_fs_benchmark_preprocessing_pipelines,
)
from selectarena.evaluation import read_benchmark_results


def make_task(task_cache_dir: Path) -> tuple[UserTask, object]:
    X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=0)
    dataset = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])]).assign(target=y)
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(
            dataset.drop(columns="target"), dataset["target"]
        ),
        n_splits=3,
    )
    task = UserTask(task_name="selectarena_toy", task_cache_path=task_cache_dir)
    wrapper = task.create_task(dataset=dataset, target_feature="target", problem_type="classification", splits=splits)
    task.save_task(wrapper)
    return task, wrapper.metadata


if __name__ == "__main__":
    here = Path(__file__).parent / "local_run"
    task, metadata = make_task(here / "tasks")

    pipelines = get_fs_benchmark_preprocessing_pipelines(
        fs_methods=["MIFeatureSelector", "LOCOFeatureSelector"],
        proxy_model_config=["lgbm"],
        time_limit=[300],
        total_budget=3,
        include_default=True,
    )
    experiments = SelectArenaExperimentBundle(
        models=[("LightGBM", 0)],
        preprocessing_pipelines=pipelines,
    ).build_experiments(time_limit=300)

    context = AbstractArenaContext(
        task_metadata=TaskMetadataCollection.from_source([metadata]),
        methods=[],
        validation_protocol=SELECTARENA_VALIDATION_PROTOCOL,
    )
    context.build_and_run_jobs(
        experiments,
        expname=str(here / "results"),
        user_tasks=[task],
        build_kwargs={"split_indices": "lite"},
        debug_mode=True,
    )

    table = read_benchmark_results(here / "results" / "data", use_ray=False)
    columns = ["experiment_method_name_string", "metric_error", "max_features", "feature_selection_fit_time"]
    print(table[columns].sort_values("experiment_method_name_string").to_string(index=False))
