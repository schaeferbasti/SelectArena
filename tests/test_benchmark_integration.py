"""End-to-end check that a feature selection pipeline runs through TabArena's experiment runner.

Builds a small local ``UserTask``, runs the default pipeline and two ``FSBench__*`` pipelines
with SelectArena's bundle, and checks that every result carries the feature selection metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.contexts import AbstractArenaContext
from tabarena.utils.pickle_utils import load_pickle

from selectarena.benchmark import (
    SELECTARENA_VALIDATION_PROTOCOL,
    SelectArenaBagExperiment,
    SelectArenaExperimentBundle,
    get_fs_benchmark_preprocessing_pipelines,
)
from selectarena.evaluation import read_benchmark_results

MODEL = "LightGBM" if importlib.util.find_spec("lightgbm") is not None else "RandomForest"
N_FEATURES = 12


def _make_task(task_cache_dir: Path) -> tuple[UserTask, object]:
    X, y = make_classification(n_samples=150, n_features=N_FEATURES, n_informative=4, random_state=0)
    dataset = pd.DataFrame(X, columns=[f"num_{i}" for i in range(N_FEATURES)]).assign(target=y)
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=0).split(
            dataset.drop(columns="target"), dataset["target"]
        ),
        n_splits=2,
    )
    task = UserTask(task_name="selectarena_toy_classification", task_cache_path=task_cache_dir)
    wrapper = task.create_task(dataset=dataset, target_feature="target", problem_type="classification", splits=splits)
    task.save_task(wrapper)
    return task, wrapper.metadata


@pytest.mark.slow
def test_fs_pipelines_run_and_record_metadata(tmp_path):
    task, metadata = _make_task(tmp_path / "tasks")
    pipelines = get_fs_benchmark_preprocessing_pipelines(
        fs_methods=["MIFeatureSelector", "RandomFeatureSelector"],
        proxy_model_config=["lgbm"],
        time_limit=[60],
        total_budget=2,
        include_default=True,
    )
    experiments = SelectArenaExperimentBundle(
        models=[(MODEL, 0)],
        preprocessing_pipelines=pipelines,
        model_artifacts_base_path=str(tmp_path / "ag_models"),
    ).build_experiments(time_limit=60)
    assert len(experiments) == len(pipelines)
    assert all(isinstance(exp, SelectArenaBagExperiment) for exp in experiments)

    # The experiments must survive TabArena's YAML round trip (how they reach cluster nodes).
    for exp in experiments:
        restored = SelectArenaBagExperiment.from_yaml(**{k: v for k, v in exp.to_yaml_dict().items() if k != "type"})
        assert restored.preprocessing_pipeline == exp.preprocessing_pipeline

    context = AbstractArenaContext(
        task_metadata=TaskMetadataCollection.from_source([metadata]),
        methods=[],
        validation_protocol=SELECTARENA_VALIDATION_PROTOCOL,
    )
    results_dir = tmp_path / "results"
    context.build_and_run_jobs(
        experiments,
        expname=str(results_dir),
        user_tasks=[task],
        build_kwargs={"split_indices": "lite"},
        debug_mode=True,
    )

    result_files = sorted(results_dir.rglob("results.pkl"))
    assert len(result_files) == len(pipelines)
    n_selected = {0: 4, 1: 8}  # get_num_selected_features(d=12, b=2)
    for path in result_files:
        result = load_pickle(path)
        fs = result.get("preprocessing_artifacts", {}).get("feature_selection")
        if "FSBench__" not in result["framework"]:
            assert fs is None
            continue
        budget_index = int(result["framework"].split("FSBench__")[1].split("__")[2])
        assert fs is not None, result["framework"]
        assert fs["max_features"] == n_selected[budget_index]
        assert len(fs["selected_feature_names"]) == n_selected[budget_index]
        assert set(fs["selected_feature_names"]) <= set(fs["original_feature_names"])
        assert fs["feature_selection_time_limit"] == 60

    table = read_benchmark_results(results_dir / "data", use_ray=False)
    assert len(table) == len(pipelines)
    fs_rows = table[table["feature_selection_method"].notna()]
    assert set(fs_rows["feature_selection_method"]) == {"MIFeatureSelector", "RandomFeatureSelector"}
    assert set(fs_rows["feature_selection_budget_total"]) == {2}
    assert set(fs_rows["feature_selection_budget_index"]) == {0, 1}
