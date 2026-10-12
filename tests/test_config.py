"""The YAML configs build a tabflow_slurm plan over the SelectArena suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits

from selectarena.benchmark import SelectArenaBagExperiment
from selectarena.config import build_plan, load_benchmark_config, load_cluster_config
from selectarena.datasets import COLLECTION_NAME

REPO_ROOT = Path(__file__).parent.parent


def _write_suite_metadata(openml_cache: Path) -> None:
    """Put a one-task suite where ``prepare_datasets`` would write the real one."""
    from tabarena.benchmark.task.data_foundry.metadata_cache import get_path_to_metadata_cache
    from tabarena.caching import CacheConfig

    CacheConfig(openml=openml_cache).apply()
    X, y = make_classification(n_samples=60, n_features=6, random_state=0)
    dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]).assign(target=y)
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(X, y), n_splits=3
    )
    task = UserTask(task_name="toy/0000")
    wrapper = task.create_task(dataset=dataset, target_feature="target", problem_type="classification", splits=splits)
    task.save_task(wrapper)
    path = get_path_to_metadata_cache() / f"{COLLECTION_NAME}_tasks_metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    TaskMetadataCollection.from_source([wrapper.metadata]).to_dataframe().to_csv(path, index=False)


def test_example_configs_build_a_plan(tmp_path):
    cluster = yaml.safe_load((REPO_ROOT / "configs" / "cluster.example.yaml").read_text())
    cluster.update(
        workspace=str(tmp_path / "workspace"),
        caches={"openml": str(tmp_path / "openml")},
        data_foundry_warehouse=None,
    )
    cluster_path = tmp_path / "cluster.yaml"
    cluster_path.write_text(yaml.safe_dump(cluster))
    _write_suite_metadata(tmp_path / "openml")

    benchmark = load_benchmark_config(REPO_ROOT / "configs" / "benchmark.yaml")
    plan = build_plan(benchmark, load_cluster_config(cluster_path))

    assert len(plan.context.task_metadata_collection.dataset_names()) == 1
    assert plan.resources_setup.time_limit_for_model_agnostic_preprocessing == 3600
    pipelines = benchmark.feature_selection.pipelines()
    assert len(pipelines) == 15 * 5

    (setup,) = plan.build_setups(num_ray_cpus=1)  # one model job in the example config
    experiments = setup.experiment_bundle.build_experiments()
    assert len(experiments) == 2 * len(pipelines)  # TabICLv2 + RealMLP, default configs
    assert all(isinstance(exp, SelectArenaBagExperiment) for exp in experiments)
    assert {exp.preprocessing_pipeline for exp in experiments} == set(pipelines)

    jobs = plan.context.build_jobs(experiments, task_subset=setup.task_subset)
    assert len(jobs) == len(experiments) * 3  # splits r0f0, r0f1, r0f2
