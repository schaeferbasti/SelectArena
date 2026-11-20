"""Two real downstream models must apply one identical fitted STG selection."""

from __future__ import annotations

import pytest
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts import AbstractArenaContext
from tabarena.utils.pickle_utils import load_pickle

from selectarena.benchmark import SELECTARENA_VALIDATION_PROTOCOL, SelectArenaExperimentBundle
from test_benchmark_integration import _make_task


@pytest.mark.slow
def test_stg_selection_is_shared_across_downstream_models(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTARENA_STG_CACHE_DIR", str(tmp_path / "selections"))
    monkeypatch.setenv("SELECTARENA_STG_CACHE_ROLE", "auto")
    task, metadata = _make_task(tmp_path / "tasks")
    experiments = SelectArenaExperimentBundle(
        models=[("LightGBM", 0), ("RandomForest", 0)],
        preprocessing_pipelines=["FSBench__STGFeatureSelector__2__0__lgbm__10"],
        model_artifacts_base_path=str(tmp_path / "models"),
    ).build_experiments(time_limit=60)
    context = AbstractArenaContext(
        task_metadata=TaskMetadataCollection.from_source([metadata]),
        methods=[],
        validation_protocol=SELECTARENA_VALIDATION_PROTOCOL,
    )
    output = tmp_path / "results"
    context.build_and_run_jobs(
        experiments, expname=str(output), user_tasks=[task], build_kwargs={"split_indices": "lite"}, debug_mode=True
    )
    files = sorted(output.rglob("results.pkl"))
    assert len(files) == 2
    results = [load_pickle(path) for path in files]
    selections = [r["preprocessing_artifacts"]["feature_selection"] for r in results]
    assert selections[0]["selected_features"] == selections[1]["selected_features"]
    assert all(not s["random_feature_names"] for s in selections)
    caches = [s["selection_diagnostics"]["cache"] for s in selections]
    assert caches[0]["key"] == caches[1]["key"]
    assert sorted(c["hit"] for c in caches) == [False, True]
    assert len(list((tmp_path / "selections").glob("*.json"))) == 1
    assert selections[0]["feature_selection_fit_time"] == selections[1]["feature_selection_fit_time"]
    assert all("time_train_s_wall" in r for r in results)
