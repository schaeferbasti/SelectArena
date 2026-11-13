from __future__ import annotations

import ast

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits

from selectarena.extra_benchmark import ExtraBenchmarkJob, collect_results, run_extra_benchmark_job
from selectarena.extra_benchmark.slurm import ExtraBenchmarkSlurmSettings, build_job_commands, write_slurm_arrays

PIPELINE = "FSBench__MIFeatureSelector__2__0__lgbm__60"


def _task_id(tmp_path) -> str:
    X, y = make_classification(n_samples=80, n_features=6, random_state=0)
    dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]).assign(target=y)
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=0).split(X, y), n_splits=2
    )
    task = UserTask(task_name="extra_toy", task_cache_path=tmp_path / "tasks")
    task.save_task(
        task.create_task(dataset=dataset, target_feature="target", problem_type="classification", splits=splits)
    )
    return task.task_id_str


def test_stability_and_validity_jobs_resume(tmp_path):
    task_id = _task_id(tmp_path)
    results_dir = tmp_path / "results"
    for mode in ("stability", "validity"):
        job = ExtraBenchmarkJob(mode=mode, method_name=PIPELINE, task_id=task_id, results_dir=str(results_dir))
        assert run_extra_benchmark_job(job, stability_repeats=3) is not None
        assert run_extra_benchmark_job(job, stability_repeats=3) is None  # nothing left to do

    results = collect_results(results_dir)
    stability, validity = results["stability"], results["validity"]
    assert sorted(stability["repeat"]) == [0, 1, 2]
    assert len(validity) == 3
    assert (stability["max_features"] == 2).all()  # ceil(6 * 1/3) for the 6 bootstrapped features
    # Validity adds 50/75/100% noise columns, so the budget is computed on 9, 10 and 12 features.
    assert sorted(validity["max_features"]) == [3, 4, 4]


def test_autogluon_preprocessing_handles_raw_dtypes(tmp_path, monkeypatch):
    """Selectors not in ``EXTRA_BENCHMARK_RAW_DATA_METHODS`` run on AutoGluon-preprocessed data.

    Datetime, pandas ``string`` and all-NaN columns broke the shared preprocessing on raw data.
    """
    from selectarena.extra_benchmark import runner
    from selectarena.methods.registry import EXTRA_BENCHMARK_RAW_DATA_METHODS

    assert "SHAPFeatureSelector" not in EXTRA_BENCHMARK_RAW_DATA_METHODS
    assert "LassoNetFeatureSelector" not in EXTRA_BENCHMARK_RAW_DATA_METHODS
    assert "CPIFeatureSelector" not in EXTRA_BENCHMARK_RAW_DATA_METHODS
    monkeypatch.setattr(runner, "EXTRA_BENCHMARK_RAW_DATA_METHODS", frozenset())  # MI is fast

    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]).assign(
        date=pd.date_range("2020-01-01", periods=80, freq="7D"),
        text=pd.array([f"c{i % 3}" for i in range(80)], dtype="string"),
        empty=float("nan"),
        target=y,
    )
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=0).split(X, y), n_splits=2
    )
    task = UserTask(task_name="extra_dtypes", task_cache_path=tmp_path / "tasks")
    task.save_task(
        task.create_task(dataset=dataset, target_feature="target", problem_type="classification", splits=splits)
    )
    pipeline = "FSBench__MIFeatureSelector__1__0__lgbm__60"  # keep all features
    for mode in ("stability", "validity"):
        job = ExtraBenchmarkJob(
            mode=mode, method_name=pipeline, task_id=task.task_id_str, results_dir=str(tmp_path / "results")
        )
        assert run_extra_benchmark_job(job, stability_repeats=2) is not None

    results = collect_results(tmp_path / "results")
    noise = {f"__noise_feature_{i}__" for i in range(10)}
    for mode in ("stability", "validity"):
        df = results[mode]
        assert (df["preprocessing"] == "autogluon").all()
        for selected, original in zip(df["selected_features"], df["original_features"], strict=True):
            # Selections are mapped back to raw columns (e.g. ``date.year`` -> ``date``).
            assert set(ast.literal_eval(selected)) <= set(ast.literal_eval(original)) | noise


def test_slurm_array_scripts(tmp_path):
    commands = build_job_commands(
        modes=["stability", "validity"],
        method_names=[PIPELINE],
        task_ids=["UserTask|1|a/1", "UserTask|2|b/2"],
        results_dir=tmp_path / "results",
    )
    assert len(commands) == 4
    scripts = write_slurm_arrays(
        commands,
        out_dir=tmp_path / "setup",
        python_path="/venv/bin/python",
        settings=ExtraBenchmarkSlurmSettings(partition="cpu", max_array_size=3),
    )
    assert len(scripts) == 2
    assert "#SBATCH --array=0-2%100" in scripts[0].read_text()
    assert "-m selectarena.extra_benchmark.runner" in scripts[1].read_text()
