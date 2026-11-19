"""Convergence evidence must survive both benchmark output paths, including failed fits."""

from __future__ import annotations

import ast
import json
from types import SimpleNamespace

import pandas as pd

from selectarena.benchmark.wrapper import feature_selection_metadata
from selectarena.extra_benchmark.runner import ExtraBenchmarkJob, FeatureSelectionResult, save_results
from selectarena.methods.mi import MIFeatureSelector
from selectarena.methods.stg import STGFeatureSelector


def test_optional_diagnostics_do_not_change_other_selectors():
    selector = MIFeatureSelector(max_features=2)
    selector._original_features = ["a", "b", "c"]
    selector.max_features = 2
    assert "selection_diagnostics" not in feature_selection_metadata(selector)


def test_stg_metadata_is_compact_and_serializable():
    selector = STGFeatureSelector(max_features=2)
    selector._original_features = ["a", "b", "c"]
    selector.max_features = 2
    selector.budget_converged_, selector.native_count_ = True, 2
    selector.best_params_, selector.best_epoch_ = {"lambda": 0.1, "hidden_dims": (8, 8, 8), "learning_rate": 0.001}, 700
    selector.gate_diagnostics_ = {
        "zero_gates": 1,
        "one_gates": 2,
        "expected_active": 2.1,
        "gate_change": 0,
        "probability_change": 0.01,
    }
    selector.search_results_ = [dict(selector.best_params_, status="early_stopping", history=[{"epoch": 700}])]
    diagnostics = feature_selection_metadata(selector)["selection_diagnostics"]
    assert diagnostics["budget_converged"] and diagnostics["native_active_features"] == 2
    assert diagnostics["zero_gates"] == 1
    assert "history" not in diagnostics["trials"][0]
    json.dumps(diagnostics, allow_nan=False)


def test_extra_benchmark_retains_failed_and_successful_audits(tmp_path):
    selector = STGFeatureSelector(max_features=2)
    selector.max_features = 2
    job = ExtraBenchmarkJob(mode="stability", method_name="STG", task_id="UserTask|1|toy", results_dir=str(tmp_path))
    for repeat, matched in enumerate([False, True]):
        selector.budget_converged_ = matched
        row = FeatureSelectionResult(
            method="STG",
            data_foundry_task_id=job.task_id,
            mode=job.mode,
            mode_kwargs={},
            repeat=repeat,
            original_features=["a", "b", "c"],
            max_features=2,
            selected_features=["a", "b"],
            num_classes=2,
            num_samples=20,
            min_samples_per_class=10,
            elapsed_time_fs=1,
            selection_diagnostics=selector.get_selection_diagnostics(),
        )
        path = save_results(job, [row])
    frame = pd.read_csv(path)
    assert [ast.literal_eval(v)["budget_converged"] for v in frame["selection_diagnostics"]] == [False, True]


def test_performance_table_keeps_failed_convergence_metadata(monkeypatch):
    from tabarena.benchmark.result.baseline_result import BaselineResult

    from selectarena.evaluation import read_results

    selector = STGFeatureSelector(max_features=2)
    selector._original_features, selector.max_features = ["a", "b", "c"], 2
    metadata = feature_selection_metadata(selector)
    result = SimpleNamespace(
        framework="LightGBM",
        problem_type="binary",
        task_metadata={"name": "toy"},
        result={
            "metric_error": 0.2,
            "metric": "roc_auc",
            "time_train_s": 1,
            "time_infer_s": 0.01,
            "preprocessing_artifacts": {"feature_selection": metadata},
        },
    )
    monkeypatch.setattr(read_results, "load_result", lambda _: {})
    monkeypatch.setattr(BaselineResult, "from_dict", lambda **_: result)
    row = read_results.parse_result("unused.pkl")
    assert row["selection_diagnostics"]["budget_converged"] is False
