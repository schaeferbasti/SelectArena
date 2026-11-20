"""A fitted STG selection is shared only for identical training problems."""

from __future__ import annotations

import concurrent.futures
import threading
import time

import pandas as pd
import pytest

from selectarena.methods.checks import synthetic_data
from selectarena.methods.stg_cache import cached_selection, selection_key
from test_stg import small_selector


def test_real_selection_is_reused_without_training_and_is_invalidated_by_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTARENA_STG_CACHE_DIR", str(tmp_path))
    X, y = synthetic_data("regression", n_samples=40)
    producer = small_selector(gate_tolerance=0)
    first = producer.fit_transform(X=X, y=y, problem_type="regression", time_limit=10)
    assert not producer.selection_cache_["hit"]
    consumer = small_selector(gate_tolerance=0, device="auto")
    monkeypatch.setattr(consumer, "_search_feature_scores", lambda **_: pytest.fail("cached selection retrained"))
    monkeypatch.setenv("SELECTARENA_STG_CACHE_ROLE", "consumer")
    second = consumer.fit_transform(X=X, y=y, problem_type="regression", time_limit=10)
    pd.testing.assert_frame_equal(first, second)
    assert consumer.selection_cache_["hit"]
    assert consumer._feature_selection_fit_time == producer._feature_selection_fit_time
    assert consumer.get_selection_diagnostics()["selection_mode"] == producer.selection_mode_
    original = selection_key(producer, X, y, 10)
    assert selection_key(consumer, X, y, 10) == original
    assert selection_key(producer, X, y + 1, 10) != original
    assert selection_key(producer, X.iloc[::-1], y.iloc[::-1], 10) != original
    changed = X.copy()
    changed["f0"] += 1
    assert selection_key(producer, changed, y, 10) != original
    assert selection_key(producer, X, y, 11) != original
    consumer.max_features = 2
    assert selection_key(consumer, X, y, 10) != original
    consumer.max_features = 3
    consumer.encoding_permutation_seed = 123
    assert selection_key(consumer, X, y, 10) != original


def test_concurrent_producers_compute_only_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTARENA_STG_CACHE_DIR", str(tmp_path))
    X, y = synthetic_data("regression", n_samples=40)
    calls, lock = [], threading.Lock()

    def run(_):
        selector = small_selector()
        selector.problem_type, selector.max_features = "regression", 3

        def compute(**kwargs):
            with lock:
                calls.append(1)
            time.sleep(0.2)
            for name in selector.CACHE_STATE:
                setattr(selector, name, [] if name == "search_results_" else None)
            selector.search_results_ = []
            return {name: float(i) for i, name in enumerate(X.columns)}

        return cached_selection(selector, X, y, 10, compute)

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(2)))
    assert len(calls) == 1 and results[0] == results[1]


def test_training_failure_does_not_publish_partial_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SELECTARENA_STG_CACHE_DIR", str(tmp_path))
    X, y = synthetic_data("regression", n_samples=40)
    selector = small_selector()
    selector.problem_type, selector.max_features = "regression", 3
    with pytest.raises(RuntimeError, match="refusing random padding"):
        cached_selection(selector, X, y, 10, lambda **_: {})
    assert not list(tmp_path.glob("*.json"))


def test_cache_key_includes_category_schema_and_seed():
    selector = small_selector()
    selector.problem_type, selector.max_features = "binary", 1
    X, y = pd.DataFrame({"a": pd.Categorical(["a", "b"])}), pd.Series([0, 1])
    key = selection_key(selector, X, y, 10)
    changed = X.copy()
    changed["a"] = changed["a"].cat.add_categories(["unseen"])
    assert selection_key(selector, changed, y, 10) != key
    selector.random_state = 44
    assert selection_key(selector, X, y, 10) != key


def test_cached_runtime_charges_original_selection_compute(monkeypatch):
    from types import SimpleNamespace

    from tabarena.benchmark.experiment import OOFExperimentRunner

    from selectarena.benchmark.runner import SelectArenaExperimentRunner

    fs = {"selection_diagnostics": {"cache": {"enabled": True, "fit_wall_seconds": 2, "source_fit_seconds": 100}}}
    runner = object.__new__(SelectArenaExperimentRunner)
    runner.compute_preprocessing_artifacts = True
    runner.model = SimpleNamespace(
        can_get_preprocessing=True, get_preprocessing_results=lambda: {"feature_selection": fs}
    )
    monkeypatch.setattr(OOFExperimentRunner, "post_evaluate", lambda self, out: out)
    result = runner.post_evaluate({"time_train_s": 12})
    assert result["time_train_s_wall"] == 12
    assert result["time_train_s"] == 110
