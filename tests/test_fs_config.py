from __future__ import annotations

import io
import pickle
import sys
import types
from functools import partial

import pytest

from selectarena.benchmark.fs_config import (
    FSBenchConfig,
    _get_budget_feature_count,
    get_fs_benchmark_preprocessing_pipelines,
    get_num_selected_features,
    selector_and_config_from_string,
    with_encoding_permutation_variants,
)
from selectarena.evaluation.read_results import _CompatUnpickler
from selectarena.methods import FEATURE_SELECTION_METHODS, ProxyModelConfig


def test_config_string_round_trip():
    config = FSBenchConfig(
        fs_method="MIFeatureSelector", total_budget=5, budget_index=2, proxy_model="lgbm", fs_time=3600
    )
    assert config.to_string() == "FSBench__MIFeatureSelector__5__2__lgbm__3600"
    assert FSBenchConfig.from_string(config.to_string()) == config
    assert FSBenchConfig.from_string("FSBench__MIFeatureSelector__5__2__lgbm__0.5").fs_time == 0.5


@pytest.mark.parametrize(
    ("d", "b", "expected"),
    [
        (10, 4, [2, 4, 6, 8]),  # uniform regime: ceil(d * i / (b + 1))
        (1000, 2, [10, 100]),  # exponential regime: ceil(d ** (i / (b + 1)))
    ],
)
def test_budget_schedule(d, b, expected):
    assert get_num_selected_features(d=d, b=b) == expected


def test_pipelines_cover_methods_budgets_and_ablations():
    pipelines = get_fs_benchmark_preprocessing_pipelines(
        fs_methods=["MIFeatureSelector", "LassoFeatureSelector"],
        proxy_model_config=["lgbm"],
        time_limit=[60],
        total_budget=3,
    )
    assert len(pipelines) == 2 * 3 + 1
    assert pipelines[-1] == "default"
    variants = with_encoding_permutation_variants(pipelines, permutation_seeds=[1, 2])
    assert "default" in variants
    assert len(variants) == 2 * 3 * 3 + 1


@pytest.mark.parametrize("method", FEATURE_SELECTION_METHODS)
def test_selector_from_pipeline_string(method):
    selector, config = selector_and_config_from_string(
        preprocessing_name=f"FSBench__{method}__5__1__lgbm__60__nofallback__permenc3"
    )
    assert type(selector).__name__ == method
    assert config.budget_index == 1
    assert selector.skip_random_fallback
    assert selector.encoding_permutation_seed == 3
    assert selector.max_features_input(10) == 4
    if method in {
        "LOCOFeatureSelector",
        "SequentialBackwardEliminationFeatureSelector",
        "SequentialForwardSelectionFeatureSelector",
    }:
        assert isinstance(selector.proxy_mode_config, ProxyModelConfig)
    else:
        assert selector.proxy_mode_config is None


def test_fork_era_results_unpickle():
    """Results from tabarena_fork_fs reference the budget function under its old module path."""
    legacy_name = "tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils"
    legacy = types.ModuleType(legacy_name)
    legacy._get_budget_feature_count = _get_budget_feature_count
    original_module = _get_budget_feature_count.__module__
    sys.modules[legacy_name] = legacy
    try:
        _get_budget_feature_count.__module__ = legacy_name
        payload = pickle.dumps({"max_features_input": partial(_get_budget_feature_count, b=5, idx=1)})
    finally:
        _get_budget_feature_count.__module__ = original_module
        del sys.modules[legacy_name]

    loaded = _CompatUnpickler(io.BytesIO(payload)).load()
    assert loaded["max_features_input"].keywords == {"b": 5, "idx": 1}
    assert loaded["max_features_input"](10) == 4
