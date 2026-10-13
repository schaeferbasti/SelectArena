from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from selectarena.evaluation.leaderboard import (
    build_leaderboard,
    method_label,
    nogueira_stability,
    parse_method_name,
)


def _results() -> pd.DataFrame:
    rows = []
    errors = {"MIFeatureSelector": 0.1, "RandomFeatureSelector": 0.3, "LassoFeatureSelector": 0.2}
    for dataset in ["a", "b"]:
        for budget in [0, 1]:
            for method, error in errors.items():
                rows.append(
                    {
                        "experiment_method_name_string": f"LightGBM_c1_FSBench__{method}__2__{budget}__lgbm__60_BAG_L1",
                        "name": dataset,
                        "split_idx": 0,
                        "problem_type": "binary",
                        "metric_error": error,
                        "feature_selection_budget_index": budget,
                        "feature_selection_fit_time": 1.0,
                        "random_feature_names": "[]",
                    }
                )
        rows.append(
            {
                "experiment_method_name_string": "LightGBM_c1_default_BAG_L1",
                "name": dataset,
                "split_idx": 0,
                "problem_type": "binary",
                "metric_error": 0.2,
                "feature_selection_budget_index": np.nan,
                "feature_selection_fit_time": np.nan,
                "random_feature_names": np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_method_names():
    parsed = parse_method_name("TabICLv2_r3_FSBench__MIFeatureSelector__5__0__lgbm__3600__nofallback_BAG_L1")
    assert parsed == {
        "model": "TabICLv2",
        "config": "r3",
        "pipeline": "FSBench__MIFeatureSelector__5__0__lgbm__3600__nofallback",
    }
    assert method_label(parsed["pipeline"]) == "MI [nofallback]"
    assert method_label("default") is None


def test_leaderboard_tables():
    tables = build_leaderboard(_results())
    overall = tables["overall"].set_index("method")
    assert list(overall.index) == ["MI", "Lasso", "Random"]
    assert overall.loc["MI", "avg_rank"] == 1
    assert overall.loc["MI", "win_rate"] == 1
    assert overall.loc["Random", "improvability"] == pytest.approx((0.3 - 0.1) / 0.3, abs=1e-4)
    assert overall.loc["MI", "rel_error_vs_all_features"] == pytest.approx(0.5)
    assert overall.loc["MI", "n_comparisons"] == 4
    assert set(tables["by_budget"]["budget_index"]) == {0, 1}
    assert tables["meta"]["n_datasets"] == 2


def test_nogueira_stability():
    features = ["a", "b", "c", "d"]
    assert nogueira_stability([["a", "b"]] * 3, features) == pytest.approx(1.0)
    assert nogueira_stability([["a", "b"], ["c", "d"]], features) < 0


def test_fork_era_names():
    assert parse_method_name("LightGBM_c1_BAG_L1") == {"model": "LightGBM", "config": "c1", "pipeline": "default"}
    assert method_label("FSBench__AccuracyFeatureSelector__5__0__lgbm__3600") == "LOCO"
    assert method_label("FSBench__ANOVAFeatureSelector__5__0__lgbm__3600") == "F-Test"
