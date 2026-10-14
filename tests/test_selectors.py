from __future__ import annotations

import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from selectarena.methods import FEATURE_SELECTION_METHODS, ProxyModelConfig, get_feature_selector_from_name

logger = logging.getLogger(__name__)

# LightGBM is the benchmark's downstream and proxy model. Fall back to a random forest so the
# selectors can still be exercised in an environment without the compiled LightGBM wheel.
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
if HAS_LIGHTGBM:
    from autogluon.tabular.models import LGBModel as ProxyModel

    MODEL_HYPERPARAMETERS = {"GBM": {"num_boost_round": 10}}
    PROXY_HYPERPARAMETERS = {"num_boost_round": 1}
else:
    from autogluon.tabular.models import RFModel as ProxyModel

    MODEL_HYPERPARAMETERS = {"RF": {"n_estimators": 10}}
    PROXY_HYPERPARAMETERS = {"n_estimators": 5}

# Three small synthetic datasets (binary, multiclass, regression; with missing values) x every method.
DATASET_CONFIGS: list[tuple[str, str]] = [
    ("binary", "roc_auc"),
    ("multiclass", "log_loss"),
    ("regression", "rmse"),
]
test_params = []
for problem_type, evaluation_metric in DATASET_CONFIGS:
    for method_name in FEATURE_SELECTION_METHODS:
        test_params.append((problem_type, evaluation_metric, method_name))


@pytest.mark.parametrize(
    ("problem_type", "evaluation_metric", "method_name"),
    test_params,
    ids=[f"{method_name}_{problem_type}" for problem_type, evaluation_metric, method_name in test_params],
)
def test_feature_selector_dataset_combo(problem_type, evaluation_metric, method_name, tmp_path, verbosity=0):
    missing_values = True
    only_positive = True
    missing_values_target = False
    only_positive_target = True
    if problem_type == "binary":
        X, y = make_classification(n_samples=20, n_features=5, n_classes=2, random_state=0)
    elif problem_type == "multiclass":
        X, y = make_classification(
            n_samples=20, n_features=5, n_classes=3, n_informative=4, n_redundant=1, n_repeated=0, random_state=0
        )
    else:
        X, y = make_regression(n_samples=20, n_features=5, random_state=0)
        if only_positive_target:
            y = np.abs(y.astype(int))
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    rng = np.random.default_rng(0)  # fixed data: a failure is the method's, not an unlucky draw
    for col in X.columns:
        if missing_values:
            missing_mask = rng.random(len(X)) < 0.1
            X.loc[missing_mask, col] = np.nan
        if only_positive:
            X[col] = np.abs(X[col].astype(float))
    if missing_values_target:
        y_mask = rng.random(len(y)) < 0.1
        y[y_mask] = np.nan
    data = pd.concat([X, pd.Series(y, name="target")], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=0)

    proxy_config = ProxyModelConfig(
        model_class=ProxyModel,
        problem_type=problem_type,
        eval_metric=evaluation_metric,
        model_hyperparameters=PROXY_HYPERPARAMETERS,
    )

    max_features = 3
    feature_selector = get_feature_selector_from_name(
        name=method_name,
    )
    feature_selector = feature_selector(max_features=max_features, proxy_mode_config=proxy_config)

    predictor = TabularPredictor(
        label="target",
        default_base_path=str(tmp_path / "ag_out"),
        eval_metric=evaluation_metric,
        problem_type=problem_type,
    ).fit(
        train_data=train_data,
        hyperparameters=MODEL_HYPERPARAMETERS,
        num_bag_folds=2,
        num_bag_sets=1,
        verbosity=verbosity,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        _feature_generator_kwargs={
            "post_generators": [feature_selector],
        },
    )

    leaderboard = predictor.leaderboard(data=test_data, silent=True)
    assert not leaderboard.empty, f"Leaderboard empty for {method_name} on a {problem_type} dataset"

    assert feature_selector._selected_features is not None
    assert len(feature_selector._selected_features) <= max_features
    # Each selection entry is a {"feature": name, "value": ...} dict.
    assert all(set(entry) == {"feature", "value"} for entry in feature_selector._selected_features)
    selected_names = [entry["feature"] for entry in feature_selector._selected_features]
    assert all(f in train_data.columns for f in selected_names)
