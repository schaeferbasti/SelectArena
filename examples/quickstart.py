"""Quickstart: run every SelectArena feature selector inside an AutoGluon pipeline.

Each selector is an AutoGluon feature generator, so it plugs into ``TabularPredictor.fit`` as a
post generator. Downloads OpenML dataset 55 (hepatitis) on first run.

    python examples/quickstart.py                          # every registered method
    python examples/quickstart.py MIFeatureSelector        # only the named methods
"""

from __future__ import annotations

import sys
import tempfile

import openml
import pandas as pd
from autogluon.core.data import LabelCleaner
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from selectarena.methods import (
    FEATURE_SELECTION_METHODS,
    FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL,
    ProxyModelConfig,
    get_feature_selector_from_name,
)


def run_example(method_names: list[str] | None = None) -> None:
    dataset_id = 55
    problem_type = "binary"
    eval_metric = "roc_auc"

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    y = LabelCleaner.construct(problem_type=problem_type, y=y, verbose=False).transform(y)
    data = pd.concat([X, y.rename("class")], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=0)

    max_features = 5
    proxy_model_config = ProxyModelConfig(
        problem_type=problem_type,
        eval_metric=eval_metric,
        model_hyperparameters={"num_boost_round": 5},
    )

    for method_name in method_names or FEATURE_SELECTION_METHODS:
        selector_cls = get_feature_selector_from_name(name=method_name)
        proxy = proxy_model_config if method_name in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL else None
        feature_selector = selector_cls(max_features=max_features, proxy_mode_config=proxy)

        print(f"\n####### Running feature selector: {method_name}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            predictor = TabularPredictor(
                label="class", path=tmp_dir, eval_metric=eval_metric, problem_type=problem_type
            ).fit(
                train_data=train_data,
                hyperparameters={"GBM": {"num_boost_round": 100}},
                num_bag_folds=8,
                num_bag_sets=1,
                verbosity=0,
                dynamic_stacking=False,
                fit_weighted_ensemble=False,
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
                _feature_generator_kwargs={"post_generators": [feature_selector]},
                time_limit=3600,
                time_limit_preprocessing=3600,
            )
            print(predictor.leaderboard(data=test_data)[["model", "score_test", "score_val"]])

        print("Selected features:", [entry["feature"] for entry in feature_selector._selected_features])
        print("Feature scores:", feature_selector._feature_scores)


if __name__ == "__main__":
    run_example(sys.argv[1:] or None)
