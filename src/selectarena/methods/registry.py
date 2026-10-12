"""Registry of the feature selection methods benchmarked in SelectArena."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from selectarena.methods.abstract import AbstractFeatureSelector

#: Class name of each method mapped to the module (under ``selectarena.methods``) defining it.
NAME_TO_MODULE_MAP: dict[str, str] = {
    "LOCOFeatureSelector": "loco",
    "RandomFeatureSelector": "random_baseline",
    "FTestFeatureSelector": "f_test",
    "CARTFeatureSelector": "cart",
    "ElasticNetFeatureSelector": "elastic_net",
    "GainRatioFeatureSelector": "gain_ratio",
    "LaplacianScoreFeatureSelector": "laplacian_score",
    "LassoFeatureSelector": "lasso",
    "MarkovBlanketFeatureSelector": "markov_blanket",
    "MIFeatureSelector": "mi",
    "mRMRFeatureSelector": "mrmr",
    "ReliefFFeatureSelector": "relief_f",
    "RFImportanceFeatureSelector": "rf_importance",
    "SequentialBackwardEliminationFeatureSelector": "sbe",
    "SequentialForwardSelectionFeatureSelector": "sfs",
}
FEATURE_SELECTION_METHODS: list[str] = list(NAME_TO_MODULE_MAP.keys())
#: Methods that fit a proxy model internally and therefore need a ``ProxyModelConfig``.
FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL: list[str] = [
    "LOCOFeatureSelector",
    "SequentialBackwardEliminationFeatureSelector",
    "SequentialForwardSelectionFeatureSelector",
]

#: Names used in the paper for methods whose class name differs.
PAPER_NAMES: dict[str, str] = {
    "LOCOFeatureSelector": "LOCO",
    "RandomFeatureSelector": "Random",
    "FTestFeatureSelector": "F-Test",
    "CARTFeatureSelector": "CART",
    "ElasticNetFeatureSelector": "Elastic Net",
    "GainRatioFeatureSelector": "Gain Ratio",
    "LaplacianScoreFeatureSelector": "Laplacian Score",
    "LassoFeatureSelector": "Lasso",
    "MarkovBlanketFeatureSelector": "Markov Blanket",
    "MIFeatureSelector": "MI",
    "mRMRFeatureSelector": "mRMR",
    "ReliefFFeatureSelector": "(R)ReliefF",
    "RFImportanceFeatureSelector": "RF Importance",
    "SequentialBackwardEliminationFeatureSelector": "RFE",
    "SequentialForwardSelectionFeatureSelector": "SFS",
}


def get_feature_selector_from_name(*, name: str) -> type[AbstractFeatureSelector]:
    """Return the feature selector class registered under ``name``.

    Raises:
        ValueError: If ``name`` is not a registered method.
    """
    if name not in NAME_TO_MODULE_MAP:
        raise ValueError(f"Method name '{name}' is not recognized. Options are: {FEATURE_SELECTION_METHODS}")
    module = importlib.import_module(f"selectarena.methods.{NAME_TO_MODULE_MAP[name]}")
    return getattr(module, name)
