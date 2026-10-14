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

#: Method names as used in the paper (Table 2, Figure 1).
PAPER_NAMES: dict[str, str] = {
    "LOCOFeatureSelector": "LOCO",
    "RandomFeatureSelector": "Random",
    "FTestFeatureSelector": "F-Test",
    "CARTFeatureSelector": "CART",
    "ElasticNetFeatureSelector": "ElasticNet",
    "GainRatioFeatureSelector": "GainRatio",
    "LaplacianScoreFeatureSelector": "LaplacianScore",
    "LassoFeatureSelector": "Lasso",
    "MarkovBlanketFeatureSelector": "MarkovBlanket",
    "MIFeatureSelector": "MI",
    "mRMRFeatureSelector": "mRMR",
    "ReliefFFeatureSelector": "(R)ReliefF",
    "RFImportanceFeatureSelector": "RFImportance",
    "SequentialBackwardEliminationFeatureSelector": "SBE",
    "SequentialForwardSelectionFeatureSelector": "SFS",
}

#: Class names in results written by tabarena_fork_fs before the methods were renamed, mapped to
#: their current class. Reading old results through these keeps one name per method.
LEGACY_NAMES: dict[str, str] = {
    "AccuracyFeatureSelector": "LOCOFeatureSelector",
    "ANOVAFeatureSelector": "FTestFeatureSelector",
}

#: Subgroup of each method in the paper's functional taxonomy (Table 2 and Appendix A): search
#: methods evaluate many feature subsets with a model-dependent utility, non-search methods rely on a
#: proxy (univariate relevance, distances, information theory, trees, regularization). Shown in the
#: leaderboard's Type column; every registered method needs an entry, or it shows as "Other".
METHOD_SUBGROUPS: dict[str, str] = {
    "CARTFeatureSelector": "Tree",
    "ElasticNetFeatureSelector": "Regularization",
    "FTestFeatureSelector": "Correlation",
    "GainRatioFeatureSelector": "Information theory",
    "LaplacianScoreFeatureSelector": "Distance",
    "LassoFeatureSelector": "Regularization",
    "LOCOFeatureSelector": "Leave-one-out",
    "MarkovBlanketFeatureSelector": "Backward search",
    "mRMRFeatureSelector": "Information theory",
    "MIFeatureSelector": "Information theory",
    "RandomFeatureSelector": "Baseline",
    "ReliefFFeatureSelector": "Distance",
    "RFImportanceFeatureSelector": "Tree",
    "SequentialBackwardEliminationFeatureSelector": "Backward search",
    "SequentialForwardSelectionFeatureSelector": "Forward search",
}

#: Category of each subgroup (Figure A.1). A new subgroup needs an entry here as well.
SUBGROUP_CATEGORIES: dict[str, str] = {
    "Forward search": "Search",
    "Backward search": "Search",
    "Leave-one-out": "Search",
    "Correlation": "Non-search",
    "Distance": "Non-search",
    "Information theory": "Non-search",
    "Tree": "Non-search",
    "Regularization": "Non-search",
    "Baseline": "Baseline",
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
