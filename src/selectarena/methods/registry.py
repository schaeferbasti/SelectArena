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
    "LassoNetFeatureSelector": "lasso_net",
    "MarkovBlanketFeatureSelector": "markov_blanket",
    "MIFeatureSelector": "mi",
    "mRMRFeatureSelector": "mrmr",
    "ReliefFFeatureSelector": "relief_f",
    "RFImportanceFeatureSelector": "rf_importance",
    "SequentialBackwardEliminationFeatureSelector": "sbe",
    "SequentialForwardSelectionFeatureSelector": "sfs",
    "SHAPFeatureSelector": "shap",
}
FEATURE_SELECTION_METHODS: list[str] = list(NAME_TO_MODULE_MAP.keys())
#: Methods that fit a proxy model internally (``evaluate_proxy_model`` or ``fitted_proxy_model``) and
#: therefore need a ``ProxyModelConfig``.
FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL: list[str] = [
    "LOCOFeatureSelector",
    "SequentialBackwardEliminationFeatureSelector",
    "SequentialForwardSelectionFeatureSelector",
    "SHAPFeatureSelector",
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
    "LassoNetFeatureSelector": "LassoNet",
    "MarkovBlanketFeatureSelector": "MarkovBlanket",
    "MIFeatureSelector": "MI",
    "mRMRFeatureSelector": "mRMR",
    "ReliefFFeatureSelector": "(R)ReliefF",
    "RFImportanceFeatureSelector": "RFImportance",
    "SequentialBackwardEliminationFeatureSelector": "SBE",
    "SequentialForwardSelectionFeatureSelector": "SFS",
    "SHAPFeatureSelector": "SHAP",
}

#: Link to the paper each method is cited with in the paper (Table 2 and Appendix B.4; the links of
#: Appendix C.1, a DOI where there is one). The leaderboard links the method's name to it. Optional: a method
#: without a paper (Random, or a new method not published yet) shows its name without a link.
METHOD_PAPERS: dict[str, str] = {
    "CARTFeatureSelector": "https://doi.org/10.1201/9781315139470",  # Breiman et al., 1984
    "ElasticNetFeatureSelector": "https://doi.org/10.1111/j.1467-9868.2005.00503.x",  # Zou & Hastie, 2005
    "FTestFeatureSelector": "https://doi.org/10.1016/0169-7439(89)80095-4",  # Ståhle et al., 1989
    "GainRatioFeatureSelector": "https://doi.org/10.1007/BF00116251",  # Quinlan, 1986
    "LaplacianScoreFeatureSelector": (  # He et al., 2005
        "https://proceedings.neurips.cc/paper_files/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf"
    ),
    "LassoFeatureSelector": "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",  # Tibshirani, 1996
    "LassoNetFeatureSelector": "https://arxiv.org/abs/1907.12207",  # Lemhadri et al., 2021 (JMLR)
    "LOCOFeatureSelector": "https://arxiv.org/abs/1604.04173",  # Lei et al., 2017 (Appendix B.4.7)
    "MarkovBlanketFeatureSelector": "http://ilpubs.stanford.edu:8090/208/",  # Koller & Sahami, 1996
    "mRMRFeatureSelector": "https://doi.org/10.1109/TPAMI.2005.159",  # Peng et al., 2005
    "MIFeatureSelector": "https://doi.org/10.1109/72.298224",  # Battiti, 1994
    "ReliefFFeatureSelector": "https://doi.org/10.1023/A:1008280620621",  # Kononenko et al., 1997
    "RFImportanceFeatureSelector": "https://doi.org/10.1023/A:1010933404324",  # Breiman, 2001
    "SequentialBackwardEliminationFeatureSelector": "https://doi.org/10.1023/A:1012487302797",  # Guyon et al., 2002
    "SequentialForwardSelectionFeatureSelector": "https://doi.org/10.1109/TSMC.1977.4309803",  # Cover & Van Campenhout, 1977
    "SHAPFeatureSelector": "https://arxiv.org/abs/1705.07874",  # Lundberg & Lee, 2017 (NeurIPS)
}

#: Former class names of renamed methods (as in the published stability / validity results), mapped
#: to their current class, so every method keeps one name.
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
    "LassoNetFeatureSelector": "Regularization",
    "LOCOFeatureSelector": "Leave-one-out",
    "MarkovBlanketFeatureSelector": "Backward search",
    "mRMRFeatureSelector": "Information theory",
    "MIFeatureSelector": "Information theory",
    "RandomFeatureSelector": "Baseline",
    "ReliefFFeatureSelector": "Distance",
    "RFImportanceFeatureSelector": "Tree",
    "SequentialBackwardEliminationFeatureSelector": "Backward search",
    "SequentialForwardSelectionFeatureSelector": "Forward search",
    "SHAPFeatureSelector": "Game theory",
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
    "Game theory": "Non-search",
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
