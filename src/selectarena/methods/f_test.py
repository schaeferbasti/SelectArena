"""ANOVA feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from selectarena.methods.abstract import AbstractFeatureSelector


class FTestFeatureSelector(AbstractFeatureSelector):
    """ANOVA Feature Selection.

    Reference: St, Lars, and Svante Wold. "Analysis of variance(ANOVA)."
    Chemometrics and intelligent laboratory systems 6.4 (1989): 259-272.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
    """

    name = "FTestFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
    ) -> dict[str, float]:
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)

        score_func = f_regression if self.problem_type == "regression" else f_classif
        anova = SelectKBest(score_func=score_func, k="all")
        anova.fit(X_pre, y)
        scores = np.nan_to_num(anova.scores_, nan=0.0)  # in case of constant features that return nan
        return dict(zip(X_pre.columns, scores))
