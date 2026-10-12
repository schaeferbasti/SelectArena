"""Lasso feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier

from selectarena.methods.abstract import AbstractFeatureSelector


class LassoFeatureSelector(AbstractFeatureSelector):
    """Lasso Feature Selection.

    Reference: Tibshirani, Robert. "Regression shrinkage and selection via the lasso."
    Journal of the Royal Statistical Society Series B:  Statistical Methodology 58.1 (1996): 267-288.
    Implementation Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    name = "LassoFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
    ) -> dict[str, float]:
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True, scale=True)

        if self.problem_type == "regression":
            lasso = LassoCV(random_state=self.random_state)
        else:
            lasso = LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                random_state=self.random_state,
            )
        if self.problem_type == "multiclass":
            # liblinear is binary-only. scikit-learn < 1.8 silently fitted one-vs-rest models for
            # multiclass targets; newer versions raise, so the one-vs-rest scheme is explicit here.
            lasso = OneVsRestClassifier(lasso).fit(X_pre, y)
            coef = np.vstack([estimator.coef_ for estimator in lasso.estimators_])
        else:
            lasso.fit(X_pre, y)
            coef = lasso.coef_
        if coef.ndim == 2:
            scores = np.abs(coef).mean(axis=0)  # for multiclass average over classes
        else:
            scores = np.abs(coef)
        return dict(zip(X_pre.columns, scores))
