"""Elastic net feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV

from selectarena.methods.abstract import AbstractFeatureSelector


class ElasticNetFeatureSelector(AbstractFeatureSelector):
    """ElasticNet Feature Selection.

    Reference: Zou, Hui, and Trevor Hastie. "Regularization and variable selection via the elastic net."
    Journal of the Royal Statistical Society Series B: Statistical Methodology 67.2 (2005): 301-320.
    Implementation Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    """

    name = "ElasticNetFeatureSelector"
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
            model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], random_state=self.random_state)
        else:
            model = LogisticRegressionCV(
                penalty="elasticnet", solver="saga", l1_ratios=[0.1, 0.5, 0.7, 0.9], random_state=self.random_state
            )
        model.fit(X_pre, y)

        coef = model.coef_
        if coef.ndim == 2:
            scores = np.abs(coef).mean(axis=0)  # for multiclass average over classes
        else:
            scores = np.abs(coef)
        return dict(zip(X_pre.columns, scores))
