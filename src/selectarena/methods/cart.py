"""CART (decision tree) feature selection."""

from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from selectarena.methods.abstract import AbstractFeatureSelector


class CARTFeatureSelector(AbstractFeatureSelector):
    """CART Feature Selection.

    Reference: Breiman, Leo, et al. Classification and regression trees. Chapman and Hall/CRC, 2017.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    name = "CARTFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
    ) -> dict[str, float]:
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)

        if self.problem_type == "regression":
            cart = DecisionTreeRegressor(random_state=self.random_state)
        else:
            cart = DecisionTreeClassifier(random_state=self.random_state)
        cart.fit(X_pre, y)
        importances = cart.feature_importances_
        return dict(zip(X_pre.columns, importances))
