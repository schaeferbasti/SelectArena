"""Random forest importance feature selection."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from selectarena.methods.abstract import AbstractFeatureSelector


class RFImportanceFeatureSelector(AbstractFeatureSelector):
    """RFImportance Feature Selection.

    Reference: Breiman, Leo. "Random forests."
    Machine learning 45.1 (2001): 5-32.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    name = "RFImportanceFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
    ) -> dict[str, float]:
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)

        rf_class = RandomForestRegressor if self.problem_type == "regression" else RandomForestClassifier
        rf_model = rf_class(random_state=self.random_state, n_jobs=-1)
        rf_model.fit(X_pre, y)

        return dict(zip(X_pre.columns, rf_model.feature_importances_))
