"""Mutual information feature selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class MIFeatureSelector(AbstractFeatureSelector):
    """MI Feature Selection.

    Reference: Battiti, Roberto. "Using mutual information for selecting features in supervised neural net learning."
    IEEE Transactions on neural networks 5.4 (1994): 537-550.

    Implementation: sklearn (uses k-NN based MI estimator (Kraskov et al. 2004) for continuous attributes).
    """

    name = "MIFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        # categorical columns treated as discrete (ordinal encoding), binary prob represented as ints and processed as continuous (incorrectly)
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
        discrete_mask = X.columns.isin(cat_cols)
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)

        mi_func = mutual_info_regression if self.problem_type == "regression" else mutual_info_classif
        scores = mi_func(
            X_pre,
            y,
            discrete_features=discrete_mask,
            random_state=self.random_state,
        )
        return dict(zip(X.columns, scores))
