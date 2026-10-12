"""Gain ratio feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from selectarena.methods.abstract import AbstractITFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class GainRatioFeatureSelector(AbstractITFeatureSelector):
    """GainRatio Feature Selection.

    Reference: Quinlan, J. Ross. "Induction of decision trees."
    Machine learning 1.1 (1986): 81-106.
    IGR(Y,X) = IG(Y,X)/SI(X)
    IG(Y,X) = H(Y) - H(Y|X) = MI(Y,X)
    SI(X) = H(X)
    Changes to the algorithm:
        - Time constraint
    """

    name = "GainRatioFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        igr_scores = {}

        for col in X_pre.columns:
            if self._timed_out(time_limit, start_time):
                break
            feature = X_pre[col]
            info_gain = self._mutual_information_dispatched(feature, y)
            split_info = self._entropy(feature)
            igr_scores[col] = info_gain / split_info if split_info > 0 else 0.0

        return igr_scores
