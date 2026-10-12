"""Accuracy-based feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from autogluon.core.utils.exceptions import TimeLimitExceeded

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class LOCOFeatureSelector(AbstractFeatureSelector):
    """Leave-One-Covariate-Out Feature Selection."""

    name = "LOCOFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        feature_scores = {}  # Store feature scores, higher is better

        try:  # safeguard for AG model throwing a TimeExceededlimit error that would reset the already selected feature set
            time_to_fit = max(0.0, time_limit - (time.monotonic() - start_time)) if time_limit is not None else None
            baseline_score = self.evaluate_proxy_model(X=X, y=y, time_limit=time_to_fit)

            # columns are permutated per split to vary what gets evaluated in case time limit is exceeded
            for feature in self._seeded_feature_order():
                if self._timed_out(time_limit, start_time):
                    break

                evaluate_X = X.drop(columns=[feature]).copy()  # Evaluate proxy model without the feature
                time_to_fit = max(0.0, time_limit - (time.monotonic() - start_time)) if time_limit is not None else None
                score = self.evaluate_proxy_model(X=evaluate_X, y=y, time_limit=time_to_fit)
                del evaluate_X  # free up memory

                if score is None:
                    continue
                # how much accuracy is lost without the feature (the higher the difference, the more important the feature)
                feature_scores[feature] = baseline_score - score
        except TimeLimitExceeded:
            self._log(30, f"TimeLimitExceeded during scoring. Returning {len(feature_scores)} partial scores.")

        return feature_scores
