"""Sequential backward elimination feature selection."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.utils.exceptions import TimeLimitExceeded

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SequentialBackwardEliminationFeatureSelector(AbstractFeatureSelector):
    """SequentialBackwardElimination Feature Selection.

    Implementation Source: Algorithm implemented by Bastian Schäfer
    (including time constraint using the autogluon model)
    """

    name = "SequentialBackwardEliminationFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[str]:
        start_time = time.monotonic()
        current_features = self._original_features.copy()

        if self.max_features >= len(self._original_features):  # trivial case
            return [{"feature": str(feat), "value": float("nan")} for feat in self._original_features]

        try:  # safeguard for AG model throwing a TimeExceededlimit error that would reset the already selected feature set
            while len(current_features) > self.max_features and current_features:
                if self._timed_out(time_limit, start_time):
                    break
                worst_score = -np.inf
                worst_feature = None

                for feature in current_features:
                    if self._timed_out(time_limit, start_time):
                        break

                    time_to_fit = None
                    if time_limit is not None:
                        remaining = time_limit - (time.monotonic() - start_time)
                        time_to_fit = max(0.0, remaining * 0.9)

                    test_X = X[[f for f in current_features if f != feature]]
                    score = self.evaluate_proxy_model(X=test_X, y=y, time_limit=time_to_fit)
                    del test_X

                    if score is None:
                        continue

                    if score > worst_score:
                        worst_score = score
                        worst_feature = feature

                if worst_feature is None:
                    break

                current_features.remove(worst_feature)

        except TimeLimitExceeded:
            self._log(30, f"TimeLimitExceeded during elimination. Returning {len(current_features)} selected features.")

        # Backward elimination exposes no natural per-feature value for the survivors -> NaN
        # (they were selected by the method, unlike random-fallback fills which are None).
        return [{"feature": str(feat), "value": float("nan")} for feat in current_features]
