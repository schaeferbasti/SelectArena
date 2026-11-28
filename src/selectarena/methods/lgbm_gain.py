"""LightGBM gain feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class LGBMGainFeatureSelector(AbstractFeatureSelector):
    """LGBMGain Feature Selection.

    Scores each feature by the total gain of its splits in a LightGBM model with the library's default
    hyperparameters (100 trees), fitted on all training rows. Two departures from the defaults:

    * Gain replaces the default importance type ``split`` (the number of splits, which favors features
      with many distinct values).
    * Below 80 rows ``min_child_samples`` is a quarter of the rows: with the default of 20, no split is
      possible below 40 rows and every feature would score 0.

    Features without a split tie at 0 and are ranked in a seeded random order. The model is not the
    benchmark's proxy model, which trains 10000 trees on two thirds of the rows.

    Time limit: boosting stops after the first tree that ends past the limit, and the scores come from
    the trees grown so far. Building LightGBM's histograms and growing one tree are not interrupted.

    Reference: Ke, Guolin, et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
    Advances in Neural Information Processing Systems 30 (2017).
    Implementation: ``lightgbm.LGBMClassifier`` / ``LGBMRegressor``, https://github.com/microsoft/LightGBM,
    MIT licence.
    """

    name = "LGBMGainFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from lightgbm.callback import EarlyStopException

        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)

        def stop_at_time_limit(env) -> None:
            if self._timed_out(time_limit, start_time):
                raise EarlyStopException(env.iteration, [])

        model_cls = LGBMRegressor if self.problem_type == "regression" else LGBMClassifier
        model = model_cls(
            importance_type="gain",
            min_child_samples=max(1, min(20, len(X) // 4)),
            random_state=self.random_state,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,  # without it LightGBM times both histogram layouts, which is not deterministic
            verbosity=-1,
        )
        # As an array: LightGBM rejects feature names with JSON characters.
        model.fit(X_pre.to_numpy(dtype=float), y, callbacks=[stop_at_time_limit])
        gains = dict(zip(X_pre.columns, model.feature_importances_.tolist(), strict=True))
        # The ranking sort is stable, so this order breaks ties (not the column order).
        return {feature: gains[feature] for feature in self._seeded_feature_order()}
