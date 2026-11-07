"""Conditional Permutation Importance (CPI) feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.utils.exceptions import NoValidFeatures, TimeLimitExceeded

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    from selectarena.methods.abstract import FittedProxyModel


class CPIFeatureSelector(AbstractFeatureSelector):
    """Conditional Permutation Importance (CPI) Feature Selection.

    Scores each feature by how much the proxy model's held-out loss rises when the feature is
    replaced by a draw from its distribution given the other features (Algorithm 1 of the paper).
    The proxy model (LightGBM by default; FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL) is fitted once
    on all features. Up to ``n_eval_samples`` of its held-out rows are the evaluation rows. For each
    feature ``j``, a random forest (``n_estimators`` trees of depth at most ``max_depth``) is fitted
    on these rows to predict ``x_j`` from the other features. The conditional draws are then
    generated in one of two ways:

    * Continuous ``x_j``: the forest's prediction plus a permutation of its residuals (the paper's
      additive construction).
    * Discrete ``x_j``: values sampled from the forest's predicted class probabilities. A feature
      is discrete if it is categorical or has at most ``categorical_max_cardinality`` distinct
      values, as in hidimstat.

    The score is the per-row loss increase, averaged over the evaluation rows and
    ``n_permutations`` draws. The loss is the squared error for regression and the log loss of the
    true class for classification (the paper's Eq. 6). Higher is more important.

    The paper's inference (a z-test on the per-row loss increases) and its 2-fold cross-fitting are
    not used: features are ranked by their importance estimate, on the proxy model's single split.
    Inputs are imputed and ordinal-encoded before the proxy fit, as the conditional forests need
    numeric data.

    Time limit: the proxy fit gets the remaining time, then the features are scored one at a time in
    a per-split random order until the time runs out. The random fallback fills the budget from the
    features left unscored. If the proxy fit times out, the method returns no scores.

    Reference: Chamma, Ahmad, Denis A. Engemann, and Bertrand Thirion. "Statistically Valid Variable
    Importance Assessment through Conditional Permutations." Advances in Neural Information Processing
    Systems 36 (2023).
    Implementation: own implementation of the paper's Algorithm 1 with the proxy model as the learner.
    The conditional sampling follows ``hidimstat.CFI`` (https://github.com/mind-inria/hidimstat,
    BSD-3-Clause), and the random-forest conditional model follows the paper's reference code
    (https://github.com/achamma723/Variable_Importance).
    """

    name = "CPIFeatureSelector"
    feature_scoring_method: bool = True

    n_permutations: int = 50
    """Conditional draws per feature (the default of the reference code and of hidimstat)."""
    n_eval_samples: int = 1000
    """Maximum number of held-out rows of the proxy split on which the loss increase is measured."""
    n_estimators: int = 100
    """Trees of each conditional random forest."""
    max_depth: int = 5
    """Depth of the conditional random forests (the reference code's default)."""
    categorical_max_cardinality: int = 10
    """Numeric features with at most this many distinct values are sampled as discrete (hidimstat's default)."""
    max_rows_per_predict: int = 20_000
    """Rows per proxy prediction call: the draws of one feature are predicted in chunks of this size."""

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)
        categorical = set(X.select_dtypes(include=["object", "category"]).columns)
        discrete = {
            col: col in categorical or X_pre[col].nunique() <= self.categorical_max_cardinality for col in X_pre.columns
        }
        remaining = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - start_time))
        try:
            with self.fitted_proxy_model(X=X_pre, y=y, time_limit=remaining) as proxy:
                return self._cpi_scores(proxy=proxy, discrete=discrete, time_limit=time_limit, start_time=start_time)
        except (TimeLimitExceeded, NoValidFeatures) as error:
            self._log(30, f"Proxy model could not be fitted ({type(error).__name__}). Returning no scores.")
            return {}

    def _cpi_scores(
        self, *, proxy: FittedProxyModel, discrete: dict[str, bool], time_limit: int | None, start_time: float
    ) -> dict[str, float]:
        columns = list(proxy.X_val.columns)
        eval_idx = self._rng.permutation(len(proxy.X_val))[: self.n_eval_samples]
        X_eval = proxy.X_val.to_numpy(dtype=float)[eval_idx]
        if proxy.problem_type in ("binary", "multiclass"):
            labels = proxy.encode_labels(proxy.y_val)[eval_idx]
            known = np.isfinite(labels)  # drop rows of a class absent from the proxy's training rows
            X_eval, labels = X_eval[known], labels[known].astype(int)

            def row_losses(prediction: np.ndarray) -> np.ndarray:  # (draws, rows, classes) -> (draws, rows)
                return -np.log(np.clip(prediction[:, np.arange(len(labels)), labels], 1e-10, 1.0))
        else:
            target = proxy.y_val.to_numpy(dtype=float)[eval_idx]

            def row_losses(prediction: np.ndarray) -> np.ndarray:  # (draws, rows) -> (draws, rows)
                return (target - prediction) ** 2

        n_rows = len(X_eval)
        if n_rows < 2:
            self._log(30, "Fewer than two held-out rows to evaluate on. Returning no scores.")
            return {}

        rows_per_call = max(1, self.max_rows_per_predict // n_rows) * n_rows  # whole copies per call

        def predict(X_values: np.ndarray) -> np.ndarray:
            """Proxy predictions for stacked copies of the evaluation rows, shape (draws, rows, ...)."""
            flat = X_values.reshape(-1, X_values.shape[-1])
            prediction = np.concatenate(
                [
                    proxy.predict(pd.DataFrame(flat[i : i + rows_per_call], columns=columns))
                    for i in range(0, len(flat), rows_per_call)
                ]
            )
            return prediction.reshape(len(X_values), n_rows, *prediction.shape[1:])

        reference_loss = row_losses(predict(X_eval[np.newaxis]))  # (1, rows)
        scores = {}
        for feature in self._seeded_feature_order():
            if self._timed_out(time_limit, start_time):
                break
            j = columns.index(feature)
            draws = self._conditional_draws(X_eval=X_eval, j=j, discrete=discrete[feature])  # (draws, rows)
            X_perturbed = np.repeat(X_eval[np.newaxis], len(draws), axis=0)
            X_perturbed[:, :, j] = draws
            loss_increase = row_losses(predict(X_perturbed)) - reference_loss
            scores[feature] = float(np.nan_to_num(loss_increase.mean(), nan=-np.inf))
        self._log(20, f"\tScored {len(scores)} of {len(columns)} features on {n_rows} held-out rows.")
        return scores

    def _conditional_draws(self, *, X_eval: np.ndarray, j: int, discrete: bool) -> np.ndarray:
        """``n_permutations`` draws of column ``j`` given the other columns, shape (draws, rows)."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        x_j, others = X_eval[:, j], np.delete(X_eval, j, axis=1)
        forest_kwargs = dict(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=-1
        )
        if not discrete:
            prediction = RandomForestRegressor(**forest_kwargs).fit(others, x_j).predict(others)
            residuals = x_j - prediction
            return prediction + np.stack([self._rng.permutation(residuals) for _ in range(self.n_permutations)])
        values, codes = np.unique(x_j, return_inverse=True)  # integer classes, also for float-valued columns
        forest = RandomForestClassifier(**forest_kwargs).fit(others, codes)
        cumulative = np.cumsum(forest.predict_proba(others), axis=1)  # (rows, classes)
        uniform = self._rng.random((self.n_permutations, len(x_j), 1))
        drawn = np.minimum((uniform > cumulative[np.newaxis]).sum(axis=2), len(forest.classes_) - 1)
        return values[forest.classes_[drawn]]
