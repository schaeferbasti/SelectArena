"""SHAP feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.utils.exceptions import NoValidFeatures, TimeLimitExceeded

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    from selectarena.methods.abstract import FittedProxyModel


class SHAPFeatureSelector(AbstractFeatureSelector):
    """SHAP Feature Selection.

    Scores each feature by its mean absolute Shapley value for the proxy model (LightGBM by default,
    fitted once on all features; FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL). The Shapley values
    (index ``SV``, order 1) are estimated by KernelSHAP with marginal imputation from
    ``n_background_samples`` training rows of the proxy split. They explain up to
    ``n_explained_samples`` held-out rows of that split. For classification each row's explained
    output is the predicted probability of its true class: for binary tasks this gives the same
    absolute values as explaining the positive class, and for multiclass tasks it costs one
    explanation per row instead of one per class. For regression the explained output is the
    prediction. The KernelSHAP budget is ``min(2**d, 2048 + 2 * d)`` coalitions for ``d`` features,
    which is exact for up to 11 features.

    Time limit: the proxy fit gets the remaining time, then rows are explained one at a time until
    the time runs out. The scores average over the rows explained so far, and each row gives a value
    for every feature. If the proxy fit times out or no row is explained, the method returns no
    scores and the random fallback selects the features.

    Reference: Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions."
    Advances in Neural Information Processing Systems 30 (2017).
    Implementation: ``shapiq.TabularExplainer`` (Muschalik et al., "shapiq: Shapley Interactions for
    Machine Learning", NeurIPS 2024), https://github.com/mmschlk/shapiq, MIT licence.
    """

    name = "SHAPFeatureSelector"
    feature_scoring_method: bool = True

    n_explained_samples: int = 100
    """Maximum number of held-out rows whose predictions are explained (the scores average over them)."""
    n_background_samples: int = 100
    """Training rows the marginal imputer draws replacement values from (shapiq's default sample size)."""

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True, scale=True)
        remaining = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - start_time))
        try:
            with self.fitted_proxy_model(X=X_pre, y=y, time_limit=remaining) as proxy:
                return self._shap_scores(proxy=proxy, time_limit=time_limit, start_time=start_time)
        except (TimeLimitExceeded, NoValidFeatures) as error:
            self._log(30, f"Proxy model could not be fitted ({type(error).__name__}). Returning no scores.")
            return {}

    def _shap_scores(self, *, proxy: FittedProxyModel, time_limit: int | None, start_time: float) -> dict[str, float]:
        import shapiq

        columns = list(proxy.X_train.columns)
        n_features = len(columns)
        budget = int(min(2**n_features, 2048 + 2 * n_features))
        background_idx = self._rng.permutation(len(proxy.X_train))[: self.n_background_samples]
        background = proxy.X_train.to_numpy(dtype=float)[background_idx]
        explained_idx = self._rng.permutation(len(proxy.X_val))[: self.n_explained_samples]
        X_explained = proxy.X_val.to_numpy(dtype=float)[explained_idx]
        if proxy.problem_type in ("binary", "multiclass"):
            targets = proxy.encode_labels(proxy.y_val)[explained_idx]
        else:
            targets = np.zeros(len(explained_idx))  # one output to explain

        explainers = {}  # target class -> explainer (its empty-coalition value depends on the class)

        def explainer_for(target: int) -> shapiq.TabularExplainer:
            if target not in explainers:

                def model_output(data: np.ndarray) -> np.ndarray:
                    prediction = proxy.predict(pd.DataFrame(data, columns=columns))
                    return prediction if prediction.ndim == 1 else prediction[:, target]

                explainers[target] = shapiq.TabularExplainer(
                    model=model_output,
                    data=background,
                    index="SV",
                    max_order=1,
                    random_state=self.random_state,
                    sample_size=self.n_background_samples,
                )
            return explainers[target]

        abs_sum, n_explained = np.zeros(n_features), 0
        for x, target in zip(X_explained, targets, strict=True):
            if self._timed_out(time_limit, start_time):
                break
            if not np.isfinite(target):  # class absent from the proxy's training rows
                continue
            values = explainer_for(int(target)).explain(x, budget=budget).get_n_order_values(1)
            abs_sum += np.abs(values)
            n_explained += 1
        self._log(20, f"\tExplained {n_explained} rows with a budget of {budget} coalitions.")
        if n_explained == 0:
            return {}
        scores = np.nan_to_num(abs_sum / n_explained, nan=-np.inf)
        return dict(zip(columns, scores.tolist(), strict=True))
