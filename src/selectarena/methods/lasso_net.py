"""LassoNet feature selection."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class _TimeUp(Exception):
    """Raised inside LassoNet's training loop once the selection time has run out."""


def _optimizer_with_hook(optimizer_class: type, hook: Callable, **kwargs) -> Callable:
    """Factory for LassoNet's ``optim`` argument: the optimizer runs ``hook`` before every step."""

    def make(params):
        optimizer = optimizer_class(params, **kwargs)
        optimizer.register_step_pre_hook(hook)
        return optimizer

    return make


class LassoNetFeatureSelector(AbstractFeatureSelector):
    """LassoNet Feature Selection.

    LassoNet is a feed-forward network (one hidden layer of ``hidden_dims`` units) with a linear skip
    layer from the inputs to the outputs. An L1 penalty with weight lambda on the skip layer, combined
    with the hierarchy constraint (a feature's weights into the hidden layer are bounded by ``M`` times
    the norm of its skip-layer weights), removes a feature from the whole network once its skip-layer
    weights are zero. The method fits the dense network (lambda = 0), then fits a path of increasing
    lambda (a factor ``path_multiplier`` per step, each fit warm-started from the previous one) until
    no feature is left. A feature's score is the largest lambda on the path at which it is still in
    the model. LassoNet's ``feature_importances_`` is the next lambda, the one at which the feature
    leaves, so both give the same ranking. As in ``feature_importances_``, a feature that leaves and
    later comes back counts as removed. Features with the same score are ordered by the norm of their
    skip-layer weights at the last lambda at which they were in the model.

    Inputs are imputed, ordinal-encoded and standardized, and a regression target is standardized as
    well. A ``val_fraction`` of the rows is held out as the validation data of LassoNet's early
    stopping. For classification this split is stratified and keeps every class in the training rows.
    Everything else uses the library defaults. The network trains on the CPU, seeded with
    ``random_state``.

    Time limit: the training loop checks the time before every optimizer step (one epoch, since the
    batch is the full data). When the time runs out, the path ends at the last lambda whose fit was
    completed. The features still in the model then share that lambda as their score and are ordered
    by their skip-layer weights. If the dense fit does not finish, the method returns no scores and
    the random fallback selects the features.

    Reference: Lemhadri, Ismael, Feng Ruan, Louis Abraham, and Robert Tibshirani. "LassoNet: A Neural
    Network with Feature Sparsity." Journal of Machine Learning Research 22 (2021): 1-29.
    Implementation: ``lassonet.LassoNetClassifier`` / ``lassonet.LassoNetRegressor`` (``path``),
    https://github.com/lasso-net/lassonet, MIT licence.
    """

    name = "LassoNetFeatureSelector"
    feature_scoring_method: bool = True

    hidden_dims: tuple[int, ...] = (100,)
    """Hidden layer sizes of the network (library default)."""
    M: float = 10.0
    """Hierarchy parameter: bound on the hidden-layer weights relative to the skip-layer weights (library default)."""
    path_multiplier: float = 1.02
    """Factor between consecutive lambdas on the path (library default)."""
    val_fraction: float = 0.1
    """Fraction of the rows held out for early stopping (the library's ``val_size`` default)."""

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        import torch
        from autogluon.core.utils.utils import generate_train_test_split
        from lassonet import LassoNetClassifier, LassoNetRegressor, NonFiniteLossError

        start_time = time.monotonic()
        regression = self.problem_type == "regression"
        X_pre, y_pre = self._preprocess(
            X, y=y, impute=True, encode_ordinal=True, scale=True, encode_target=not regression
        )
        # Our own split instead of LassoNet's (unstratified) one: the classifier sizes its output layer
        # from the training labels, so every class must be among the training rows.
        X_train, X_val, y_train, y_val = generate_train_test_split(
            X=X_pre,
            y=y_pre,
            problem_type=self.problem_type,
            test_size=self.val_fraction,
            random_state=self.random_state,
        )
        X_train, X_val = X_train.to_numpy(dtype=np.float32), X_val.to_numpy(dtype=np.float32)
        if regression:
            y_train, y_val = y_train.to_numpy(dtype=np.float32), y_val.to_numpy(dtype=np.float32)
            mean, std = float(y_train.mean()), float(y_train.std())
            std = std if np.isfinite(std) and std > 0 else 1.0
            y_train, y_val = (y_train - mean) / std, (y_val - mean) / std
        else:
            y_train, y_val = y_train.to_numpy(dtype=np.int64), y_val.to_numpy(dtype=np.int64)

        def check_time(*_):
            if self._timed_out(time_limit, start_time):
                raise _TimeUp

        lambdas, skip_norms = [], []  # one entry per completed fit on the path, the dense fit first

        def record(estimator, history):
            lambdas.append(float(history[-1].lambda_))
            skip_weight = estimator.model.skip.weight.detach()  # (n_outputs, n_features)
            skip_norms.append(torch.linalg.vector_norm(skip_weight, dim=0).cpu().numpy())

        estimator_class = LassoNetRegressor if regression else LassoNetClassifier
        estimator = estimator_class(
            hidden_dims=self.hidden_dims,
            M=self.M,
            path_multiplier=self.path_multiplier,
            # The library's default optimizers, with the time check before every step.
            optim=(
                _optimizer_with_hook(torch.optim.Adam, check_time, lr=1e-3),
                _optimizer_with_hook(torch.optim.SGD, check_time, lr=1e-3, momentum=0.9),
            ),
            device="cpu",
            verbose=0,
            random_state=self.random_state,
            torch_seed=self.random_state,
        )
        try:
            with torch.random.fork_rng(devices=[]):  # torch_seed reseeds torch's global generator
                estimator.path(X_train, y_train, X_val=X_val, y_val=y_val, callback=record, disable_lambda_warning=True)
        except _TimeUp:
            pass  # logged by _timed_out; the completed fits are recorded
        except NonFiniteLossError as error:
            self._log(30, f"LassoNet's loss became non-finite ({error}). Using the path up to there.")

        if not lambdas:
            self._log(30, "LassoNet's dense fit did not finish. Returning no scores.")
            return {}
        self._log(20, f"\tLassoNet path: {len(lambdas)} fits, last lambda {lambdas[-1]:.3g}.")

        in_model = skip_norms[0] > 0
        score = np.where(in_model, lambdas[0], -np.inf)
        tie_break = np.where(in_model, skip_norms[0], 0.0)
        for lambda_, norms in zip(lambdas[1:], skip_norms[1:], strict=True):
            in_model &= norms > 0
            score[in_model] = lambda_
            tie_break[in_model] = norms[in_model]

        # Insert the features by decreasing (score, skip-layer norm). The base class ranks the scores with
        # Python's stable sort, so features with equal scores keep this order.
        order = np.lexsort((-tie_break, -score))
        return {X_pre.columns[i]: float(score[i]) for i in order}
