"""STG with native gate convergence and fold-local budget calibration."""

from __future__ import annotations

import math
import time
from collections import deque
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


def _make_model(input_dim, output_dim, hidden_dims, sigma):
    import torch
    from torch import nn

    class GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.mu = nn.Parameter(0.01 * torch.randn(input_dim))
            layers = []
            for i, (left, right) in enumerate(pairwise((input_dim, *hidden_dims, output_dim))):
                layers.append(nn.Linear(left, right))
                if i < len(hidden_dims):
                    layers.append(nn.Tanh())
            self.network = nn.Sequential(*layers)

        def forward(self, X):
            z = self.mu + sigma * torch.randn_like(self.mu) if self.training else self.mu
            return self.network(X * (z + 0.5).clamp(0, 1))

        def regularizer(self):
            return (0.5 * (1 + torch.erf((self.mu + 0.5) / (sigma * math.sqrt(2))))).mean()

    return GatedMLP()


def _gate_snapshot(mu, sigma):
    from scipy.special import ndtr

    mu = np.asarray(mu, dtype=float)
    gates, probabilities = np.clip(mu + 0.5, 0, 1), ndtr((mu + 0.5) / sigma)
    return gates, probabilities


def _stability(window, tolerance):
    """Require stable support, deterministic gates AND stochastic open probabilities."""
    gates = np.stack([s[0] for s in window])
    probabilities = np.stack([s[1] for s in window])
    support_stable = bool(np.all((gates > 0) == (gates[-1] > 0)))
    gate_change = float(np.ptp(gates, axis=0).max())
    probability_change = float(np.ptp(probabilities, axis=0).max())
    return support_stable and max(gate_change, probability_change) <= tolerance, gate_change, probability_change


def _next_lambda(observations, initial, max_trials, target):
    """Propose lambda from observed counts; only converged terminal fits can be selected."""
    if len(observations) >= max_trials:
        return None
    if any(r["budget_matched"] and r["native_count"] == target for r in observations):
        return None
    if len(observations) < len(initial):
        return initial[len(observations)]
    measured = []
    for record in observations:
        count = record["calibration_count"]
        if count is None:
            count = (record.get("terminal_checkpoint") or {}).get("native_count")
        if count is not None:
            measured.append(dict(record, calibration_count=count))
    if not measured:
        return None
    dense = [r for r in measured if r["calibration_count"] > target]
    sparse = [r for r in measured if r["calibration_count"] < target]
    brackets = [(a["lambda"], b["lambda"]) for a in dense for b in sparse if a["lambda"] < b["lambda"]]
    if brackets:
        low, high = min(brackets, key=lambda p: math.log(p[1] / p[0]))
        proposed = math.sqrt(low * high)
    elif dense and not sparse:
        proposed = max(r["lambda"] for r in observations) * 10
    elif sparse and not dense:
        proposed = min(r["lambda"] for r in observations) / 10
    elif len(observations) == 1:
        proposed = observations[0]["lambda"] / math.sqrt(10)
    else:
        # Count need not be monotone in lambda. Explore the widest untried log interval.
        values = sorted(r["lambda"] for r in observations)
        low, high = max(pairwise(values), key=lambda p: math.log(p[1] / p[0]))
        proposed = math.sqrt(low * high)
    if (
        not math.isfinite(proposed)
        or proposed <= 0
        or any(math.isclose(proposed, r["lambda"], rel_tol=1e-8) for r in observations)
    ):
        return None
    return proposed


def _balanced_groups(architectures, learning_rates, rng):
    """Cover depths and rates before revisiting a depth/rate combination."""
    depths = [architectures[i] for i in rng.permutation(len(architectures))]
    rates = [learning_rates[i] for i in rng.permutation(len(learning_rates))]
    depths.sort(key=lambda dims: (len(dims), sum(a * b for a, b in pairwise(dims))))
    middle = float(np.median(np.log(rates)))
    rates.sort(key=lambda rate: abs(math.log(rate) - middle))
    return [
        {"hidden_dims": dims, "learning_rate": rates[(index + shift) % len(rates)]}
        for shift in range(len(rates))
        for index, dims in enumerate(depths)
    ]


def _calibration_seed(trials, target, upper, min_epochs):
    """Borrow only a lambda proposal from mature terminal counts, never a fitted gate state."""
    measured = [t for t in trials if t.get("epochs", 0) >= min_epochs and t.get("terminal_checkpoint") is not None]
    if not measured:
        return None

    def priority(trial):
        terminal = trial["terminal_checkpoint"]
        count = terminal["native_count"]
        return (not target <= count <= upper, abs(count - target), terminal["validation_loss"])

    return min(measured, key=priority)["lambda"]


def _epoch_deadline(start, now, epoch, planned, hard, required_epochs):
    """Protect a usable training horizon, bounded by the actual group/outer deadline."""
    seconds_per_epoch = (now - start) / epoch
    return min(hard, max(planned, start + 1.25 * seconds_per_epoch * required_epochs))


class STGFeatureSelector(AbstractFeatureSelector):
    """Feature Selection using Stochastic Gates, calibrated to a native feature budget.

    Tanh MLPs with 3, 4 or 5 hidden layers jointly learn weights and stochastic gates
    ``clip(mu + 0.5 + sigma * N(0,1), 0, 1)``. Loss is cross entropy or standardized
    target MSE plus ``lambda * mean(Phi((mu + 0.5) / sigma))``. Adam learning rate,
    depth/width and lambda are tuned inside the outer training fold. All inputs are
    imputed and z-scored using INNER TRAINING statistics only; validation receives
    the same transformation. Shared ordinal encoding honours the permutation ablation.

    Validation uses the native gated network, with NO artificial top-k gate mask.
    Before a checkpoint is eligible, train at least ``min_epochs`` and observe
    unchanged native support and <= ``gate_tolerance`` peak-to-peak changes in BOTH
    deterministic gate values and stochastic open probabilities over ``gate_window``
    epochs (sampled every ``evaluate_every`` epochs). Predictive early stopping is
    disabled until this operational gate-convergence criterion holds.

    A usable checkpoint has between k and k + ceil(k * budget_tolerance) native
    nonzero gates. This one-sided tolerance
    permits trimming a small surplus to the benchmark's exact k without adding
    closed gates. Exact k is preferred, then native validation loss. Lambda is
    calibrated separately for each architecture/rate: initial values, followed by
    log-space bracket refinement or expansion. Finish a group's calibration before
    moving to the next group, and use a successful lambda as the next group's seed.
    Without a winner, mature terminal counts nearest the budget provide a proposal
    for the next group; they cannot qualify a fit. Every trial is trained from scratch.
    The balanced order starts with the shallowest MLP and the middle log learning
    rate, and covers different depths and rates first. A group
    gets at most a third of the remaining time while three or more groups remain.
    Terminal counts of unfinished fits
    may guide exploration, but can never qualify a checkpoint for selection.
    Sparsity need not be strictly monotone; every attempted fit is recorded.

    The last validated checkpoint must be converged and budget-matched. Restore an
    earlier best checkpoint only if its support is identical to the terminal support
    and its gate values AND open probabilities remain within the convergence tolerance
    of the terminal values; otherwise use the terminal checkpoint. This rejects an
    earlier sparse plateau that was subsequently lost. Only active gates receive scores.
    If none qualifies, return no scores, set ``budget_converged_=False``, and let
    the benchmark's separately labelled random fallback operate. Such a fallback
    is NOT evidence of STG convergence and the smoke evaluator declines to score it.
    ``search_results_`` records histories, gate values, probabilities, native counts,
    convergence tests, epochs and timeout status. No test rows enter this process.

    Large training folds use larger batches, capped by ``max_batch_size``, to target
    ``max_batches_per_epoch`` batches per epoch (never smaller than ``batch_size``).
    All training rows are still used in each completed epoch. CUDA scalar transfers
    are accumulated within each epoch instead of synchronizing twice per minibatch.

    Time limit: check every training/validation batch. Measured epoch cost protects
    enough time for min_epochs + gate_window + patience, with a 25% timing margin,
    inside the group and outer deadlines. Once the terminal count reaches the budget
    band, compare other architectures/rates; reject the fit if it is not converged.
    No epoch minimum or gate-convergence
    threshold is relaxed. The search may be incomplete.
    CPU/GPU RNG state and CPU thread count are restored; optional
    torch imports stay lazy. ``device='auto'`` uses CUDA when available.

    Reference: Yamada, Yutaro, Ofir Lindenbaum, Sahand Negahban, and Yuval Kluger.
    "Feature Selection using Stochastic Gates." ICML (2020), PMLR 119:10648-10659.
    Implementation: own PyTorch implementation of the equations in the paper and
    https://github.com/runopti/stg (MIT licence), with explicit convergence tracking.
    """

    name = "STGFeatureSelector"
    feature_scoring_method = True

    def get_selection_diagnostics(self) -> dict:
        """Compact, serializable audit metadata for performance and selection-only results."""
        checkpoint = getattr(self, "gate_diagnostics_", None) or {}
        trials = getattr(self, "search_results_", [])
        keys = (
            "lambda",
            "hidden_dims",
            "learning_rate",
            "seed",
            "epochs",
            "status",
            "converged",
            "budget_matched",
            "native_count",
            "best_epoch",
            "validation_loss",
            "calibration_count",
            "terminal_native_count",
            "terminal_gate_change",
            "terminal_probability_change",
            "elapsed_seconds",
            "effective_batch_size",
            "planned_seconds",
        )
        return {
            "schema_version": 2,
            "budget_converged": getattr(self, "budget_converged_", False),
            "target_features": getattr(self, "max_features", None),
            "native_active_features": getattr(self, "native_count_", None),
            "zero_gates": checkpoint.get("zero_gates"),
            "one_gates": checkpoint.get("one_gates"),
            "expected_stochastic_active": checkpoint.get("expected_active"),
            "best_params": getattr(self, "best_params_", None),
            "selected_epoch": getattr(self, "best_epoch_", None),
            "gate_window_change": checkpoint.get("gate_change"),
            "probability_window_change": checkpoint.get("probability_change"),
            "protocol": {
                "activation": "tanh",
                "architectures": self.architectures,
                "learning_rates": self.learning_rates,
                "initial_lambdas": self.lambdas,
                "max_lambda_trials": self.max_lambda_trials,
                "sigma": self.sigma,
                "max_epochs": self.max_epochs,
                "min_epochs": self.min_epochs,
                "patience": self.patience,
                "gate_window": self.gate_window,
                "evaluate_every": self.evaluate_every,
                "gate_tolerance": self.gate_tolerance,
                "budget_tolerance": self.budget_tolerance,
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "max_batches_per_epoch": self.max_batches_per_epoch,
                "effective_batch_size": getattr(self, "effective_batch_size_", None),
                "search_schedule": "shallow/middle-rate first, sequential lambda calibration, cross-group proposals, epoch-cost floor",
                "preprocessing": "inner-training-only imputation and z-score",
                "validation": "native gates, no artificial top-k mask",
            },
            "trials": [{key: trial.get(key) for key in keys} for trial in trials],
        }

    def __init__(
        self,
        max_features,
        *,
        lambdas=(0.1, 1.0),
        architectures=((128, 64, 32), (128, 128, 64, 32), (128, 128, 64, 32, 16)),
        learning_rates=(0.001, 0.003, 0.01),
        max_lambda_trials=8,
        sigma=0.5,
        val_fraction=0.2,
        max_epochs=6000,
        min_epochs=500,
        patience=100,
        gate_window=50,
        evaluate_every=10,
        gate_tolerance=0.02,
        budget_tolerance=0.1,
        min_delta=1e-4,
        batch_size=256,
        max_batch_size=8192,
        max_batches_per_epoch=32,
        weight_decay=1e-3,
        device="auto",
        **kwargs,
    ):
        super().__init__(max_features=max_features, **kwargs)
        self.lambdas = tuple(float(x) for x in lambdas)
        self.architectures = tuple(tuple(dims) for dims in architectures)
        self.learning_rates = tuple(float(x) for x in learning_rates)
        if not self.lambdas or any(not np.isfinite(x) or x <= 0 for x in self.lambdas):
            raise ValueError("lambdas must be a nonempty sequence of finite positive values")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("initial lambdas must be distinct")
        if not self.architectures or any(
            not 3 <= len(dims) <= 5 or any(not isinstance(d, int) or d < 1 for d in dims) for dims in self.architectures
        ):
            raise ValueError("architectures must have 3 to 5 positive integer hidden-layer widths")
        if not self.learning_rates or any(not np.isfinite(x) or x <= 0 for x in self.learning_rates):
            raise ValueError("learning_rates must be finite and positive")
        if not np.isfinite(sigma) or sigma <= 0 or not 0 < val_fraction < 1:
            raise ValueError("sigma must be positive and val_fraction must lie in (0, 1)")
        if any(
            not isinstance(x, int) or x < 1
            for x in (
                max_epochs,
                min_epochs,
                patience,
                batch_size,
                max_batch_size,
                gate_window,
                evaluate_every,
                max_lambda_trials,
            )
        ):
            raise ValueError("epoch, batch and trial settings must be positive integers")
        if max_batch_size < batch_size or (
            max_batches_per_epoch is not None
            and (not isinstance(max_batches_per_epoch, int) or max_batches_per_epoch < 1)
        ):
            raise ValueError("max_batch_size must cover batch_size; max_batches_per_epoch must be positive or None")
        if min_epochs > max_epochs or gate_window < evaluate_every or gate_window % evaluate_every:
            raise ValueError("min_epochs must not exceed max_epochs; gate_window must be a multiple of evaluate_every")
        if max_lambda_trials < len(self.lambdas):
            raise ValueError("max_lambda_trials must accommodate every initial lambda")
        if any(not np.isfinite(x) or x < 0 for x in (min_delta, weight_decay, gate_tolerance, budget_tolerance)):
            raise ValueError("tolerances and weight_decay must be finite and nonnegative")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")
        self.sigma, self.val_fraction = sigma, val_fraction
        self.max_epochs, self.min_epochs, self.patience = max_epochs, min_epochs, patience
        self.gate_window, self.evaluate_every = gate_window, evaluate_every
        self.gate_tolerance, self.budget_tolerance = gate_tolerance, budget_tolerance
        self.max_lambda_trials, self.device = max_lambda_trials, device
        self.min_delta, self.batch_size, self.weight_decay = min_delta, batch_size, weight_decay
        self.max_batch_size, self.max_batches_per_epoch = max_batch_size, max_batches_per_epoch

    def _batch_size(self, n_train):
        if self.max_batches_per_epoch is None:
            return self.batch_size
        target = math.ceil(n_train / self.max_batches_per_epoch)
        rounded = self.batch_size * math.ceil(target / self.batch_size)
        return max(self.batch_size, min(self.max_batch_size, rounded))

    def _prepare_data(self, X_train, X_val, y_train, y_val):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        encoded, _ = self._preprocess(X_train, encode_ordinal=True)
        validation = X_val.copy()
        for col in X_train.select_dtypes(include=["object", "category"]).columns:
            present = X_train[col].notna()
            mapping = dict(zip(X_train.loc[present, col], encoded.loc[present, col], strict=True))
            validation[col] = validation[col].astype(object).map(mapping).fillna(-1).where(X_val[col].notna())
        imputer, scaler = SimpleImputer(keep_empty_features=True), StandardScaler()
        train = scaler.fit_transform(imputer.fit_transform(encoded)).astype(np.float32)
        val = scaler.transform(imputer.transform(validation)).astype(np.float32)
        regression = self.problem_type == "regression"
        dtype = np.float32 if regression else np.int64
        train_y, val_y = y_train.to_numpy(dtype=dtype), y_val.to_numpy(dtype=dtype)
        if regression:
            mean, std = float(train_y.mean()), float(train_y.std())
            std = std if std > 0 else 1.0
            train_y, val_y = ((train_y - mean) / std)[:, None], ((val_y - mean) / std)[:, None]
        return train, val, train_y, val_y

    def _validation_loss(self, model, X, y, loss_fn, deadline):
        import torch

        model.eval()
        total = torch.zeros((), device=X.device)
        batch_size = getattr(self, "effective_batch_size_", self.batch_size)
        with torch.no_grad():
            for offset in range(0, len(X), batch_size):
                if time.monotonic() >= deadline:
                    return None
                batch = slice(offset, offset + batch_size)
                total += loss_fn(model(X[batch]), y[batch]) * len(X[batch])
        return float(total) / len(X)

    def _fit_trial(self, data, params, output_dim, seed, deadline, *, hard_deadline=None):
        import torch

        X_train, X_val, y_train, y_val = data
        started = time.monotonic()
        hard_deadline = deadline if hard_deadline is None else hard_deadline
        planned_seconds = deadline - started
        planned_seconds = planned_seconds if math.isfinite(planned_seconds) else None
        batch_size = getattr(self, "effective_batch_size_", self.batch_size)
        torch.random.default_generator.manual_seed(seed)
        if X_train.is_cuda:
            torch.cuda.default_generators[X_train.device.index].manual_seed(seed)
        model = _make_model(X_train.shape[1], output_dim, params["hidden_dims"], self.sigma).to(X_train.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss() if self.problem_type == "regression" else torch.nn.CrossEntropyLoss()
        window = deque(maxlen=self.gate_window // self.evaluate_every + 1)
        best_state, best_record, last_record, terminal_state = None, None, None, None
        best_key, patience_loss, stale = (math.inf, math.inf), math.inf, 0
        history, status, epochs = [], "max_epochs", 0
        upper = self.max_features + math.ceil(self.max_features * self.budget_tolerance)
        for epoch in range(1, self.max_epochs + 1):
            if time.monotonic() >= deadline:
                status = "time_limit"
                break
            model.train()
            order = torch.randperm(len(X_train), device=X_train.device)
            train_total = torch.zeros((), device=X_train.device)
            for offset in range(0, len(order), batch_size):
                if time.monotonic() >= deadline:
                    status = "time_limit"
                    break
                indices = order[offset : offset + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(X_train[indices]), y_train[indices]) + params["lambda"] * model.regularizer()
                loss.backward()
                optimizer.step()
                train_total += loss.detach() * len(indices)
            if status != "max_epochs":
                break
            if not torch.isfinite(train_total):
                status = "nonfinite_loss"
                break
            epochs = epoch
            if epoch % self.evaluate_every and epoch != self.max_epochs:
                continue
            value = self._validation_loss(model, X_val, y_val, loss_fn, deadline)
            if value is None or not math.isfinite(value):
                status = "time_limit" if value is None else "nonfinite_loss"
                break
            mu = model.mu.detach().cpu().numpy().copy()
            gates, probabilities = _gate_snapshot(mu, self.sigma)
            window.append((gates, probabilities))
            stable, gate_change, probability_change = _stability(window, self.gate_tolerance)
            converged = epoch >= self.min_epochs and len(window) == window.maxlen and stable
            count = int(np.sum(gates > 0))
            matched = converged and self.max_features <= count <= upper
            observed_now = time.monotonic()
            horizon = min(self.max_epochs, self.min_epochs + self.gate_window + self.patience)
            deadline = _epoch_deadline(started, observed_now, epoch, deadline, hard_deadline, horizon)
            record = {
                "epoch": epoch,
                "validation_loss": value,
                "training_objective": float(train_total) / len(X_train),
                "native_count": count,
                "zero_gates": int(np.sum(gates == 0)),
                "one_gates": int(np.sum(gates == 1)),
                "expected_active": float(probabilities.sum()),
                "gate_change": gate_change,
                "probability_change": probability_change,
                "converged": converged,
                "budget_matched": matched,
            }
            history.append(record)
            last_record = dict(record, mu=mu.tolist(), gates=gates.tolist(), open_probabilities=probabilities.tolist())
            terminal_state = (
                {name: tensor.detach().clone() for name, tensor in model.state_dict().items()} if matched else None
            )
            key = (count - self.max_features, value)
            if matched and key < best_key:
                best_key, best_record = key, last_record
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            if not converged:
                patience_loss, stale = math.inf, 0
            elif value < patience_loss - self.min_delta:
                patience_loss, stale = value, 0
            else:
                stale += self.evaluate_every
            if converged and stale >= self.patience:
                status = "early_stopping"
                break
        selected_mu = None
        if status == "nonfinite_loss" or last_record is None or not last_record["budget_matched"]:
            best_state, best_record = None, None
        elif best_record is not None:
            a, b = np.asarray(best_record["gates"]), np.asarray(last_record["gates"])
            pa, pb = np.asarray(best_record["open_probabilities"]), np.asarray(last_record["open_probabilities"])
            if (
                not np.array_equal(a > 0, b > 0)
                or max(np.max(np.abs(a - b)), np.max(np.abs(pa - pb))) > self.gate_tolerance
            ):
                best_state, best_record = terminal_state, last_record
        if best_state is not None:
            model.load_state_dict(best_state)
            selected_mu = model.mu.detach().cpu().numpy().copy()
        result = dict(
            params,
            seed=seed,
            epochs=epochs,
            status=status,
            history=history,
            budget_matched=best_record is not None,
            converged=last_record is not None and last_record["converged"],
            native_count=best_record["native_count"] if best_record else None,
            best_epoch=best_record["epoch"] if best_record else None,
            validation_loss=best_record["validation_loss"] if best_record else None,
            calibration_count=last_record["native_count"] if last_record and last_record["converged"] else None,
            selected_checkpoint=best_record,
            terminal_checkpoint=last_record,
            terminal_native_count=last_record["native_count"] if last_record else None,
            terminal_gate_change=last_record["gate_change"] if last_record else None,
            terminal_probability_change=last_record["probability_change"] if last_record else None,
            elapsed_seconds=time.monotonic() - started,
            effective_batch_size=batch_size,
            planned_seconds=planned_seconds,
        )
        return selected_mu, result

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start = time.monotonic()
        self.search_results_, self.best_params_, self.best_epoch_ = [], None, None
        self.best_validation_loss_, self.gate_means_ = math.inf, None
        self.budget_converged_, self.native_count_, self.gate_diagnostics_ = False, None, None
        if self._timed_out(time_limit, start):
            return {}
        import torch
        from autogluon.core.utils.utils import generate_train_test_split

        if self._timed_out(time_limit, start):
            return {}
        device = torch.device(
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.device == "auto"
            else self.device
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        _, labels = self._preprocess(X, y=y, encode_target=self.problem_type != "regression")
        X_train, X_val, y_train, y_val = generate_train_test_split(
            X=X, y=labels, problem_type=self.problem_type, test_size=self.val_fraction, random_state=self.random_state
        )
        if len(X_train) == 0 or len(X_val) == 0:
            return {}
        data = tuple(torch.from_numpy(a).to(device) for a in self._prepare_data(X_train, X_val, y_train, y_val))
        self.effective_batch_size_ = self._batch_size(len(X_train))
        output_dim = 1 if self.problem_type == "regression" else int(labels.max()) + 1
        groups = _balanced_groups(self.architectures, self.learning_rates, self._rng)
        seed = int(self._rng.integers(0, 2**31 - 1))
        outer_deadline = math.inf if time_limit is None else start + time_limit
        threads, best_key = torch.get_num_threads(), (math.inf, math.inf)
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                # Warm lazy optimizer imports within the overall limit, before slicing trial budgets.
                torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, device=device))])
                for group_index, group in enumerate(groups):
                    now = time.monotonic()
                    remaining = max(0, outer_deadline - now)
                    if remaining <= 0:
                        break
                    group_deadline = min(outer_deadline, now + remaining / min(3, len(groups) - group_index))
                    observations = []
                    candidate = (
                        self.best_params_["lambda"]
                        if self.best_params_
                        else _calibration_seed(
                            self.search_results_,
                            self.max_features,
                            self.max_features + math.ceil(self.max_features * self.budget_tolerance),
                            self.min_epochs,
                        )
                    )
                    initial = (candidate,) if candidate is not None else self.lambdas
                    for _ in range(self.max_lambda_trials):
                        now = time.monotonic()
                        remaining = max(0, group_deadline - now)
                        if remaining <= 0:
                            break
                        lam = _next_lambda(observations, initial, self.max_lambda_trials, self.max_features)
                        if lam is None:
                            break
                        params = {
                            "lambda": lam,
                            "hidden_dims": group["hidden_dims"],
                            "learning_rate": group["learning_rate"],
                        }
                        allowance = min(remaining, max(30.0, remaining / (self.max_lambda_trials - len(observations))))
                        mu, result = self._fit_trial(
                            data, params, output_dim, seed, now + allowance, hard_deadline=group_deadline
                        )
                        self.search_results_.append(result)
                        observations.append(result)
                        self._log(
                            20,
                            f"STG trial: lambda={lam:.6g}, layers={len(group['hidden_dims'])}, "
                            f"lr={group['learning_rate']}, epochs={result['epochs']}, "
                            f"native={result['terminal_native_count']}, "
                            f"converged={result['converged']}, budget_matched={result['budget_matched']}",
                        )
                        if mu is not None:
                            key = (result["native_count"] - self.max_features, result["validation_loss"])
                            if key < best_key:
                                best_key = key
                                self.best_params_, self.best_epoch_ = params, result["best_epoch"]
                                self.best_validation_loss_, self.gate_means_ = result["validation_loss"], mu
                                self.native_count_, self.gate_diagnostics_ = (
                                    result["native_count"],
                                    result["selected_checkpoint"],
                                )
                        terminal_count = result.get("terminal_native_count")
                        if result["budget_matched"] or (
                            result["epochs"] >= self.min_epochs
                            and terminal_count is not None
                            and self.max_features
                            <= terminal_count
                            <= self.max_features + math.ceil(self.max_features * self.budget_tolerance)
                        ):
                            break  # Spend the remaining time comparing architectures/rates.
        finally:
            torch.set_num_threads(threads)
        self.budget_converged_ = self.gate_means_ is not None
        if not self.budget_converged_:
            self._log(30, "STG found no converged native-gate fit near the target budget; returning no scores.")
            return {}
        active = self.gate_means_ > -0.5
        return {name: float(mu) for name, mu, keep in zip(X.columns, self.gate_means_, active, strict=True) if keep}
