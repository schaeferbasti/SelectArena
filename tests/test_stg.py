"""Native sparsity and convergence are prerequisites for STG evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from selectarena.methods.checks import synthetic_data
from selectarena.methods.stg import STGFeatureSelector, _gate_snapshot, _make_model, _next_lambda, _stability


def small_selector(**kwargs):
    defaults = dict(
        max_features=3,
        lambdas=(0.1, 1.0),
        architectures=((8, 8, 8),),
        learning_rates=(0.01,),
        max_lambda_trials=2,
        max_epochs=30,
        min_epochs=10,
        gate_window=10,
        evaluate_every=5,
        patience=10,
        random_state=0,
        verbosity=0,
        device="cpu",
    )
    return STGFeatureSelector(**(defaults | kwargs))


def test_native_gate_equations_and_tanh_architecture():
    model = _make_model(4, 1, (), 0.5)
    with torch.no_grad():
        model.mu.copy_(torch.tensor([-1.0, -0.25, 0.25, 1.0]))
        model.network[0].weight.fill_(1)
        model.network[0].bias.zero_()
    model.eval()
    torch.testing.assert_close(model(torch.ones(2, 4)), torch.full((2, 1), 2.0))
    expected = torch.distributions.Normal(0, 1).cdf((model.mu + 0.5) / 0.5).mean()
    torch.testing.assert_close(model.regularizer(), expected)
    for dims in STGFeatureSelector(max_features=1).architectures:
        network = _make_model(4, 2, dims, 0.5)
        assert 3 <= len(dims) <= 5
        assert sum(isinstance(m, torch.nn.Tanh) for m in network.modules()) == len(dims)
        assert not any(isinstance(m, torch.nn.ReLU) for m in network.modules())


def test_training_only_zscore():
    selector = small_selector()
    selector.problem_type = "regression"
    train = pd.DataFrame({"num": [0.0, 2.0, np.nan], "cat": pd.Categorical(["a", "b", "a"])})
    val = pd.DataFrame({"num": [1e6], "cat": pd.Categorical(["unseen"])})
    y = pd.Series([0.0, 2.0, 1.0])
    first = selector._prepare_data(train, val, y, pd.Series([1e6]))
    second = selector._prepare_data(train, val.assign(num=-1e6), y, pd.Series([-1e6]))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_allclose(first[0].mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(first[0].std(axis=0), 1, atol=1e-7)
    assert first[1][0, 0] > 1e5 and np.isfinite(first[1]).all()


def test_stability_checks_support_values_and_probabilities():
    a = _gate_snapshot(np.array([-1.0, 1.0]), 0.5)
    assert _stability([a, a], 0.02)[0]
    # Deterministic zeros alone do not imply stable stochastic gates.
    b = _gate_snapshot(np.array([-2.0, 1.0]), 0.5)
    assert not _stability([a, b], 0.02)[0]
    c = _gate_snapshot(np.array([-0.49, 1.0]), 0.5)
    assert not _stability([a, c], 1.0)[0]


def test_lambda_calibration_uses_converged_counts():
    observations = [
        {"lambda": 0.1, "calibration_count": 20, "budget_matched": False, "native_count": None},
        {"lambda": 1.0, "calibration_count": 3, "budget_matched": False, "native_count": None},
    ]
    assert _next_lambda(observations, (0.1, 1.0), 6, 10) == pytest.approx(math.sqrt(0.1))
    for row in observations:
        row["calibration_count"] = None
    assert _next_lambda(observations, (0.1, 1.0), 6, 10) is None


def test_early_good_loss_cannot_restore_unconverged_gates(monkeypatch):
    selector = small_selector(max_features=1, max_epochs=60, min_epochs=20)
    selector.problem_type, selector.max_features = "regression", 1
    data = (torch.ones(8, 3), torch.ones(2, 3), torch.ones(8, 1), torch.ones(2, 1))
    calls = []

    def validation(model, *args):
        index = len(calls)
        with torch.no_grad():
            model.mu.copy_(torch.tensor([0.0, 0.0, 0.0] if index == 0 else [-1.0, -1.0, 1.0]))
        calls.append(index)
        return 0.01 if index == 0 else 2.0

    monkeypatch.setattr(selector, "_validation_loss", validation)
    mu, result = selector._fit_trial(
        data, {"lambda": 0.1, "hidden_dims": (8, 8, 8), "learning_rate": 0.01}, 1, 0, math.inf
    )
    assert result["status"] == "early_stopping"
    assert result["budget_matched"] and result["converged"]
    assert result["best_epoch"] >= 20 and result["validation_loss"] == 2
    assert result["epochs"] >= 30  # predictive patience starts only after gate convergence
    assert np.sum(mu > -0.5) == 1


def test_unfinished_dense_fit_can_guide_exploration_without_qualifying():
    observations = [
        {
            "lambda": 0.1,
            "calibration_count": None,
            "terminal_checkpoint": {"native_count": 20},
            "budget_matched": False,
            "native_count": None,
        },
        {"lambda": 1.0, "calibration_count": 3, "budget_matched": False, "native_count": None},
    ]
    assert _next_lambda(observations, (0.1, 1.0), 6, 10) == pytest.approx(math.sqrt(0.1))
    assert not any(r["budget_matched"] for r in observations)


def test_unstable_support_cannot_early_stop_or_qualify(monkeypatch):
    selector = small_selector(max_features=1)
    selector.problem_type, selector.max_features = "regression", 1
    data = (torch.ones(8, 3), torch.ones(2, 3), torch.ones(8, 1), torch.ones(2, 1))
    calls = []

    def validation(model, *args):
        with torch.no_grad():
            model.mu.copy_(torch.tensor([-0.4 if len(calls) % 2 else -0.6, -1.0, 1.0]))
        calls.append(1)
        return 1.0

    monkeypatch.setattr(selector, "_validation_loss", validation)
    mu, result = selector._fit_trial(
        data, {"lambda": 0.1, "hidden_dims": (8, 8, 8), "learning_rate": 0.01}, 1, 0, math.inf
    )
    assert mu is None and not result["converged"] and not result["budget_matched"]
    assert result["status"] == "max_epochs"


def test_later_loss_of_sparse_support_invalidates_earlier_checkpoint(monkeypatch):
    selector = small_selector(max_features=1, max_epochs=60, min_epochs=20, patience=100)
    selector.problem_type, selector.max_features = "regression", 1
    data = (torch.ones(8, 3), torch.ones(2, 3), torch.ones(8, 1), torch.ones(2, 1))
    calls = []

    def validation(model, *args):
        with torch.no_grad():
            model.mu.copy_(torch.tensor([-1.0, -1.0, 1.0] if len(calls) < 6 else [1.0, 1.0, 1.0]))
        calls.append(1)
        return 1.0

    monkeypatch.setattr(selector, "_validation_loss", validation)
    mu, result = selector._fit_trial(
        data, {"lambda": 0.1, "hidden_dims": (8, 8, 8), "learning_rate": 0.01}, 1, 0, math.inf
    )
    assert any(row["budget_matched"] for row in result["history"])
    assert mu is None and not result["budget_matched"]
    assert result["converged"] and result["calibration_count"] == 3


@pytest.mark.parametrize("problem", ["binary", "multiclass", "regression"])
def test_failed_budget_is_explicit_and_rng_is_preserved(problem):
    X, y = synthetic_data(problem, n_samples=40)
    first, second = small_selector(gate_tolerance=0), small_selector(gate_tolerance=0)
    state, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
    output = first.fit_transform(X=X, y=y, problem_type=problem)
    second.fit_transform(X=X, y=y, problem_type=problem)
    assert output.shape == (40, 3)
    assert not first.budget_converged_ and first.gate_means_ is None
    assert len(first._feature_selection_random_features) == 3
    assert first._selected_features == second._selected_features
    assert all(not r["budget_matched"] for r in first.search_results_)
    assert torch.equal(state, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()


def test_timeout_does_not_claim_convergence():
    X, y = synthetic_data("regression", n_samples=40)
    selector = small_selector()
    selector.fit_transform(X=X, y=y, problem_type="regression", time_limit=0)
    assert not selector.budget_converged_ and selector.search_results_ == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_rng_is_preserved():
    X, y = synthetic_data("binary", n_samples=40)
    state = torch.cuda.get_rng_state().clone()
    selector = small_selector(device="cuda", gate_tolerance=0)
    selector.fit_transform(X=X, y=y, problem_type="binary")
    assert torch.equal(state, torch.cuda.get_rng_state())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambdas": ()},
        {"lambdas": (-1,)},
        {"architectures": ((64,),)},
        {"architectures": ((8, 8, 0),)},
        {"learning_rates": (0,)},
        {"patience": 0},
        {"sigma": 0},
    ],
)
def test_invalid_search_space(kwargs):
    with pytest.raises(ValueError):
        STGFeatureSelector(max_features=1, **kwargs)
