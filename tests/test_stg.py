"""Strict convergence and ranked recovery must remain distinguishable and reproducible."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from selectarena.methods.checks import synthetic_data
from selectarena.methods.stg import (
    STGFeatureSelector,
    _balanced_groups,
    _calibration_seed,
    _epoch_deadline,
    _gate_snapshot,
    _make_model,
    _next_lambda,
    _stability,
)


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


def test_large_fold_batching_uses_all_rows_without_tiny_batches():
    selector = small_selector()
    assert selector._batch_size(40) == 256
    assert selector._batch_size(128_000) == 4096
    assert math.ceil(128_000 / selector._batch_size(128_000)) <= 32
    assert selector._batch_size(10_000_000) == 8192
    assert small_selector(max_batches_per_epoch=None)._batch_size(128_000) == 256


def test_measured_epoch_cost_protects_training_but_never_extends_the_hard_budget():
    # Ten measured one-second epochs cannot justify a 20-second trial for 650 required epochs.
    assert 650 <= _epoch_deadline(0, 10, 10, 20, 1000, 650) <= 1000
    assert _epoch_deadline(0, 10, 10, 20, 100, 650) == 100


def test_first_groups_cover_depths_and_rates():
    architectures = ((8, 8, 8), (8, 8, 8, 8), (8, 8, 8, 8, 8))
    rates = (0.001, 0.003, 0.01)
    groups = _balanced_groups(architectures, rates, np.random.default_rng(0))
    assert len({(g["hidden_dims"], g["learning_rate"]) for g in groups}) == 9
    assert {len(g["hidden_dims"]) for g in groups[:3]} == {3, 4, 5}
    assert {g["learning_rate"] for g in groups[:3]} == set(rates)
    assert len(groups[0]["hidden_dims"]) == 3 and groups[0]["learning_rate"] == 0.003


def test_calibration_proposals_can_use_mature_unconverged_counts_but_skip_immature_fits():
    trials = [
        {"lambda": 0.1, "epochs": 600, "terminal_checkpoint": {"native_count": 20, "validation_loss": 0.5}},
        {"lambda": 1.0, "epochs": 600, "terminal_checkpoint": {"native_count": 4, "validation_loss": 0.7}},
        {"lambda": 3.0, "epochs": 20, "terminal_checkpoint": {"native_count": 3, "validation_loss": 0.1}},
    ]
    assert _calibration_seed(trials, 3, 4, 500) == 1.0
    assert _calibration_seed(trials, 3, 4, 700) is None


def test_calibrate_lambda_before_switching_group_and_reuse_a_successful_seed(monkeypatch):
    selector = small_selector(architectures=((8, 8, 8), (8, 8, 8, 8), (8, 8, 8, 8, 8)))
    X, y = synthetic_data("regression", n_samples=40)
    attempts = []

    def fake_trial(data, params, output_dim, seed, deadline, *, hard_deadline):
        assert deadline <= hard_deadline
        attempts.append(params)
        matched = params["lambda"] >= 1
        mu = np.array([0.0] * 3 + [-1.0] * (data[0].shape[1] - 3)) if matched else None
        return mu, dict(
            params,
            epochs=30,
            converged=True,
            budget_matched=matched,
            native_count=3 if matched else None,
            calibration_count=3 if matched else 5,
            validation_loss=1.0 if matched else None,
            best_epoch=30 if matched else None,
            selected_checkpoint={},
            terminal_native_count=3 if matched else 5,
        )

    monkeypatch.setattr(selector, "_fit_trial", fake_trial)
    selector.fit_transform(X=X, y=y, problem_type="regression", time_limit=10)
    assert selector.budget_converged_
    assert attempts[0]["hidden_dims"] == attempts[1]["hidden_dims"]
    assert [p["lambda"] for p in attempts] == [0.1, 1.0, 1.0, 1.0]
    assert {len(p["hidden_dims"]) for p in attempts} == {3, 4, 5}


def test_seed_lambda_with_unfinished_exact_count_does_not_crash():
    observation = {
        "lambda": 1.0,
        "calibration_count": None,
        "native_count": None,
        "budget_matched": False,
        "terminal_checkpoint": {"native_count": 3},
    }
    assert _next_lambda([observation], (1.0,), 8, 3) == pytest.approx(1 / math.sqrt(10))


def test_divergence_invalidates_an_earlier_eligible_checkpoint(monkeypatch):
    selector = small_selector(max_features=1)
    selector.problem_type, selector.max_features = "regression", 1
    data = (torch.ones(8, 3), torch.ones(2, 3), torch.ones(8, 1), torch.ones(2, 1))
    calls = []

    def validation(model, *args):
        with torch.no_grad():
            model.mu.copy_(torch.tensor([-1.0, -1.0, 1.0]))
            if len(calls) == 3:
                model.network[0].weight.fill_(math.nan)
        calls.append(1)
        return float(len(calls)) * -1  # Keep predictive patience from firing first.

    monkeypatch.setattr(selector, "_validation_loss", validation)
    mu, record = selector._fit_trial(
        data, {"lambda": 0.1, "hidden_dims": (8, 8, 8), "learning_rate": 0.01}, 1, 0, math.inf
    )
    assert any(h["budget_matched"] for h in record["history"])
    assert record["status"] == "nonfinite_loss" and not record["budget_matched"] and mu is None


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
def test_failed_strict_budget_uses_flagged_ranking_without_random_padding(problem):
    X, y = synthetic_data(problem, n_samples=40)
    first, second = small_selector(gate_tolerance=0), small_selector(gate_tolerance=0)
    state, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
    output = first.fit_transform(X=X, y=y, problem_type=problem)
    second.fit_transform(X=X, y=y, problem_type=problem)
    assert output.shape == (40, 3)
    assert not first.budget_converged_ and first.gate_means_ is not None
    assert first._feature_selection_random_features == []
    assert first.selection_mode_ == "ranked_recovery"
    assert first.get_selection_diagnostics()["recovery_used"]
    assert any(t["phase"] == "recovery" for t in first.search_results_)
    expected = sorted(zip(X.columns, first.gate_means_, strict=True), key=lambda pair: pair[1], reverse=True)[:3]
    assert list(output.columns) == [name for name, _ in expected]
    assert first._selected_features == second._selected_features
    assert all(not r["budget_matched"] for r in first.search_results_)
    assert torch.equal(state, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()


def test_timeout_does_not_claim_convergence():
    X, y = synthetic_data("regression", n_samples=40)
    selector = small_selector()
    with pytest.raises(RuntimeError, match="refusing random"):
        selector.fit_transform(X=X, y=y, problem_type="regression", time_limit=0)
    assert not selector.budget_converged_ and selector.search_results_ == []


def test_total_training_failure_is_not_replaced_with_random_features(monkeypatch):
    X, y = synthetic_data("regression", n_samples=40)
    selector = small_selector()
    monkeypatch.setattr(selector, "_validation_loss", lambda *args: math.nan)
    with pytest.raises(RuntimeError, match="no finite trained gate ranking"):
        selector.fit_transform(X=X, y=y, problem_type="regression", time_limit=3)
    assert selector.selection_mode_ == "training_failed"
    assert not selector._feature_selection_random_features


def test_relaxed_checkpoint_does_not_claim_stability_from_one_observation(monkeypatch):
    selector = small_selector(max_epochs=5, min_epochs=5, ranking_min_epochs=5)
    selector.problem_type, selector.max_features = "regression", 1
    data = (torch.ones(8, 3), torch.ones(2, 3), torch.ones(8, 1), torch.ones(2, 1))
    _, result = selector._fit_trial(
        data, {"lambda": 0.1, "hidden_dims": (8, 8, 8), "learning_rate": 0.01}, 1, 0, math.inf
    )
    assert result["ranking_checkpoint"] is not None
    assert not result["ranking_checkpoint"]["window_complete"]


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
        {"max_batch_size": 0},
        {"max_batches_per_epoch": 0},
    ],
)
def test_invalid_search_space(kwargs):
    with pytest.raises(ValueError):
        STGFeatureSelector(max_features=1, **kwargs)
