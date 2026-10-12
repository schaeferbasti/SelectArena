"""Elo ratings of feature selection methods, as in the SelectArena paper (Figure 1, App. D.2).

Port of ``performance/baseline/elo_v4.py`` from the paper's plotting code:

* A *cell* is one dataset, downstream model (configuration), budget and metric; the metric error
  is averaged over seeds (splits) within a cell.
* In each cell every pair of methods plays a match; the lower error wins. Tied pairs are skipped.
* A Bradley-Terry model, ``P(i beats j) = 1 / (1 + exp(beta_j - beta_i))``, is fit by maximum
  likelihood on all matches (strengths normalized to mean 0).
* Ratings are ``(beta - beta_Random) * 400 / ln(10) + 1000`` using the point estimate's Random
  strength for the point estimate and every bootstrap sample alike.
* 95% intervals come from 1000 cluster-bootstrap resamples of the datasets.

The fit minimizes the same penalized likelihood as the script's logistic regression, but on a
matrix of win counts instead of one row per match, which keeps 1000 bootstrap refits on the full
benchmark fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ELO_SCALE = 400 / np.log(10)
ANCHOR_ELO = 1000.0
DEFAULT_ANCHOR = "Random"
CELL_COLUMNS = ["dataset", "model", "config", "budget_index"]


def cell_errors(scores: pd.DataFrame) -> pd.DataFrame:
    """One seed-averaged ``metric_error`` per (cell, method)."""
    return (
        scores.dropna(subset=["method", "metric_error"])
        .groupby([*CELL_COLUMNS, "method"], as_index=False, dropna=False)["metric_error"]
        .mean()
    )


def win_matrices(cells: pd.DataFrame, methods: list[str]) -> tuple[np.ndarray, list]:
    """Per-dataset win counts ``W[d, i, j]`` (matches method i won against j; ties are skipped)."""
    index = {m: k for k, m in enumerate(methods)}
    datasets = sorted(cells["dataset"].unique())
    dataset_index = {d: k for k, d in enumerate(datasets)}
    W = np.zeros((len(datasets), len(methods), len(methods)))
    for keys, cell in cells.groupby(CELL_COLUMNS, dropna=False):
        idx = cell["method"].map(index).to_numpy()
        err = cell["metric_error"].to_numpy()
        W[dataset_index[keys[0]]][np.ix_(idx, idx)] += (err[:, None] < err[None, :]).astype(float)
    return W, datasets


def fit_bradley_terry(W: np.ndarray, *, C: float = 1e9) -> np.ndarray:
    """Bradley-Terry strengths (natural-log scale) from win counts ``W[i, j]``.

    Minimizes the objective of the script's ``LogisticRegression(fit_intercept=False, C=1e9)`` on
    the symmetrized match design, ``C * 2 * sum_ij W_ij * log(1 + exp(beta_j - beta_i)) +
    ||beta||^2 / 2``, with L-BFGS. The tiny penalty fixes the free offset (strengths sum to about
    zero) and keeps a method that never loses finite.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        diff = beta[:, None] - beta[None, :]
        # The objective divided by 2C (same minimizer, better conditioned).
        loss = np.sum(W * np.logaddexp(0, -diff)) + beta @ beta / (4 * C)
        weight = W * expit(-diff)  # d loss_ij / d beta_i = -W_ij * sigmoid(-(beta_i - beta_j))
        grad = weight.T.sum(axis=1) - weight.sum(axis=1) + beta / (2 * C)
        return loss, grad

    n = W.shape[0]
    result = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    beta = result.x
    return beta - beta.mean()


def compute_elo(
    scores: pd.DataFrame,
    *,
    anchor: str = DEFAULT_ANCHOR,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Elo per method with its 95% cluster-bootstrap interval, sorted best first.

    ``scores`` needs ``dataset``, ``model``, ``config``, ``budget_index``, ``method`` and
    ``metric_error`` (e.g. :func:`~selectarena.evaluation.leaderboard.per_group_scores`). Without
    the ``anchor`` method in the results, the mean rating is set to 1000 instead.
    """
    cells = cell_errors(scores)
    methods = sorted(cells["method"].unique())
    if len(methods) < 2:
        return pd.DataFrame(columns=["method", "elo", "elo_ci_low", "elo_ci_high"])
    W, datasets = win_matrices(cells, methods)

    beta = fit_bradley_terry(W.sum(axis=0))
    shift = beta[methods.index(anchor)] if anchor in methods else 0.0
    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, len(methods)))
    for b in range(n_bootstrap):
        sample = rng.choice(len(datasets), size=len(datasets), replace=True)
        boot[b] = fit_bradley_terry(W[sample].sum(axis=0))

    elo = (beta - shift) * ELO_SCALE + ANCHOR_ELO
    boot_elo = (boot - shift) * ELO_SCALE + ANCHOR_ELO
    low, high = (np.percentile(boot_elo, [2.5, 97.5], axis=0)) if n_bootstrap else (elo, elo)
    return (
        pd.DataFrame({"method": methods, "elo": elo, "elo_ci_low": low, "elo_ci_high": high})
        .sort_values("elo", ascending=False)
        .round(1)
        .reset_index(drop=True)
    )
