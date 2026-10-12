"""Elo ratings match the paper's script (elo_v4.py: logistic-regression Bradley-Terry fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from selectarena.evaluation.elo import CELL_COLUMNS, ELO_SCALE, cell_errors, compute_elo
from selectarena.evaluation.plot import plot_main_figure

METHODS = ["Random", "MI", "CART", "Lasso", "LOCO"]
QUALITY = {"Random": 0.0, "MI": 0.6, "CART": 0.9, "Lasso": 0.4, "LOCO": 0.2}


def _scores(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(12):
        for model in ["LightGBM", "RealMLP"]:
            for budget in range(3):
                for split in range(3):
                    for m in METHODS:
                        error = 1.0 - 0.1 * QUALITY[m] + rng.normal(0, 0.05)
                        # Rounded errors create ties, which the script skips.
                        rows.append(
                            {
                                "dataset": f"d{d}",
                                "split_idx": split,
                                "model": model,
                                "config": "c1",
                                "budget_index": budget,
                                "method": m,
                                "metric_error": round(error, 2),
                            }
                        )
    return pd.DataFrame(rows)


def _reference_elo(scores: pd.DataFrame) -> dict[str, float]:
    """``build_matches`` + ``fit_bt`` + anchoring from elo_v4.py."""
    cells = cell_errors(scores)
    methods = sorted(cells["method"].unique())
    index = {m: i for i, m in enumerate(methods)}
    winners, losers = [], []
    for _, group in cells.groupby(CELL_COLUMNS):
        errs = group.set_index("method")["metric_error"]
        ms = errs.index.tolist()
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                if errs[a] == errs[b]:
                    continue
                w, lo = (a, b) if errs[a] < errs[b] else (b, a)
                winners.append(index[w])
                losers.append(index[lo])
    winners, losers = np.array(winners), np.array(losers)
    M = len(winners)
    X = np.zeros((2 * M, len(methods)))
    X[np.arange(M), winners] = 1
    X[np.arange(M), losers] = -1
    X[M + np.arange(M), winners] = -1
    X[M + np.arange(M), losers] = 1
    y = np.concatenate([np.ones(M), np.zeros(M)])
    beta = LogisticRegression(fit_intercept=False, C=1e9, solver="lbfgs", max_iter=1000).fit(X, y).coef_.ravel()
    elo = (beta - beta[index["Random"]]) * ELO_SCALE + 1000
    return dict(zip(methods, elo))


def test_elo_matches_paper_script():
    scores = _scores()
    ours = compute_elo(scores, n_bootstrap=0).set_index("method")["elo"]
    reference = _reference_elo(scores)
    for method, value in reference.items():
        assert ours[method] == pytest.approx(value, abs=0.5)
    assert ours["Random"] == 1000
    assert list(ours.index[:2]) == ["CART", "MI"]


def test_bootstrap_interval_contains_point_estimate():
    elo = compute_elo(_scores(), n_bootstrap=50)
    assert (elo["elo_ci_low"] <= elo["elo"] + 1e-6).all()
    assert (elo["elo"] <= elo["elo_ci_high"] + 1e-6).all()


def test_main_figure_is_written(tmp_path):
    elo = compute_elo(_scores(), n_bootstrap=20)
    main = elo.assign(stability=np.linspace(0, 1, len(elo)), validity=[0.5, None, 0.7, 0.9, 1.0])
    paths = [tmp_path / "main_figure.png", tmp_path / "main_figure.pdf"]
    plot_main_figure(main, paths)
    assert all(p.stat().st_size > 0 for p in paths)
