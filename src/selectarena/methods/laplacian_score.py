"""Laplacian score feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from selectarena.methods.abstract import (
    AbstractFeatureSelector,
)

if TYPE_CHECKING:
    import pandas as pd


class LaplacianScoreFeatureSelector(AbstractFeatureSelector):
    """LaplacianScore Feature Selection.

    Reference: He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature
    selection." Advances in neural information processing systems 18 (2005).
    Implementation Source:
    https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/similarity_based/lap_score.py#L6
    The author of the code is Li, Jundong, Associate Professor at the University of
    Virginia and main-author of 'Feature selection: A data perspective' (2017).

    Changes to the implementation:
        - Efficiency refactoring.
        - Time constraint.
        - Median heat-kernel bandwidth instead of a fixed one.
        - Guard against a degenerate (all-isolated) graph.
    """

    name = "LaplacianScoreFeatureSelector"
    feature_scoring_method: bool = True

    knn_k: int = 5
    """Neighbours per sample in the affinity graph."""

    max_graph_samples: int = 4096
    """Row cap for graph construction and scoring."""

    fallback_graph_samples: int = 512
    """Row cap used when the time budget is already exhausted on entry."""

    def _fit_feature_scoring(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
    ) -> dict[str, float]:
        """Score L_r = f̃_r^T (D - W) f̃_r / f̃_r^T D f̃_r, returned as 1 - L_r."""
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True, scale=True)
        columns = X_pre.columns
        X_arr = X_pre.to_numpy(dtype=float)

        self._lap_timed_out = bool(self._timed_out(time_limit, start_time))
        cap = self.fallback_graph_samples if self._lap_timed_out else self.max_graph_samples
        X_arr = self._subsample(X_arr, cap=cap)
        self._lap_n_graph_samples = X_arr.shape[0]

        W = self._construct_W(X_arr, k=self.knn_k)

        d = np.asarray(W.sum(axis=1)).ravel()
        d_total = d.sum()
        if d_total <= 1e-12:  # no edges (n < 2), so return a flat ranking instead of NaNs
            self._lap_degenerate_graph = True
            return dict.fromkeys(columns, 0.0)
        self._lap_degenerate_graph = False

        weighted_means = (X_arr * d[:, None]).sum(axis=0) / d_total
        X_centered = X_arr - weighted_means
        denom = ((X_centered**2) * d[:, None]).sum(axis=0)
        quad_W = ((W @ X_centered) * X_centered).sum(axis=0)
        numer = denom - quad_W

        const = denom < 1e-12
        laplacian_scores = np.where(const, np.inf, numer / np.where(const, 1.0, denom))

        # higher is better, can go negative, constant features get -inf
        return dict(zip(columns, 1 - laplacian_scores))

    def _subsample(self, X: np.ndarray, *, cap: int) -> np.ndarray:
        n = X.shape[0]
        if n <= cap:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = np.sort(rng.choice(n, size=cap, replace=False))
        return X[idx]

    @staticmethod
    def _construct_W(X: np.ndarray, k: int = 5, t: float | None = None) -> sparse.csr_matrix:
        """kNN affinity graph with heat-kernel weights."""
        n = X.shape[0]
        if n < 2:
            return sparse.csr_matrix((n, n))

        k_eff = min(k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
        nbrs.fit(X)
        dist, idx = nbrs.kneighbors(X)

        # Drop exactly one self-edge per row. If ties pushed the point out of its own
        # neighbour list, drop the farthest neighbour instead.
        self_mask = idx == np.arange(n)[:, None]
        self_mask[~self_mask.any(axis=1), -1] = True
        keep = ~self_mask
        dist_k = dist[keep].reshape(n, -1)
        idx_k = idx[keep].reshape(n, -1)

        if idx_k.shape[1] == 0:
            return sparse.csr_matrix((n, n))

        d2 = dist_k**2
        if t is None:
            t = float(np.median(d2))
            if not np.isfinite(t) or t <= 0.0:
                t = 1.0
        aff = np.exp(-d2 / t)

        rows = np.repeat(np.arange(n), idx_k.shape[1])
        W = sparse.coo_matrix((aff.ravel(), (rows, idx_k.ravel())), shape=(n, n)).tocsr()
        # Symmetrise on affinities rather than distances, so an edge kept either way survives.
        return W.maximum(W.T).tocsr()
