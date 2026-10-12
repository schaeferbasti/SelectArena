"""Minimum Redundancy Maximum Relevance (mRMR) feature selection."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from selectarena.methods.abstract import AbstractITFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class mRMRFeatureSelector(AbstractITFeatureSelector):
    """mRMR Feature Selection.

    Reference: Peng, Hanchuan, Fuhui Long, and Chris Ding.
    "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy."
    IEEE Transactions on pattern analysis and machine intelligence 27.8 (2005): 1226-1238.

    Changes to the implementation:
        - Time constraint
    """

    name = "mRMRFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[dict]:
        """
        At each step, select the candidate that maximizes:
            score(X_i) = I(X_i; Y) - (1/|F|) * sum_{X_j in F} I(X_i; X_j)
        where F is the set of already-selected features.

        For the first selected feature (F empty), the criterion reduces to plain relevance: argmax_i I(X_i; Y).
        """
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        n_features = len(X_pre.columns)
        cols = [X_pre.iloc[:, i] for i in range(n_features)]

        relevance = np.array(
            [self._mutual_information_dispatched(col, y) for col in cols]
        )  # mi for all features and target
        mi_pairs: dict[tuple[int, int], float] = {}  # mi_pairs[(i, j)] = I(X_i; X_j); symmetric so we only store i < j

        def get_pair_mi(i: int, j: int) -> float:
            key = (min(i, j), max(i, j))
            if key not in mi_pairs:
                mi_pairs[key] = self._mutual_information(cols[i], cols[j])
            return mi_pairs[key]

        selected: list[int] = []
        selected_mask = np.zeros(n_features, dtype=bool)

        # step 1: select feature with highest relevance
        first = int(np.argmax(relevance))
        selected.append(first)
        selected_mask[first] = True
        values: list[float] = [float(relevance[first])]  # criterion value at selection (first: plain relevance)

        # step 2: greedy forward with mrmr criterion
        while len(selected) < self.max_features:
            if self._timed_out(time_limit, start_time):
                break

            best_score = -np.inf
            best_idx = None
            for i in range(n_features):
                if selected_mask[i]:
                    continue
                if self._timed_out(time_limit, start_time):
                    break
                avg_redundancy = sum(get_pair_mi(i, j) for j in selected) / len(selected)
                score = relevance[i] - avg_redundancy
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            values.append(float(best_score))  # mRMR criterion value at selection

        return [{"feature": str(self._original_features[i]), "value": v} for i, v in zip(selected, values)]
