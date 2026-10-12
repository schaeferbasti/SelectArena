"""Koller-Sahami feature selection via chi-square conditional-independence tests."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import chi2

from selectarena.methods.abstract import (
    AbstractITFeatureSelector,
)

_LN10 = math.log(10.0)


class MarkovBlanketFeatureSelector(AbstractITFeatureSelector):
    """Koller-Sahami backward elimination (ICML 1996).

    For each round of the backward search:

    1. Give every survivor a candidate blanket, up to K of its best subsumers
       among the survivors, ranked by the paper's gamma_ij = I(C; F_i | F_j).
    2. Test whether the feature still informs the target given that blanket.
    3. Drop the one with the weakest evidence.

    Repeat until exactly ``max_features`` remain.

    Where this differs from the paper.
    - Scores are log p-values of a chi-square test with adjusted df (bnlearn's
      `x2-adf`), not raw plug-in CMI. Plug-in CMI is biased upward by ~df/2n,
      which flatters high-cardinality features. The p-value removes that bias,
      but it is not an effect size, so it under-rates them instead.
    - Regression targets are cut into ``y_bins`` equal-frequency bins.
    - A test counts only if it is powered (n >= min_obs_per_cell * df).
      Unpowered blanket additions are skipped, so on small samples the blanket
      shrinks toward 1 or 0. ``_ks_mean_blanket_size`` reports the average.
    - On timeout the rest is filled from the marginal ranking, so the budget is
      always met exactly and the random fallback never fires. See
      ``_ks_timed_out``.

    """

    name = "MarkovBlanketFeatureSelector"
    feature_scoring_method: bool = False

    markov_blanket_size: int = 2
    """Max blanket size."""
    y_bins: int = 5
    """Equal-frequency bins for a regression target."""
    max_conditioner_levels: int = 0
    """Coarsen conditioners to at most this many levels, or 0 to turn it off."""
    min_obs_per_cell: int = 5
    """A test with n < min_obs_per_cell * df counts as unpowered."""
    max_skips: int = 8
    """Stop growing a blanket after this many rejected candidates in a row."""
    max_table_bytes: int = 512_000_000
    """Peak bytes for one contingency-table computation, counted over every
    array that is live at once."""

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[dict]:
        start_time = time.monotonic()
        K = self.markov_blanket_size

        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        X_pre = X_pre.reset_index(drop=True)
        feats = list(X_pre.columns)
        n = len(X_pre)
        k = min(self.max_features, len(feats))

        codes = {f: self._compact(X_pre[f].to_numpy()) for f in feats}
        # Conditioner-side view. Candidate and target keep full resolution.
        zcodes = {f: self._coarsen(codes[f], self.max_conditioner_levels) for f in feats}
        yb = self._encode_y(y)
        no_z = (np.zeros(n, dtype=np.int64), 1)

        def weakness(stat: tuple[float, float, bool]) -> tuple[float, float]:
            """Sort key. Larger means a weaker association with the target."""
            return (stat[0], -stat[1])

        marginal = {f: self._ci(codes[f], yb, no_z, n) for f in feats}
        current = list(feats)
        timed_out = False

        # Pairwise gamma, computed once. It does not depend on which survive, so
        # restricting the ranking to survivors each round is the paper's step 3.
        pair_logp, pair_stat, pair_powered, timed_out = self._pairwise(
            feats, codes, zcodes, yb, n, time_limit, start_time
        )
        subsumers: dict[str, list[str]] = {}
        if not timed_out:
            for i, fi in enumerate(feats):
                # `powered` leads the key. An untested pair's log p means "not
                # measured", not "independent", so it must rank below every
                # measured pair instead of on top of them.
                order = sorted(
                    (j for j in range(len(feats)) if j != i),
                    key=lambda j: (pair_powered[i, j], pair_logp[i, j], -pair_stat[i, j]),
                    reverse=True,
                )
                subsumers[fi] = [feats[j] for j in order]

        blanket_sizes: list[int] = []

        def compute_delta(f: str) -> tuple[tuple[float, float, bool], tuple[str, ...]]:
            """Evidence that f still informs the target given its best blanket.

            Also returns the survivors whose later removal could change it.
            """
            survivors = set(current)
            mb: list[str] = []
            examined: list[str] = []
            res = marginal[f]
            skips = 0
            ended_by_skips = False
            for fj in subsumers[f]:
                if len(mb) == K:
                    break
                if skips >= self.max_skips:
                    ended_by_skips = True
                    break
                if fj not in survivors:
                    continue
                examined.append(fj)
                z = self._joint([zcodes[g] for g in [*mb, fj]], n)
                trial = self._ci(codes[f], yb, z, n)
                if trial[2]:
                    mb.append(fj)
                    res = trial
                    skips = 0
                else:
                    skips += 1
            blanket_sizes.append(len(mb))
            # Dropping a rejected candidate only lowers the skip counter, so it
            # can delay the max_skips break but never advance it. If the scan
            # stopped for any other reason the replay is identical and only the
            # accepted members matter.
            return res, tuple(examined) if ended_by_skips else tuple(mb)

        # Backward elimination. A cached delta stays valid while every survivor
        # it depended on is still present. The ranking is fixed and the pool
        # only shrinks, so the greedy scan would replay identically.
        cache: dict[str, tuple] = {}
        while len(current) > k and not timed_out:
            for f in current:
                if f in cache:
                    continue
                if self._timed_out(time_limit, start_time):
                    timed_out = True
                    break
                cache[f] = compute_delta(f)
            if timed_out:
                break
            worst = max(current, key=lambda f: weakness(cache[f][0]))
            current.remove(worst)
            cache.pop(worst, None)
            for f in [g for g in cache if worst in cache[g][1]]:
                del cache[f]

        if len(current) > k:  # timed out, so finish by marginal ranking
            current.sort(key=lambda f: weakness(marginal[f]))
            del current[k:]

        self._ks_timed_out = timed_out
        self._ks_mean_blanket_size = float(np.mean(blanket_sizes)) if blanket_sizes else 0.0

        # value = -log10 p of the final delta. 0 means no evidence.
        out = []
        for f in current:
            res = cache[f][0] if f in cache else marginal[f]
            out.append({"feature": str(f), "value": float(-res[0] / _LN10)})
        out.sort(key=lambda d: d["value"], reverse=True)
        return out

    def _pairwise(self, feats, codes, zcodes, yb, n, time_limit, start_time):
        """log p, X2 and poweredness for every (F_i ind C | F_j) pair, x2-adf.

        One bincount per conditioner over a (p, strata, R, ry) table, then one
        vectorized chi-square, sharing the target margins across candidates.
        Unpowered, degenerate and size-skipped pairs get log p = -inf and
        powered = False, so they sort last as subsumers.
        """
        p = len(feats)
        yc, ry = yb
        Xc = np.stack([codes[f][0] for f in feats])  # (p, n)
        rx = np.array([codes[f][1] for f in feats])
        R = int(rx.max()) if p else 1
        pair_logp = np.full((p, p), -np.inf)
        pair_stat = np.zeros((p, p))
        pair_powered = np.zeros((p, p), dtype=bool)
        inner_x = Xc * ry + yc[None, :]  # z-part added per fj
        offsets = (np.arange(p) * R * ry)[:, None]
        for j, fj in enumerate(feats):
            if self._timed_out(time_limit, start_time):
                return pair_logp, pair_stat, pair_powered, True
            z, rz = zcodes[fj]
            # counts, expected and dev are live at once, plus the (p, n) index.
            if 8 * (p * n + 3 * p * rz * R * ry) > self.max_table_bytes:
                continue  # stays unpowered
            nyz = np.bincount(z * ry + yc, minlength=rz * ry).astype(np.float64).reshape(rz, ry)
            nz = nyz.sum(axis=1)  # >0 by construction
            obs_y = (nyz > 0).sum(axis=1)
            idx = (offsets * rz + (z * R * ry)[None, :] + inner_x).ravel()
            counts = np.bincount(idx, minlength=p * rz * R * ry).astype(np.float64)
            counts = counts.reshape(p, rz, R, ry)
            nxz = counts.sum(axis=3)  # (p, rz, R)
            expected = nxz[..., None] * (nyz / nz[:, None])[None, :, None, :]
            mask = expected > 0
            dev = np.zeros_like(counts)
            dev[mask] = (counts[mask] - expected[mask]) ** 2 / expected[mask]
            stat = dev.sum(axis=(1, 2, 3))
            obs_x = (nxz > 0).sum(axis=2)  # (p, rz)
            df = (np.maximum(obs_x - 1, 0) * np.maximum(obs_y - 1, 0)[None, :]).sum(axis=1)
            powered = (df > 0) & (n >= self.min_obs_per_cell * df)
            pair_stat[:, j] = stat
            good = np.flatnonzero(powered)
            if good.size:
                pair_logp[good, j] = self._log_sf(stat[good], df[good])
                pair_powered[good, j] = True
        return pair_logp, pair_stat, pair_powered, False

    @staticmethod
    def _log_sf(stat: np.ndarray, df: np.ndarray) -> np.ndarray:
        """log P(X2_df > stat), valid deep into the tail.

        scipy's chi2.logsf logs an already-underflowed sf, so it hits -inf at
        X2 ~ 1.4e3 for any df, which real data clears easily. Past that, use the
        incomplete-gamma asymptotic, which matches scipy wherever scipy is finite.
        """
        stat = np.atleast_1d(np.asarray(stat, dtype=np.float64))
        df = np.broadcast_to(np.atleast_1d(np.asarray(df, dtype=np.float64)), stat.shape)
        out = np.asarray(chi2.logsf(stat, df), dtype=np.float64).copy()
        bad = np.isneginf(out) & (stat > 0)
        if bad.any():
            a = df[bad] / 2.0
            z = stat[bad] / 2.0
            series = 1.0 + (a - 1.0) / z + (a - 1.0) * (a - 2.0) / (z * z)
            out[bad] = (a - 1.0) * np.log(z) - z - gammaln(a) + np.log(np.clip(series, 1e-300, None))
        return out

    @staticmethod
    def _compact(values: np.ndarray) -> tuple[np.ndarray, int]:
        levels, inv = np.unique(values, return_inverse=True)
        return inv.astype(np.int64), len(levels)

    @classmethod
    def _coarsen(cls, block: tuple[np.ndarray, int], max_levels: int) -> tuple[np.ndarray, int]:
        """Merge an ordinal code array into <= max_levels equal-frequency groups."""
        arr, r = block
        if max_levels <= 0 or r <= max_levels:
            return block
        cum = np.cumsum(np.bincount(arr, minlength=r).astype(np.int64))
        edges = np.searchsorted(cum, (np.arange(1, max_levels) * int(cum[-1])) / max_levels, side="left")
        return cls._compact(np.searchsorted(edges, np.arange(r), side="left")[arr])

    _relabel_max: int = 1 << 20
    """Largest intermediate code range ``_joint`` will relabel by counting."""

    @staticmethod
    def _relabel(code: np.ndarray, upper: int) -> tuple[np.ndarray, int]:
        """Relabel codes bounded by ``upper`` without sorting.

        Same labels as ``_compact``, in O(n + upper) instead of O(n log n).
        """
        seen = np.zeros(upper, dtype=bool)
        seen[code] = True
        mapping = np.cumsum(seen) - 1
        return mapping[code], int(mapping[-1]) + 1

    @classmethod
    def _joint(cls, blocks: list[tuple[np.ndarray, int]], n: int) -> tuple[np.ndarray, int]:
        code = np.zeros(n, dtype=np.int64)
        r = 1
        for arr, ra in blocks:
            upper = r * ra
            code = code * ra + arr
            # Blanket joins have a tiny code range, so counting beats the
            # argsort in _compact, which would otherwise dominate the run.
            code, r = cls._relabel(code, upper) if upper <= cls._relabel_max else cls._compact(code)
        return code, r

    def _encode_y(self, y: pd.Series) -> tuple[np.ndarray, int]:
        y = y.reset_index(drop=True)
        if self.problem_type == "regression":
            b = int(min(self.y_bins, max(2, y.nunique())))
            binned = pd.qcut(y.astype(float).rank(method="first"), q=b, labels=False, duplicates="drop")
            return self._compact(binned.to_numpy())
        return self._compact(self._encode_target(y).reset_index(drop=True).to_numpy())

    def _ci(
        self,
        xb: tuple[np.ndarray, int],
        yb: tuple[np.ndarray, int],
        zb: tuple[np.ndarray, int],
        n: int,
    ) -> tuple[float, float, bool]:
        """(log p, X2, powered) for X independent of Y given Z (bnlearn's x2-adf)."""
        x, rx = xb
        yc, ry = yb
        z, rz = zb
        if rx < 2 or ry < 2 or 8 * 2 * rz * rx * ry > self.max_table_bytes:
            return 0.0, 0.0, False
        idx = (z * rx + x) * ry + yc
        nxyz = np.bincount(idx, minlength=rz * rx * ry).astype(np.float64).reshape(rz, rx, ry)
        nz = nxyz.sum(axis=(1, 2))
        nxz = nxyz.sum(axis=2)
        nyz = nxyz.sum(axis=1)
        nz_col = np.where(nz > 0, nz, 1.0)[:, None, None]
        expected = nxz[:, :, None] * nyz[:, None, :] / nz_col
        mask = expected > 0
        stat = float((((nxyz - expected)[mask]) ** 2 / expected[mask]).sum())
        obs_x = (nxz > 0).sum(axis=1)
        obs_y = (nyz > 0).sum(axis=1)
        df = int((np.maximum(obs_x - 1, 0) * np.maximum(obs_y - 1, 0)).sum())
        if df <= 0:
            return 0.0, stat, False
        if n < self.min_obs_per_cell * df:
            return 0.0, stat, False
        logp = float(chi2.logsf(stat, df))
        if logp == -np.inf and stat > 0:  # deep tail, see _log_sf
            logp = float(self._log_sf(stat, df)[0])
        return logp, stat, True
