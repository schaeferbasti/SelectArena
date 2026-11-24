"""Six PACE/Raw selectors implementing SelectArena's native scoring contract."""
from __future__ import annotations

import gc
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import os
import time

import numpy as np
import torch
from selectarena.methods.abstract import AbstractFeatureSelector

from .pace_encoder import PaceEncoder
from .pace_kernels import ScreeningTimeout, check_deadline, kernel_score, scalar_bandwidth, sis_score, target_tensor


def representation_bandwidth(z):
    """Lower median over strictly positive unordered Euclidean distances."""
    distances = torch.pdist(z.contiguous())
    positive = distances[distances > 0]
    return positive.median().item() if positive.numel() else 1.0


def error_kind(message):
    if "float() argument" in message and "'NAType'" in message:
        return "nullable_missing_value"
    if "could not convert string to float" in message:
        return "unencoded_text"
    if "Cannot cast object dtype to float64" in message:
        return "categorical_cast"
    if "Columns must be same length as key" in message:
        return "empty_column"
    return None


class _PaceRawSelector(AbstractFeatureSelector):
    feature_scoring_method = True
    pace = False
    utility = 'sis'

    def _preprocess(self, X, **kwargs):
        try:
            result, y = super()._preprocess(X, **kwargs)
            result.to_numpy(dtype=np.float32)
            return result, y
        except (ValueError, TypeError) as exc:
            if not (kwargs.get("impute") and kwargs.get("encode_ordinal") and error_kind(str(exc))):
                raise
            print("PREPROCESS_FALLBACK", self.name, error_kind(str(exc)), flush=True)

        columns = {}
        for name, series in X.items():
            if series.isna().all():
                # A constant keeps the candidate identity without inventing variation.
                columns[name] = np.zeros(len(series), dtype=np.float32)
            elif is_numeric_dtype(series.dtype):
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                fill = values[np.isfinite(values)].mean() if np.isfinite(values).any() else 0.0
                columns[name] = np.where(np.isfinite(values), values, fill)
            else:
                values = series.astype(object).to_numpy(copy=True)
                missing = series.isna().to_numpy()
                values[missing] = np.nan
                observed = values[~missing]
                if not (
                    all(isinstance(value, str) for value in observed)
                    or all(isinstance(value, (int, float, np.number)) for value in observed)
                ):
                    values[~missing] = [f"{type(value).__name__}:{value}" for value in observed]
                values = SimpleImputer(strategy="most_frequent").fit_transform(values.reshape(-1, 1))
                columns[name] = OrdinalEncoder().fit_transform(values).ravel()
        return pd.DataFrame(columns, index=X.index), kwargs.get("y")

    def _fit_feature_scoring(self, *, X, y, time_limit=None):
        start = time.monotonic()
        deadline = start + time_limit if time_limit is not None else None
        # Same training-only type handling for each member of the Raw/PACE pair.
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)
        n_train = len(X_pre)
        seed = getattr(self, '_outer_random_state', None)
        seed = 42 if seed is None else int(seed)
        if n_train > 2000:
            rows = np.sort(np.random.default_rng(seed).choice(n_train, 2000, replace=False))
            X_pre, y = X_pre.iloc[rows], y.iloc[rows]
        values = X_pre.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite features after official training-only preprocessing')
        device = os.environ.get('PACE_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA device is unavailable')
        target = target_tensor(y, self.problem_type, device)
        encoder = None
        scores = {}
        try:
            check_deadline(deadline)
            if self.pace:
                checkpoint = os.environ.get("PACE_ENCODER_CHECKPOINT")
                if checkpoint is None:
                    from huggingface_hub import hf_hub_download
                    checkpoint = hf_hub_download("jingang/TabICL", "tabicl-classifier-v1.1-20250506.ckpt")
                encoder = PaceEncoder(checkpoint, device)
            sy = 1.0
            if self.utility == 'hsic' and self.problem_type == 'regression':
                sy = scalar_bandwidth(y, deadline)
            positions = {name: i for i, name in enumerate(X_pre.columns)}
            for name in self._seeded_feature_order():
                check_deadline(deadline)
                x = values[:, positions[name]]
                z = encoder.encode(x) if encoder else torch.as_tensor(x.copy(), device=device).reshape(-1, 1)
                if self.utility == 'sis':
                    score = sis_score(z, target)
                else:
                    sx = 1.0
                    if self.utility == 'hsic':
                        sx = representation_bandwidth(z) if self.pace else scalar_bandwidth(x, deadline)
                    score = kernel_score(z, target, self.utility, sigma_x=sx, sigma_y=sy,
                        categorical=self.problem_type != 'regression', pace=self.pace, deadline=deadline)
                if not np.isfinite(score):
                    raise ValueError(f'Non-finite score: {name}')
                scores[name] = score
                del z
            if device.startswith('cuda'):
                torch.cuda.synchronize()
        except ScreeningTimeout:
            # The official base class fills unscored features using seeded random fallback.
            self._log(30, f'{self.name}: screening time limit, {len(scores)}/{len(X_pre.columns)} scored')
        finally:
            del encoder, target
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        return scores


class RawSISFeatureSelector(_PaceRawSelector):
    name = 'RawSISFeatureSelector'


class PACESISFeatureSelector(_PaceRawSelector):
    name = 'PACESISFeatureSelector'
    pace = True


class RawDCFeatureSelector(_PaceRawSelector):
    name = 'RawDCFeatureSelector'
    utility = 'dc'


class PACEDCFeatureSelector(_PaceRawSelector):
    name = 'PACEDCFeatureSelector'
    pace = True
    utility = 'dc'


class RawHSICFeatureSelector(_PaceRawSelector):
    name = 'RawHSICFeatureSelector'
    utility = 'hsic'


class PACEHSICFeatureSelector(_PaceRawSelector):
    name = 'PACEHSICFeatureSelector'
    pace = True
    utility = 'hsic'
