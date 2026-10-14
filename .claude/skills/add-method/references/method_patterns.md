# Method patterns for SelectArena

Templates and the base-class contract for a new feature selection method. The authoritative code is
`src/selectarena/methods/abstract.py`; `scripts/check_method.py` checks most of what is below.

## How a method is called

In the benchmark each selector is an AutoGluon **post generator**: AutoGluon preprocesses the data
of one bagging fold, then calls the selector's `fit_transform`, which runs

1. `self._rng = np.random.default_rng(self.random_state)` (the wrapper sets `random_state` per
   outer split), `self.max_features` resolved from the budget, `self.problem_type` set,
   the proxy config completed with the task's metric;
2. `_fit_feature_scoring` (scoring methods) or `_fit_feature_selection` (subset methods), timed;
3. completion to exactly `max_features` features by the **random fallback** if the method returned
   fewer (`_feature_selection_random_features` records them; their `value` is `None`);
4. `X[selected]` as the output.

The stability / validity benchmark calls `fit_transform(X=X, y=y, time_limit=..., eval_metric=...,
problem_type=...)` directly on a task's data.

Ablations reuse the same class: `__nofallback` (`skip_random_fallback=True`: the method may return
fewer features) and `__permenc<seed>` (`encoding_permutation_seed`: `self._preprocess(...,
encode_ordinal=True)` permutes the category codes). Use `self._preprocess` and they work for free.

## What the method can use

| Attribute / helper | Meaning |
|---|---|
| `self.max_features` | Number of features to return (resolved; never more than the input has). |
| `self.problem_type` | `"binary"`, `"multiclass"` or `"regression"`. |
| `self._original_features` | Input column names, in input order. |
| `self.random_state`, `self._rng` | Per-split seed and its generator. The only allowed randomness. |
| `self._seeded_feature_order()` | The input columns in a per-split random order, for loops a timeout may truncate. |
| `self._timed_out(time_limit, start_time)` | True (and logs) once `time_limit` seconds have passed. |
| `self._preprocess(X, y=None, impute=, discretize=, encode_ordinal=, encode_target=, scale=)` | Imputation (mean / most frequent), quantile discretization (Sturges' bins), ordinal encoding of categoricals (honours the permutation ablation), label encoding, standardization. Returns `(X, y)`. |
| `self.evaluate_proxy_model(X=, y=, time_limit=)` | Validation score (higher is better) of the proxy model (LightGBM by default) on a 67/33 split of the given columns; `None` if no valid feature. Needs the class in `FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL`. |
| `AbstractITFeatureSelector` helpers | `_mutual_information_dispatched(x, y)` (MI with the right estimator per problem type), `_entropy(*xs)`, `_mutual_information`, `_conditional_mutual_information(x, y, z)`, `_joint_mutual_information`, `_symmetrical_uncertainty`, `_symmetrical_relevance`. Expect discretized inputs. |

## Output contract

- **Scoring** (`feature_scoring_method = True`): `_fit_feature_scoring` returns `{feature: score}`
  for the input columns, **higher is better**, finite floats. Convert p-values or costs (lower is
  better) yourself (`-p`, `1 - p`), and map `nan` (e.g. constant features) to `-inf` or `0.0`:
  `nan` breaks the ranking. A partial dict (time ran out) is fine; the fallback fills the rest.
- **Subset** (`feature_scoring_method = False`): `_fit_feature_selection` returns an ordered list
  (strongest / first selected first) of `{"feature": name, "value": v}` with `v` a float under the
  method's criterion or `float("nan")` if it has none. At most `max_features` entries for a forward
  method; a backward method may return its current set when the time runs out (the fallback then
  keeps a random `max_features` of it). Never `None` values; those mark fallback fills.
- Names are column names of `X` (strings); no duplicates.

## Template: scoring method (statistic or library score)

```python
"""<MethodName> feature selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class <ClassName>(AbstractFeatureSelector):
    """<MethodName> Feature Selection.

    Reference: <Authors>. "<Title>." <Venue> (<Year>).
    Implementation: <library function or "own implementation following the paper">.
    """

    name = "<ClassName>"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)
        if self.problem_type == "regression":
            scores = ...  # regression variant of the criterion
        else:
            scores = ...  # classification variant (binary and multiclass)
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
        return dict(zip(X.columns, scores.tolist()))
```

## Template: per-feature loop with a time limit

```python
    def _fit_feature_scoring(self, *, X, y, time_limit=None):
        start_time = time.monotonic()
        X_pre, _ = self._preprocess(X, impute=True, discretize=True, encode_ordinal=True)
        scores = {}
        for feature in self._seeded_feature_order():  # a timeout leaves a random subset scored
            if self._timed_out(time_limit, start_time):
                break
            scores[feature] = float(<criterion>(X_pre[feature], y))
        return scores
```

## Template: model-based scoring (importances / coefficients)

```python
    def _fit_feature_scoring(self, *, X, y, time_limit=None):
        from somelib import SomeClassifier, SomeRegressor  # optional dependency: import lazily

        X_pre, _ = self._preprocess(X, impute=True, encode_ordinal=True)
        model_cls = SomeRegressor if self.problem_type == "regression" else SomeClassifier
        model = model_cls(random_state=self.random_state).fit(X_pre, y)
        importances = np.abs(np.asarray(model.feature_importances_, dtype=float))
        if importances.ndim == 2:  # one row per class (multiclass coefficients)
            importances = importances.mean(axis=0)
        return dict(zip(X_pre.columns, importances.tolist()))
```

Libraries whose binary-only solver rejects multiclass targets need an explicit one-vs-rest scheme
(see `methods/lasso.py`).

## Template: search with the proxy model (forward)

```python
import time

import numpy as np
from autogluon.core.utils.exceptions import TimeLimitExceeded

from selectarena.methods.abstract import AbstractFeatureSelector


class <ClassName>(AbstractFeatureSelector):
    """<MethodName> Feature Selection. ... Uses the proxy model (FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL)."""

    name = "<ClassName>"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X, y, time_limit=None):
        start_time = time.monotonic()
        selected, values = [], []
        try:  # keep what was selected if the proxy fit hits the time limit
            while len(selected) < self.max_features:
                best_feature, best_score = None, -np.inf
                for feature in self._seeded_feature_order():
                    if feature in selected:
                        continue
                    if self._timed_out(time_limit, start_time):
                        raise TimeLimitExceeded
                    remaining = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - start_time))
                    score = self.evaluate_proxy_model(X=X[[*selected, feature]], y=y, time_limit=remaining)
                    if score is not None and score > best_score:
                        best_feature, best_score = feature, score
                if best_feature is None:
                    break
                selected.append(best_feature)
                values.append(float(best_score))
        except TimeLimitExceeded:
            self._log(30, f"Time limit reached with {len(selected)} features selected.")
        return [{"feature": f, "value": v} for f, v in zip(selected, values)]
```

For a backward method start from all features, remove the least useful one per step, and return
the remaining set (`value` = `float("nan")` when there is no per-feature number); see `methods/sbe.py`.

## Registry entries

```python
NAME_TO_MODULE_MAP = {..., "<ClassName>": "<module>"}
PAPER_NAMES = {..., "<ClassName>": "<MethodName>"}
METHOD_SUBGROUPS = {..., "<ClassName>": "<Subgroup>"}
FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL = [..., "<ClassName>"]  # only if it calls evaluate_proxy_model
# SUBGROUP_CATEGORIES = {..., "<New subgroup>": "Search" | "Non-search"}  # only for a new subgroup
```

## Common mistakes the checks catch

| Mistake | Check that fails |
|---|---|
| `name` differs from the class name, or contains `__` | `naming` |
| missing `PAPER_NAMES` / `METHOD_SUBGROUPS` entry, duplicate paper name | `paper name`, `taxonomy` |
| calls `evaluate_proxy_model` but not listed as a proxy method | `proxy model` |
| returns plain names or more than `max_features` entries | `fit <problem type>` |
| `np.random.*` / `random.*` instead of `self._rng` / `self.random_state` | `determinism` |
| `nan` scores | `fit <problem type>` |
| no `self._timed_out` check in a long loop | `time limit` |
| works only for classification | `fit regression` |
