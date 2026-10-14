"""Checks a feature selection method against the benchmark's contract (``scripts/check_method.py``).

Run before benchmarking a new method; the ``add-method`` skill uses it as its gate:

* **registry**: the class is registered under its own ``name``, has a paper name and a subgroup of
  the paper's taxonomy, and is listed as a proxy-model method exactly when it fits a proxy model;
* **naming**: names fit the ``FSBench__<method>__...`` pipeline strings and leaderboard labels;
* **fit**: on small binary, multiclass and regression data (with missing values and a categorical
  column) the method returns exactly ``max_features`` columns of the input, in the
  ``{"feature", "value"}`` format, with ``None`` values only for random-fallback fills;
* **determinism**: the same ``random_state`` gives the same selection;
* **time limit**: on a larger dataset the fit stops near its ``time_limit`` (the benchmark gives
  every method a fixed selection time and fills the budget at random when it runs out).

Each check returns ``CheckResult``s; ``ok`` is False only for failures, warnings do not fail.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from selectarena.methods.registry import (
    FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL,
    METHOD_SUBGROUPS,
    NAME_TO_MODULE_MAP,
    PAPER_NAMES,
    SUBGROUP_CATEGORIES,
)

PROBLEM_TYPES = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}


@dataclass
class CheckResult:
    check: str
    ok: bool
    message: str
    warning: bool = False

    def __str__(self) -> str:
        status = "WARN" if self.warning else ("ok  " if self.ok else "FAIL")
        return f"[{status}] {self.check}: {self.message}"


def _fail(check: str, message: str) -> CheckResult:
    return CheckResult(check, False, message)


def _ok(check: str, message: str) -> CheckResult:
    return CheckResult(check, True, message)


def _warn(check: str, message: str) -> CheckResult:
    return CheckResult(check, True, message, warning=True)


# --------------------------------------------------------------------------------------------------
# Static checks


def check_registry(name: str) -> list[CheckResult]:
    """Registration, naming and taxonomy of method ``name`` (its class name)."""
    results = []
    if name not in NAME_TO_MODULE_MAP:
        return [_fail("registry", f"{name} is not in NAME_TO_MODULE_MAP (methods/registry.py)")]
    module_name = f"selectarena.methods.{NAME_TO_MODULE_MAP[name]}"
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        return [_fail("import", f"{module_name} does not import: {e!r}")]
    cls = getattr(module, name, None)
    if cls is None:
        return [_fail("import", f"{module_name} defines no class {name}")]
    results.append(_ok("import", f"{module_name}.{name}"))

    source = inspect.getsource(module)
    if "from __future__ import annotations" not in source:
        results.append(_fail("style", f"{module_name} lacks 'from __future__ import annotations'"))
    if getattr(cls, "name", None) != name:
        results.append(_fail("naming", f"{name}.name is {getattr(cls, 'name', None)!r}; it must equal the class name"))
    if "__" in name:
        results.append(_fail("naming", f"{name} contains '__', the separator of FSBench pipeline strings"))
    if not name.endswith("FeatureSelector"):
        results.append(_warn("naming", f"{name} does not end with 'FeatureSelector' like the other methods"))

    label = PAPER_NAMES.get(name)
    if label is None:
        results.append(_fail("paper name", f"{name} has no entry in PAPER_NAMES"))
    else:
        duplicates = [c for c, n in PAPER_NAMES.items() if n == label and c != name]
        if duplicates:
            results.append(_fail("paper name", f"{label!r} is also the paper name of {duplicates}"))
        if " [" in label or "__" in label:
            results.append(_fail("paper name", f"{label!r} must not contain ' [' (ablation labels) or '__'"))
        if not duplicates and " [" not in label and "__" not in label:
            results.append(_ok("paper name", repr(label)))

    subgroup = METHOD_SUBGROUPS.get(name)
    if subgroup is None:
        results.append(_fail("taxonomy", f"{name} has no entry in METHOD_SUBGROUPS (it would show as 'Other')"))
    elif subgroup not in SUBGROUP_CATEGORIES:
        results.append(_fail("taxonomy", f"subgroup {subgroup!r} has no category in SUBGROUP_CATEGORIES"))
    else:
        results.append(_ok("taxonomy", f"{subgroup} ({SUBGROUP_CATEGORIES[subgroup].lower()})"))

    class_source = inspect.getsource(cls)
    uses_proxy = "evaluate_proxy_model" in class_source
    listed = name in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL
    if uses_proxy and not listed:
        results.append(
            _fail("proxy model", "calls evaluate_proxy_model but is not in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL")
        )
    elif listed and not uses_proxy:
        results.append(
            _warn(
                "proxy model",
                "listed in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL but never calls evaluate_proxy_model",
            )
        )
    else:
        results.append(_ok("proxy model", "uses a proxy model" if uses_proxy else "no proxy model"))

    from selectarena.methods.abstract import AbstractFeatureSelector, AbstractITFeatureSelector

    scoring = getattr(cls, "feature_scoring_method", False)
    implemented = "_fit_feature_scoring" if scoring else "_fit_feature_selection"
    owner = next((base for base in cls.__mro__ if implemented in vars(base)), None)
    if owner in (None, AbstractFeatureSelector, AbstractITFeatureSelector):
        results.append(
            _fail("interface", f"feature_scoring_method={scoring} but {name} does not implement {implemented}")
        )
    else:
        results.append(_ok("interface", f"feature_scoring_method={scoring}, implements {implemented}"))
    return results


# --------------------------------------------------------------------------------------------------
# Fit checks


def synthetic_data(problem_type: str, *, n_samples: int = 200, n_features: int = 12, seed: int = 0):
    """Data as a selector sees it in the pipeline: numeric and categorical columns, missing values.

    The first three columns are the informative ones.
    """
    from sklearn.datasets import make_classification, make_regression

    if problem_type == "regression":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_informative=3, shuffle=False, noise=5.0, random_state=seed
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            n_redundant=0,
            n_repeated=0,
            n_classes=2 if problem_type == "binary" else 3,
            n_clusters_per_class=1,
            shuffle=False,
            random_state=seed,
        )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    rng = np.random.default_rng(seed)
    X[f"f{n_features - 1}"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n_samples))
    for col in (f"f{n_features - 2}", f"f{n_features - 3}"):
        X.loc[rng.random(n_samples) < 0.1, col] = np.nan
    return X, pd.Series(y, name="target")


def _proxy_config(problem_type: str):
    from selectarena.methods.abstract import ProxyModelConfig

    if importlib.util.find_spec("lightgbm") is not None:
        from autogluon.tabular.models import LGBModel

        return ProxyModelConfig(model_class=LGBModel, model_hyperparameters={"num_boost_round": 10})
    from autogluon.tabular.models import RFModel

    return ProxyModelConfig(model_class=RFModel, model_hyperparameters={"n_estimators": 10})


def _make_selector(name: str, *, max_features: int, problem_type: str, random_state: int = 0):
    from selectarena.methods.registry import get_feature_selector_from_name

    cls = get_feature_selector_from_name(name=name)
    proxy = _proxy_config(problem_type) if name in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL else None
    return cls(max_features=max_features, proxy_mode_config=proxy, random_state=random_state, verbosity=0)


def _fit(name: str, X: pd.DataFrame, y: pd.Series, problem_type: str, *, max_features: int, time_limit, random_state=0):
    selector = _make_selector(name, max_features=max_features, problem_type=problem_type, random_state=random_state)
    start = time.monotonic()
    X_out = selector.fit_transform(
        X=X, y=y, time_limit=time_limit, eval_metric=PROBLEM_TYPES[problem_type], problem_type=problem_type
    )
    return selector, X_out, time.monotonic() - start


def check_fit(name: str, *, max_features: int = 3, time_limit: float = 60) -> list[CheckResult]:
    """Fit on binary, multiclass and regression data; check the output, determinism and signal."""
    results = []
    informative_hits = []
    for problem_type in PROBLEM_TYPES:
        check = f"fit {problem_type}"
        X, y = synthetic_data(problem_type)
        try:
            selector, X_out, elapsed = _fit(name, X, y, problem_type, max_features=max_features, time_limit=time_limit)
        except Exception as e:
            results.append(_fail(check, f"raised {e!r}"))
            continue
        selected = selector._selected_features
        problems = []
        if not isinstance(selected, list) or not all(
            isinstance(e, dict) and set(e) == {"feature", "value"} for e in selected
        ):
            problems.append("_selected_features is not a list of {'feature', 'value'} dicts")
        else:
            names = [e["feature"] for e in selected]
            if len(names) != max_features or len(set(names)) != max_features:
                problems.append(f"selected {len(names)} features ({len(set(names))} distinct), budget {max_features}")
            if not set(names) <= set(X.columns):
                problems.append(f"selected features not in the input: {sorted(set(names) - set(X.columns))}")
            if list(X_out.columns) != names:
                problems.append("the transformed data's columns differ from the selection")
            fills = {e["feature"] for e in selected if e["value"] is None}
            if not fills <= set(selector._feature_selection_random_features or []):
                problems.append("None values must be reserved for random-fallback fills")
            bad_values = [
                e["value"]
                for e in selected
                if e["value"] is not None and not isinstance(e["value"], (float, int, np.floating, np.integer))
            ]
            if bad_values:
                problems.append(f"values must be float, NaN or None, got {bad_values[:3]}")
            if selector.feature_scoring_method:
                scores = selector._feature_scores
                if not isinstance(scores, dict) or not set(scores) <= set(X.columns):
                    problems.append("_feature_scores must map input feature names to scores")
                elif any(isinstance(v, float) and math.isnan(v) for v in scores.values()):
                    problems.append("_feature_scores contains NaN, which breaks the ranking (use -inf)")
            informative_hits.append(len({"f0", "f1", "f2"} & set(names)) - len(fills & {"f0", "f1", "f2"}))
        if problems:
            results.append(_fail(check, "; ".join(problems)))
            continue
        n_random = len(selector._feature_selection_random_features or [])
        note = f", {n_random} random fills" if n_random else ""
        results.append(_ok(check, f"{[e['feature'] for e in selected]} in {elapsed:.1f}s{note}"))

        # Determinism: the same random_state gives the same selection.
        try:
            again, _, _ = _fit(name, X, y, problem_type, max_features=max_features, time_limit=time_limit)
            first = [e["feature"] for e in selected]
            second = [e["feature"] for e in again._selected_features]
            if first != second:
                results.append(_fail(f"determinism {problem_type}", f"{first} then {second} with the same seed"))
        except Exception as e:
            results.append(_fail(f"determinism {problem_type}", f"second fit raised {e!r}"))

    if informative_hits and name != "RandomFeatureSelector":
        mean_hits = float(np.mean(informative_hits))
        # A random pick of 3 out of 12 hits 0.75 informative features on average.
        if mean_hits <= 0.75:
            results.append(
                _warn("signal", f"{mean_hits:.2f} of 3 informative features found on average (random: 0.75)")
            )
        else:
            results.append(_ok("signal", f"{mean_hits:.2f} of 3 informative features found on average"))
    return results


def check_time_limit(
    name: str,
    *,
    time_limit: float = 5,
    n_samples: int = 3000,
    n_features: int = 80,
    tolerance: float = 2.0,
    slack: float = 10.0,
) -> list[CheckResult]:
    """On a larger dataset the fit must stop near ``time_limit`` (at most ``tolerance`` x limit + ``slack`` s)."""
    X, y = synthetic_data("binary", n_samples=n_samples, n_features=n_features, seed=1)
    max_features = max(1, math.ceil(n_features / 4))
    try:
        selector, _, elapsed = _fit(name, X, y, "binary", max_features=max_features, time_limit=time_limit)
    except Exception as e:
        return [_fail("time limit", f"raised {e!r}")]
    allowed = tolerance * time_limit + slack
    fills = len(selector._feature_selection_random_features or [])
    detail = f"{elapsed:.1f}s with a {time_limit:g}s limit ({n_samples}x{n_features}, budget {max_features})"
    if elapsed > allowed:
        return [_fail("time limit", f"{detail}: check time_limit in the selection loop (self._timed_out)")]
    if fills:
        detail += f", {fills} random fills after the limit"
    return [_ok("time limit", detail)]


def check_method(name: str, *, fit: bool = True, time_limit_check: bool = True) -> list[CheckResult]:
    """All checks for method ``name``; fit checks run only if the static ones pass."""
    results = check_registry(name)
    if not all(r.ok for r in results):
        return results
    if fit:
        results += check_fit(name)
    if time_limit_check and all(r.ok for r in results):
        results += check_time_limit(name)
    return results
