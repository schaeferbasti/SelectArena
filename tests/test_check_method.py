"""The method checker passes good methods and catches the common mistakes of new ones."""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from selectarena.methods import registry
from selectarena.methods.abstract import AbstractFeatureSelector
from selectarena.methods.checks import check_fit, check_method, check_registry, check_time_limit


class ListOfNamesFeatureSelector(AbstractFeatureSelector):
    """Returns plain names instead of {"feature", "value"} dicts."""

    name = "ListOfNamesFeatureSelector"

    def _fit_feature_selection(self, *, X, y, time_limit=None):
        return list(X.columns[: self.max_features])


class GlobalRandomFeatureSelector(AbstractFeatureSelector):
    """Draws from the global numpy RNG instead of self._rng / self.random_state."""

    name = "GlobalRandomFeatureSelector"
    feature_scoring_method = True

    def _fit_feature_scoring(self, *, X, y, time_limit=None):
        return dict(zip(X.columns, np.random.rand(X.shape[1])))


class SlowFeatureSelector(AbstractFeatureSelector):
    """Ignores its time limit."""

    name = "SlowFeatureSelector"
    feature_scoring_method = True

    def _fit_feature_scoring(self, *, X, y, time_limit=None):
        time.sleep(1.5)
        return {c: float(i) for i, c in enumerate(X.columns)}


@pytest.fixture
def register(monkeypatch):
    """Register a class of this module like a method module under selectarena.methods."""
    monkeypatch.setitem(sys.modules, "selectarena.methods._check_fixture", sys.modules[__name__])

    def _register(cls, *, paper_name=None, subgroup="Tree"):
        monkeypatch.setitem(registry.NAME_TO_MODULE_MAP, cls.name, "_check_fixture")
        if paper_name:
            monkeypatch.setitem(registry.PAPER_NAMES, cls.name, paper_name)
        if subgroup:
            monkeypatch.setitem(registry.METHOD_SUBGROUPS, cls.name, subgroup)
        return cls.name

    return _register


@pytest.mark.parametrize("name", ["MIFeatureSelector", "RandomFeatureSelector", "CARTFeatureSelector"])
def test_registered_methods_pass(name):
    results = check_method(name)
    assert all(r.ok for r in results), "\n".join(map(str, results))


def test_unregistered_method_fails():
    assert not any(r.ok for r in check_registry("NotAFeatureSelector"))


def test_missing_paper_name_and_subgroup_fail(register):
    name = register(ListOfNamesFeatureSelector, paper_name=None, subgroup=None)
    failed = {r.check for r in check_registry(name) if not r.ok}
    assert {"paper name", "taxonomy"} <= failed


def test_duplicate_paper_name_fails(register):
    name = register(ListOfNamesFeatureSelector, paper_name="MI")
    assert any(r.check == "paper name" and not r.ok for r in check_registry(name))


def test_wrong_output_format_fails(register):
    name = register(ListOfNamesFeatureSelector, paper_name="Names")
    assert all(r.ok for r in check_registry(name))
    assert not all(r.ok for r in check_fit(name))


def test_global_randomness_fails_determinism(register):
    name = register(GlobalRandomFeatureSelector, paper_name="GlobalRandom")
    failed = [r for r in check_fit(name) if not r.ok]
    assert failed and all(r.check.startswith("determinism") for r in failed)


def test_ignoring_the_time_limit_fails(register):
    name = register(SlowFeatureSelector, paper_name="Slow")
    (result,) = check_time_limit(name, time_limit=0.2, n_samples=200, n_features=12, tolerance=1, slack=0.5)
    assert not result.ok and "time_limit" in result.message
