"""Feature selection methods, implemented as AutoGluon feature generators."""

from __future__ import annotations

from selectarena.methods.abstract import AbstractFeatureSelector, AbstractITFeatureSelector, ProxyModelConfig
from selectarena.methods.registry import (
    FEATURE_SELECTION_METHODS,
    FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL,
    PAPER_NAMES,
    get_feature_selector_from_name,
)

__all__ = [
    "FEATURE_SELECTION_METHODS",
    "FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL",
    "PAPER_NAMES",
    "AbstractFeatureSelector",
    "AbstractITFeatureSelector",
    "ProxyModelConfig",
    "get_feature_selector_from_name",
]
