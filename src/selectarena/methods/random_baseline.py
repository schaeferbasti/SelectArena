"""Random feature selection."""

from __future__ import annotations

import logging

from selectarena.methods.abstract import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class RandomFeatureSelector(AbstractFeatureSelector):
    """Random Feature Selection."""

    name = "RandomFeatureSelector"

    def _fit_feature_selection(self, **kwargs) -> list[dict[str, float | None]]:
        # Every pick comes from the random fallback -> all values are None.
        return self.fallback_feature_selection()
