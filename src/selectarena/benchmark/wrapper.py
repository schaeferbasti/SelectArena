"""TabArena exec-model wrapper that seeds feature selectors and reports what they selected.

TabArena's ``AGSingleBagWrapper`` fits one bagged AutoGluon model. SelectArena subclasses it
instead of patching TabArena, adding two things:

* the per-split seed (``split_seed``, one per outer fold and repeat) is handed to every
  feature selector in the AutoGluon preprocessing pipeline before the fit, and
* :meth:`get_preprocessing_results` collects the fitted selector's metadata (selected features,
  scores, fit time), which :class:`~selectarena.benchmark.runner.SelectArenaExperimentRunner`
  stores under ``results["preprocessing_artifacts"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.exec_models import AGSingleBagWrapper

from selectarena.methods.abstract import AbstractFeatureSelector

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd


def iter_feature_selectors(generator) -> Iterator[AbstractFeatureSelector]:
    """Yield every :class:`AbstractFeatureSelector` nested inside an AutoGluon feature generator."""
    if isinstance(generator, AbstractFeatureSelector):
        yield generator
        return
    for group in getattr(generator, "generators", None) or []:
        children = group if isinstance(group, list | tuple) else [group]
        for child in children:
            yield from iter_feature_selectors(child)


def feature_selection_metadata(selector: AbstractFeatureSelector) -> dict:
    """The metadata SelectArena records for one fitted feature selector."""
    selected = selector._selected_features
    return {
        "method_type": selector.name,
        "feature_scoring_method": selector.feature_scoring_method,
        "original_feature_names": selector._original_features,
        "selected_feature_names": [entry["feature"] for entry in selected] if selected is not None else None,
        "selected_features": selected,
        "random_feature_names": selector._feature_selection_random_features,
        "feature_scores": selector._feature_scores,
        "max_features": getattr(selector, "max_features", None),
        "max_features_input": selector.max_features_input,
        "feature_selection_fit_time": selector._feature_selection_fit_time,
        "feature_selection_time_limit": selector._feature_selection_time_limit,
    }


class SelectArenaBagWrapper(AGSingleBagWrapper):
    """``AGSingleBagWrapper`` with feature selection seeding and metadata collection."""

    can_get_preprocessing = True

    def _build_predictor_args(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        data_owned: bool = False,
    ) -> tuple[pd.DataFrame, dict, dict]:
        train_data, init_kwargs, fit_kwargs = super()._build_predictor_args(
            X=X, y=y, X_val=X_val, y_val=y_val, data_owned=data_owned
        )
        if self._split_seed not in ("NOTSET", None):
            post_generators = fit_kwargs.get("_feature_generator_kwargs", {}).get("post_generators", [])
            for generator in post_generators:
                if isinstance(generator, AbstractFeatureSelector):
                    generator._outer_random_state = self._split_seed
        return train_data, init_kwargs, fit_kwargs

    def get_preprocessing_results(self) -> dict:
        """Return ``{"feature_selection": {...}}`` for the fitted selector, or ``{}`` without one.

        Raises:
            NotImplementedError: If the pipeline contains more than one feature selector.
        """
        selectors = list(iter_feature_selectors(self.predictor._learner.feature_generator))
        if not selectors:
            return {}
        if len(selectors) > 1:
            raise NotImplementedError("Getting metadata for multiple feature selectors is not supported!")
        return {"feature_selection": feature_selection_metadata(selectors[0])}
