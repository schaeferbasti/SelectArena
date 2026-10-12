"""TabArena experiment runner that also stores the feature selection metadata of each fit."""

from __future__ import annotations

from tabarena.benchmark.experiment import OOFExperimentRunner


class SelectArenaExperimentRunner(OOFExperimentRunner):
    """``OOFExperimentRunner`` that adds ``results["preprocessing_artifacts"]``.

    The entry holds whatever the model's ``get_preprocessing_results()`` returns (for
    :class:`~selectarena.benchmark.wrapper.SelectArenaBagWrapper`, the selector metadata under
    ``"feature_selection"``). Models without that method are left unchanged.
    """

    def __init__(self, *, compute_preprocessing_artifacts: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.compute_preprocessing_artifacts = compute_preprocessing_artifacts

    def post_evaluate(self, out: dict) -> dict:
        out = super().post_evaluate(out=out)
        if self.compute_preprocessing_artifacts and getattr(self.model, "can_get_preprocessing", False):
            out["preprocessing_artifacts"] = self.model.get_preprocessing_results()
        return out
