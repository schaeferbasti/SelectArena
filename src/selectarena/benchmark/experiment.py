"""SelectArena experiments: TabArena's bagged single-model experiment plus a feature selector.

A feature selection run is identified by its preprocessing pipeline name,
``FSBench__<method>__<total_budget>__<budget_index>__<proxy_model>__<fs_time>`` (see
:class:`~selectarena.benchmark.fs_config.FSBenchConfig`). TabArena only knows its own named
pipelines, so SelectArena layers the feature selection on top through subclasses rather than by
editing TabArena:

* :class:`SelectArenaBagExperiment` resolves ``FSBench__*`` pipelines itself and runs the fit
  with :class:`~selectarena.benchmark.wrapper.SelectArenaBagWrapper` and
  :class:`~selectarena.benchmark.runner.SelectArenaExperimentRunner`.
* :class:`SelectArenaExperimentBundle` is TabArena-v0.1's experiment bundle whose bagged
  experiments are built as :class:`SelectArenaBagExperiment`.

Both classes are importable by path, so they survive the YAML round trip TabArena uses to ship
experiments to cluster nodes, provided ``selectarena`` is installed in the node's environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabarena.benchmark.experiment import (
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
    AGModelBagExperiment,
    Experiment,
    TabArenaV0pt1ExperimentBundle,
)

from selectarena.benchmark.fs_config import FSBenchConfig, apply_fs_bench_preprocessing
from selectarena.benchmark.runner import SelectArenaExperimentRunner
from selectarena.benchmark.wrapper import SelectArenaBagWrapper

#: The inner validation protocol SelectArena runs under (TabArena-v0.1: 8 bagging folds x 1 set).
SELECTARENA_VALIDATION_PROTOCOL = TABARENA_V0PT1_VALIDATION_PROTOCOL


def is_fs_pipeline(name: str | None) -> bool:
    """Whether ``name`` is a SelectArena ``FSBench__*`` preprocessing pipeline."""
    return name is not None and FSBenchConfig.is_fs_bench_string(name)


class SelectArenaBagExperiment(AGModelBagExperiment):
    """``AGModelBagExperiment`` that understands ``FSBench__*`` preprocessing pipelines."""

    _method_cls = SelectArenaBagWrapper
    _experiment_cls = SelectArenaExperimentRunner

    def _apply_preprocessing(self, method_kwargs: dict) -> dict:
        if is_fs_pipeline(self.preprocessing_pipeline):
            return apply_fs_bench_preprocessing(
                preprocessing_name=self.preprocessing_pipeline, method_kwargs=method_kwargs
            )
        return super()._apply_preprocessing(method_kwargs)

    @classmethod
    def from_experiment(cls, experiment: AGModelBagExperiment) -> SelectArenaBagExperiment:
        """Rebuild a TabArena ``AGModelBagExperiment`` as a SelectArena experiment (same arguments)."""
        return cls(**experiment._locals)


def to_selectarena_experiments(experiments: list[Experiment]) -> list[Experiment]:
    """Convert every ``AGModelBagExperiment`` in ``experiments``; other experiment types pass through."""
    return [
        SelectArenaBagExperiment.from_experiment(exp)
        if isinstance(exp, AGModelBagExperiment) and not isinstance(exp, SelectArenaBagExperiment)
        else exp
        for exp in experiments
    ]


@dataclass(kw_only=True)
class SelectArenaExperimentBundle(TabArenaV0pt1ExperimentBundle):
    """TabArena-v0.1's experiment bundle with SelectArena's defaults and experiment class.

    Inherits the TabArena-v0.1 settings the feature selection benchmark ran with (AutoGluon's
    default preprocessing, no feature shuffling, no text-embedding cache) and lowers the number
    of random configurations to 25.

    ``preprocessing_pipelines`` takes the ``FSBench__*`` names from
    :func:`~selectarena.benchmark.fs_config.get_fs_benchmark_preprocessing_pipelines`
    (``"default"`` runs the model without feature selection). Each pipeline multiplies the
    number of experiments, and its name is appended to the config name, e.g.
    ``LightGBM_c1_FSBench__MIFeatureSelector__5__0__lgbm__3600_BAG_L1``.
    """

    n_random_configs: int = 25
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["default"])

    def build_experiments(self, **kwargs) -> list[Experiment]:
        return to_selectarena_experiments(super().build_experiments(**kwargs))
