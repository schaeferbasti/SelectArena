"""SelectArena's benchmark layer on top of TabArena's experiment runner."""

from __future__ import annotations

from selectarena.benchmark.experiment import (
    SELECTARENA_VALIDATION_PROTOCOL,
    SelectArenaBagExperiment,
    SelectArenaExperimentBundle,
    is_fs_pipeline,
    to_selectarena_experiments,
)
from selectarena.benchmark.fs_config import (
    FSBenchConfig,
    apply_fs_bench_preprocessing,
    get_fs_benchmark_preprocessing_pipelines,
    get_num_selected_features,
    selector_and_config_from_string,
    with_encoding_permutation_variants,
)
from selectarena.benchmark.runner import SelectArenaExperimentRunner
from selectarena.benchmark.wrapper import SelectArenaBagWrapper

__all__ = [
    "SELECTARENA_VALIDATION_PROTOCOL",
    "FSBenchConfig",
    "SelectArenaBagExperiment",
    "SelectArenaBagWrapper",
    "SelectArenaExperimentBundle",
    "SelectArenaExperimentRunner",
    "apply_fs_bench_preprocessing",
    "get_fs_benchmark_preprocessing_pipelines",
    "get_num_selected_features",
    "is_fs_pipeline",
    "selector_and_config_from_string",
    "to_selectarena_experiments",
    "with_encoding_permutation_variants",
]
