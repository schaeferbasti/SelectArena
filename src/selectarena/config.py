"""YAML configuration for SelectArena runs.

Two files keep what is benchmarked apart from where it runs:

* a **benchmark config** (``configs/benchmark.yaml``, committed): feature selection methods,
  budgets, downstream models, splits;
* a **cluster config** (``configs/cluster.yaml``, git-ignored; start from
  ``configs/cluster.example.yaml`` for SLURM or ``configs/cluster.calculco.example.yaml`` for OAR):
  workspace, Python environment, cache locations, scheduler settings and resources.

``load_config`` reads both. The ``build_*`` functions turn them into the TabArena / tabflow_slurm
objects the scripts use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from tabarena.caching import CacheConfig
    from tabarena.contexts import AbstractArenaContext
    from tabflow_slurm import TabArenaBenchmarkPlan

#: Environment variable naming the cluster config when ``--cluster-config`` is not given.
CLUSTER_CONFIG_ENV = "SELECTARENA_CLUSTER_CONFIG"
DEFAULT_CLUSTER_CONFIG = Path("configs/cluster.yaml")


@dataclass
class FeatureSelectionConfig:
    """Which ``FSBench__*`` pipelines to run (see ``get_fs_benchmark_preprocessing_pipelines``)."""

    methods: list[str] | str = "all"
    """Registry names of the methods, or ``"all"``."""
    total_budget: int = 5
    """Number of feature-count budgets per dataset (``b`` in the budget formula)."""
    proxy_models: list[str] = field(default_factory=lambda: ["lgbm"])
    fs_time_limits: list[float] = field(default_factory=lambda: [3600])
    """Feature selection time limits: seconds when >= 1, a fraction of the fit time limit when < 1."""
    include_default: bool = False
    """Also run each model without feature selection (pipeline ``"default"``)."""
    no_fallback: bool = False
    """Disable the random fallback that fills the budget (``__nofallback`` ablation)."""
    encoding_permutation_seeds: list[int] = field(default_factory=list)
    """Ordinal-encoding permutation ablation seeds (``__permenc<seed>``); empty for none."""

    def method_names(self) -> list[str]:
        from selectarena.methods import FEATURE_SELECTION_METHODS, get_feature_selector_from_name

        if self.methods == "all":
            return list(FEATURE_SELECTION_METHODS)
        for name in self.methods:
            get_feature_selector_from_name(name=name)  # validates the name
        return list(self.methods)

    def pipelines(self) -> list[str]:
        from selectarena.benchmark.fs_config import (
            get_fs_benchmark_preprocessing_pipelines,
            with_encoding_permutation_variants,
        )

        pipelines = get_fs_benchmark_preprocessing_pipelines(
            fs_methods=self.method_names(),
            proxy_model_config=self.proxy_models,
            time_limit=self.fs_time_limits,
            total_budget=self.total_budget,
            include_default=False,
        )
        if self.no_fallback:
            pipelines = [f"{p}__nofallback" for p in pipelines]
        if self.encoding_permutation_seeds:
            pipelines = with_encoding_permutation_variants(pipelines, permutation_seeds=self.encoding_permutation_seeds)
        if self.include_default:
            pipelines.append("default")
        return pipelines


@dataclass
class ExtraBenchmarkConfig:
    modes: list[str] = field(default_factory=lambda: ["stability", "validity"])
    noise_type: str = "gaussian"
    results_subdir: str = "extra_benchmark"
    """Results go to ``<workspace>/output/<benchmark_name>/<results_subdir>``."""


@dataclass
class BenchmarkConfig:
    benchmark_name: str
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    time_limit: int = 3600
    """Time limit of one model fit in seconds (feature selection time comes on top)."""
    experiment: dict[str, Any] = field(default_factory=dict)
    """Extra ``SelectArenaExperimentBundle`` fields (e.g. ``n_random_configs``, ``custom_model_constraints``)."""
    task_subset: dict[str, Any] = field(default_factory=dict)
    """``TaskSubset`` fields scoping the run (e.g. ``split_indices``, ``dataset_names``)."""
    model_jobs: list[dict[str, Any]] = field(default_factory=list)
    """``tabflow_slurm.ModelJob`` fields; ``models`` is a list of ``[name, n_configs]`` pairs."""
    extra_benchmark: ExtraBenchmarkConfig = field(default_factory=ExtraBenchmarkConfig)

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkConfig:
        data = dict(data)
        data["feature_selection"] = FeatureSelectionConfig(**data.get("feature_selection", {}))
        data["extra_benchmark"] = ExtraBenchmarkConfig(**data.get("extra_benchmark", {}))
        return cls(**data)


@dataclass
class ClusterConfig:
    workspace: str
    """Persistent directory every job can reach; holds ``output/``, ``slurm_out/``, ``setup_out/``."""
    python_path: str
    """Python of the environment SelectArena is installed in, as seen from the compute nodes."""
    caches: dict[str, str | None] = field(default_factory=dict)
    """``tabarena.caching.CacheConfig`` fields (``openml``, ``huggingface``, ``data_foundry``, ``tabarena``)."""
    data_foundry_warehouse: str | None = None
    """Directory holding the suite's Data Foundry containers (``<dataset>/<uuid>``)."""
    slurm: dict[str, Any] = field(default_factory=dict)
    """``tabflow_slurm.SlurmSetup`` fields (``cpu_partition``, ``gpu_partition``, ``extra_gres``, ...)."""
    resources: dict[str, Any] = field(default_factory=dict)
    """``tabflow_slurm.ResourcesSetup`` fields overriding the TabArena-v0.1 defaults."""
    extra_benchmark_slurm: dict[str, Any] = field(default_factory=dict)
    """``ExtraBenchmarkSlurmSettings`` fields for the stability/validity array jobs."""
    scheduler: str = "slurm"
    """``"slurm"`` (``slurm`` and ``extra_benchmark_slurm`` apply) or ``"oar"`` (``oar`` and ``extra_benchmark_oar``)."""
    oar: dict[str, Any] = field(default_factory=dict)
    """``selectarena.benchmark.oar.OarSetup`` fields (resources, queue, properties, modules, ...)."""
    extra_benchmark_oar: dict[str, Any] = field(default_factory=dict)
    """``ExtraBenchmarkOarSettings`` fields for the stability/validity array jobs."""

    def __post_init__(self) -> None:
        if self.scheduler not in ("slurm", "oar"):
            raise ValueError(f"Unknown scheduler {self.scheduler!r}; options are 'slurm' and 'oar'.")

    @property
    def output_dir(self) -> Path:
        return Path(self.workspace) / "output"


def _read_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return BenchmarkConfig.from_dict(_read_yaml(path))


def load_cluster_config(path: str | Path | None = None) -> ClusterConfig:
    """Read the cluster config from ``path``, ``$SELECTARENA_CLUSTER_CONFIG`` or ``configs/cluster.yaml``."""
    path = path or os.environ.get(CLUSTER_CONFIG_ENV) or DEFAULT_CLUSTER_CONFIG
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Cluster config {path} not found. Copy configs/cluster.example.yaml to configs/cluster.yaml "
            f"and edit it, or pass --cluster-config / set ${CLUSTER_CONFIG_ENV}."
        )
    return ClusterConfig(**_read_yaml(path))


# --- Building TabArena / tabflow_slurm objects ------------------------------------------------


def build_cache_config(cluster: ClusterConfig) -> CacheConfig | None:
    from tabarena.caching import CacheConfig

    caches = {k: v for k, v in cluster.caches.items() if v is not None}
    return CacheConfig(**caches) if caches else None


def build_context(cluster: ClusterConfig) -> AbstractArenaContext:
    """A self-contained arena over the SelectArena suite (no TabArena baselines)."""
    from tabarena.contexts import AbstractArenaContext

    from selectarena.benchmark.experiment import SELECTARENA_VALIDATION_PROTOCOL
    from selectarena.datasets import make_task_metadata_collection

    cache_config = build_cache_config(cluster)
    if cache_config is not None:
        cache_config.apply()  # the suite's metadata CSV lives in the OpenML cache
    collection = make_task_metadata_collection(
        warehouse_dir=cluster.data_foundry_warehouse,
        cache_dir=cluster.caches.get("data_foundry"),
    )
    return AbstractArenaContext(
        task_metadata=collection,
        methods=[],
        validation_protocol=SELECTARENA_VALIDATION_PROTOCOL,
        cache_config=cache_config,
    )


def _preprocessing_time_overhead(benchmark: BenchmarkConfig) -> int | None:
    limits = benchmark.feature_selection.fs_time_limits
    if not limits:
        return None
    return int(max(t if t >= 1 else t * benchmark.time_limit for t in limits))


def build_plan(benchmark: BenchmarkConfig, cluster: ClusterConfig) -> TabArenaBenchmarkPlan:
    """The ``tabflow_slurm`` plan running ``benchmark`` on the SLURM or OAR cluster in ``cluster``."""
    from tabarena.benchmark.experiment import ModelConstraints
    from tabarena.benchmark.task.metadata import TaskSubset
    from tabflow_slurm import ModelJob, PathSetup, SlurmSetup, TabArenaBenchmarkPlan, TabArenaV0pt1ResourcesSetup

    from selectarena.benchmark.experiment import SelectArenaExperimentBundle

    if not benchmark.model_jobs:
        raise ValueError("The benchmark config lists no model_jobs.")
    model_jobs = []
    for job in benchmark.model_jobs:
        job = dict(job)
        job["models"] = [tuple(entry) for entry in job["models"]]
        model_jobs.append(ModelJob(**job))

    if cluster.scheduler == "oar":
        from selectarena.benchmark.oar import OarSetup

        scheduler_setup = OarSetup(**cluster.oar)
    else:
        slurm = {"gpu_partition": None, "cpu_partition": None, "extra_gres": None, **cluster.slurm}
        scheduler_setup = SlurmSetup(**slurm)
    resources = {
        "time_limit": benchmark.time_limit,
        "time_limit_for_model_agnostic_preprocessing": _preprocessing_time_overhead(benchmark),
        **cluster.resources,
    }
    experiment = dict(benchmark.experiment)
    if "custom_model_constraints" in experiment:
        experiment["custom_model_constraints"] = {
            key: ModelConstraints(**value) for key, value in experiment["custom_model_constraints"].items()
        }
    return TabArenaBenchmarkPlan(
        benchmark_name=benchmark.benchmark_name,
        model_jobs=model_jobs,
        context=build_context(cluster),
        experiment_bundle=SelectArenaExperimentBundle(
            preprocessing_pipelines=benchmark.feature_selection.pipelines(),
            **experiment,
        ),
        path_setup=PathSetup(workspace=cluster.workspace, python_path=cluster.python_path),
        scheduler_setup=scheduler_setup,
        resources_setup=TabArenaV0pt1ResourcesSetup(**resources),
        task_subset=TaskSubset(**benchmark.task_subset) if benchmark.task_subset else None,
    )
