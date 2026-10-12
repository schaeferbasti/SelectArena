"""Utilities for the feature selection benchmark pipeline."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class FSBenchConfig:
    """Parsed representation of an FSBench preprocessing pipeline config string.

    Config string format::

        FSBench__<fs_method>__<total_budget>__<budget_index>__<proxy_model>__<fs_time>

    ``fs_time`` is seconds when >= 1, a fraction of the total time limit when < 1.
    """

    fs_method: str
    total_budget: int
    budget_index: int
    proxy_model: str
    fs_time: int | float

    _PREFIX: ClassVar[str] = "FSBench"
    _SEP: ClassVar[str] = "__"

    def to_string(self) -> str:
        """Serialise to the config string used in benchmark pipelines."""
        parts = [self._PREFIX, self.fs_method, self.total_budget, self.budget_index, self.proxy_model, self.fs_time]
        return self._SEP.join(str(p) for p in parts)

    @classmethod
    def from_string(cls, s: str) -> FSBenchConfig:
        """Parse a config string into an :class:`FSBenchConfig`."""
        _, fs_method, total_budget, budget_index, proxy_model, raw_time = s.split(cls._SEP)
        fs_time: int | float = float(raw_time) if "." in raw_time else int(raw_time)
        return cls(
            fs_method=fs_method,
            total_budget=int(total_budget),
            budget_index=int(budget_index),
            proxy_model=proxy_model,
            fs_time=fs_time,
        )

    @classmethod
    def is_fs_bench_string(cls, s: str) -> bool:
        """Return True if ``s`` looks like an FSBench config string."""
        return s.startswith(cls._PREFIX + cls._SEP)


def get_num_selected_features(d: int, b: int, thr: int = 100) -> list[int]:
    """Return the list of feature counts to evaluate for a given dataset dimensionality and budget.

    Parameters
    ----------
    d : int
        Dataset feature dimensionality.
    b : int
        Budget — number of different feature counts to evaluate.
    thr : int
        Threshold separating the uniform (d <= thr) from the exponential (d > thr) regime.

    Returns:
    -------
    list[int]
        List of b feature counts in increasing order.
    """
    if d <= thr:
        return [math.ceil(d * (i / (b + 1))) for i in range(1, b + 1)]
    return [math.ceil(math.pow(d, i / (b + 1))) for i in range(1, b + 1)]


def _get_budget_feature_count(d: int, *, b: int, idx: int) -> int:
    """Return the ``idx``-th feature count from the budget schedule for a dataset with ``d`` features."""
    return get_num_selected_features(d=d, b=b)[idx]


def get_fs_benchmark_preprocessing_pipelines(
    fs_methods: list[str],
    proxy_model_config: list[str],
    time_limit: list[int],
    total_budget: int,
    *,
    include_default: bool = True,
) -> list[str]:
    """Build the list of preprocessing pipeline config strings for the feature selection benchmark.

    Parameters
    ----------
    fs_methods : list[str]
        Feature selector method names.
    proxy_model_config : list[str]
        Proxy model identifiers (e.g. ``["lgbm"]``).
    time_limit : list[int]
        Time limits in seconds to include.
    total_budget : int
        Total number of budget indices (b in the feature-count formula).
    include_default : bool
        If True, append ``"default"`` as the last pipeline entry.

    Returns:
    -------
    list[str]
        Ordered list of pipeline config strings.
    """
    pipelines = [
        FSBenchConfig(
            fs_method=fs_method,
            total_budget=total_budget,
            budget_index=budget_index,
            proxy_model=proxy_model,
            fs_time=fs_time,
        ).to_string()
        for fs_method, budget_index, proxy_model, fs_time in product(
            fs_methods, range(total_budget), proxy_model_config, time_limit
        )
    ]
    if include_default:
        pipelines.append("default")
    return pipelines


def with_encoding_permutation_variants(
    pipelines: list[str],
    *,
    permutation_seeds: list[int],
    include_baseline: bool = True,
) -> list[str]:
    """Expand FSBench pipelines into ordinal-encoding-permutation ablation variants.

    For every FSBench pipeline string, appends one ``__permenc<seed>`` variant per seed in
    ``permutation_seeds``. The unpermuted baseline (default sorted ordinal encoding) is kept when
    ``include_baseline`` is True. Non-FSBench entries (e.g. ``"default"``) pass through unchanged.

    The ``__permenc<seed>`` token is appended after any existing ablation token (such as
    ``__nofallback``), keeping each variant a distinct run identity / output path.

    Parameters
    ----------
    pipelines : list[str]
        Pipeline config strings, optionally already carrying an ablation token.
    permutation_seeds : list[int]
        Seeds for the category -> integer permutation; each yields one variant per pipeline.
    include_baseline : bool
        Keep the original (unpermuted) pipeline alongside the permuted variants.

    Returns:
    -------
    list[str]
        Expanded list of pipeline config strings.
    """
    out: list[str] = []
    for pipeline in pipelines:
        if not FSBenchConfig.is_fs_bench_string(pipeline):
            out.append(pipeline)
            continue
        if include_baseline:
            out.append(pipeline)
        out.extend(f"{pipeline}__permenc{seed}" for seed in permutation_seeds)
    return out


def selector_and_config_from_string(*, preprocessing_name: str):
    """Parse a preprocessing pipeline config string and return the corresponding feature selector instance and config."""
    from selectarena.methods.abstract import ProxyModelConfig
    from selectarena.methods.registry import (
        FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL,
        get_feature_selector_from_name,
    )

    # Ablation tokens are appended after the standard 6-field FSBench string and stripped before
    # parsing, so FSBenchConfig still sees the standard string while the full name (with tokens)
    # stays the run identity for the cache path and the result "method" field:
    #   "__nofallback"    -> disable the random fallback (skip_random_fallback)
    #   "__permenc<seed>" -> permute the ordinal category->integer mapping with the given seed
    # Both may be present together (e.g. "...__nofallback__permenc2").
    parse_name = preprocessing_name
    skip_random_fallback = "__nofallback" in parse_name
    parse_name = parse_name.replace("__nofallback", "")

    encoding_permutation_seed: int | None = None
    permenc_match = re.search(r"__permenc(\d+)", parse_name)
    if permenc_match:
        encoding_permutation_seed = int(permenc_match.group(1))
        parse_name = parse_name[: permenc_match.start()] + parse_name[permenc_match.end() :]

    config = FSBenchConfig.from_string(parse_name)
    logger.info(
        "Using preprocessing pipeline %s (fs_method=%s, budget index %d of %d, proxy_model=%s, fs_time=%s)",
        config.to_string(),
        config.fs_method,
        config.budget_index,
        config.total_budget,
        config.proxy_model,
        config.fs_time,
    )

    proxy_mode_config = None
    if config.fs_method in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL:
        if config.proxy_model == "lgbm":
            proxy_mode_config = ProxyModelConfig(model_hyperparameters={})
        else:
            raise ValueError(f"Proxy model '{config.proxy_model}' not recognised for pipeline '{preprocessing_name}'.")

    max_features_fn = partial(_get_budget_feature_count, b=config.total_budget, idx=config.budget_index)
    selector_cls = get_feature_selector_from_name(name=config.fs_method)
    return (
        selector_cls(
            max_features=max_features_fn,
            proxy_mode_config=proxy_mode_config,
            skip_random_fallback=skip_random_fallback,
            encoding_permutation_seed=encoding_permutation_seed,
        ),
        config,
    )


def apply_fs_bench_preprocessing(*, preprocessing_name: str, method_kwargs: dict) -> dict:
    """Attach the feature selector described by ``preprocessing_name`` to ``method_kwargs`` (in place).

    The selector (with a deferred ``max_features`` callable) is appended as an AutoGluon post
    generator, after AutoGluon's default unique/duplicate dropping, and the preprocessing time
    limit is set to the configured feature selection time.

    Parameters
    ----------
    preprocessing_name : str
        Config string of the form
        ``FSBench__<fs_method>__<total_budget>__<budget_index>__<proxy_model>__<fs_time>``,
        optionally followed by the ``__nofallback`` / ``__permenc<seed>`` ablation tokens.
    method_kwargs : dict
        The experiment's resolved method kwargs (``Experiment.init_method_kwargs``).

    Returns:
    -------
    dict
        ``method_kwargs``, edited in place.
    """
    from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator
    from autogluon.features.generators.drop_unique import DropUniqueFeatureGenerator

    selector, config = selector_and_config_from_string(preprocessing_name=preprocessing_name)

    fit_kwargs = method_kwargs.setdefault("fit_kwargs", {})
    prep_pipeline = fit_kwargs.get("_feature_generator_kwargs", {})
    prep_pipeline["post_generators"] = [
        *prep_pipeline.get("post_generators", []),
        # AutoGluon's default post generators, which must run before the selector.
        DropUniqueFeatureGenerator(),
        DropDuplicatesFeatureGenerator(post_drop_duplicates=False),
        selector,
    ]
    prep_pipeline["post_drop_duplicates"] = False  # already done by the generator above
    fit_kwargs["_feature_generator_kwargs"] = prep_pipeline
    fit_kwargs["time_limit_preprocessing"] = config.fs_time
    return method_kwargs
