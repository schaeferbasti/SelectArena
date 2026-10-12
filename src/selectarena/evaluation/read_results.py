"""Collect SelectArena ``results.pkl`` files into one table, including the feature selection metadata.

Results land at ``<workspace>/output/<benchmark_name>/data/<method>/<task>/<repeat>_<fold>/results.pkl``.
Every row of the table is one result: the task and split, the downstream metric, train/infer
time, and (for ``FSBench__*`` runs) what the feature selector did.
"""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Module paths of the tabarena_fork_fs era, so results written by the fork can still be read.
_LEGACY_MODULES = {
    "tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils": "selectarena.benchmark.fs_config",
}


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        return super().find_class(_LEGACY_MODULES.get(module, module), name)


def load_result(path: str | Path) -> dict:
    """Load a (possibly gzipped) ``results.pkl``, also when it was written by tabarena_fork_fs."""
    from tabarena.utils.pickle_utils import read_pickle_bytes

    return _CompatUnpickler(io.BytesIO(read_pickle_bytes(path))).load()


def parse_result(path: str | Path) -> dict:
    """Return the table row for one ``results.pkl``."""
    from tabarena.benchmark.result.baseline_result import BaselineResult

    result = BaselineResult.from_dict(result=load_result(path))
    fs = (result.result.get("preprocessing_artifacts") or {}).get("feature_selection")
    budget = getattr(fs["max_features_input"], "keywords", {}) if fs is not None else {}
    return {
        "experiment_method_name_string": result.framework,
        "model_details": result.result.get("hyperparameters"),
        **result.task_metadata,
        "metric_error": result.result["metric_error"],
        "metric_error_val": result.result.get("metric_error_val"),
        "metric": result.result["metric"],
        "problem_type": result.problem_type,
        "time_train_s": result.result["time_train_s"],
        "time_infer_s": result.result["time_infer_s"],
        "feature_selection_method": fs["method_type"] if fs else None,
        "feature_selection_is_scoring_method": fs["feature_scoring_method"] if fs else None,
        "selected_features": fs.get("selected_features") if fs else None,
        "random_feature_names": fs.get("random_feature_names") if fs else None,
        "max_features": fs["max_features"] if fs else None,
        "feature_selection_budget_total": budget.get("b"),
        "feature_selection_budget_index": budget.get("idx"),
        "feature_selection_fit_time": fs["feature_selection_fit_time"] if fs else None,
        "feature_selection_time_limit": fs["feature_selection_time_limit"] if fs else None,
        "path": str(path),
    }


def _parse_batch(file_paths: list[str]) -> list[dict]:
    return [parse_result(path) for path in file_paths]


def read_benchmark_results(
    data_dir: str | Path,
    *,
    only_default: bool = False,
    num_cpus: int = 8,
    batch_size: int = 1000,
    use_ray: bool = True,
) -> pd.DataFrame:
    """Find every ``results.pkl`` under ``data_dir`` and return one row per result.

    Parameters
    ----------
    data_dir
        A run's ``data/`` directory (``<workspace>/output/<benchmark_name>/data``).
    only_default
        Keep only default-config results (``_c1_`` in the path), dropping random configs.
    num_cpus, batch_size
        Ray workers and pickle files per worker call.
    use_ray
        Parse in parallel with Ray; ``False`` parses sequentially in this process.
    """
    from tabarena.utils.pickle_utils import fetch_all_pickles

    file_paths = [str(p) for p in fetch_all_pickles(dir_path=Path(data_dir), suffix="results.pkl")]
    logger.info("Found %d results.pkl files under %s", len(file_paths), data_dir)
    if only_default:
        file_paths = [p for p in file_paths if "_c1_" in p]
    if not file_paths:
        return pd.DataFrame()

    batches = [file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)]
    if not use_ray:
        nested = [_parse_batch(batch) for batch in batches]
    else:
        import ray
        from tabarena.utils.ray_utils import ray_map_list

        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, include_dashboard=False)
        nested = ray_map_list(
            batches,
            func=_parse_batch,
            func_element_key_string="file_paths",
            num_workers=num_cpus,
            num_cpus_per_worker=1,
            track_progress=True,
            tqdm_kwargs={"desc": "Parsing results"},
            ray_remote_kwargs={"max_calls": 1},
        )
    return pd.DataFrame([row for batch in nested for row in batch])
