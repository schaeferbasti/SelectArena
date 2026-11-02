"""Convert only the comparison's datasets into TabArena tasks. Run it with SelectArena's environment.

    python prepare_datasets.py --config benchmark.yaml --cluster-config <workspace>/cluster.yaml

Same conversion as SelectArena's ``scripts/prepare_datasets.py``, restricted to the datasets in
``task_subset.dataset_names``: the containers are downloaded from Hugging Face
(``selectarena.datasets.HF_REPO_ID`` at ``HF_REVISION``), or read from ``data_foundry_warehouse``
when the cluster config sets it. It writes into the comparison's own OpenML cache, so a later full
SelectArena run never picks up this two-dataset task list.
"""

from __future__ import annotations

import argparse
import pandas as pd
from selectarena.config import (
    build_cache_config,
    load_benchmark_config,
    load_cluster_config,
)
from selectarena.datasets import (
    BENCHMARK_DATA_FOUNDRY_TASKS,
    COLLECTION_DESCRIPTION,
    COLLECTION_NAME,
    make_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cluster-config", required=True)
    args = parser.parse_args()

    from data_foundry.collections import DatasetCollection
    from tabarena.benchmark.task.data_foundry.metadata_cache import (
        get_path_to_metadata_cache,
        prepare_collection_for_tabarena,
    )

    benchmark = load_benchmark_config(args.config)
    cluster = load_cluster_config(args.cluster_config)
    datasets = set(benchmark.task_subset["dataset_names"])
    paths = [p for p in BENCHMARK_DATA_FOUNDRY_TASKS if p.split("/")[0] in datasets]
    if len(paths) != len(datasets):
        raise SystemExit(
            f"Not in the SelectArena suite: {sorted(datasets - {p.split('/')[0] for p in paths})}"
        )

    cache_config = build_cache_config(cluster)
    cache_config.apply()
    csv = get_path_to_metadata_cache() / f"{COLLECTION_NAME}_tasks_metadata.csv"
    if csv.exists():
        listed = set(pd.read_csv(csv)["dataset_name"])
        if listed == datasets:
            print(f"Tasks already prepared: {csv}")
            return
        if listed - datasets:
            raise SystemExit(
                f"{csv} lists other datasets too; use an OpenML cache of its own for the comparison."
            )

    collection = DatasetCollection.from_relative_paths(
        name=COLLECTION_NAME,
        description=COLLECTION_DESCRIPTION,
        relative_paths=paths,
        source=make_collection(cluster.data_foundry_warehouse).source,
    )
    path = prepare_collection_for_tabarena(
        collection=collection,
        benchmark_suite_name=COLLECTION_NAME,
        openml_cache=cluster.caches["openml"],
    )
    print(f"Wrote task metadata to {path}")


if __name__ == "__main__":
    main()
