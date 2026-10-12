"""Convert the SelectArena Data Foundry containers into TabArena tasks (run once per cluster).

    python scripts/prepare_datasets.py [--cluster-config configs/cluster.yaml]

Reads the containers from ``data_foundry_warehouse`` (or downloads them from the Hugging Face dataset
repo when it is not set), writes the ``UserTask`` pickles and the
suite's metadata CSV into the configured OpenML cache. Needs ``tabarena[data-foundry]``.
"""

from __future__ import annotations

import argparse

from selectarena.config import build_cache_config, load_cluster_config
from selectarena.datasets import prepare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster-config", default=None)
    args = parser.parse_args()

    cluster = load_cluster_config(args.cluster_config)
    cache_config = build_cache_config(cluster)
    if cache_config is not None:
        cache_config.apply()
    path = prepare_datasets(warehouse_dir=cluster.data_foundry_warehouse, openml_cache=cluster.caches.get("openml"))
    print(f"Wrote task metadata to {path}")


if __name__ == "__main__":
    main()
