"""Run the SelectArena benchmark on SLURM or OAR: ``setup`` the jobs, then ``eval`` the results.

    python scripts/run_benchmark.py setup --config configs/benchmark.yaml
    python scripts/run_benchmark.py eval  --config configs/benchmark.yaml

``setup`` runs on the head node: it builds every (model config x FSBench pipeline x split) job,
skips the ones with cached results, writes the job batch and prints the ``sbatch`` command(s), or
the ``oarsub`` command(s) with ``scheduler: oar`` in the cluster config (``--submit`` submits those).
``eval`` collects all ``results.pkl`` of the run into ``<output>/<benchmark_name>/results.csv``.
The cluster config is read from ``--cluster-config``, ``$SELECTARENA_CLUSTER_CONFIG`` or
``configs/cluster.yaml``.
"""

from __future__ import annotations

import argparse
import logging

from selectarena.config import build_plan, load_benchmark_config, load_cluster_config


def setup(args: argparse.Namespace) -> None:
    benchmark = load_benchmark_config(args.config)
    cluster = load_cluster_config(args.cluster_config)
    print(f"{len(benchmark.feature_selection.pipelines())} preprocessing pipelines")
    commands = build_plan(benchmark, cluster).setup_jobs()
    if args.submit:
        if cluster.scheduler != "oar":
            raise SystemExit("--submit is only supported with `scheduler: oar`; run the printed sbatch commands.")
        from selectarena.oar import submit

        for line in submit(commands):
            print(line)


def evaluate(args: argparse.Namespace) -> None:
    from selectarena.evaluation import read_benchmark_results

    benchmark = load_benchmark_config(args.config)
    cluster = load_cluster_config(args.cluster_config)
    run_dir = cluster.output_dir / benchmark.benchmark_name
    table = read_benchmark_results(
        run_dir / "data", only_default=args.only_default, num_cpus=args.num_cpus, use_ray=not args.no_ray
    )
    out_path = args.output or run_dir / "results.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved {len(table)} results to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["setup", "eval"])
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Benchmark config (YAML).")
    parser.add_argument("--cluster-config", default=None, help="Cluster config (YAML).")
    parser.add_argument("--submit", action="store_true", help="setup: submit the OAR array jobs with oarsub.")
    parser.add_argument("--only-default", action="store_true", help="eval: only default configs (_c1_).")
    parser.add_argument("--num-cpus", type=int, default=8, help="eval: Ray workers for parsing results.")
    parser.add_argument("--no-ray", action="store_true", help="eval: parse results sequentially.")
    parser.add_argument("--output", default=None, help="eval: CSV path (default: <run dir>/results.csv).")
    args = parser.parse_args()
    {"setup": setup, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
