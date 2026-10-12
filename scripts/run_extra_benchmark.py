"""Stability / validity benchmark on SLURM: ``setup`` the array jobs, then ``collect`` the results.

    python scripts/run_extra_benchmark.py setup   --config configs/benchmark.yaml [--submit]
    python scripts/run_extra_benchmark.py collect --config configs/benchmark.yaml

``setup`` writes one array job per chunk of (mode x FSBench pipeline x task) jobs and prints the
``sbatch`` commands (``--submit`` submits them). ``collect`` merges the per-job CSVs into
``stability_results.csv`` and ``validity_results.csv``. A single job can also be run directly with
``python -m selectarena.extra_benchmark.runner``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from selectarena.config import build_context, load_benchmark_config, load_cluster_config


def _results_dir(benchmark, cluster) -> Path:
    return cluster.output_dir / benchmark.benchmark_name / benchmark.extra_benchmark.results_subdir


def setup(args: argparse.Namespace) -> None:
    from selectarena.extra_benchmark.slurm import (
        ExtraBenchmarkSlurmSettings,
        build_job_commands,
        submit,
        write_slurm_arrays,
    )

    benchmark = load_benchmark_config(args.config)
    cluster = load_cluster_config(args.cluster_config)
    collection = build_context(cluster).task_metadata_collection
    task_ids = sorted({task.task_id_str for task in collection.tasks})
    commands = build_job_commands(
        modes=benchmark.extra_benchmark.modes,
        method_names=[p for p in benchmark.feature_selection.pipelines() if p != "default"],
        task_ids=task_ids,
        results_dir=_results_dir(benchmark, cluster),
        openml_cache=cluster.caches.get("openml"),
        noise_type=benchmark.extra_benchmark.noise_type,
    )
    scripts = write_slurm_arrays(
        commands,
        out_dir=Path(cluster.workspace) / "setup_out" / f"{benchmark.benchmark_name}_extra",
        python_path=cluster.python_path,
        settings=ExtraBenchmarkSlurmSettings(**cluster.extra_benchmark_slurm),
    )
    print(f"{len(commands)} jobs in {len(scripts)} array script(s).")
    if args.submit:
        for line in submit(scripts):
            print(line)
    else:
        for script in scripts:
            print(f"sbatch {script}")


def collect(args: argparse.Namespace) -> None:
    from selectarena.extra_benchmark import collect_results

    benchmark = load_benchmark_config(args.config)
    cluster = load_cluster_config(args.cluster_config)
    results_dir = _results_dir(benchmark, cluster)
    for mode, frame in collect_results(results_dir).items():
        out_path = results_dir / f"{mode}_results.csv"
        frame.to_csv(out_path, index=False)
        print(f"Saved {len(frame)} {mode} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["setup", "collect"])
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--cluster-config", default=None)
    parser.add_argument("--submit", action="store_true", help="setup: submit the array jobs with sbatch.")
    args = parser.parse_args()
    {"setup": setup, "collect": collect}[args.mode](args)


if __name__ == "__main__":
    main()
