"""Run the TabArena job batch on an OAR cluster (e.g. calculco) instead of SLURM.

``OarSetup`` is a ``tabflow_slurm`` scheduler: ``TabArenaBenchmarkPlan`` builds and bundles the
jobs exactly as for SLURM, and ``OarSetup.get_run_commands`` writes, per bundle size,

* the job JSON TabArena's SLURM path writes (``{"defaults": ..., "jobs": [{"items": [...]}]}``),
* ``<json>.params``: one ``<json> <array index>`` line per array task,
* ``<json>.oar``: the batch script (resources, walltime, environment),

and returns the ``oarsub -S <script> --array-param-file <params>`` commands. TabArena's
``submit_template.sh`` needs SLURM variables and ``jq``, so each subjob runs
``python -m selectarena.benchmark.oar <json> <index>`` instead: it runs the bundle's items one
``run_tabarena_experiment.py`` process at a time inside a node-local scratch directory, with the
same arguments, environment hygiene and failure handling as the template.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.utils.ray_utils import to_batch_list
from tabflow_slurm.setup.scheduler import SchedulerSetup

from selectarena.oar import DEFAULT_ENV_SCRIPT, format_walltime, oarsub_command, write_oar_script

if TYPE_CHECKING:
    from tabflow_slurm.setup.paths import PathSetup
    from tabflow_slurm.setup.resources import ResourcesSetup

#: Thread-pool variables unset before each item: the models size their pools themselves.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
)
JOB_ENV = {"TABPFN_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
#: A bundle stops after this many failed items in a row (a broken GPU should not burn every item).
MAX_CONSECUTIVE_FAILURES = 3


@dataclass(kw_only=True)
class OarSetup(SchedulerSetup):
    """``tabflow_slurm`` scheduler emitting OAR parametric array jobs. Set it in ``cluster.yaml`` under ``oar:``."""

    cpu_resources: str = "/nodes=1/core={num_cpus}"
    """``-l`` resources of a CPU job; ``{num_cpus}`` is filled from the resources setup."""
    gpu_resources: str = "/nodes=1/gpu={num_gpus}"
    """``-l`` resources of a GPU job; ``{num_gpus}`` and ``{num_cpus}`` are filled in."""
    queue: str | None = None
    """``-q`` queue; ``None`` uses the cluster's default queue."""
    besteffort: bool = False
    """Submit as ``besteffort`` (preemptible). A killed item is rerun by the next ``setup``."""
    cpu_properties: str | None = None
    """``-p`` SQL clause for CPU jobs, e.g. ``"cputype='Milan'"``."""
    gpu_properties: str | None = None
    """``-p`` SQL clause for GPU jobs, e.g. ``"gpumodel IN ('A100','H100')"``."""
    env_script: str | None = DEFAULT_ENV_SCRIPT
    """Sourced before ``module load`` in the batch script."""
    modules: list[str] = field(default_factory=list)
    """Modules loaded in the batch script (e.g. ``cuda/cuda-12.1``)."""
    time_limit_overhead: int = 1
    """Hours added to the walltime of every subjob (startup, warm-up, evaluation)."""
    max_array_size: int = 10_000
    """Maximum number of subjobs per ``oarsub``; larger groups are split."""
    scratch_root: str | None = None
    """Node-local directory for the per-subjob scratch; ``None`` uses ``$TMPDIR`` or ``/tmp``."""

    def get_extra_default_args(self) -> dict:
        # OAR nodes are shared like SLURM nodes: Ray needs its own temp dir and explicit resources.
        return {"setup_ray_for_slurm_shared_resources_environment": True}

    def describe_target(self, resources: ResourcesSetup) -> str:
        properties = self._properties(resources)
        return f"OAR -l {self._resources(resources)}" + (f" -p {properties}" if properties else "")

    def _resources(self, resources: ResourcesSetup) -> str:
        template = self.gpu_resources if resources.num_gpus > 0 else self.cpu_resources
        if resources.num_cpus is None and "{num_cpus}" in template:
            return "/nodes=1"  # num_cpus=None means all CPUs of the node
        return template.format(num_cpus=resources.num_cpus, num_gpus=resources.num_gpus)

    def _properties(self, resources: ResourcesSetup) -> str | None:
        return self.gpu_properties if resources.num_gpus > 0 else self.cpu_properties

    def get_run_commands(
        self,
        *,
        jobs_dict: dict,
        path_setup: PathSetup,
        benchmark_name: str,
        parallel_safe_benchmark_name: str,
        resources_setup: ResourcesSetup,
        print_summary: bool = True,
    ) -> list[str] | None:
        base_json_path = path_setup.get_slurm_job_json_path(
            benchmark_name=benchmark_name, safe_benchmark_name=parallel_safe_benchmark_name
        ).replace("slurm_run_data_", "oar_run_data_")
        all_jobs = jobs_dict["jobs"]
        if not all_jobs:
            if print_summary:
                print("No jobs to run.")
            Path(base_json_path).unlink(missing_ok=True)
            return None

        log_dir = Path(path_setup.workspace) / "oar_out" / benchmark_name
        # As for SLURM: one array per bundle size, so each gets a walltime sized for its bundles.
        jobs_by_size: dict[int, list] = {}
        for job in all_jobs:
            jobs_by_size.setdefault(job["bundle_size"], []).append(job)

        commands = []
        for size in sorted(jobs_by_size):
            batches = [list(b) for b in to_batch_list(jobs_by_size[size], self.max_array_size)]
            for batch_idx, batch in enumerate(batches):
                suffix = (f"_size{size}" if len(jobs_by_size) > 1 else "") + (
                    f"_batch{batch_idx}" if len(batches) > 1 else ""
                )
                json_path = Path(base_json_path.replace(".json", f"{suffix}.json"))
                shipped = {
                    "defaults": {**jobs_dict["defaults"], "log_dir": str(log_dir), "scratch_root": self.scratch_root},
                    "jobs": [{"items": job["items"]} for job in batch],
                }
                json_path.write_text(json.dumps(shipped))
                param_file = json_path.with_suffix(".params")
                param_file.write_text("".join(f"{json_path} {i}\n" for i in range(len(batch))))
                script = write_oar_script(
                    json_path.with_suffix(".oar"),
                    command=f"{shlex.quote(path_setup.python_path)} -P -m selectarena.benchmark.oar",
                    job_name=f"{parallel_safe_benchmark_name}{suffix}",
                    resources=self._resources(resources_setup),
                    walltime=format_walltime(
                        resources_setup.time_limit_per_config * size + self.time_limit_overhead * 3600
                    ),
                    log_dir=log_dir,
                    queue=self.queue,
                    properties=self._properties(resources_setup),
                    besteffort=self.besteffort,
                    env_script=self.env_script,
                    modules=self.modules,
                    env=JOB_ENV,
                )
                commands.append(oarsub_command(script, param_file))

        if print_summary:
            print("##### Setup Jobs\nRun the following command(s) to start the jobs:\n" + "\n".join(commands) + "\n")
        return commands


# --- One array task on the node ----------------------------------------------------------------


def _item_command(defaults: dict, item: dict, ray_temp_root: str) -> list[str]:
    """The ``run_tabarena_experiment.py`` call of one item (the arguments ``submit_template.sh`` passes)."""
    return [
        defaults["python"],
        "-P",  # keeps the script directory and cwd off sys.path
        defaults["run_script"],
        "--experiment", item["experiment"],
        "--dataset", item["dataset"],
        "--fold", str(item["fold"]),
        "--repeat", str(item["repeat"]),
        "--job_batch_dir", defaults["job_batch_dir"],
        "--output_dir", defaults["output_dir"],
        "--num_cpus", str(defaults["num_cpus"]),
        "--num_gpus", str(defaults["num_gpus"]),
        "--memory_limit", str(defaults["memory_limit"]),
        "--setup_ray_for_slurm_shared_resources_environment",
        str(defaults["setup_ray_for_slurm_shared_resources_environment"]),
        "--ignore_cache", str(defaults["ignore_cache"]),
        "--ray_temp_root", ray_temp_root,
        "--offline_weights", str(defaults.get("offline_weights", False)),
        "--require_warmup", str(defaults.get("require_warmup", True)),
    ]  # fmt: skip


def run_array_task(json_path: str | Path, index: int) -> int:
    """Run every item of ``jobs[index]`` of the job JSON; return the number of failed items."""
    spec = json.loads(Path(json_path).read_text())
    defaults = spec["defaults"]
    items = spec["jobs"][index]["items"]

    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV_VARS}
    env.update(JOB_ENV)
    job_id = os.environ.get("OAR_JOB_ID", str(os.getpid()))
    if defaults.get("log_dir"):
        # Ray worker logs of a failed item land next to the OAR logs.
        env["TABARENA_RAY_LOG_DIR"] = str(Path(defaults["log_dir"]) / "ray_logs" / job_id)

    # Short scratch path: Ray's AF_UNIX socket paths must stay under 107 bytes.
    scratch_root = defaults.get("scratch_root") or os.environ.get("TMPDIR") or "/tmp"
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"sa_{job_id}_", dir=scratch_root))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # walltime / oardel: still clean up
    n_failed = consecutive = 0
    try:
        for sub in ("tmp", "ag", "ray"):
            (scratch / sub).mkdir()
        env["TMPDIR"] = str(scratch / "tmp")
        # AGWrapper roots the predictor artifacts here instead of the shared filesystem.
        env["TABARENA_MODEL_ARTIFACTS_BASE_PATH"] = str(scratch / "ag")
        print(f"Bundle {index} of {json_path}: {len(items)} item(s), scratch {scratch}", flush=True)
        for position, item in enumerate(items):
            desc = " ".join(f"{k}={item[k]}" for k in ("experiment", "dataset", "fold", "repeat"))
            # Ray forwards the driver's cwd into its workers' sys.path, so run from the scratch dir.
            rc = subprocess.run(_item_command(defaults, item, str(scratch / "ray")), env=env, cwd=scratch).returncode
            if rc == 0:
                consecutive = 0
                print(f"##### item OK: {desc}", flush=True)
                continue
            n_failed += 1
            consecutive += 1
            print(f"##### item FAILED (exit {rc}): {desc}", flush=True)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print(f"##### stopping bundle: {consecutive} consecutive failures", flush=True)
                n_failed += len(items) - position - 1  # not run
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"##### bundle summary: failed={n_failed} total={len(items)}", flush=True)
    return n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one OAR array task of a SelectArena job JSON.")
    parser.add_argument("json_path")
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    sys.exit(1 if run_array_task(args.json_path, args.index) else 0)


if __name__ == "__main__":
    main()
