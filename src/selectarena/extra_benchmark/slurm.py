"""Write (and optionally submit) SLURM array jobs for the stability/validity benchmark.

TabArena's ``tabflow_slurm`` schedules model fits; the extra benchmark fits no model, so it uses
a plain SLURM array: one line of ``job_commands.txt`` per ``(mode, pipeline, task)`` job, run by
``python -m selectarena.extra_benchmark.runner``.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from selectarena.extra_benchmark.runner import MODES


@dataclass(frozen=True)
class ExtraBenchmarkSlurmSettings:
    """SLURM resources for one array task (one job runs all its repeats / noise levels)."""

    partition: str
    cpus_per_task: int = 8
    mem: str = "32G"
    time: str = "10:00:00"
    max_array_size: int = 10_000
    parallel: int = 100
    extra_sbatch_lines: tuple[str, ...] = ()


def build_job_commands(
    *,
    modes: list[str],
    method_names: list[str],
    task_ids: list[str],
    results_dir: str | Path,
    openml_cache: str | Path | None = None,
    noise_type: str = "gaussian",
) -> list[str]:
    """One runner argument string per ``(mode, pipeline, task)``."""
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"Unknown modes {sorted(unknown)}; options are {MODES}.")
    commands = []
    for mode, method_name, task_id in product(modes, method_names, task_ids):
        args = [
            "--mode", mode,
            "--method_name", method_name,
            "--task_id", task_id,
            "--results_dir", str(Path(results_dir).absolute()),
            "--noise_type", noise_type,
        ]  # fmt: skip
        if openml_cache is not None:
            args += ["--openml_cache", str(openml_cache)]
        commands.append(shlex.join(args))
    return commands


def write_slurm_arrays(
    commands: list[str],
    *,
    out_dir: str | Path,
    python_path: str,
    settings: ExtraBenchmarkSlurmSettings,
    job_name: str = "selectarena_extra",
) -> list[Path]:
    """Write one sbatch array script per chunk of ``settings.max_array_size`` commands."""
    out_dir = Path(out_dir).absolute()
    scripts = []
    for start in range(0, len(commands), settings.max_array_size):
        chunk = commands[start : start + settings.max_array_size]
        chunk_dir = out_dir / f"submit_{start:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / "job_commands.txt").write_text("\n".join(chunk) + "\n")
        extra = "\n".join(settings.extra_sbatch_lines)
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{start:06d}
#SBATCH --partition={settings.partition}
#SBATCH --cpus-per-task={settings.cpus_per_task}
#SBATCH --mem={settings.mem}
#SBATCH --time={settings.time}
#SBATCH --array=0-{len(chunk) - 1}%{settings.parallel}
#SBATCH --output={chunk_dir}/slurm-%A_%a.out
#SBATCH --error={chunk_dir}/slurm-%A_%a.err
{extra}
set -euo pipefail
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{chunk_dir}/job_commands.txt")
if [ -z "$ARGS" ]; then
  echo "No command for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
eval "set -- $ARGS"
"{python_path}" -m selectarena.extra_benchmark.runner "$@"
"""
        script_path = chunk_dir / "array.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        scripts.append(script_path)
    return scripts


def submit(scripts: list[Path]) -> list[str]:
    """Submit the scripts with ``sbatch`` and return its output lines."""
    outputs = []
    for script in scripts:
        result = subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True)
        outputs.append(result.stdout.strip())
    return outputs
