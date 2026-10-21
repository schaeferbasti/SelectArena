"""Write OAR array jobs for the stability/validity benchmark (the OAR counterpart of ``slurm.py``).

The job commands are the same as for SLURM (``build_job_commands``): one line of runner
arguments per ``(mode, pipeline, task)`` job. On OAR that file is the ``--array-param-file``, so
each subjob receives its line as the arguments of ``python -m selectarena.extra_benchmark.runner``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from selectarena.oar import DEFAULT_ENV_SCRIPT, oarsub_command, write_oar_script


@dataclass(frozen=True)
class ExtraBenchmarkOarSettings:
    """OAR resources for one subjob (one job runs all its repeats / noise levels)."""

    resources: str = "/nodes=1/core=8"
    walltime: str = "10:00:00"
    queue: str | None = None
    properties: str | None = None
    besteffort: bool = False
    env_script: str | None = DEFAULT_ENV_SCRIPT
    modules: list[str] = field(default_factory=list)
    max_array_size: int = 10_000


def write_oar_arrays(
    commands: list[str],
    *,
    out_dir: str | Path,
    python_path: str,
    settings: ExtraBenchmarkOarSettings,
    job_name: str = "selectarena_extra",
) -> list[str]:
    """Write one batch script + param file per chunk of ``settings.max_array_size`` commands.

    Returns the ``oarsub`` commands submitting them.
    """
    out_dir = Path(out_dir).absolute()
    oarsub_commands = []
    for start in range(0, len(commands), settings.max_array_size):
        chunk = commands[start : start + settings.max_array_size]
        chunk_dir = out_dir / f"submit_{start:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        param_file = chunk_dir / "job_commands.txt"
        param_file.write_text("\n".join(chunk) + "\n")
        script = write_oar_script(
            chunk_dir / "array.oar",
            command=f"{shlex.quote(python_path)} -m selectarena.extra_benchmark.runner",
            job_name=f"{job_name}_{start:06d}",
            resources=settings.resources,
            walltime=settings.walltime,
            log_dir=chunk_dir,
            queue=settings.queue,
            properties=settings.properties,
            besteffort=settings.besteffort,
            env_script=settings.env_script,
            modules=settings.modules,
        )
        oarsub_commands.append(oarsub_command(script, param_file))
    return oarsub_commands
