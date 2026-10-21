"""OAR building blocks shared by the main benchmark and the stability/validity benchmark.

Both submit *parametric array jobs*: one ``oarsub -S <script> --array-param-file <params>`` per
batch, where every non-empty line of ``<params>`` becomes the arguments of one subjob. The script
starts with ``#OAR`` directives (resources, walltime, queue, logs), loads the cluster environment
and runs one Python entry point with ``"$@"``.
"""

from __future__ import annotations

import math
import shlex
import subprocess
from pathlib import Path

#: Sourced before ``module load`` in batch jobs (calculco: ``/nfs/opt/env/env.sh``).
DEFAULT_ENV_SCRIPT = "/nfs/opt/env/env.sh"


def format_walltime(seconds: float) -> str:
    """OAR walltime ``H:MM:SS``, rounded up to the next minute."""
    minutes = math.ceil(seconds / 60)
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def write_oar_script(
    path: str | Path,
    *,
    command: str,
    job_name: str,
    resources: str,
    walltime: str,
    log_dir: str | Path,
    queue: str | None = None,
    properties: str | None = None,
    besteffort: bool = False,
    env_script: str | None = DEFAULT_ENV_SCRIPT,
    modules: list[str] | tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> Path:
    """Write an executable OAR batch script that runs ``command "$@"`` and return its path.

    ``resources`` is the resource part of ``-l`` (e.g. ``/nodes=1/core=8`` or ``/gpu=1``); the
    walltime is appended. ``properties`` is an SQL ``WHERE`` clause for ``-p``
    (e.g. ``gpumodel IN ('A100','H100')``).
    """
    path = Path(path)
    log_dir = Path(log_dir).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    directives = [
        f"#OAR -n {job_name}",
        f"#OAR -l {resources},walltime={walltime}",
        f"#OAR -O {log_dir}/OAR.%jobid%.out",
        f"#OAR -E {log_dir}/OAR.%jobid%.err",
    ]
    if queue:
        directives.append(f"#OAR -q {queue}")
    if besteffort:
        directives.append("#OAR -t besteffort")
    if properties:
        directives.append(f'#OAR -p "{properties}"')
    setup = []
    if env_script:
        setup.append(f"[ -f {shlex.quote(env_script)} ] && . {shlex.quote(env_script)}")
    if modules:
        setup.append("module load " + " ".join(shlex.quote(m) for m in modules))
    setup += [f"export {key}={shlex.quote(value)}" for key, value in (env or {}).items()]
    lines = [
        "#!/bin/bash",
        *directives,
        "",
        "set -eo pipefail",
        *setup,
        "set -u",
        'echo "OAR job ${OAR_JOB_ID:-?} (array ${OAR_ARRAY_ID:-?}, index ${OAR_ARRAY_INDEX:-?}) on $(hostname)"',
        'echo "Arguments: $*"',
        f'{command} "$@"',
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    return path


def oarsub_command(script: str | Path, param_file: str | Path) -> str:
    """The ``oarsub`` command submitting ``script`` as a parametric array job over ``param_file``."""
    return f"oarsub -S {shlex.quote(str(script))} --array-param-file {shlex.quote(str(param_file))}"


def submit(commands: list[str]) -> list[str]:
    """Run the ``oarsub`` commands and return their output."""
    outputs = []
    for command in commands:
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, check=True)
        outputs.append(result.stdout.strip())
    return outputs
