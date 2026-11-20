"""Run the stability/validity jobs on this machine, one after the other (``scheduler: local``).

The local counterpart of ``slurm.py`` / ``oar.py``: ``write_job_file`` writes the runner arguments
of every ``(mode, pipeline, task)`` job (``build_job_commands``) to ``job_commands.txt``, and

    python -m selectarena.extra_benchmark.local <job_commands.txt>

runs ``python -m selectarena.extra_benchmark.runner`` once per line in a subprocess. A failed job
is reported and skipped; jobs with complete cached results finish immediately, so rerunning the
command resumes the run.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def write_job_file(commands: list[str], *, out_dir: str | Path) -> Path:
    """Write ``commands`` (one runner argument string per line) to ``<out_dir>/job_commands.txt``."""
    out_dir = Path(out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "job_commands.txt"
    path.write_text("\n".join(commands) + "\n")
    return path


def run_commands(job_file: str | Path, *, python: str = sys.executable) -> int:
    """Run every job of ``job_file`` sequentially; return the number of failed jobs."""
    lines = [line for line in Path(job_file).read_text().splitlines() if line.strip()]
    failed = 0
    for i, line in enumerate(lines, start=1):
        start = time.monotonic()
        rc = subprocess.run([python, "-P", "-m", "selectarena.extra_benchmark.runner", *shlex.split(line)]).returncode
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"##### [{i}/{len(lines)}] {status} in {time.monotonic() - start:.0f}s: {line}", flush=True)
        failed += rc != 0
    print(f"##### summary: failed={failed} total={len(lines)}", flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job_file", help="job_commands.txt written by `run_extra_benchmark.py setup`.")
    args = parser.parse_args()
    sys.exit(1 if run_commands(args.job_file) else 0)


if __name__ == "__main__":
    main()
