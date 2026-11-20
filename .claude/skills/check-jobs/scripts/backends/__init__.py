"""Scheduler backends of the check-jobs skill.

A backend turns its scheduler's (read-only) queries into the records the core in ``collect.py`` works with:

- ``collect(since, extra_arrays) -> dict[array_id, dict]``: every task of the user's arrays that are queued, ended
  after ``since`` or are listed in ``extra_arrays``; each ``{"name", "acct", "tasks": {task_id: task}}`` with
  ``task = dict(state, raw, elapsed, end, node, part, cpus, gpus, limit, exit, reason)``.
  ``state`` is normalized: PENDING, RUNNING, COMPLETED, TIMEOUT, OUT_OF_MEMORY, FAILED, NODE_FAIL, LAUNCH_FAIL,
  CANCELLED or the scheduler's own name; ``raw`` keeps the scheduler's details. ``end`` is ISO ``YYYY-MM-DDTHH:MM:SS``
  (empty while not ended), ``elapsed``/``limit`` are seconds.
- ``array_info(array_id, tasks) -> dict(throttle, partitions, notes)``; ``throttle`` is None if the scheduler has no
  per-array concurrency limit.
- ``partition_info(partition, accounts) -> dict`` with the cluster's and the account's usage of that partition/queue.
- ``throttle_word``, ``throttle_hint``: how to describe and change the concurrency in the estimates.
"""

from __future__ import annotations

import shutil


def detect(name: str = "auto"):
    """The backend for ``name`` (slurm, oar) or the first scheduler found on PATH."""
    if name == "auto":
        if shutil.which("squeue") and shutil.which("sacct"):
            name = "slurm"
        elif shutil.which("oarstat"):
            name = "oar"
        else:
            raise SystemExit("check-jobs: no supported scheduler found (squeue/sacct for SLURM, oarstat for OAR).")
    if name == "slurm":
        from backends.slurm import SlurmBackend

        return SlurmBackend()
    if name == "oar":
        from backends.oar import OarBackend

        return OarBackend()
    raise SystemExit(f"check-jobs: unknown scheduler {name!r} (supported: slurm, oar).")
