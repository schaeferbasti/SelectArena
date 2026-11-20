"""Helpers shared by the backends."""

from __future__ import annotations

import os
import re
import subprocess

USER = os.environ.get("USER", "")


def run(cmd: list[str]) -> str:
    """stdout of a read-only query; empty on errors (the caller treats missing data as unknown)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def gpus_of(tres: str) -> int:
    """'gres/gpu:1,gres/localtmp:10' or 'gres/gpu=2' or 'gpu:l40s:1' -> 1/2/1."""
    n = 0
    for m in re.finditer(r"gpu(?::[A-Za-z0-9_-]+)?[:=](\d+)", tres or ""):
        n += int(m.group(1))
    return n


def task(**kw) -> dict:
    """A task record with defaults for the fields a backend does not know."""
    base = dict(
        state="?",
        raw="",
        elapsed=0,
        end="",
        node="",
        part="",
        cpus=0,
        gpus=0,
        limit=0,
        exit="",
        reason="",
        name="",
        acct="",
    )
    base.update(kw)
    return base
