"""``pip install`` must ship every subpackage (the src layout needs ``selectarena/__init__.py``)."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages

SRC = Path(__file__).resolve().parent.parent / "src"


def test_every_subpackage_is_installed():
    found = set(find_packages(str(SRC)))
    folders = {
        ".".join(p.relative_to(SRC).parts)
        for p in SRC.rglob("*")
        if p.is_dir() and p.name != "__pycache__" and any(p.glob("*.py"))
    }
    assert "selectarena" in found
    assert folders <= found, f"not installed (missing __init__.py): {sorted(folders - found)}"
