"""The dataset release checks a local warehouse and writes the Hub dataset card."""

from __future__ import annotations

import json

import pandas as pd

from selectarena.data_release import CORE_FILES, METADATA_FILES, inspect_suite, write_repo_files

PATHS = ["anneal/0001", "abalone/0002"]


def _warehouse(tmp_path, *, incomplete: str | None = None):
    for i, relative in enumerate(PATHS):
        path = tmp_path / relative
        path.mkdir(parents=True)
        for name in CORE_FILES:
            (path / name).write_text("{}")
        dataset = {
            "unique_name": relative.split("/")[0],
            "license": "CC BY 4.0",
            "original_dataset_source_download_link": "https://example.org",
            "academic_reference_bibtex": f"@misc{{ref{i}, title={{T{i}}}}}",
        }
        task = {"problem_type": ["multiclass", "regression"][i], "target_column_name": "y"}
        (path / METADATA_FILES["dataset"]).write_text(json.dumps(dataset))
        (path / METADATA_FILES["task"]).write_text(json.dumps(task))
    if incomplete:
        (tmp_path / incomplete / "dtypes.json").unlink()
    return tmp_path


def test_missing_and_incomplete_containers_are_reported(tmp_path):
    warehouse = _warehouse(tmp_path, incomplete="abalone/0002")
    infos = inspect_suite(warehouse, [*PATHS, "missing/0003"])
    assert [i.exists for i in infos] == [True, True, False]
    assert infos[1].missing_files == ["dtypes.json"]


def test_repo_files(tmp_path):
    infos = inspect_suite(_warehouse(tmp_path / "wh"), PATHS)
    files = write_repo_files(infos, tmp_path / "out", repo_id="Org/data", excluded=["automobile"])
    assert {f.name for f in files} == {"README.md", "LICENSE.md", "CITATIONS.bib", "datasets.csv"}
    card = (tmp_path / "out" / "README.md").read_text()
    assert card.startswith("---\n")
    assert "- tabular-classification" in card and "- tabular-regression" in card
    assert 'HuggingFaceSource(repo_id="Org/data")' in card
    assert "`automobile`" in card
    assert (tmp_path / "out" / "CITATIONS.bib").read_text().count("@misc") == 2
    table = pd.read_csv(tmp_path / "out" / "datasets.csv")
    assert list(table["dataset"]) == ["abalone", "anneal"]
