"""The suite's containers come from the Hugging Face repository, or from a local warehouse."""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from selectarena import datasets

pytest.importorskip("data_foundry")


def test_collection_reads_from_hugging_face_by_default(tmp_path):
    from data_foundry.collections import HuggingFaceSource, LocalWarehouseSource

    collection = datasets.make_collection()
    assert isinstance(collection.source, HuggingFaceSource)
    assert collection.source.repo_id == datasets.HF_REPO_ID
    assert collection.source.revision == datasets.HF_REVISION
    assert datasets.HF_REPO_ID == "ANONYMIZED/SelectArena_Datasets"
    assert len(datasets.HF_REVISION) == 40  # a commit, not a branch
    assert len(collection) == len(datasets.BENCHMARK_DATA_FOUNDRY_TASKS)

    local = datasets.make_collection(tmp_path)
    assert isinstance(local.source, LocalWarehouseSource)
    assert local.source.base_dir == tmp_path


def test_prepare_datasets_from_the_hugging_face_cache(tmp_path, monkeypatch):
    """A container already in the download cache is converted without contacting the Hub."""
    pytest.importorskip("pyarrow")  # reads the container's parquet file
    from data_foundry.examples import TOY_CONTAINER_UNIQUE_NAME, TOY_CONTAINER_UUID, get_toy_container_path

    path = f"{TOY_CONTAINER_UNIQUE_NAME}/{TOY_CONTAINER_UUID}"
    monkeypatch.setattr(datasets, "BENCHMARK_DATA_FOUNDRY_TASKS", [path])
    # huggingface_hub's cache layout, below the per-collection folder of the Data Foundry cache
    repo_folder = f"datasets--{datasets.HF_REPO_ID.replace('/', '--')}"
    snapshot = tmp_path / "data_foundry" / datasets.COLLECTION_NAME / repo_folder / "snapshots" / ("0" * 40)
    shutil.copytree(get_toy_container_path(), snapshot / path)
    monkeypatch.setattr(datasets, "HF_REVISION", None)

    csv = datasets.prepare_datasets(cache_dir=tmp_path / "data_foundry", openml_cache=tmp_path / "openml")

    metadata = pd.read_csv(csv)
    assert list(metadata["dataset_name"]) == [TOY_CONTAINER_UNIQUE_NAME]
    assert csv.is_relative_to(tmp_path / "openml")
