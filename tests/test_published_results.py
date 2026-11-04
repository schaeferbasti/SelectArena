from __future__ import annotations

import json

import pandas as pd
import pytest

from selectarena import published_results
from selectarena.published_results import RESULT_FILES, RESULTS_REPO_ID, ensure_published_results, merge_results
from selectarena.website import preview


def test_missing_results_are_downloaded_into_the_results_folder(tmp_path, monkeypatch):
    calls = []

    def fake_download(*, repo_id, filename, repo_type, revision, local_dir):
        calls.append((repo_id, filename, repo_type))
        path = tmp_path / "results" / filename
        assert str(local_dir) == str(tmp_path / "results")
        path.write_text("x\n1\n")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / RESULT_FILES["performance"]).write_text("already here\n")  # kept, not downloaded

    paths = ensure_published_results(tmp_path / "results", log=lambda _: None)
    assert sorted(f for _, f, _ in calls) == [RESULT_FILES["stability"], RESULT_FILES["validity"]]
    assert {repo for repo, _, _ in calls} == {RESULTS_REPO_ID} and {t for _, _, t in calls} == {"dataset"}
    assert all(path.is_file() for path in paths.values())
    assert paths["performance"].read_text() == "already here\n"

    calls.clear()
    ensure_published_results(tmp_path / "results", log=lambda _: None)  # everything there now
    assert calls == []


def test_the_results_folder_is_created(tmp_path, monkeypatch):
    def fake_download(*, local_dir, filename, **_):
        (tmp_path / "new" / filename).write_text("x\n")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    ensure_published_results(tmp_path / "new", log=lambda _: None)
    assert sorted(p.name for p in (tmp_path / "new").iterdir()) == sorted(RESULT_FILES.values())


def test_own_results_replace_the_published_ones_of_the_same_method_and_task():
    published = pd.DataFrame(
        {
            "experiment_method_name_string": ["A", "A", "B", "default"],
            "name": ["t1", "t2", "t1", "t1"],
            "metric_error": [0.1, 0.2, 0.3, 0.4],
        }
    )
    own = pd.DataFrame(
        {"experiment_method_name_string": ["A", "New"], "name": ["t1", "t1"], "metric_error": [0.9, 0.5]}
    )
    merged = merge_results(published, own, "performance").set_index(["experiment_method_name_string", "name"])
    assert merged.loc[("A", "t1"), "metric_error"] == 0.9  # replaced by the rerun
    assert merged.loc[("A", "t2"), "metric_error"] == 0.2  # other tasks keep the published result
    assert merged.loc[("default", "t1"), "metric_error"] == 0.4
    assert ("New", "t1") in merged.index and len(merged) == 5
    assert merge_results(published, None, "performance") is published
    assert merge_results(None, own, "performance") is own


def test_extra_results_merge_by_method_and_task():
    published = pd.DataFrame({"method": ["m1", "m2"], "data_foundry_task_id": ["a", "a"], "repeat": [0, 0]})
    own = pd.DataFrame({"method": ["m2"], "data_foundry_task_id": ["a"], "repeat": [5]})
    merged = merge_results(published, own, "stability")
    assert sorted(zip(merged["method"], merged["repeat"], strict=True)) == [("m1", 0), ("m2", 5)]


def test_preview_combines_the_page_with_the_generated_data(tmp_path):
    site = tmp_path / "space"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<html>page</html>")
    (site / "assets" / "app.js").write_text("// app")
    (site / "data").mkdir()
    (site / "data" / "meta.json").write_text('{"published": true}')  # the Space's own data: not used
    data = tmp_path / "website_data"
    (data / "subsets").mkdir(parents=True)
    (data / "meta.json").write_text(json.dumps({"local": True}))
    out = tmp_path / "preview"
    (out / "data").mkdir(parents=True)
    (out / "data" / "stale.csv").write_text("old")

    preview.prepare_preview(data, out, site_dir=site, log=lambda _: None)
    assert (out / "index.html").read_text() == "<html>page</html>"
    assert (out / "assets" / "app.js").is_file()
    assert json.loads((out / "data" / "meta.json").read_text()) == {"local": True}
    assert not (out / "data" / "stale.csv").exists()  # the old data is deleted first


def test_preview_needs_generated_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_generate_website_artifacts"):
        preview.prepare_preview(tmp_path / "missing", tmp_path / "out", site_dir=tmp_path)


def test_preview_downloads_only_the_page(tmp_path, monkeypatch):
    seen = {}

    def fake_snapshot(*, repo_id, repo_type, allow_patterns, local_dir):
        seen.update(repo_id=repo_id, repo_type=repo_type, patterns=allow_patterns)
        (local_dir / "index.html").write_text("page")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    data = tmp_path / "website_data"
    data.mkdir()
    (data / "meta.json").write_text("{}")
    preview.prepare_preview(data, tmp_path / "out", log=lambda _: None)
    assert seen == {"repo_id": preview.SPACE_REPO_ID, "repo_type": "space", "patterns": ["index.html", "assets/*"]}
    assert not any("data" in p for p in seen["patterns"])


def test_revision_is_used(tmp_path, monkeypatch):
    revisions = set()

    def fake_download(*, revision, filename, **_):
        revisions.add(revision)
        (tmp_path / filename).write_text("x\n")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(published_results, "RESULTS_REVISION", "abc123")
    ensure_published_results(tmp_path, revision="abc123", log=lambda _: None)
    assert revisions == {"abc123"}
