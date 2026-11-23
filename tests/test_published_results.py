from __future__ import annotations

import json

import pandas as pd
import pytest

from selectarena.published_results import (
    METHODS_DIR,
    RESULTS_REPO_ID,
    ensure_published_results,
    merge_results,
    read_published_results,
    write_method_folders,
)
from selectarena.website import preview


def _perf(method: str, task: str = "t1") -> dict:
    return {"experiment_method_name_string": f"LightGBM_c1_FSBench__{method}__5__0__lgbm__3600_BAG_L1", "name": task}


def test_result_files_are_sorted_into_method_folders(tmp_path):
    published = tmp_path / "performance_results.csv"
    pd.DataFrame([_perf("MIFeatureSelector"), _perf("CPIFeatureSelector"), _perf("MIFeatureSelector", "t2")]).to_csv(
        published, index=False
    )
    stability = tmp_path / "stability.csv"
    pd.DataFrame({"method": ["FSBench__ANOVAFeatureSelector__5__0__lgbm__3600"], "mode": ["stability"]}).to_csv(
        stability, index=False
    )
    rerun = tmp_path / "performance_cpi_r0f0.csv"
    pd.DataFrame([_perf("CPIFeatureSelector", "t9")]).to_csv(rerun, index=False)

    out = tmp_path / "published"
    write_method_folders([published, stability, rerun], out, chunksize=1, log=lambda _: None)
    methods = out / METHODS_DIR
    assert sorted(p.relative_to(methods).as_posix() for p in methods.glob("*/*")) == [
        "CPI/performance_CPI.csv",
        "F-Test/stability_F-Test.csv",  # an old class name goes to the current method's folder
        "MI/performance_MI.csv",
    ]
    assert list(pd.read_csv(methods / "MI" / "performance_MI.csv")["name"]) == ["t1", "t2"]
    assert list(pd.read_csv(methods / "CPI" / "performance_CPI.csv")["name"]) == ["t9"]  # the rerun replaced it

    frames = read_published_results(out)
    assert len(frames["performance"]) == 3 and len(frames["stability"]) == 1 and frames["validity"] is None


def test_unregistered_methods_have_no_folder(tmp_path):
    path = tmp_path / "performance.csv"
    pd.DataFrame([_perf("NoSuchFeatureSelector")]).to_csv(path, index=False)
    with pytest.raises(KeyError, match="not registered"):
        write_method_folders([path], tmp_path / "out", log=lambda _: None)


def test_the_method_folders_are_synced_into_results_published(tmp_path, monkeypatch):
    seen = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    local = ensure_published_results(tmp_path, revision="abc123", log=lambda _: None)
    assert local == tmp_path / "published" and seen["local_dir"] == local
    assert (seen["repo_id"], seen["repo_type"], seen["revision"]) == (RESULTS_REPO_ID, "dataset", "abc123")
    assert seen["allow_patterns"] == ["methods/*/*.csv"]


def test_the_local_copy_is_used_without_a_connection(tmp_path, monkeypatch):
    def offline(**_):
        raise OSError("no connection")

    monkeypatch.setattr("huggingface_hub.snapshot_download", offline)
    with pytest.raises(OSError):
        ensure_published_results(tmp_path, log=lambda _: None)  # nothing local either
    folder = tmp_path / "published" / METHODS_DIR / "MI"
    folder.mkdir(parents=True)
    (folder / "performance_MI.csv").write_text("x\n1\n")
    assert ensure_published_results(tmp_path, log=lambda _: None) == tmp_path / "published"


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


def test_extra_results_merge_ignores_the_cache_path_of_task_ids():
    """The published extra results append the cluster's cache path to the task id; a rerun without it
    still replaces them (instead of being added next to them)."""
    task = "UserTask|1087039797|polish_companies_bankruptcy/019d9d2a"
    published = pd.DataFrame(
        {
            "method": ["SHAP", "SHAP", "MI"],
            "data_foundry_task_id": [f"{task}|/work/cache/tabarena_tasks", "UserTask|2|other/uuid|/x", task],
            "repeat": [0, 0, 0],
        }
    )
    own = pd.DataFrame({"method": ["SHAP"], "data_foundry_task_id": [task], "repeat": [7]})
    merged = merge_results(published, own, "validity")
    assert sorted(zip(merged["method"], merged["repeat"], strict=True)) == [("MI", 0), ("SHAP", 0), ("SHAP", 7)]
    assert (merged["method"] == "SHAP").sum() == 2  # the other task's published SHAP result stays


def test_own_results_can_replace_all_published_results_of_their_methods():
    """With replace_methods, a rerun replaces every published row of its method (other tasks and
    time limits included); other methods keep theirs."""
    published = pd.DataFrame(
        {
            "method": [
                "FSBench__SHAPFeatureSelector__5__0__lgbm__3600",
                "FSBench__SHAPFeatureSelector__5__0__lgbm__80000",
                "FSBench__MIFeatureSelector__5__0__lgbm__3600",
            ],
            "data_foundry_task_id": ["UserTask|1|a/u|/cache", "UserTask|2|b/u|/cache", "UserTask|1|a/u|/cache"],
        }
    )
    own = pd.DataFrame(
        {"method": ["FSBench__SHAPFeatureSelector__5__0__lgbm__3600"], "data_foundry_task_id": ["UserTask|3|c/u"]}
    )
    merged = merge_results(published, own, "stability", replace_methods=True)
    assert sorted(merged["data_foundry_task_id"]) == ["UserTask|1|a/u|/cache", "UserTask|3|c/u"]
    assert len(merge_results(published, own, "stability")) == 4  # by default only the same task is replaced

    perf_published = pd.DataFrame(
        {
            "experiment_method_name_string": [
                "LightGBM_c1_FSBench__SHAPFeatureSelector__5__0__lgbm__3600_BAG_L1",
                "LightGBM_c1_default_BAG_L1",
            ],
            "name": ["a", "a"],
        }
    )
    perf_own = pd.DataFrame(
        {
            "experiment_method_name_string": ["LightGBM_c1_FSBench__SHAPFeatureSelector__5__0__lgbm__3600_BAG_L1"],
            "name": ["b"],
        }
    )
    merged = merge_results(perf_published, perf_own, "performance", replace_methods=True)
    assert sorted(merged["name"]) == ["a", "b"] and "LightGBM_c1_default_BAG_L1" in set(
        merged["experiment_method_name_string"]
    )
