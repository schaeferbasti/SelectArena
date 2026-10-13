"""The website artifacts the leaderboard Space serves."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from selectarena.methods.registry import FEATURE_SELECTION_METHODS, METHOD_FAMILIES, PAPER_NAMES
from selectarena.website import Subset, generate_website_artifacts
from selectarena.website.generate import SUBSET_FILES
from selectarena.website.html import elo_figure_svg
from selectarena.website.website_format import LEADERBOARD_COLUMNS, method_family

METHODS = {
    "MIFeatureSelector": 0.6,
    "RandomFeatureSelector": 0.0,
    "LassoFeatureSelector": 0.4,
    "CARTFeatureSelector": 0.8,
}
DATASETS = {"anneal": "multiclass", "credit": "binary", "diabetes": "binary", "abalone": "regression"}
FEATURES = [f"f{i}" for i in range(10)]


def _pipeline(method: str, budget: int) -> str:
    return f"FSBench__{method}__2__{budget}__lgbm__60"


def results() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for dataset, problem_type in DATASETS.items():
        for split in range(2):
            for model in ["LightGBM", "TabICLv2"]:
                for budget in range(3):
                    for method, quality in METHODS.items():
                        rows.append(
                            {
                                "experiment_method_name_string": f"{model}_c1_{_pipeline(method, budget)}_BAG_L1",
                                "name": f"{dataset}/0199-uuid",
                                "split_idx": split,
                                "problem_type": problem_type,
                                "metric_error": 1 - 0.1 * quality + rng.normal(0, 0.03),
                                "feature_selection_budget_index": budget,
                                "feature_selection_fit_time": 2.0,
                                "random_feature_names": "[]",
                            }
                        )
                rows.append(
                    {
                        "experiment_method_name_string": f"{model}_c1_default_BAG_L1",
                        "name": f"{dataset}/0199-uuid",
                        "split_idx": split,
                        "problem_type": problem_type,
                        "metric_error": 0.97,
                        "feature_selection_budget_index": np.nan,
                        "feature_selection_fit_time": np.nan,
                        "random_feature_names": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def extra_results(mode: str) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            for budget in range(3):
                for repeat in range(3 if mode == "stability" else 1):
                    for noise in [0.5, 1.0] if mode == "validity" else [None]:
                        selected = [
                            str(f) for f in rng.choice([*FEATURES, "noise_0", "noise_1"], size=4, replace=False)
                        ]
                        rows.append(
                            {
                                "method": _pipeline(method, budget),
                                "data_foundry_task_id": f"UserTask|123|{dataset}/0199-uuid",
                                "mode": mode,
                                "mode_kwargs": str({"noise": noise, "noise_type": "gaussian"}) if noise else "{}",
                                "repeat": repeat,
                                "original_features": str(FEATURES),
                                "max_features": 4,
                                "selected_features": str(selected),
                            }
                        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("website")
    return generate_website_artifacts(
        results(),
        out_dir=out,
        stability=extra_results("stability"),
        validity=extra_results("validity"),
        n_bootstrap=20,
        log=lambda _: None,
    )


def test_every_registered_method_has_a_family():
    assert set(METHOD_FAMILIES) == set(FEATURE_SELECTION_METHODS)
    assert all(method_family(PAPER_NAMES[m]) != "other" for m in FEATURE_SELECTION_METHODS)
    assert method_family("MI [nofallback]") == "filter"
    assert method_family("NewMethod") == "other"


def test_layout(artifacts):
    data = artifacts.website_data
    meta = json.loads((data / "meta.json").read_text())
    assert meta["n_datasets"] == 4 and meta["n_methods"] == 4
    assert meta["models"] == {"lightgbm": "LightGBM", "tabiclv2": "TabICLv2"}
    assert meta["budgets"] == [0, 1, 2] and meta["total_budget"] == 2
    assert set(meta["task_types"]) == {"all", "classification", "binary", "multiclass", "regression"}
    # 5 task types x (all + 2 models) x (all + 3 budgets)
    assert len(meta["subsets"]) == 5 * 3 * 4
    assert meta["subsets"][0]["path"] == "subsets/" + Subset().rel_path
    for subset in meta["subsets"]:
        for name in SUBSET_FILES:
            assert (data / subset["path"] / name).is_file()
    for name in [
        "per_dataset.csv",
        "per_dataset_table.html",
        "stability.csv",
        "validity.csv",
        "stability_table.html",
        "validity_table.html",
    ]:
        assert (data / name).is_file()
    assert not list(data.rglob("*.png"))
    assert (artifacts.raw_dir / "main_figure.pdf").is_file()
    with zipfile.ZipFile(artifacts.zip_path) as z:
        assert "website_data/meta.json" in z.namelist()


def test_leaderboards(artifacts):
    data = artifacts.website_data
    overall = pd.read_csv(data / "subsets" / Subset().rel_path / "website_leaderboard.csv")
    assert list(overall.columns) == LEADERBOARD_COLUMNS
    assert list(overall["Method"]) == ["CART", "MI", "Lasso", "Random"]
    assert overall.set_index("Method").loc["Random", "Elo"] == 1000
    assert overall["Stability"].notna().all() and overall["Validity"].between(0, 1).all()
    assert set(overall["TypeName"]) == {"Embedded", "Filter", "Baseline"}
    assert (overall["#Datasets"] == 4).all()

    regression = pd.read_csv(
        data / "subsets" / Subset("regression", "lightgbm", "1").rel_path / "website_leaderboard.csv"
    )
    assert (regression["#Datasets"] == 1).all()
    assert (regression["#Comparisons"] == 2).all()  # 2 splits


def test_html_is_self_contained(artifacts):
    data = artifacts.website_data
    for path in [*data.rglob("*.html")]:
        text = path.read_text()
        assert "<script src" not in text and "http" not in text.replace("http://www.w3.org", "")
        assert "${" not in text and "{{" not in text  # Gradio's gr.HTML templates
    table = (data / "subsets" / Subset().rel_path / "leaderboard_table.html").read_text()
    assert "CART" in table and "🌳 Embedded" in table


def test_figure_handles_missing_scores():
    leaderboard = pd.DataFrame(
        {
            "Method": ["A", "B"],
            "TypeName": ["Filter", "Wrapper"],
            "Elo": [1000.0, 1100.0],
            "Elo CI low": [990.0, 1080.0],
            "Elo CI high": [1010.0, 1120.0],
            "Stability": [0.5, np.nan],
            "Validity": [np.nan, 1.0],
        }
    )
    svg = elo_figure_svg(leaderboard)
    assert svg.count("url(#sa-na)") == 2 + 1  # two missing bars + the legend swatch
    assert svg.index(">A<") < svg.index(">B<")  # worst to best


def test_ablations_are_left_out_by_default(tmp_path):
    df = results()
    ablation = df[df["experiment_method_name_string"].str.contains("MIFeatureSelector")].copy()
    ablation["experiment_method_name_string"] = ablation["experiment_method_name_string"].str.replace(
        "__60_BAG_L1", "__60__nofallback_BAG_L1"
    )
    both = pd.concat([df, ablation])
    default = generate_website_artifacts(
        both, out_dir=tmp_path / "a", n_bootstrap=2, paper_figures=False, log=lambda _: None
    )
    meta = json.loads((default.website_data / "meta.json").read_text())
    assert "MI [nofallback]" not in meta["methods"]
    kept = generate_website_artifacts(
        both, out_dir=tmp_path / "b", n_bootstrap=2, paper_figures=False, include_ablations=True, log=lambda _: None
    )
    assert "MI [nofallback]" in json.loads((kept.website_data / "meta.json").read_text())["methods"]


def test_fork_era_results(tmp_path):
    """Results written by tabarena_fork_fs: old class names, a retired method, duplicated extra rows."""
    df = results()
    df["experiment_method_name_string"] = df["experiment_method_name_string"].str.replace(
        "LassoFeatureSelector", "AccuracyFeatureSelector"
    )
    retired = df[df["experiment_method_name_string"].str.contains("CARTFeatureSelector")].copy()
    retired["experiment_method_name_string"] = retired["experiment_method_name_string"].str.replace(
        "CARTFeatureSelector", "AccuracyLinearFeatureSelector"
    )
    stability = extra_results("stability")
    artifacts = generate_website_artifacts(
        pd.concat([df, retired]),
        out_dir=tmp_path,
        stability=pd.concat([stability, stability]),
        n_bootstrap=2,
        paper_figures=False,
        log=lambda _: None,
    )
    meta = json.loads((artifacts.website_data / "meta.json").read_text())
    assert meta["methods"] == ["CART", "LOCO", "MI", "Random"]
    single = generate_website_artifacts(
        df, out_dir=tmp_path / "single", stability=stability, n_bootstrap=2, paper_figures=False, log=lambda _: None
    )
    path = "subsets/tasks_all/models_all/budgets_all/website_leaderboard.csv"
    doubled = pd.read_csv(artifacts.website_data / path).set_index("Method")["Stability"]
    once = pd.read_csv(single.website_data / path).set_index("Method")["Stability"]
    pd.testing.assert_series_equal(doubled.sort_index(), once.sort_index())
