"""The website artifacts the leaderboard Space serves."""

from __future__ import annotations

import ast
import json
import re
import zipfile

import numpy as np
import pandas as pd
import pytest

from selectarena.methods.registry import FEATURE_SELECTION_METHODS, METHOD_SUBGROUPS, PAPER_NAMES, SUBGROUP_CATEGORIES
from selectarena.website import Subset, generate_website_artifacts
from selectarena.website.generate import SUBSET_FILES
from selectarena.website.html import elo_figure_svg
from selectarena.website.website_format import LEADERBOARD_COLUMNS, method_category, method_paper, method_subgroup

METHODS = {
    "MIFeatureSelector": 0.6,
    "RandomFeatureSelector": 0.0,
    "LassoFeatureSelector": 0.4,
    "CARTFeatureSelector": 0.8,
}
DATASETS = {"anneal": "multiclass", "credit": "binary", "diabetes": "binary", "abalone": "regression"}
SAMPLES = {"anneal": (800, 40), "credit": (600, 250), "diabetes": (700, 200), "abalone": (4000, 1)}
"""(samples, samples of the smallest class) of each dataset, as stored in the extra results."""
FEATURES = [f"f{i}" for i in range(10)]


def _pipeline(method: str, budget: int) -> str:
    return f"FSBench__{method}__2__{budget}__lgbm__60"


def results() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for dataset, problem_type in DATASETS.items():
        for split in range(2):
            for model in ["LightGBM", "TA-TabICLv2"]:  # TabArena's prefix is dropped on the leaderboard
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
                                # Random fills its budget with random features by design.
                                "random_feature_names": "['f0']" if method == "RandomFeatureSelector" else "[]",
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
                                "num_samples": SAMPLES[dataset][0],
                                "min_samples_per_class": SAMPLES[dataset][1],
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


def test_every_registered_method_has_a_subgroup():
    assert set(METHOD_SUBGROUPS) == set(FEATURE_SELECTION_METHODS)
    assert set(METHOD_SUBGROUPS.values()) <= set(SUBGROUP_CATEGORIES)
    assert all(method_category(PAPER_NAMES[m]) != "Other" for m in FEATURE_SELECTION_METHODS)
    assert method_subgroup("MI [nofallback]") == "Information theory"
    assert method_category("SBE") == "Search" and method_category("CART") == "Non-search"
    assert method_category("NewMethod") == "Other"


def test_method_papers():
    from selectarena.methods.registry import METHOD_PAPERS

    assert set(METHOD_PAPERS) <= set(FEATURE_SELECTION_METHODS)
    assert all(url.startswith(("https://", "http://")) for url in METHOD_PAPERS.values())
    assert method_paper("Lasso") == METHOD_PAPERS["LassoFeatureSelector"]
    assert method_paper("MI [nofallback]") == METHOD_PAPERS["MIFeatureSelector"]
    assert method_paper("Random") is None and method_paper("NewMethod") is None


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
        "stability.csv",
        "validity.csv",
        "stability_table.html",
        "validity_noise_0.5_table.html",
        "validity_noise_1_table.html",
    ]:
        assert (data / name).is_file()
    assert meta["validity_tables"] == [
        {"noise": 0.5, "path": "validity_noise_0.5_table.html"},
        {"noise": 1.0, "path": "validity_noise_1_table.html"},
    ]
    assert "Noise level" not in (data / "validity_noise_1_table.html").read_text()
    assert meta["has_results_explorer"] and (data / "results_explorer.html").is_file()
    assert meta["subsets"][0]["pareto"]
    for subset in meta["subsets"]:
        assert (data / subset["path"] / "pareto.html").is_file() == subset["pareto"]
    lasagna = meta["lasagna"]
    assert lasagna["variants"] == ["plain", "overlay_top_3", "binary_thr_003"]
    # Regression has a single dataset, i.e. a single EPV: no plot.
    assert lasagna["stability"]["task_types"] == ["all", "classification", "binary"]
    assert [level["noise"] for level in lasagna["validity"]] == [0.5, 1.0]
    for folder in [lasagna["stability"], *lasagna["validity"]]:
        for tasks in folder["task_types"]:
            for variant in lasagna["variants"]:
                assert (data / folder["path"] / f"tasks_{tasks}" / f"{variant}.html").is_file()
    assert not (data / "per_dataset_table.html").exists()  # the site no longer shows the per-dataset ranks
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
    assert set(overall["Subgroup"]) == {"Tree", "Information theory", "Regularization", "Baseline"}
    assert set(overall["Category"]) == {"Non-search", "Baseline"}
    assert (overall["#Datasets"] == 4).all()

    regression = pd.read_csv(
        data / "subsets" / Subset("regression", "lightgbm", "1").rel_path / "website_leaderboard.csv"
    )
    assert (regression["#Datasets"] == 1).all()
    assert (regression["#Comparisons"] == 2).all()  # 2 splits


def test_html_is_self_contained(artifacts):
    data = artifacts.website_data
    for path in [*data.rglob("*.html")]:
        # Links to method papers are clicked, not loaded: the page stays self-contained.
        text = re.sub(r"<a class='paper' href='[^']*'", "", path.read_text())
        assert "<script src" not in text and "http" not in text.replace("http://www.w3.org", "")
        assert "${" not in text and "{{" not in text  # Gradio's gr.HTML templates
    table = (data / "subsets" / Subset().rel_path / "leaderboard_table.html").read_text()
    assert "CART" in table and ">Tree<" in table
    header = table.split("<thead", 1)[1].split("</thead>", 1)[0]
    assert ">Improvement over Random" in header
    for hidden in ("Win rate", "Error vs. all", "#Datasets", "Avg. rank", "Improvability"):
        assert hidden not in table  # neither shown nor in the CSV download
    overall = pd.read_csv(data / "subsets" / Subset().rel_path / "website_leaderboard.csv").set_index("Method")
    assert overall.loc["Random", "Improvement over Random (%)"] == 0
    assert overall["Improvement over Random (%)"].notna().all()
    # The same numbers as the Pareto front (both average per dataset, then over datasets).
    pareto = pd.read_csv(data / "subsets" / Subset().rel_path / "pareto.csv").set_index("method")
    pd.testing.assert_series_equal(
        overall["Improvement over Random (%)"].sort_index(),
        pareto["improvement"].round(1).rename("Improvement over Random (%)").rename_axis("Method").sort_index(),
        check_exact=False,
        atol=0.11,
    )


def test_figure_tooltip(artifacts):
    """The Elo figure shows its own tooltip at once (a native SVG <title> waits 1-3 s)."""
    page = (artifacts.website_data / "subsets" / Subset().rel_path / "elo_explorer.html").read_text()
    assert page.count("class='sa-hit'") == 4 and "data-tip='CART (Tree, non-search)&#10;Elo" in page
    assert "<title>" not in page.split("<svg", 1)[1].split("</svg>", 1)[0]
    script = page.rsplit("<script>", 1)[1]
    for line in script.splitlines():  # a raw line break inside a JS string literal is a syntax error
        code = line.split("//", 1)[0] if line.lstrip().startswith("//") else line
        assert code.count("'") % 2 == 0, line


def test_figure_handles_missing_scores():
    leaderboard = pd.DataFrame(
        {
            "Method": ["A", "B"],
            "Category": ["Non-search", "Search"],
            "Subgroup": ["Tree", "Forward search"],
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


def test_former_names_unregistered_methods_and_duplicates(tmp_path):
    """Former class names (as in the published results), a method not in the registry, duplicated extra rows."""
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


def test_selected_features_as_entries(tmp_path):
    """Selected features stored as {"feature", "value"} entries (as in the published results)."""
    stability = extra_results("stability")
    as_entries = stability.copy()
    as_entries["selected_features"] = [
        str([{"feature": f, "value": f"np.float64({0.5 + i})"} for i, f in enumerate(ast.literal_eval(v))])
        .replace("'np.float64", "np.float64")
        .replace(")'", ")")
        for v in stability["selected_features"]
    ]
    assert "np.float64(0.5)" in as_entries["selected_features"].iloc[0]
    validity = extra_results("validity")
    validity_entries = validity.copy()
    validity_entries["selected_features"] = [
        str([{"feature": f, "value": None} for f in ast.literal_eval(v)]) for v in validity["selected_features"]
    ]

    def leaderboard(stab, val, out):
        artifacts = generate_website_artifacts(
            results(),
            out_dir=tmp_path / out,
            stability=stab,
            validity=val,
            n_bootstrap=2,
            paper_figures=False,
            log=lambda _: None,
        )
        path = artifacts.website_data / "subsets/tasks_all/models_all/budgets_all/website_leaderboard.csv"
        return pd.read_csv(path).set_index("Method")[["Stability", "Validity"]].sort_index()

    names = leaderboard(stability, validity, "names")
    entries = leaderboard(as_entries, validity_entries, "entries")
    assert names.notna().all().all()
    pd.testing.assert_frame_equal(names, entries)


def test_task_ids_are_named():
    from tabarena.benchmark.task import UserTask

    from selectarena.datasets import (
        BENCHMARK_DATA_FOUNDRY_TASKS,
        RENAMED_DATA_FOUNDRY_TASKS,
        dataset_name,
        user_task_id,
    )

    assert user_task_id("anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792") == 1386903908  # TabArena's id of a UserTask
    long_name = next(p for p in BENCHMARK_DATA_FOUNDRY_TASKS if p.startswith("predict_students"))  # slug is cut
    for path in [BENCHMARK_DATA_FOUNDRY_TASKS[0], long_name]:
        name = path.split("/")[0]
        assert dataset_name(user_task_id(path)) == name
        assert dataset_name(str(user_task_id(path))) == name
        assert dataset_name(float(user_task_id(path))) == name
        assert dataset_name(path) == name
        assert dataset_name(f"UserTask|{user_task_id(path)}|{path}") == name
        assert dataset_name(f"Task-{user_task_id(path)}") == name  # task name in the published results
        assert dataset_name(UserTask(task_name=path).slug) == name  # TabArena's tabarena_task_name
    assert dataset_name("anneal") == "anneal"
    assert dataset_name("123") == "123"
    # A container renamed to snake_case keeps its dataset name for results from before the rename.
    for path in RENAMED_DATA_FOUNDRY_TASKS:
        renamed = path.lower()
        assert renamed in BENCHMARK_DATA_FOUNDRY_TASKS
        assert dataset_name(user_task_id(path)) == dataset_name(user_task_id(renamed)) == renamed.split("/")[0]
        assert dataset_name(path.split("/")[0]) == renamed.split("/")[0]


def test_results_with_task_ids(tmp_path):
    from selectarena.datasets import BENCHMARK_DATA_FOUNDRY_TASKS, user_task_id

    anneal = next(p for p in BENCHMARK_DATA_FOUNDRY_TASKS if p.startswith("anneal/"))
    credit = next(p for p in BENCHMARK_DATA_FOUNDRY_TASKS if p.startswith("credit_approval/"))
    df = results().replace(
        {"name": {"anneal/0199-uuid": user_task_id(anneal), "credit/0199-uuid": user_task_id(credit)}}
    )
    artifacts = generate_website_artifacts(df, out_dir=tmp_path, n_bootstrap=2, paper_figures=False, log=lambda _: None)
    per_dataset = pd.read_csv(artifacts.website_data / "per_dataset.csv")
    assert sorted(per_dataset["dataset"]) == ["abalone", "anneal", "credit_approval", "diabetes"]


def test_task_ids_named_from_extra_results(tmp_path):
    """Task ids of containers not listed in selectarena.datasets are named from the extra results."""
    df = results().replace({"name": {"anneal/0199-uuid": "Task-1234567890"}})
    stability = extra_results("stability")
    stability["data_foundry_task_id"] = stability["data_foundry_task_id"].str.replace(
        "UserTask|123|anneal/0199-uuid", "UserTask|1234567890|anneal/0199-other-uuid", regex=False
    )
    artifacts = generate_website_artifacts(
        df, out_dir=tmp_path, stability=stability, n_bootstrap=2, paper_figures=False, log=lambda _: None
    )
    per_dataset = pd.read_csv(artifacts.website_data / "per_dataset.csv")
    assert "anneal" in set(per_dataset["dataset"]) and "Task-1234567890" not in set(per_dataset["dataset"])


def test_results_table(artifacts):
    table = pd.read_csv(artifacts.website_data / "results.csv")
    # 4 datasets x 2 models x (4 methods x 3 budgets + all features)
    assert len(table) == 4 * 2 * (4 * 3 + 1)
    assert set(table["method"]) == {"CART", "MI", "Lasso", "Random", "All features"}
    assert (table["n_splits"] == 2).all()
    assert table.loc[table["method"] == "All features", "fallback_pct"].isna().all()
    assert (table.loc[table["method"] == "Random", "fallback_pct"] == 0).all()  # by definition
    assert set(table["model"]) == {"LightGBM", "TabICLv2"}
    html = (artifacts.website_data / "results_explorer.html").read_text()
    assert "id='sa-data'" in html and "Lasso" in html


def test_events_per_variable():
    from selectarena.evaluation.leaderboard import events_per_variable

    rows = pd.DataFrame(
        {
            "dataset": ["a", "b", "c"],
            "num_features": [10, 4, 5],
            "num_samples": [100, 40, 50],
            "min_samples_per_class": [20, np.nan, 1],
        }
    )
    assert list(events_per_variable(rows)) == [2.0, 10.0, 0.2]
    # Regression datasets count all samples, whatever their per-class count says.
    assert list(events_per_variable(rows, {"a": "binary", "b": "binary", "c": "regression"})) == [2.0, 10.0, 10.0]


def test_pareto():
    from selectarena.evaluation.pareto import improvement_and_runtime, pareto_front

    rows = pd.DataFrame(
        [
            {"dataset": d, "method": m, "metric_error": e, "feature_selection_fit_time": t}
            for d in ["a", "b"]
            for m, e, t in [("Random", 0.5, 0.01), ("Fast", 0.4, 0.1), ("Good", 0.25, 10.0), ("Bad", 0.45, 20.0)]
        ]
        + [{"dataset": "a", "method": None, "metric_error": 0.3, "feature_selection_fit_time": np.nan}]
    )
    summary = improvement_and_runtime(rows, {"a": 100, "b": 1000}).set_index("method")
    assert summary.loc["Random", "improvement"] == 0
    assert summary.loc["Good", "improvement"] == pytest.approx(50)
    assert summary.loc["Fast", "time_per_1k"] == pytest.approx((1.0 + 0.1) / 2)
    assert (summary["n_datasets"] == 2).all()
    assert summary["pareto_optimal"].to_dict() == {"Good": True, "Fast": True, "Bad": False, "Random": True}
    tie = pd.DataFrame({"time_per_1k": [1.0, 1.0], "improvement": [5.0, 5.0]})
    assert pareto_front(tie).all()  # neither strictly dominates the other


def test_lasagna_masks_and_ticks():
    from selectarena.evaluation.lasagna import _spread_ticks, epv_matrix, per_epv, threshold_mask, top_k_mask

    rows = pd.DataFrame(
        {
            "method": ["A", "A", "B", "B", "C", "C", "C"],
            "epv": [1.0, 10.0, 1.0, 10.0, 1.0, 10.0, -1.0],
            "stability": [0.9, 0.5, 0.88, 0.8, 0.1, 0.2, 0.99],
        }
    )
    matrix = epv_matrix(per_epv(rows, "stability"), "stability")
    assert list(matrix.index) == ["B", "A", "C"]  # best mean on top; the negative EPV is dropped
    assert list(matrix.columns) == [1.0, 10.0]
    assert top_k_mask(matrix, k=1).to_numpy().tolist() == [[0, 1], [1, 0], [0, 0]]
    assert threshold_mask(matrix).to_numpy().tolist() == [[1, 1], [1, 0], [0, 0]]
    assert list(_spread_ticks(np.array([0.0, 0.5, 0.55, 1.0, 9.95, 10.0]))) == [0.0, 1.0, 10.0]


def test_leaderboard_table_medals_and_shading(artifacts):
    data = artifacts.website_data
    overall = pd.read_csv(data / "subsets" / Subset().rel_path / "website_leaderboard.csv")
    assert (overall["Fallback rate (%)"] == 0).all()  # Random included
    table = (data / "subsets" / Subset().rel_path / "leaderboard_table.html").read_text()
    body = table.split("<tbody>", 1)[1]
    assert body.index("gold medal") < body.index("silver medal") < body.index("bronze medal")
    assert body.count("class='medal' role='img'") == 3
    assert "background:rgba(0,114,178," in body and "background:rgba(213,94,0," in body


def test_extra_tables_medals_and_paper_links(artifacts):
    from selectarena.methods.registry import METHOD_PAPERS

    data = artifacts.website_data
    for name in [
        "stability_table.html",
        *(t["path"] for t in json.loads((data / "meta.json").read_text())["validity_tables"]),
    ]:
        text = (data / name).read_text()
        body = text.split("<tbody>", 1)[1]
        assert body.count("class='medal' role='img'") == 3, name
        assert "gold medal" in body.split("<tr>")[1]  # sorted best first; ties share a medal
        assert f"href='{METHOD_PAPERS['LassoFeatureSelector']}'" in body
        assert "_rank" not in text  # helper column: not shown, not in the CSV


def test_leaderboard_type_column():
    from selectarena.website.html import leaderboard_table_html

    table = pd.DataFrame(
        {
            "Rank": [1, 2, 3, 4],
            "Method": ["SFS", "LOCO", "CART", "Random"],
            "Category": ["Search", "Search", "Non-search", "Baseline"],
            "Subgroup": ["Forward search", "Leave-one-out", "Tree", "Baseline"],
            "Elo": [1100, 1050, 1020, 1000],
            "Elo CI low": [1080, 1030, 1000, 980],
            "Elo CI high": [1120, 1070, 1040, 1020],
        }
    )
    page = leaderboard_table_html(table, title="t")
    body = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert body.count(">Search</span>") == 2 and ">Tree</span>" in body and ">Baseline</span>" in body
    assert "search</span>" not in body.replace(">Search</span>", "")  # no search / non-search label
    assert "Forward search" not in body and "Leave-one-out" not in body
    csv_header = page.split("Rank,Method,", 1)[1].split("\n", 1)[0]
    assert csv_header.startswith("Type,Elo") and "Subgroup" not in csv_header and "Category" not in csv_header


def test_per_model_tables(artifacts):
    data = artifacts.website_data
    meta = json.loads((data / "meta.json").read_text())
    for subset in meta["subsets"]:
        assert subset.get("per_model", False) == (subset["model"] == "all")
        assert (data / subset["path"] / "per_model_table.html").is_file() == (subset["model"] == "all")
    per_model = pd.read_csv(data / "subsets" / Subset().rel_path / "per_model.csv").set_index("Method")
    assert list(per_model.columns) == [
        "Rank", "Category", "Subgroup", "Elo", "Elo LightGBM", "Rank LightGBM", "Elo TabICLv2", "Rank TabICLv2",
    ]  # fmt: skip
    lightgbm = pd.read_csv(data / "subsets" / Subset(model="lightgbm").rel_path / "website_leaderboard.csv")
    assert per_model["Elo LightGBM"].to_dict() == lightgbm.set_index("Method")["Elo"].to_dict()
    html = (data / "subsets" / Subset().rel_path / "per_model_table.html").read_text()
    assert ">TabICLv2 ↑<" in html and html.count("gold medal") == 3  # overall + one per model


def test_clean_model_names():
    from selectarena.evaluation.leaderboard import clean_model_names

    models = pd.Series(["TA-RealMLP", "TA-TabICLv2", "TabICLv2", "LightGBM", None])
    cleaned = clean_model_names(models)
    assert cleaned[:4].tolist() == ["RealMLP", "TA-TabICLv2", "TabICLv2", "LightGBM"] and pd.isna(cleaned[4])


def test_interactive_pareto_and_lasagna(artifacts):
    """The Pareto front and the lasagna plots are interactive pages like the Elo figure."""
    data = artifacts.website_data
    meta = json.loads((data / "meta.json").read_text())
    pareto = (data / "subsets" / Subset().rel_path / "pareto.html").read_text()
    front = pd.read_csv(data / "subsets" / Subset().rel_path / "pareto.csv")
    shown = front[~front["method"].isin(["LaplacianScore", "Random"])]  # hidden in the figure, as in the paper
    assert pareto.count("class='sa-hit'") == len(shown)
    assert "Improvement over Random" in pareto and "saSvg('selectarena_pareto')" in pareto
    lasagna = meta["lasagna"]["stability"]
    folder = data / lasagna["path"] / "tasks_all"
    long = pd.read_csv(folder / "per_epv.csv")
    for variant in meta["lasagna"]["variants"]:
        page = (folder / f"{variant}.html").read_text()
        assert page.count("class='sa-hit'") == len(long)  # one tooltip per method and EPV
        assert "Datasets: " in page or "Dataset: " in page
        assert ("data:image/png;base64," in page) == (not variant.startswith("binary_"))
    # Downloads use the light colours, whatever the page's theme.
    assert "var SA_LIGHT" in pareto and "getComputedStyle(document.documentElement)" not in pareto


def test_analysis_figures(artifacts):
    data = artifacts.website_data
    analysis = json.loads((data / "meta.json").read_text())["analysis"]
    keys = [g["key"] for g in analysis["groupings"]]
    # Dataset sizes come from the extra results; groupings with a single group are left out.
    assert {"task_type", "samples", "model", "budget"} <= set(keys)
    folder = data / analysis["path"]
    for key in keys:
        page = (folder / f"heatmap_{key}.html").read_text()
        assert "id='sa-figure'" in page and ">All<" in page
    table = pd.read_csv(folder / "heatmap.csv")
    assert set(table["grouping"]) == {g["label"] for g in analysis["groupings"]}
    assert analysis["improvement_by_model"] and (folder / "improvement_by_model.html").is_file()
    assert analysis["improvement_by_task_type"] == (folder / "improvement_by_task_type.html").is_file()
    # Removed figures (per domain, selection time, stability and validity per domain and size) stay out.
    assert not {"improvement_by_domain", "improvement_by_data_foundry_domain", "selection_time", "stability"} & set(
        analysis
    )
    assert not any(folder.glob("*domain*")) and not (folder / "selection_time.html").exists()
    assert analysis["elo_without_timeouts"] == (folder / "elo_no_timeouts.html").is_file()
    # The rank correlation needs at least five datasets (the fixture has fewer).
    assert analysis["rank_by_size"] == (folder / "rank_by_size.html").is_file()


def test_leaderboard_without_timeouts(tmp_path):
    """Lasso runs out of time (random fallback) on abalone: the Elo figure of the other datasets is written."""
    df = results()
    timed_out = df["experiment_method_name_string"].str.contains("LassoFeatureSelector") & (
        df["name"] == "abalone/0199-uuid"
    )
    df.loc[timed_out, "random_feature_names"] = "['f3']"
    artifacts = generate_website_artifacts(df, out_dir=tmp_path, n_bootstrap=2, paper_figures=False, log=lambda _: None)
    analysis = json.loads((artifacts.website_data / "meta.json").read_text())["analysis"]
    assert analysis["elo_without_timeouts"] is True
    folder = artifacts.website_data / analysis["path"]
    leaderboard = pd.read_csv(folder / "leaderboard_no_timeouts.csv")
    assert len(leaderboard) == len(METHODS) and set(leaderboard["#Datasets"]) == {3}
    page = (folder / "elo_no_timeouts.html").read_text()
    assert "id='sa-figure'" in page and "3 datasets where no method timed out" in page
    # Without any time-out, there is no such figure.
    plain = generate_website_artifacts(
        results(), out_dir=tmp_path / "plain", n_bootstrap=2, paper_figures=False, log=lambda _: None
    )
    assert json.loads((plain.website_data / "meta.json").read_text())["analysis"]["elo_without_timeouts"] is False
    assert not (plain.website_data / "analysis" / "elo_no_timeouts.html").exists()
