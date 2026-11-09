from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from selectarena.evaluation import characteristics as C
from selectarena.website.characteristic_figures import (
    group_improvement_html,
    improvement_heatmap_html,
    method_order,
    rank_sensitivity_html,
    selection_time_html,
)

FEATURES = [10, 15, 25, 60, 150, 400, 900, 2000]
SAMPLES = [200, 500, 900, 2000, 5000, 9000, 20000, 50000]


@pytest.fixture
def data():
    """Eight datasets of growing width. "Wrapper" ranks first on narrow datasets (half the error of
    Random) and last on wide ones (1.5 times its error), "Filter" the other way round; "Random" is
    always in the middle."""
    rows = []
    for d, width in enumerate(FEATURES):
        for split in range(2):
            for model in ["LightGBM", "LinearModel"]:
                for budget in [0, 1]:
                    narrow = width < 100
                    ranks = {"Wrapper": 1 if narrow else 3, "Filter": 3 if narrow else 1, "Random": 2}
                    for method, rank in ranks.items():
                        rows.append(
                            {
                                "dataset_key": f"d{d}",
                                "split_idx": split,
                                "model": model,
                                "budget_index": budget,
                                "problem_type": "binary" if d % 2 else "regression",
                                "method": method,
                                "rank": float(rank),
                                "metric_error": 0.5 * rank,
                            }
                        )
    scores = pd.DataFrame(rows)
    stats = pd.DataFrame(
        {
            "dataset": [f"d{d}" for d in range(len(FEATURES))],
            "num_samples": SAMPLES,
            "num_features": FEATURES,
            "min_samples_per_class": [10 * (d + 1) for d in range(len(FEATURES))],
        }
    )
    problem_types = scores.groupby("dataset_key")["problem_type"].first()
    return scores, C.dataset_characteristics([stats], problem_types)


def test_dataset_characteristics(data):
    _, chars = data
    assert list(chars.columns) == ["num_samples", "num_features", "epv"]
    # Regression datasets count all samples per feature, classification ones the smallest class.
    assert chars.loc["d0", "epv"] == pytest.approx(200 / 10)  # d0: regression
    assert chars.loc["d1", "epv"] == pytest.approx(20 / 15)  # d1: binary
    empty = C.dataset_characteristics([None], pd.Series(dtype=object))
    assert empty.empty and list(empty.columns) == ["num_samples", "num_features", "epv"]


def test_first_source_wins():
    problem_types = pd.Series({"a": "binary"})
    validity = pd.DataFrame(
        {"dataset": ["a"], "num_samples": [100], "num_features": [10], "min_samples_per_class": [40]}
    )
    stability = pd.DataFrame(
        {"dataset": ["a"], "num_samples": [100], "num_features": [10], "min_samples_per_class": [1]}
    )
    chars = C.dataset_characteristics([validity, stability], problem_types)
    assert chars.loc["a", "epv"] == pytest.approx(4.0)


def test_improvement_by_grouping(data):
    results, chars = data
    with_chars = C.add_characteristics(results, chars)
    keys = [g.key for g in C.available_groupings(with_chars)]
    assert keys == ["task_type", "samples", "features", "epv", "model", "budget"]

    features = next(g for g in C.GROUPINGS if g.key == "features")
    table = C.improvement_by_grouping(with_chars, features)
    assert list(dict.fromkeys(table["group"])) == ["All", "< 20", "20–99", "≥ 100"]
    cell = table.set_index(["group", "method"])["improvement"]
    assert cell[("< 20", "Wrapper")] == pytest.approx(50) and cell[("≥ 100", "Wrapper")] == pytest.approx(-50)
    assert cell[("All", "Wrapper")] == pytest.approx(0) and cell[("All", "Random")] == 0  # 4 narrow, 4 wide
    assert table[table["group"] == "≥ 100"]["method"].tolist() == ["Filter", "Random", "Wrapper"]  # best first
    # Bins include their lower edge: 25 and 60 features are "20-99", 150 and more "≥ 100".
    assert table.groupby("group", sort=False)["n_datasets"].first().to_dict() == {
        "All": 8,
        "< 20": 2,
        "20–99": 2,
        "≥ 100": 4,
    }
    budget = C.improvement_by_grouping(with_chars, next(g for g in C.GROUPINGS if g.key == "budget"))
    assert list(dict.fromkeys(budget["group"])) == ["All", "1 (smallest)", "2 (largest)"]
    task = C.improvement_by_grouping(with_chars, next(g for g in C.GROUPINGS if g.key == "task_type"))
    assert list(dict.fromkeys(task["group"])) == ["All", "Binary", "Regression"]


def test_size_groupings_need_the_dataset_sizes(data):
    scores, _ = data
    no_sizes = C.add_characteristics(scores, C.dataset_characteristics([None], pd.Series(dtype=object)))
    assert [g.key for g in C.available_groupings(no_sizes)] == ["task_type", "model", "budget"]


def test_rank_sensitivity(data):
    scores, chars = data
    sens = C.rank_sensitivity(scores, chars, n_bootstrap=200).set_index(["characteristic", "method"])
    assert set(sens.index.get_level_values(0)) == {"Samples", "Features", "Samples per feature"}
    wrapper, filt = sens.loc[("Features", "Wrapper")], sens.loc[("Features", "Filter")]
    assert wrapper["rho"] < -0.8 and wrapper["ci_high"] < 0  # worse on wider datasets
    assert filt["rho"] > 0.8 and filt["ci_low"] > 0
    assert np.isnan(sens.loc[("Features", "Random"), "rho"])  # constant rank: no correlation
    assert (sens["n_datasets"] == 8).all()


def test_selection_time():
    df = pd.DataFrame(
        {
            "dataset_key": ["a", "a", "b", "b", "c", "c"],
            "method": ["SFS", "Lasso"] * 3,
            "feature_selection_fit_time": [10.0, 0.1, 3600.0, 0.5, 3600.0, 1.0],
            "feature_selection_time_limit": [3600] * 6,
        }
    )
    chars = pd.DataFrame({"num_features": [10, 100, 1000]}, index=pd.Index(["a", "b", "c"], name="dataset"))
    times, limit = C.selection_time(df, chars)
    assert limit == 3600 and len(times) == 6
    binned = C.binned_time(times, n_bins=3)
    sfs = binned[binned["method"] == "SFS"].sort_values("features_mid")
    assert sfs["features_mid"].tolist() == [10, 100, 1000] and sfs["time_s"].tolist() == [10, 3600, 3600]
    assert (binned["n_datasets"] == 1).all()


def test_figures(data):
    scores, chars = data
    with_chars = C.add_characteristics(scores, chars)
    table = C.improvement_by_grouping(with_chars, next(g for g in C.GROUPINGS if g.key == "features"))
    order = method_order(table)
    assert order[0] in {"Wrapper", "Filter"} and order[-1] in {"Wrapper", "Filter", "Random"}
    heatmap = improvement_heatmap_html(table, "Features", title="t", name="heatmap_features")
    assert heatmap.count("class='sa-hit'") == 4 * 3  # one per group and method
    assert "Improvement over Random +50.0%: place 1 of 3 (best)" in heatmap
    sens = C.rank_sensitivity(scores, chars, n_bootstrap=50)
    by_size = rank_sensitivity_html(sens, order, title="t")
    assert "Ranks worse with more features" in by_size and "Ranks better with more features" in by_size
    df = scores.assign(feature_selection_fit_time=1.0, feature_selection_time_limit=60.0)
    times, limit = C.selection_time(df, chars)
    timing = selection_time_html(C.binned_time(times), limit, order, title="t")
    assert "Time limit (60 s)" in timing
    for page in (heatmap, by_size, timing):
        svg = page.split("<svg", 1)[1].split("</svg>", 1)[0]
        # Theme colours only through classes: var() does not work in SVG attributes, and the
        # downloads replace var() in the figure's <style> only.
        assert "var(--" not in svg.split("</style>", 1)[1]


def test_improvement_by_model_and_task_type():
    rows = []
    for d in range(4):
        task = "regression" if d < 2 else "binary"
        for model, gain in [("LightGBM", 0.5), ("LinearModel", 0.8)]:
            for method, error in [("Random", 1.0), ("Good", gain), ("Bad", 0.9)]:
                rows.append(
                    {"dataset": f"d{d}", "model": model, "problem_type": task, "method": method, "metric_error": error}
                )
    df = pd.DataFrame(rows)
    by_model = C.improvement_by(df, "model")
    table = by_model.set_index(["group", "method"])["improvement"]
    assert table[("LightGBM", "Good")] == pytest.approx(50) and table[("LinearModel", "Good")] == pytest.approx(20)
    assert table[("All", "Good")] == pytest.approx(35) and table[("All", "Random")] == 0
    assert C.ranking_agreement(by_model) == pytest.approx(1.0)
    page = group_improvement_html(by_model, ["Good", "Bad", "Random"], 1.0, title="t", name="n")
    assert "LinearModel: +20.0%" in page and "between the downstream models: mean Spearman ρ = 1.00" in page

    by_task = C.improvement_by(df, "problem_type", C.TASK_TYPE_LABELS)
    assert list(dict.fromkeys(by_task["group"])) == ["All", "Binary", "Regression"]  # in the labels' order
    page = group_improvement_html(
        by_task, ["Good"], None, title="t", name="n", all_label="All task types", agreement_label="the task types"
    )
    assert "All task types: +35.0%" in page and "Spearman" not in page


def test_domain_labels():
    domains = {"a": "finance", "b": "finance", "c": "finance", "d": "medical & healthcare", "e": "education"}
    labels = C.domain_labels(["a", "b", "c", "d", "e", "f"], domains, min_datasets=2)
    # Small domains together as "Other" (last), the number of datasets in the label; "f" has no domain.
    assert labels.astype(str).to_dict() == {
        "a": "Finance (3)",
        "b": "Finance (3)",
        "c": "Finance (3)",
        "d": "Other (2)",
        "e": "Other (2)",
    }
    assert list(labels.cat.categories) == ["Finance (3)", "Other (2)"]


def test_every_benchmark_dataset_has_a_domain():
    from selectarena.datasets import (
        BENCHMARK_DATA_FOUNDRY_TASKS,
        DATA_FOUNDRY_DOMAINS,
        DATASET_DOMAINS,
        PAPER_DOMAINS,
    )

    assert {path.split("/")[0] for path in BENCHMARK_DATA_FOUNDRY_TASKS} == set(DATASET_DOMAINS)
    assert set(DATASET_DOMAINS.values()) == set(PAPER_DOMAINS)  # the paper's domains, each used
    assert set(DATA_FOUNDRY_DOMAINS) == set(DATASET_DOMAINS)
    assert DATA_FOUNDRY_DOMAINS["framingham_heart_study"] == "medical & healthcare"  # corrected metadata


def test_stability_by_domain_and_size(data):
    _, chars = data
    # "Steady" is equally stable everywhere; "Grows" gets more stable the more features a dataset has.
    rows = pd.DataFrame(
        [
            {"dataset": f"d{d}", "method": method, "budget_index": b, "stability": value}
            for d, width in enumerate(FEATURES)
            for b in range(2)
            for method, value in [("Steady", 0.8), ("Grows", 0.1 + 0.1 * d)]
        ]
    )
    groups = C.domain_labels(
        [f"d{d}" for d in range(8)], {f"d{d}": "finance" if d < 4 else "physics" for d in range(8)}
    )
    table = C.measure_by(rows, "stability", groups).set_index(["group", "method"])["stability"]
    assert table[("All", "Steady")] == pytest.approx(0.8)
    assert table[("Finance (4)", "Grows")] == pytest.approx(0.25) and table[("Physics (4)", "Grows")] == pytest.approx(
        0.65
    )
    sens = C.measure_sensitivity(rows, "stability", chars, n_bootstrap=100).set_index(["characteristic", "method"])
    assert (
        sens.loc[("Features", "Grows"), "rho"] == pytest.approx(1.0) and sens.loc[("Features", "Grows"), "ci_low"] > 0
    )
    assert np.isnan(sens.loc[("Features", "Steady"), "rho"])
    page = rank_sensitivity_html(
        sens.reset_index(),
        ["Grows", "Steady"],
        title="t",
        better="more stable",
        worse="less stable",
        direction="higher",
    )
    assert "More stable with more features" in page and "← higher with fewer" in page
    page = group_improvement_html(
        table.rename("stability").reset_index(),
        ["Steady", "Grows"],
        None,
        title="t",
        name="n",
        all_label="All domains",
        value="stability",
        axis_label="Mean stability",
        unit="",
    )
    assert "Physics (4): 0.650" in page and "Mean stability" in page
