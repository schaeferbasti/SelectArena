# SelectArena

SelectArena is a benchmark for feature selection on tabular data. Each feature selection method
selects a budgeted number of features, and the selection is judged by the downstream performance of
strong tabular models trained on the selected features, plus a selection-only stability and
validity benchmark. The paper is available at [arXiv:2610.13644](https://arxiv.org/abs/2610.13644). We provide a public leaderboard: [SelectArena/Leaderboard](https://huggingface.co/spaces/SelectArena/Leaderboard).


## Installation

SelectArena depends on [TabArena](https://github.com/autogluon/tabarena) for datasets, models,
the experiment runner, the validation protocol and the SLURM tooling.
Python 3.11 to 3.13. TabArena is pinned to one upstream commit (see `pyproject.toml`), and
AutoGluon comes from its pre-release builds, so allow pre-releases:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --prerelease=allow -e ".[models,dev]"
```

With pip, use `pip install --pre -e ".[models,dev]"`. The `models` extra adds the downstream
models of the default benchmark config (TabICLv2, RealMLP).

## Quickstart

```bash
python examples/quickstart.py   # every selector inside an AutoGluon TabularPredictor (OpenML dataset 55)
python examples/run_local.py    # a small benchmark through TabArena's runner on a toy task
```

`run_local.py` prints one row per run with the downstream error, the number of selected features
and the selection time.

## Running the benchmark on a SLURM cluster

1. Describe your machine: `cp configs/cluster.example.yaml configs/cluster.yaml` and set the
   workspace, the Python of your environment, the cache directories and the SLURM partitions.
   `configs/cluster.yaml` is git-ignored.
2. Optional: download the dataset suite and convert it into TabArena tasks in advance with
   `python scripts/prepare_datasets.py`; otherwise `setup` does it (see [Datasets](#datasets)).
3. Choose what to run in `configs/benchmark.yaml` (methods, budgets, time limits, models, splits).
4. On the head node: `python scripts/run_benchmark.py setup`, then launch the printed `sbatch`
   command(s). Rerunning `setup` skips jobs whose results exist.
5. When the jobs are done: `python scripts/run_benchmark.py eval` writes
   `<workspace>/output/<benchmark_name>/results.csv`, one row per result including the selected
   features, their scores and the selection time.

The stability and validity benchmark uses the same configs:

```bash
python scripts/run_extra_benchmark.py setup     # writes SLURM array scripts (add --submit to submit)
python scripts/run_extra_benchmark.py collect   # stability_results.csv, validity_results.csv
```

Both config files can be passed explicitly with `--config` and `--cluster-config`; the cluster
config is also read from `$SELECTARENA_CLUSTER_CONFIG`.

## Running the benchmark on an OAR cluster

With `scheduler: oar` in the cluster config, the same scripts write OAR parametric array jobs
and print `oarsub` commands instead of `sbatch` ones (`--submit` submits them). The jobs, their
bundling and the results are the same as on SLURM. Start from `configs/cluster.oar.example.yaml`.

## Datasets

The suite is 71 curated [Data Foundry](https://github.com/TabArena/data-foundry) containers
(`selectarena.datasets.BENCHMARK_DATA_FOUNDRY_TASKS`), published in the Hugging Face dataset
repository [SelectArena/SelectArena_Datasets](https://huggingface.co/datasets/SelectArena/SelectArena_Datasets). 
They are downloaded on first use: `run_benchmark.py setup` and `run_extra_benchmark.py setup` fetch
the containers into the `data_foundry` cache of the cluster config and convert the ones the run
needs into TabArena tasks in the `openml` cache. `scripts/prepare_datasets.py` does the same for the
whole suite in advance, e.g. on a front end when the compute nodes have no internet access. To read
the containers from a local copy instead (e.g. a Data Foundry warehouse), set
`data_foundry_warehouse` in the cluster config to the directory that holds them as
`<dataset>/<container uuid>/`.

New containers go into the Hugging Face repository first, in the same layout; then add their paths
to `BENCHMARK_DATA_FOUNDRY_TASKS` and update `HF_REVISION` to the repository's new commit.

## Leaderboard

The public leaderboard is the Hugging Face Space
[SelectArena/Leaderboard](https://huggingface.co/spaces/SelectArena/Leaderboard). The Space repo holds the site and this repo generates the data it serves:

```bash
python scripts/run_benchmark.py eval                # results.csv
python scripts/run_extra_benchmark.py collect       # optional: stability / validity
python scripts/run_generate_website_artifacts.py    # generated_website_artifacts/
```

Put the result files in `results/` and pass them explicitly; no cluster config is needed then:

```bash
python scripts/run_generate_website_artifacts.py \
    --results results/performance_results.csv --extra-dir results   # stability_results.csv, validity_results.csv
```

`generated_website_artifacts/clean_website_artifacts/website_data/` replaces the Space's `data/`
folder: one leaderboard per task type, downstream model and budget (`website_leaderboard.csv`, a
sortable `leaderboard_table.html` and `elo_explorer.html`, the paper's main figure as an interactive
SVG, and the Pareto front of improvement over Random vs. selection time as `pareto.svg`), the
paper's lasagna plots of stability and validity per task type (`lasagna/`), the per-dataset ranks,
the stability / validity tables and a filterable results explorer (`results.csv`), all as text. The
Space shows them in three tabs: Performance (Elo figure, Pareto front, leaderboard and the
results per dataset), Stability and Validity.
`raw_website_artifacts/` keeps the main figure as PNG/PDF for the paper. The generator lives in
`selectarena/website/`; the Elo computation (`selectarena/evaluation/elo.py`), the Pareto front
(`pareto.py`) and the lasagna plots (`lasagna.py`) port `elo_v4.py`, `scripts/pareto.py` and
`fsplots/lasagna.py` from the paper's plotting code. The lasagna plots place datasets by events per
variable, which needs the `num_samples` / `min_samples_per_class` columns of the validity results.

**Publishing** is done by the `update-leaderboard` skill (`.claude/skills/update-leaderboard/`):
ask Claude Code to "update the leaderboard" in this repo with the path of a clone of the Space
(`git clone https://huggingface.co/spaces/SelectArena/Leaderboard`). It checks that every method
has a subgroup of the paper's taxonomy, regenerates the artifacts, swaps them into the Space's
`data/`, adds a version history entry, runs the Space's smoke test and a local preview, and leaves
the commit and `git push` (which deploys) to you. The
manual steps are in the docstring of `scripts/run_generate_website_artifacts.py`.

## Feature selection methods

| Class | Paper name | Subgroup (paper, Table 2) | Proxy model |
|---|---|---|---|
| `RandomFeatureSelector` | Random | Baseline | |
| `FTestFeatureSelector` | F-Test | Correlation | |
| `MIFeatureSelector` | MI | Information theory | |
| `GainRatioFeatureSelector` | GainRatio | Information theory | |
| `mRMRFeatureSelector` | mRMR | Information theory | |
| `ReliefFFeatureSelector` | (R)ReliefF | Distance | |
| `LaplacianScoreFeatureSelector` | LaplacianScore | Distance | |
| `MarkovBlanketFeatureSelector` | MarkovBlanket | Backward search | |
| `LassoFeatureSelector` | Lasso | Regularization | |
| `ElasticNetFeatureSelector` | ElasticNet | Regularization | |
| `CARTFeatureSelector` | CART | Tree | |
| `RFImportanceFeatureSelector` | RFImportance | Tree | |
| `LOCOFeatureSelector` | LOCO | Leave-one-out | yes |
| `SequentialForwardSelectionFeatureSelector` | SFS | Forward search | yes |
| `SequentialBackwardEliminationFeatureSelector` | SBE | Backward search | yes |

To add a method, ask Claude Code to "add <method>" in this repo: the `add-method` skill
(`.claude/skills/add-method/`) writes the `AbstractFeatureSelector` subclass from templates,
registers it and verifies it. `python scripts/check_method.py <ClassName>` (or `--all`) checks any
method against the benchmark's contract: registration, output format on binary / multiclass /
regression data, determinism and the selection time limit.

Search methods (forward, backward, leave-one-out) evaluate many feature subsets; non-search methods
use a proxy instead (Appendix A of the paper). Methods with a proxy model fit a LightGBM model
(configurable through `ProxyModelConfig`) inside the selection.

The budget: for `d` features and total budget `b`, budget index `i` (0-based) keeps
`ceil(d * (i+1) / (b+1))` features when `d <= 100` and `ceil(d ** ((i+1) / (b+1)))` otherwise.

## Repository layout

```
src/selectarena/
  methods/          feature selectors, their base class and the registry
  benchmark/        FSBench pipelines and the TabArena subclasses
  extra_benchmark/  stability / validity benchmark (selection only)
  evaluation/       reading results.pkl files into a table, leaderboard tables, Elo
  website/          the leaderboard Space's data (subsets, tables, interactive figure)
  datasets.py       the dataset suite (Data Foundry containers)
  config.py         YAML configs and the tabflow_slurm plan built from them
configs/            benchmark.yaml, cluster.example.yaml (SLURM), cluster.oar.example.yaml (OAR)
scripts/            prepare_datasets.py, run_benchmark.py, run_extra_benchmark.py,
                    run_generate_website_artifacts.py, check_method.py
.claude/skills/     add-method: add a feature selection method;
                    update-leaderboard: regenerate and publish the leaderboard
examples/           quickstart.py, run_local.py
tests/
```

## Tests

```bash
pytest
```

`tests/test_benchmark_integration.py` runs real (small) fits through TabArena's experiment runner.
