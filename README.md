# SelectArena

SelectArena is a benchmark for feature selection on tabular data. Each feature selection method
selects a budgeted number of features, and the selection is judged by the downstream performance of
strong tabular models trained on the selected features, plus a selection-only stability and
validity benchmark. Paper: [arXiv:2610.13644](https://arxiv.org/abs/2610.13644).

SelectArena depends on [TabArena](https://github.com/autogluon/tabarena) for datasets, models,
the experiment runner, the validation protocol and the SLURM tooling. This repository only
contains what is specific to feature selection.

## How SelectArena plugs into TabArena

Feature selectors are AutoGluon feature generators (`selectarena.methods.AbstractFeatureSelector`)
that run as the last post generator of AutoGluon's preprocessing, before the model is fitted. A
run is named by a preprocessing pipeline string,
`FSBench__<method>__<total_budget>__<budget_index>__<proxy_model>__<fs_time>`, optionally with the
`__nofallback` and `__permenc<seed>` ablation tokens.

TabArena is used unmodified. The integration lives in `selectarena.benchmark` as subclasses:

| TabArena class | SelectArena subclass | What it adds |
|---|---|---|
| `AGModelBagExperiment` | `SelectArenaBagExperiment` | Resolves `FSBench__*` pipelines (attaches the selector, sets `time_limit_preprocessing`) |
| `AGSingleBagWrapper` | `SelectArenaBagWrapper` | Passes the per-split seed to the selector; collects the selector's metadata after the fit |
| `OOFExperimentRunner` | `SelectArenaExperimentRunner` | Stores that metadata in `results["preprocessing_artifacts"]["feature_selection"]` |
| `TabArenaV0pt1ExperimentBundle` | `SelectArenaExperimentBundle` | Builds the experiments above; 25 random configs by default |

The classes are referenced by import path in TabArena's YAML job files, so jobs shipped to
cluster nodes run them as long as `selectarena` is installed in the node's environment. This
follows the pattern of [RamanBench](https://github.com/ml-lab-htw/RamanBench), which layers its
domain preprocessing onto TabArena's model wrappers instead of forking TabArena.

## Installation

Python 3.11 to 3.13. TabArena is pinned to one upstream commit (see `pyproject.toml`), and
AutoGluon comes from its pre-release builds, so allow pre-releases:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --prerelease=allow -e ".[models,dev]"
```

With pip, use `pip install --pre -e ".[models,dev]"`. The `models` extra adds the downstream
models of the default benchmark config (TabICLv2, RealMLP).

Because the TabArena pin is a git URL, this package cannot be uploaded to PyPI as is. To publish
it, move the TabArena lines into a separate `requirements-tabarena-git.txt` the way RamanBench
does.

## Quickstart

```bash
python examples/quickstart.py   # every selector inside an AutoGluon TabularPredictor (OpenML dataset 55)
python examples/run_local.py    # a small benchmark through TabArena's runner on a toy task
```

`run_local.py` prints one row per run with the downstream error, the number of selected features
and the selection time.

## Running the benchmark on a SLURM cluster

1. Describe your machine: `cp configs/cluster.example.yaml configs/cluster.yaml` and set the
   workspace, the Python of your environment, the cache directories, the Data Foundry warehouse
   and the SLURM partitions. `configs/cluster.yaml` is git-ignored.
2. Convert the dataset suite into TabArena tasks (once per cluster):
   `python scripts/prepare_datasets.py`
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

## Leaderboard

The public leaderboard is the Hugging Face Space
[SelectArena/Leaderboard](https://huggingface.co/spaces/SelectArena/Leaderboard). As for TabArena,
the Space repo holds the Gradio app and this repo generates the data it serves:

```bash
python scripts/run_benchmark.py eval                # results.csv
python scripts/run_extra_benchmark.py collect       # optional: stability / validity
python scripts/run_generate_website_artifacts.py    # generated_website_artifacts/
```

With result files copied off the cluster, put them in `results/` (git-ignored) and pass them
explicitly; no cluster config is needed then:

```bash
python scripts/run_generate_website_artifacts.py \
    --results results/performance_results.csv --extra-dir results   # stability_results.csv, validity_results.csv
```

Results written by `tabarena_fork_fs` work too: the old class names (`AccuracyFeatureSelector`,
`ANOVAFeatureSelector`) are mapped to LOCO and F-Test, methods that are no longer in the registry
(e.g. `AccuracyLinearFeatureSelector`) are left out, and duplicated stability/validity rows are
counted once.

`generated_website_artifacts/clean_website_artifacts/website_data/` replaces the Space's `data/`
folder: one leaderboard per task type, downstream model and budget (`website_leaderboard.csv`, a
sortable `leaderboard_table.html` and `elo_explorer.html`, the paper's main figure as an interactive
SVG), the per-dataset ranks and the stability / validity tables, all as text. `raw_website_artifacts/`
keeps the main figure as PNG/PDF for the paper. The generator lives in `selectarena/website/`; the
Elo computation (`selectarena/evaluation/elo.py`) ports `performance/baseline/elo_v4.py` from the
paper's plotting code.

**Publishing** is done by the `update-leaderboard` skill (`.claude/skills/update-leaderboard/`):
ask Claude Code to "update the leaderboard" in this repo with the path of a clone of the Space
(`git clone https://huggingface.co/spaces/SelectArena/Leaderboard`). It checks that every method
has a family, regenerates the artifacts, swaps them into the Space's `data/`, adds a version history
entry, runs the Space's smoke test and leaves the commit and `git push` (which deploys) to you. The
manual steps are in the docstring of `scripts/run_generate_website_artifacts.py`.

## Feature selection methods

| Class | Paper name | Proxy model |
|---|---|---|
| `RandomFeatureSelector` | Random | |
| `FTestFeatureSelector` | F-Test | |
| `MIFeatureSelector` | MI | |
| `GainRatioFeatureSelector` | GainRatio | |
| `mRMRFeatureSelector` | mRMR | |
| `ReliefFFeatureSelector` | (R)ReliefF | |
| `LaplacianScoreFeatureSelector` | LaplacianScore | |
| `MarkovBlanketFeatureSelector` | MarkovBlanket | |
| `LassoFeatureSelector` | Lasso | |
| `ElasticNetFeatureSelector` | ElasticNet | |
| `CARTFeatureSelector` | CART | |
| `RFImportanceFeatureSelector` | RFImportance | |
| `LOCOFeatureSelector` | LOCO | yes |
| `SequentialForwardSelectionFeatureSelector` | SFS | yes |
| `SequentialBackwardEliminationFeatureSelector` | SBE | yes |

Methods with a proxy model fit a LightGBM model (configurable through `ProxyModelConfig`) inside
the selection.

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
configs/            benchmark.yaml, cluster.example.yaml
scripts/            prepare_datasets.py, run_benchmark.py, run_extra_benchmark.py,
                    run_generate_website_artifacts.py
.claude/skills/     update-leaderboard: regenerate and publish the leaderboard
examples/           quickstart.py, run_local.py
tests/
```

## Coming from tabarena_fork_fs

The code was moved out of the fork; method implementations are unchanged apart from import paths
and formatting, with one exception listed below.

| Fork | SelectArena |
|---|---|
| `tabarena/tabarena/benchmark/feature_selection_methods/abstract/abstract_feature_selector.py` | `selectarena/methods/abstract.py` |
| `.../feature_selection_methods/ag/<method>/<method>.py` (15 registered methods) | `selectarena/methods/<method>.py` (`random` became `random_baseline`) |
| `.../feature_selection_methods_register.py` | `selectarena/methods/registry.py` |
| `.../feature_selection_benchmark_utils.py` | `selectarena/benchmark/fs_config.py` |
| edits to `experiment_runner.py`, `AutoGluon_class.py`, `abstract_class.py` | subclasses in `selectarena/benchmark/` |
| `experimental/feature_selection_benchmark/data_integration/fs_data_constants.py` | `selectarena/datasets.py` |
| `experimental/.../tabarena_setup/*` (per-user setups and launchers) | `configs/*.yaml`, `scripts/run_benchmark.py` |
| `experimental/.../extra_benchmark/*` | `selectarena/extra_benchmark/`, `scripts/run_extra_benchmark.py` |
| `tabflow_slurm/simple_evaluation/minimal_read_benchmark_results.py` | `selectarena/evaluation/read_results.py` |
| `tst/benchmark/feature_selection_methods/fs_test.py` | `tests/test_selectors.py` |

Not carried over: the legacy method tree `tabarena/benchmark/feature_selection_methods/`, the
unregistered methods (CFS, CMIM, Consistency, DISR, Information Gain, INTERACT) and `ag_old/`,
`experimental/.../!old/`, `examples/running_tabarena_feature_selection_methods/`, the custom-model
example, the tests that depended on the legacy tree, and the local workarounds in
`model_registry.py` and `metrics/_roc_auc_cpp`.

Behavior changes:

- `LassoFeatureSelector` wraps its L1 logistic regression in `OneVsRestClassifier` for multiclass
  tasks. scikit-learn before 1.8 did this implicitly for the `liblinear` solver; newer versions
  raise instead.
- The stability/validity runner no longer writes every result twice to its CSV, and its results
  directory is configurable.
- `read_benchmark_results` reads gzipped `results.pkl` files (the format current TabArena writes)
  and results written by the fork.

## Tests

```bash
pytest
```

`tests/test_benchmark_integration.py` runs real (small) fits through TabArena's experiment runner.
