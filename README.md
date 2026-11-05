<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner-light.svg" alt="SelectArena: A modern benchmark for feature selection on real-world tabular data" width="760">
  </picture>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.13644"><img src="https://img.shields.io/badge/arXiv-2610.13644-b31b1b.svg" alt="arXiv paper"></a>
  <a href="https://huggingface.co/spaces/SelectArena/Leaderboard"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Leaderboard-SelectArena-0072b2" alt="Leaderboard"></a>
  <a href="https://huggingface.co/datasets/SelectArena/SelectArena_Datasets"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-SelectArena-0072b2" alt="Datasets"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.13644"><b>Paper</b></a> ·
  <a href="https://huggingface.co/spaces/SelectArena/Leaderboard"><b>Leaderboard</b></a> ·
  <a href="https://huggingface.co/datasets/SelectArena/SelectArena_Datasets"><b>Datasets</b></a> ·
  <a href="#add-your-method-to-selectarena"><b>Add your method</b></a>
</p>

SelectArena is a **living benchmark** for feature selection on real-world tabular IID data. Each feature selection
method selects a budgeted number of features, and the selection is judged by the downstream performance of tabular
models trained on the selected features, plus a stability and validity benchmark that captures the interpretability
of the methods. The [leaderboard](https://huggingface.co/spaces/SelectArena/Leaderboard) is updated as new methods
are added.

## Installation

SelectArena depends on [TabArena](https://github.com/autogluon/tabarena). All commands in this
README run from the repository root (the folder with `pyproject.toml`).
```bash
git clone https://github.com/schaeferbasti/SelectArena.git && cd SelectArena
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --prerelease=allow -e ".[models,dev]"
```

No `uv` on the machine (e.g. on a cluster)? Install it into your home folder with
`curl -LsSf https://astral.sh/uv/install.sh | sh`, or use pip:
`python3.12 -m venv .venv && source .venv/bin/activate && pip install --pre -e ".[models,dev]"`.
The `models` extra adds the downstream models of the default benchmark config (TabICLv2, RealMLP)
and the search spaces of their random configs.

## Quickstart

```bash
python examples/quickstart.py   # every selector inside an AutoGluon TabularPredictor (OpenML dataset 55)
python examples/run_local.py    # a small benchmark through TabArena's runner on a toy task
```

`run_local.py` prints one row per run with the downstream error, the number of selected features
and the selection time.

## Add your method to SelectArena

1. **Add it:** Use "/add-method" with an agent in this
   repo. It writes the method, registers it and checks that it works with the benchmark (you can
   rerun the check with `python scripts/check_method.py <YourMethod>`).
2. **Benchmark it:** see [Running the benchmark](#running-the-benchmark).
3. **See it next to the other methods** on your local copy of the leaderboard (see
   [Your method on the leaderboard, locally](#your-method-on-the-leaderboard-locally)).
4. **Open a pull request:** we check it and add it to the public leaderboard.

## Running the benchmark

The benchmark runs as array jobs on a SLURM or OAR cluster.

```bash
cp configs/cluster.example.yaml configs/cluster.yaml   # once: paths, Python, partitions (see its comments)
python scripts/run_benchmark.py setup                  # prints the sbatch commands to launch
python scripts/run_benchmark.py eval                   # when the jobs are done: results.csv
python scripts/run_extra_benchmark.py setup            # stability and validity (add --submit to submit)
python scripts/run_extra_benchmark.py collect          # stability_results.csv, validity_results.csv
```

`configs/benchmark.yaml` sets what runs (methods, budgets, models, splits); keep the published
values to compare with the leaderboard. Rerunning `setup` skips jobs whose results exist. Results
land in `<workspace>/output/<benchmark_name>/`. On an OAR cluster, start from
`configs/cluster.oar.example.yaml` instead: the same scripts print `oarsub` commands.

## Leaderboard

The [public leaderboard](https://huggingface.co/spaces/SelectArena/Leaderboard) holds the site and this repo generates the data it serves:

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
folder.
`raw_website_artifacts/` keeps the main figure as PNG/PDF for the paper. The generator lives in
`selectarena/website/`; the Elo computation (`selectarena/evaluation/elo.py`), the Pareto front
(`pareto.py`) and the lasagna plots (`lasagna.py`) are in `selectarena/evaluation/`. All figures on
the website are interactive SVG pages (`website/html.py`, `website/figures.py`) that follow its
light or dark theme; their SVG / PNG downloads always use the light paper look.

### Your method on the leaderboard, locally

You do not need the maintainers' result files to see a new method next to the published ones. Benchmark
your method (see [Running the benchmark](#running-the-benchmark)), then:

```bash
python scripts/run_generate_website_artifacts.py --with-published-results \
    --results <run dir>/results.csv --extra-dir <folder with stability_results.csv, validity_results.csv>
python scripts/preview_leaderboard.py      # the whole leaderboard website at http://localhost:8000
```

`--with-published-results` downloads the published results of all methods from
[SelectArena/SelectArena_Results](https://huggingface.co/datasets/SelectArena/SelectArena_Results) into
`results/` (several GB, only when they are not there yet) and adds your results to them; your results
replace the published ones of the same method on the same dataset (e.g. a rerun). The leaderboard
compares methods on the same dataset splits, so run your method with the benchmark config of the
published results (`configs/benchmark.yaml`). `preview_leaderboard.py` downloads the website's page
from the Space (not its data) and serves it with your generated data; `--site <clone of the Space>`
uses a local clone instead.

To get your method on the public leaderboard, open a pull request with it; the maintainers then
run it and publish the updated leaderboard.

## Repository layout

```
src/selectarena/
  methods/          feature selectors, their base class and the registry
  benchmark/        FSBench pipelines and the TabArena subclasses
  extra_benchmark/  stability / validity benchmark (selection only)
  evaluation/       reading results.pkl files into a table, leaderboard tables, Elo
  website/          the leaderboard Space's data (subsets, tables, interactive figures) and its local preview
  datasets.py       the dataset suite (Data Foundry containers)
  published_results.py  the published results (download, merge with your own)
  config.py         YAML configs and the tabflow_slurm plan built from them
configs/            benchmark.yaml, cluster.example.yaml (SLURM), cluster.oar.example.yaml (OAR)
scripts/            prepare_datasets.py, run_benchmark.py, run_extra_benchmark.py,
                    run_generate_website_artifacts.py, preview_leaderboard.py, check_method.py
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
