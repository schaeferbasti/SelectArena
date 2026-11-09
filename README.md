<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner-light.svg" alt="SelectArena: A modern benchmark for feature selection on real-world tabular data" width="760">
  </picture>
</p>

<p align="center">
  <a href="https://anonymous.4open.science/w/selectarena/leaderboard/"><b>Leaderboard</b></a> ·
  <a href="#add-your-method-to-selectarena"><b>Add your method</b></a>
</p>

SelectArena is a **living benchmark** for feature selection on real-world tabular IID data. Each feature selection
method selects a budgeted number of features, and the selection is judged by the downstream performance of tabular
models trained on the selected features, plus a stability and validity benchmark that captures the interpretability
of the methods. The [leaderboard](https://anonymous.4open.science/w/selectarena/leaderboard/) is updated as new methods
are added.

## Installation

SelectArena depends on [TabArena](https://github.com/autogluon/tabarena). All commands in this
README run from the repository root (the folder with `pyproject.toml`).
```bash
# download the repository from https://anonymous.4open.science/r/selectarena and cd into it
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

The [public leaderboard](https://anonymous.4open.science/w/selectarena/leaderboard/) is generated from
the benchmark results with `scripts/run_generate_website_artifacts.py`.

### Your method on the leaderboard, locally

To see your method next to the published ones, benchmark it (see
[Running the benchmark](#running-the-benchmark)), then:

```bash
python scripts/run_generate_website_artifacts.py --with-published-results \
    --results <run dir>/results.csv --extra-dir <folder with stability_results.csv, validity_results.csv>
python scripts/preview_leaderboard.py      # the whole leaderboard website at http://localhost:8000
```

`--with-published-results` downloads the published results of all methods from
ANONYMIZED/SelectArena_Results into
`results/` (several GB, only once) and adds yours; your results replace the published ones of the
same method on the same dataset (e.g. a rerun). Methods are compared on the same dataset splits, so
run your method with the published `configs/benchmark.yaml`.

## Repository layout

```
src/selectarena/
  methods/          feature selectors, their base class and the registry
  benchmark/        the benchmark pipeline on top of TabArena
  extra_benchmark/  stability / validity benchmark (selection only)
  evaluation/       results tables and the leaderboard metrics (Elo, stability, validity)
  website/          the leaderboard Space's data (subsets, tables, interactive figures) and its local preview
  datasets.py       the dataset suite (Data Foundry containers)
  published_results.py  the published results (download, merge with your own)
  config.py         YAML configs and the tabflow_slurm plan built from them
configs/            benchmark.yaml, cluster.example.yaml (SLURM), cluster.oar.example.yaml (OAR)
scripts/            prepare_datasets.py, run_benchmark.py, run_extra_benchmark.py,
                    run_generate_website_artifacts.py, preview_leaderboard.py, check_method.py
.claude/skills/     add-method: add a feature selection method;
                    update-leaderboard: publish new results (maintainers)
examples/           quickstart.py, run_local.py
tests/
```

## Tests

```bash
pytest
```

The integration test (`tests/test_benchmark_integration.py`) runs small real fits and takes about a minute.
