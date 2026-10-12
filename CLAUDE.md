# CLAUDE.md

SelectArena: a feature selection benchmark for tabular data that depends on TabArena
(`autogluon/tabarena`, pinned by commit in `pyproject.toml`). See README.md for the user-facing
overview.

## Architecture

- `src/selectarena/methods/`: one module per feature selector, each an `AbstractFeatureSelector`
  (an AutoGluon `AbstractFeatureGenerator`). A scoring method sets `feature_scoring_method = True`
  and implements `_fit_feature_scoring`; a subset method implements `_fit_feature_selection`.
  `registry.py` maps class names to modules and lists the proxy-model methods.
- `src/selectarena/benchmark/`: the `FSBench__*` pipeline strings (`fs_config.py`) and the
  TabArena subclasses (`experiment.py`, `wrapper.py`, `runner.py`). TabArena is never patched.
  Anything SelectArena needs from TabArena's internals goes through a subclass here.
- `src/selectarena/config.py`: YAML configs to a `tabflow_slurm.TabArenaBenchmarkPlan`.
- `src/selectarena/extra_benchmark/`: stability/validity runs, plain SLURM arrays.

## Adding a feature selection method

1. Add `src/selectarena/methods/<name>.py` with a subclass of `AbstractFeatureSelector`
   (or `AbstractITFeatureSelector` for information-theoretic methods) and a `name` attribute.
2. Register it in `methods/registry.py` (`NAME_TO_MODULE_MAP`, `PAPER_NAMES`, and
   `FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL` if it fits a proxy model).
3. `tests/test_selectors.py` picks it up automatically for binary, multiclass and regression.

## Working style

- Run `ruff check --fix . && ruff format .` before committing; every file needs
  `from __future__ import annotations`.
- Run `pytest`. The integration test fits small models through TabArena's runner (about a minute).
- Keep machine-specific values (paths, partitions) out of the code; they belong in
  `configs/cluster.yaml` (git-ignored).
- When bumping the TabArena pin, update every SHA in `pyproject.toml` and rerun the full test
  suite: the subclasses override `_build_predictor_args`, `_apply_preprocessing` and
  `post_evaluate`, which are not public API.
