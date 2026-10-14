---
name: add-method
description: Add a new feature selection method to SelectArena. Use whenever someone wants to integrate a feature selection method into the benchmark, e.g. "add Boruta", "implement X feature selection", "integrate this method from <paper/repo>", "support <library>'s selector". Writes the method module (an AbstractFeatureSelector subclass), registers it (registry, paper name, taxonomy subgroup, proxy-model list), adds it to the README, and verifies it with scripts/check_method.py and the selector tests. Ends with how to benchmark it; publishing results is the update-leaderboard skill.
argument-hint: <MethodName> [<paper-or-code-url>] [<pip-package>]
user-invocable: true
---

# Add a feature selection method to SelectArena

Every method is **one module** in `src/selectarena/methods/<module>.py` with one subclass of
`AbstractFeatureSelector` (an AutoGluon feature generator), plus entries in
`src/selectarena/methods/registry.py`. Everything else picks the method up from the registry: the
`FSBench__<Class>__...` pipelines (`methods: all` in `configs/benchmark.yaml`), the stability /
validity benchmark, `tests/test_selectors.py` (fits it on binary, multiclass and regression data),
`scripts/check_method.py` and the leaderboard (paper name and taxonomy subgroup).

Lifecycle: **add-method** (this skill) → benchmark the method (Step 8) → **update-leaderboard**.

The base class contract and copy-ready templates are in `references/method_patterns.md`; read it
before writing code. The base class itself is `src/selectarena/methods/abstract.py`.

## Step 0: Gather inputs

Parse `$ARGUMENTS`; ask only for what is missing or ambiguous.

| Input | Example | Notes |
|---|---|---|
| `MethodName` | `Boruta` | The name used in the paper / on the leaderboard (`PAPER_NAMES`). Unique, no ` [` and no `__`. |
| `ClassName` | `BorutaFeatureSelector` | CamelCase + `FeatureSelector`; no `__` (it is the pipeline-string separator). |
| `module` | `boruta` | snake_case file name in `src/selectarena/methods/`. |
| reference | paper + year | Goes into the class docstring (`Reference:` line, as in the other methods). |
| implementation source | library, paper pseudo-code, own code | Determines Step 1 and whether a new dependency is needed. |
| `pip_package` | `boruta>=0.4` | Only if a library is wrapped. |
| kind | scoring / subset | Step 1 decides; ask if the description allows both. |
| subgroup | `Tree`, `Forward search`, ... | The paper's taxonomy (Step 1b). Confirm with the maintainer. |

## Step 1: Understand the method

### 1a. How the method works

If a URL was given, fetch it (WebFetch) and pin down: the criterion it optimizes, whether it
**scores** features (ranks them; the top `max_features` are kept) or **selects a subset** directly,
whether it **refits a model on candidate subsets** (then it needs the proxy model), its own
randomness, and its cost in the number of features and samples (the time limit, Step 3).

- Scoring method → `feature_scoring_method = True`, implement `_fit_feature_scoring` → dict
  feature → score (higher is better; may be partial when the time runs out).
- Subset method → implement `_fit_feature_selection` → ordered list of
  `{"feature": name, "value": float}` (value `nan` when the method has no per-feature number).
- Model refits on candidate subsets → use `self.evaluate_proxy_model(X=..., y=..., time_limit=...)`
  and list the class in `FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL` (LightGBM by default).
- A library selector that fits its own model internally (e.g. a random-forest importance) is not a
  proxy-model method; fit it directly, seeded with `self.random_state`.

### 1b. Taxonomy subgroup (paper Table 2, Appendix A)

`METHOD_SUBGROUPS` holds each method's subgroup, `SUBGROUP_CATEGORIES` each subgroup's category:

- **Search** (evaluates many subsets with a model-dependent utility): `Forward search` (SFS),
  `Backward search` (SBE, MarkovBlanket), `Leave-one-out` (LOCO). Appendix A names evolutionary
  algorithms and reinforcement learning as further search subgroups.
- **Non-search** (a proxy instead of repeated subset evaluation): `Correlation` (F-Test),
  `Distance` ((R)ReliefF, LaplacianScore), `Information theory` (MI, mRMR, GainRatio), `Tree`
  (CART, RFImportance), `Regularization` (Lasso, ElasticNet). Appendix A names game-theoretic and
  knowledge-driven approaches as further non-search subgroups.

Reuse an existing subgroup when it fits; a new one also needs a `SUBGROUP_CATEGORIES` entry.
Confirm the choice with the maintainer: it is what the leaderboard shows in its Type column.

## Step 2: Read the closest existing method

| The new method is ... | Read |
|---|---|
| a univariate statistic / library score | `methods/f_test.py`, `methods/mi.py` |
| information-theoretic (entropy, MI, discretized data) | `methods/gain_ratio.py`, `methods/mrmr.py` (`AbstractITFeatureSelector`) |
| model-based (importances, coefficients) | `methods/rf_importance.py`, `methods/cart.py`, `methods/lasso.py` |
| distance / neighbourhood based | `methods/relief_f.py`, `methods/laplacian_score.py` |
| a search with a proxy model | `methods/sfs.py` (forward), `methods/sbe.py` (backward), `methods/loco.py` (scores via refits) |
| a greedy subset method without a model | `methods/mrmr.py`, `methods/markov_blanket.py` |

Read it with the Read tool and adapt it; do not copy it blindly.

## Step 3: Write `src/selectarena/methods/<module>.py`

Start from the matching template in `references/method_patterns.md`. The rules the checks enforce
or the benchmark relies on:

- `from __future__ import annotations` first; a class docstring with the method, its
  `Reference:` and, for wrapped code, the `Implementation:` source and licence.
- `name = "<ClassName>"`, equal to the class name (the registry and the pipeline strings use it).
- Inputs: `X` is the training data of one fold after AutoGluon's preprocessing (numeric and
  `category` columns, missing values possible); `y` the labels. Use `self._preprocess(...)`
  (`impute`, `encode_ordinal`, `discretize`, `scale`, `encode_target`) instead of ad-hoc encoding,
  so the ordinal-encoding ablation (`encoding_permutation_seed`) applies to the new method too.
- Available state: `self.max_features` (already resolved), `self.problem_type`, `self._original_features`,
  `self.random_state` / `self._rng` (seeded per split), `self.proxy_mode_config`.
- **Randomness** only through `self.random_state` / `self._rng`; never the global numpy / random
  state (the determinism check fails otherwise).
- **Time limit**: the benchmark gives every method a fixed selection time (`fs_time_limits`) and
  fills the budget at random when the method returns fewer features. Check
  `self._timed_out(time_limit, start_time)` inside every loop over features or candidates, iterate
  over `self._seeded_feature_order()` when a timeout truncates the loop, pass the remaining time to
  `evaluate_proxy_model`, and catch `TimeLimitExceeded` around proxy fits so partial results survive.
  A one-shot library call that cannot be interrupted is acceptable if it is fast; say so in the docstring.
- Never return `None` values yourself (reserved for the random fallback), never return more than
  `max_features` entries from a subset method, and return feature names from `X.columns`.
- Optional third-party imports go inside the fit method, so `import selectarena.methods` never
  needs them.

## Step 4: Register the method

Read `src/selectarena/methods/registry.py`, then add the class in one pass:

1. `NAME_TO_MODULE_MAP["<ClassName>"] = "<module>"`
2. `PAPER_NAMES["<ClassName>"] = "<MethodName>"`
3. `METHOD_SUBGROUPS["<ClassName>"] = "<Subgroup>"` (and `SUBGROUP_CATEGORIES` for a new subgroup)
4. `FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL` if it calls `evaluate_proxy_model`

A new library dependency goes into `dependencies` in `pyproject.toml` (every method runs in the
benchmark, so it is not an optional extra), with a lower bound and a comment naming the method.

## Step 5: Documentation

- Add a row to the methods table in `README.md` (`Class | Paper name | Subgroup | Proxy model`).
- The leaderboard Space's About tab (`index.html` in the Space repo) lists the methods per
  subgroup. If a Space checkout is available (e.g. `../SelectArenaLeaderboard`), add the method
  there too; otherwise list it as a TODO in the report.

## Step 6: Verify (Claude runs)

```bash
python scripts/check_method.py <ClassName>                 # registry, naming, fits, determinism, time limit
pytest tests/test_selectors.py -k <ClassName>              # fit inside TabularPredictor, 3 task types
pytest tests/test_website.py tests/test_check_method.py    # taxonomy and checker tests
ruff check --fix . && ruff format .
```

`check_method.py` must end with `1/1 methods pass`. A `WARN` on `signal` (the method found no
more informative features than a random pick on the synthetic data) is not a failure, but look at
it: either the method is weak on that data or it ranks features the wrong way round (lower score
= better). When a reference implementation exists (the library the method comes from, or the
paper's code), compare the selection on one dataset and report the overlap.

## Step 7: Smoke-run on real data (optional, Claude runs)

```bash
python examples/quickstart.py <ClassName>     # OpenML dataset 55 inside a bagged TabularPredictor
```

It prints the downstream leaderboard, the selected features and the scores. Needs network access
to OpenML on the first run.

## Step 8: Benchmark it (hand off: cluster)

The default config runs `methods: all`, so a full run includes the new method automatically. To run
only the new method and add it to existing results:

1. `cp configs/benchmark.yaml configs/benchmark_<module>.yaml`; set `benchmark_name:
   selectarena_<module>` and `feature_selection.methods: [<ClassName>]`; keep everything else
   (budgets, `fs_time_limits`, models, splits) **identical to the run it joins**.
2. On the cluster: `python scripts/run_benchmark.py setup --config configs/benchmark_<module>.yaml`
   (submit the printed `sbatch` command), then `... eval ...` for its `results.csv`.
3. Stability and validity: `python scripts/run_extra_benchmark.py setup --config configs/benchmark_<module>.yaml --submit`,
   then `collect`.
4. Combine with the existing results (`pd.concat` of the result CSVs) and publish with the
   **update-leaderboard** skill.

Elo compares methods within the same dataset, split, budget and downstream model, so results can
only be combined when they come from the same dataset containers and splits. Results of the fork
(`tabarena_fork_fs`) used an earlier curation of the datasets (`EARLIER_DATA_FOUNDRY_TASKS` in
`selectarena/datasets.py`); a new method run on the current containers should be compared with a
rerun of the other methods on the same containers, not with fork-era results. Raise this with the
maintainer before combining.

## Step 9: Report

Tell the maintainer: the files created and edited; the kind (scoring / subset), proxy-model use
and subgroup chosen, with the reason; new dependencies; the `check_method.py` and test results
(including any `WARN`); the comparison with a reference implementation if one was made; open
TODOs (Space About text, the benchmark run of Step 8). Commit only when asked; the commit message
names the method and its reference.
