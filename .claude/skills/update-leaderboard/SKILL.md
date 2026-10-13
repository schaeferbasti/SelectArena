---
name: update-leaderboard
description: Regenerate the SelectArena leaderboard website artifacts and refresh the leaderboard Space's `data/` with them. Use whenever a maintainer wants to publish benchmark results to the leaderboard, e.g. "update the leaderboard", "publish the new results", "refresh the Space with the latest run", "add <method> to the leaderboard". Pre-checks that every method has a family (not "❓ Other"), runs `scripts/run_generate_website_artifacts.py`, swaps the artifacts into a clone of the (static) Space repo (delete-then-copy), bumps the version and history in its `index.html`, runs its smoke test and a local preview, and hands the commit + push (which deploys) back to the maintainer.
argument-hint: <space-checkout-dir> [<results.csv>] [<extra-results-dir>]
user-invocable: true
---

# Regenerate & publish the SelectArena leaderboard

The leaderboard works like TabArena's: this repo turns `results.csv` (and the stability / validity
results) into **website artifacts** (CSV + self-contained HTML), and the **Space repo**
(`huggingface.co/spaces/<org>/Leaderboard`, a separate git repo) serves them from its `data/`
folder. The Space is a **static** site (`sdk: static`: `index.html` + `assets/app.js`), which is free;
Gradio and Docker Spaces need a paid plan. Pushing the Space repo deploys it.

The authoritative manual procedure is the docstring of `scripts/run_generate_website_artifacts.py`;
the Space side is described in the Space repo's `AGENTS.md`. This skill runs that procedure and
adds the checks that are easy to forget.

Lifecycle: `scripts/run_benchmark.py eval` (results.csv) and `scripts/run_extra_benchmark.py collect`
(stability / validity) → **this skill** → maintainer pushes the Space.

## Step 0: Gather inputs

Parse `$ARGUMENTS`. Ask only for what is missing or ambiguous.

| Input | Default | Notes |
|---|---|---|
| `space_dir` | none, **required** | A clone of the Space repo (has `index.html`, `assets/`, `data/`, `smoke_test.py`). If the maintainer has none, offer `git clone https://huggingface.co/spaces/<org>/<space> <dir>` (needs `hf auth login --add-to-git-credential` once). |
| `results` | `<output_dir>/<benchmark_name>/results.csv` | From `configs/benchmark.yaml` + the cluster config (`configs/cluster.yaml` or `$SELECTARENA_CLUSTER_CONFIG`). Pass `--results` when the CSV is elsewhere (e.g. copied from the cluster). |
| `extra_dir` | `<run dir>/<extra_benchmark.results_subdir>` | Folder with `stability_results.csv` / `validity_results.csv`. Optional: without them the Stability / Validity columns are empty and the figure's bars are hatched. |
| what changed | infer | New method(s), a rerun, more datasets? Needed for the version-history entry (Step 4). Infer from `git log`, the results, or the diff of `meta.json` (Step 3); ask if unclear. |
| Python env | current env | Generation needs SelectArena installed (`pip install --pre -e .` in this repo). The Space needs nothing: its smoke test uses the standard library. |

Run every command from the SelectArena repo root unless noted.

## Step 1: Pre-check the methods (before generating)

Every method on the leaderboard needs a family in `METHOD_FAMILIES` (`src/selectarena/methods/registry.py`),
otherwise it ships as `❓ Other`. List the methods in the results and the unclassified ones:

```bash
python - <<'PY'
import pandas as pd
from selectarena.evaluation.leaderboard import prepare_results
from selectarena.website.website_format import unclassified_methods
df = prepare_results(pd.read_csv("<results.csv>"))
methods = sorted(df["method"].dropna().unique())
print(len(methods), "methods:", methods)
print("datasets:", df["dataset"].nunique(), "| models:", sorted(df["model"].dropna().unique()))
print("unclassified:", unclassified_methods(methods))
PY
```

- **Unclassified method**: add it to `METHOD_FAMILIES` (`filter`, `wrapper`, `embedded` or `baseline`;
  confirm the family with the maintainer if unsure) and to `PAPER_NAMES`, then `pytest tests/test_website.py`.
- **Methods not in the registry** (e.g. the retired `AccuracyLinearFeatureSelector` in fork-era
  results) are left out by default and listed in the log; `--include-unregistered` keeps them.
  Fork-era names `AccuracyFeatureSelector` / `ANOVAFeatureSelector` are read as LOCO / F-Test.
- **Labels with brackets** (`MI [nofallback]`, `MI [permenc1]`) are ablation runs. They are left out
  by default; pass `--include-ablations` only if the maintainer wants them ranked.
- **Sanity**: the dataset count should match the suite (71) unless the run is partial, and every
  expected downstream model should appear. A partial run is fine to publish only if the maintainer says so.

## Step 2: Regenerate the artifacts (Claude runs; slow, no credentials)

The Elo intervals dominate the run time (1000 bootstrap refits per subset, about a hundred
subsets: minutes). Run it in the background and poll the log:

```bash
nohup python scripts/run_generate_website_artifacts.py [--results <results.csv>] [--extra-dir <dir>] \
    > <scratch>/gen_website.log 2>&1 &
```

Flags: `--elo-bootstrap-rounds 1` for a quick dry run (intervals meaningless, never publish it),
`--validity-noise` (default 1.0, as in the paper), `--include-ablations`, `--no-paper-figures`.

When the process exits, verify the output; do not trust "exited" alone:

```bash
W=generated_website_artifacts/clean_website_artifacts/website_data
tail -5 <scratch>/gen_website.log                       # "<N> subsets written to ...", no WARNING line
python - <<'PY'
import json, pandas as pd
W = "generated_website_artifacts/clean_website_artifacts/website_data"
meta = json.load(open(f"{W}/meta.json"))
print({k: meta[k] for k in ["n_results", "n_datasets", "n_methods", "models", "budgets", "has_stability", "has_validity"]})
print(len(meta["subsets"]), "subsets")
lb = pd.read_csv(f"{W}/subsets/tasks_all/models_all/budgets_all/website_leaderboard.csv")
print(lb[["Rank", "TypeName", "Method", "Elo", "Elo CI low", "Elo CI high", "Stability", "Validity"]].to_string(index=False))
assert (lb.loc[lb.Method == "Random", "Elo"] == 1000).all(), "Random must be anchored at 1000"
assert "Other" not in set(lb.TypeName), "unclassified method"
PY
find "$W" -name '*.png' | wc -l                          # must be 0: the Space gets text only
```

Also look at the main figure: `generated_website_artifacts/raw_website_artifacts/main_figure.png`
(the paper version) or open `$W/subsets/tasks_all/models_all/budgets_all/elo_explorer.html`.

## Step 3: Refresh the Space's `data/`

`data/` mirrors `website_data/`. **Delete the old content first, then copy**, so a subset that no
longer exists cannot survive as a leftover.

```bash
SRC=generated_website_artifacts/clean_website_artifacts/website_data
DST=<space_dir>/data

# 1. Only generated files may live in data/ (so the delete is safe). Expect no output:
find "$DST" -mindepth 1 -maxdepth 1 ! -name subsets ! -name meta.json ! -name 'per_dataset*' \
     ! -name 'stability*' ! -name 'validity*' ! -name .gitkeep

# 2. What changes for the version-history entry (before the swap):
diff <(python -c "import json;m=json.load(open('$DST/meta.json'));print({k:m[k] for k in ['n_datasets','n_methods','methods','models']})" 2>/dev/null) \
     <(python -c "import json;m=json.load(open('$SRC/meta.json'));print({k:m[k] for k in ['n_datasets','n_methods','methods','models']})")
diff <(cd "$SRC" && find subsets -type f | sort) <(cd "$DST" && find subsets -type f 2>/dev/null | sort) | head

# 3. Swap:
find "$DST" -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
cp -r "$SRC"/. "$DST"/

# 4. Verify:
echo "subsets: $(find "$DST/subsets" -name website_leaderboard.csv | wc -l)"   # = len(meta["subsets"])
cd <space_dir> && git status --short | awk '{print $1}' | sort | uniq -c
```

Added or deleted files under `subsets/` mean a subset appeared or disappeared (a new downstream
model, budget or task type, or one with fewer than two methods). That is fine when the run changed
accordingly; mention it in the report.

## Step 4: Bump the version (Claude does now)

In the Space repo, read then edit `index.html`:

1. Bump `<meta name="selectarena-version" content="X.Y.Z">`: the **patch** component for a data
   refresh (`0.1.0` → `0.1.1`), the **minor** one for site changes or a new benchmark release.
2. Add an entry at the **top** of `<ul id="version-history">` (newest first, today's date) with
   the same version, matching the existing format and describing what changed, e.g.:

```html
<li><strong>2026/11/02 · v0.1.1</strong>: Add new method: Boruta. Results for 71 datasets and
  3 downstream models (LightGBM, RealMLP, TabICLv2).</li>
```

The footer's "current version" renders from the meta tag; `smoke_test.py` fails if the meta tag
and the newest entry differ.

## Step 5: Smoke test and preview

From `<space_dir>` (no extra packages needed):

```bash
python smoke_test.py                     # must end with "0 errors"
nohup python -m http.server 8000 > <scratch>/lb_serve.log 2>&1 &
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/data/meta.json   # 200
```

Ask the maintainer to open http://localhost:8000 and look at the four tabs, a few selections and
dark mode. Opening `index.html` from disk does not work (browsers block `fetch()` of local
files). Stop the server afterwards.

## Step 6: Hand off (the maintainer commits and pushes)

By default the maintainer pushes, because pushing deploys the Space. Push yourself only when asked
and when `git push` is authenticated (`hf auth login --add-to-git-credential`). From `<space_dir>`:

```bash
git add -A data index.html
git commit -m "Update leaderboard data (vX.Y.Z)"
git push
```

`git add -A data` matters: a refresh modifies, adds and deletes files under `data/`. The site is
live a few seconds after the push (static Spaces have no build or start-up).

Caveats to mention:
- Keep the Space static. A "CPU Basic quota limit" message means someone switched it to Gradio or
  Docker, which need a paid plan (PRO for users, Team/Enterprise for organizations).
- `data/` is text only (no Git LFS), so the Space's storage limit is not a concern.

## Step 7: Report

Tell the maintainer: what was generated (methods, datasets, models, number of subsets), anything
left out (ablations, missing stability / validity), the code edits made (`METHOD_FAMILIES`,
`PAPER_NAMES`), the Space changes (data swap summary from Step 3, new version and history entry),
the smoke-test result, and the exact push command if it is still to be run.
