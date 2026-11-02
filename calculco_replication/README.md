# Replicating the Freiburg comparison on calculco

Runs the SelectArena side of the Freiburg comparison (`fork_comparison/`) on calculco with OAR:
the same 456 runs (`benchmark.yaml`), in a workspace of their own. Then it compares the results
with both Freiburg runs:

- `report_vs_freiburg_selectarena/`: the same SelectArena code on another cluster. Differences
  here come from the environment (library versions, CPUs).
- `report_vs_fork/`: calculco SelectArena against the fork, as in the Freiburg report.

| File | Does |
|---|---|
| `replicate.sh` | The steps below; the calculco paths are at its top |
| `benchmark.yaml` | What runs (the Freiburg comparison's runs) |
| `prepare_datasets.py` | Converts only the two containers, into this run's own OpenML cache |
| `compare_runs.py` | Matches the runs of two results.csv files and writes a report |

## Before the first run

1. **SelectArena** on calculco at commit `549433a` or later (`git -C ~/SelectArena pull`), with
   the venv from `bash scripts/oar/install_env.sh`. If the checkout or the venv live elsewhere,
   set `SELECTARENA_REPO` and `SELECTARENA_PY` at the top of `replicate.sh`.
2. **The two containers** are downloaded by `setup` from Hugging Face
   (`selectarena.datasets.HF_REPO_ID` at the pinned `HF_REVISION`) into the run's
   `.data_foundry_cache`; the front end has internet access, the compute nodes need none. To
   use a local copy instead, set `CONTAINERS` to a warehouse in the `<dataset>/<uuid>/` layout:
   ```
   <CONTAINERS>/credit_approval/019d9cd7-5cc2-72b1-8dca-a5437746e1fd/
   <CONTAINERS>/predict_students_dropout_and_academic_success/019d9d2c-d03c-7327-b13f-e4edffddb4fa/
   ```
3. **This folder** on calculco, e.g. in `~/calculco_replication`.

Each run gets its own workspace, `$DATA/fork_comparison/$RUN` (`RUN` at the top of
`replicate.sh`). Set `RUN` to a new name to repeat the replication without touching an earlier
run; `eval` then also compares with the run named in `PREVIOUS` (`report_vs_<PREVIOUS>/`),
which shows what changes from one calculco run to the next.

## Run (front end)

```bash
bash replicate.sh setup            # prints the oarsub command; or: bash replicate.sh setup --submit
bash replicate.sh status           # finished results, 456 at the end; oarstat -u $USER for the subjobs
bash replicate.sh eval             # results.csv, then the two reports
```

`setup` only converts two small datasets and writes the job files, so the front end is fine for
it. Rerunning it after killed subjobs submits the missing runs only.

Each subjob bundles 5 runs and asks for 11 h (5 x (1 h fit + 1 h selection) + 1 h); the real
runtime is much shorter. If the queue allows less, add `scheduler: {bundle_size: 1}` to the
model job in `benchmark.yaml` (3 h per subjob). It does not change the results.

## The Freiburg results

Before `eval`, copy the two Freiburg `results.csv` (written by `bash compare.sh eval` there) into
`freiburg/` here, under these names (via your Mac if the clusters cannot reach each other):

| Freiburg file | Copy to |
|---|---|
| `/work/dlclarge1/purucker-fs_benchmark/fork_comparison/selectarena_run/output/fs_compare_selectarena/results.csv` | `freiburg/selectarena_results.csv` |
| `/work/dlclarge1/purucker-fs_benchmark/output/fs_compare_fork/results.csv` | `freiburg/fork_results.csv` |

A missing file skips that report. You can run `eval` again at any time.

## Reading `report_vs_freiburg_selectarena`

The code is the same on both clusters, so most selections should be identical. Differences point
to the environments:

- **Library versions.** AutoGluon comes from pre-release builds, so two venvs installed on
  different days can differ. Compare `pip freeze | grep -iE "autogluon|lightgbm|scikit|numpy|pandas"`
  in both venvs.
- **CPUs.** LightGBM (the downstream model and the proxy of LOCO, SFS, SBE) can give slightly
  different results on different processors. That moves the errors a little and, for the proxy
  methods, can change a close choice between two features.
- **Time limits.** A run that hits the 3600 s selection limit on one cluster only stops at a
  different point; the selection times in the report show it.
