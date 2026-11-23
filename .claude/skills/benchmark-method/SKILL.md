---
name: benchmark-method
description: Start the full SelectArena benchmark of one registered feature selection method on the cluster (SLURM or OAR) or, after a warning, locally — performance on 3 splits x 5 downstream models x all datasets x 5 budgets, plus the stability and validity benchmark, with the published settings so the results rank against the leaderboard. Use when someone wants to benchmark / evaluate / run a method, e.g. "benchmark Boruta", "run the full benchmark for MI", "start the experiments for my method". Checks the method is registered (else points to add-method), checks the cluster's free CPUs/GPUs, suggests how many to use with finish-time estimates, and submits only after the user's OK. For a quick first look (1 split, no stability/validity) use benchmark-method-light.
argument-hint: <MethodName>
user-invocable: true
---

# Benchmark a method (full run)

Starts the complete benchmark of one method, as for the published results:

| part | scope |
|---|---|
| performance | all 71 datasets x 3 splits (`r0f0`, `r0f1`, `r0f2`) x 5 budgets x 5 models (TabICLv2, RealMLP on GPU; LightGBM, LinearModel, RandomForest on CPU; default configs), selection limit 1 h, model limit 1 h |
| stability + validity | all datasets x 5 budgets, selection only (CPU) |

The **light** variant (`benchmark-method-light`) runs the same steps with `--variant light`: split `r0f0`
only and no stability/validity. Both write the same `benchmark_name` (`selectarena_<module>`), so a full run
after a light one reuses the light results.

Lifecycle: add-method → **benchmark-method(-light)** → check-jobs while it runs → eval / leaderboard preview → PR.

Ground rules:
- **Never submit, cancel or change jobs without the user's explicit OK** on the concrete plan (Step 5).
- Run everything from the repository root with the SelectArena environment's Python (`python_path` of the
  cluster config, or the active venv). `P=.claude/skills/benchmark-method/scripts/benchmark_plan.py` below.
- Keep the published settings. `plan --write` writes them explicitly; do **not** copy `configs/benchmark.yaml`
  (it may hold a different, ad-hoc run). Only the concurrency is machine-specific.
- Keep the user's view about their method and their cluster: name internals only when they must act on them.

## Step 0: Method name

Parse `$ARGUMENTS` (class name, paper name or module, any case: `SHAP`, `shap`, `SHAPFeatureSelector`). Ask if
missing. Then:

```bash
python $P resolve <name>
```

- Exit 2 / `NOT_REGISTERED`: the method is not in SelectArena. Tell the user, show the close matches it
  prints (maybe a typo), and suggest **`/add-method <name>`** first (it writes, registers and checks the
  method, then comes back here). Stop.
- Otherwise note `class`, `module`, `paper_name`. If the published results already contain the method
  (a folder `results/published/methods/<paper name>/`, or ask), say that a rerun replaces the published results of that
  method in the local leaderboard preview, and confirm that is intended.

## Step 1: Context and method check

1. If `HANDOFF.md` exists (git-ignored), read its header (cluster rules: how much of the cluster to leave to
   others, deadlines, preferred partitions) and the last 2-3 sections (`grep -n '^## ' HANDOFF.md`): is this
   method already running or done? Also read the auto-memory notes about the cluster.
2. Existing results: `ls <workspace>/output/selectarena_<module>/` (workspace from the cluster config). If it
   exists, setup only adds the missing jobs (resume); say so.
3. `python scripts/check_method.py <class>` must pass (registration, output format, determinism, time limit).
   It fits small models for a few minutes: on a cluster whose login-node rules forbid that, run it through
   `srun`/`oarsub -I` on a CPU node. On failure, show the failing check and stop (fix via add-method).
   Note the selection times it reports: they feed `--fs-seconds` in Step 3.

## Step 2: Scheduler and free resources

```bash
python $P resources [--cluster-config <path>]
```

It detects the scheduler on `PATH` (`sbatch`/`squeue` → SLURM, `oarsub`/`oarstat` → OAR, neither → local),
compares it with `scheduler:` in the cluster config, and prints per partition/queue the total and idle
CPUs/GPUs, the account's usage and cap, who uses it, and the **free room** (it reuses check-jobs' read-only
queries). Handle what it flags:

- **No cluster config** (`MISSING`): on a cluster, help the user create `configs/cluster.yaml` from
  `configs/cluster.example.yaml` (SLURM) or `configs/cluster.oar.example.yaml` (OAR): workspace on a shared
  filesystem, `python_path`, caches, partitions (`sinfo -s`) / queue. Then `python scripts/prepare_datasets.py`
  once (downloads and converts the datasets, several GB).
- **Config and machine disagree** (e.g. `scheduler: slurm` on a laptop, paths that do not exist): the user is
  probably on the wrong machine. Claude runs where the jobs are submitted from, so suggest starting Claude
  Code on the cluster's login node in the repo checkout. Only go local if the user wants that (Step 3b).
- **SLURM vs OAR**: everything below works for both; the differences are in Step 5.

## Step 3: Suggest CPUs/GPUs and ask

### 3a. Cluster

The run has three arrays: GPU models (1 GPU + `resources.num_cpus` CPUs per task), CPU models, and
stability/validity (CPU). The GPU array is usually the longest: the GPU count sets the finish time; CPUs only
need to keep the CPU arrays from finishing later than it.

1. Pick a candidate: GPUs = the free GPU room capped by the user's rules from the notes (else leave about half
   of the idle GPUs to others; 4-8 is typical); CPUs = the GPU tasks' CPUs plus enough CPU tasks that the CPU
   and stability/validity arrays finish no later than the GPU array, within the free CPU room / account cap.
2. Run `python $P plan <name> --variant full --cpus C --gpus G [--fs-seconds S]` for the candidate and for a
   smaller and a larger option. `--fs-seconds` = the method's selection time per fit if known (Step 1.3, or
   the user's estimate); without it the plan shows fast/typical/slow scenarios. Adjust C until the CPU arrays
   are not the bottleneck. Act on its warnings (e.g. stability jobs longer than their walltime: raise
   `extra_benchmark_slurm.time` / `extra_benchmark_oar.walltime` in the cluster config, with the user's OK).
3. Ask with AskUserQuestion (the dialog must stand alone: name the method, the cluster and the free room in
   the question). Options: the recommended allocation (first, "(Recommended)"), a smaller one (gentler on
   the cluster, later finish), a larger one; each with its CPUs/GPUs, concurrency per array and estimated
   finish. "Other" lets the user give their own numbers; re-run `plan` for them.

### 3b. Local (no scheduler, or the user asks for it)

Warn clearly first: one machine runs one fit at a time; `plan` with a local cluster config shows the estimate
(the full run is weeks to months; without a CUDA GPU, TabICLv2 and RealMLP run on CPU, slower still). Suggest
instead: a cluster, or the light variant, or a dataset subset (`task_subset.dataset_names` in the written
config; results then rank on those datasets only). Proceed only if the user confirms. For the cluster config
copy `configs/cluster.local.example.yaml` to `configs/cluster.local.yaml` (git-ignored), adjust paths and
`resources` (CPUs and memory each fit may use; leave headroom on a laptop) and pass
`--cluster-config configs/cluster.local.yaml` to every command.

## Step 4: Write the config and set up the jobs

```bash
python $P plan <name> --variant full --cpus C --gpus G [--fs-seconds S] --write
python scripts/run_benchmark.py setup --config configs/benchmark_<module>.yaml [--cluster-config ...]
python scripts/run_extra_benchmark.py setup --config configs/benchmark_<module>.yaml --parallel <N from plan> [--cluster-config ...]
```

`plan --write` sets each model array's concurrency (`array_job_limit`) from the agreed numbers and prints the
`--parallel` value for stability/validity. `setup` (without `--submit`) only writes job files, skips jobs
with results, and prints the commands; it downloads missing datasets on first use. Check the printed commands:

- **SLURM**: one `sbatch --array=0-K%N ...` per model group *and bundle size* (large datasets get their own
  array with one fit per task). Several arrays of the same group each get `%N`, which would exceed the agreed
  concurrency: split `N` between them (proportional to their sizes, at least 1 each) by editing `%N` in the
  commands before submitting. `--partition`, `--gres=gpu:1`, `--cpus-per-task`, `--time` must match the plan.
- **OAR**: `oarsub -S <script> --array-param-file <params>` per group/bundle size. OAR has no per-array limit:
  every subjob that fits starts, so the agreed numbers are a target only. If the user wants a hard cap,
  offer to `oarhold` the subjobs beyond it after submission and `oarresume` them later (their choice).
- **Local**: one `python -P -m tabflow_slurm.run_local <json> --continue_on_error True` per model group,
  and one `python -P -m selectarena.extra_benchmark.local <job_commands.txt>`.
- "No jobs to run": everything is done already; go to Step 7.

## Step 5: Confirm and submit

Show a short plan: method, variant, scheduler and partitions/queue, per array the number of tasks, the
concurrency and CPUs/GPUs, the estimated finish (range), where results land
(`<workspace>/output/selectarena_<module>/`), and the exact commands. **Ask for an explicit OK.** Then:

- SLURM: run each `sbatch` command (perf) and `python scripts/run_extra_benchmark.py setup ... --parallel N
  --submit` (or the printed `sbatch` lines). Collect the job IDs (`Submitted batch job <id>`).
- OAR: `python scripts/run_benchmark.py setup --config ... --submit` and `python scripts/run_extra_benchmark.py
  setup --config ... --submit`; collect the `OAR_JOB_ID` / array IDs from their output.
- Local: start the commands one after the other in the background so they survive the session, e.g.
  `nohup sh -c '<cmd 1>; <cmd 2>; <extra cmd>' > logs/benchmark_<module>.log 2>&1 &` (`logs/` is git-ignored);
  note the PID.

Verify with `squeue -u $USER` / `oarstat -u $USER` (or `ps -p <pid>`) that the arrays are queued/running.

## Step 6: Record the launch in HANDOFF.md

Create `HANDOFF.md` if missing (it is git-ignored; check-jobs reads it), with a header `# SelectArena handoff
notes` and a `## Cluster rules` section holding any rules the user stated (or "none recorded"). Append:

```markdown
## <YYYY-MM-DD> benchmark-method: <paper name> (<class>), full run
- config: configs/benchmark_<module>.yaml; results: <workspace>/output/selectarena_<module>/
- arrays: <id> perf GPU (TabICLv2, RealMLP; bundle <size>; %<N>, 1 GPU + <c> CPUs/task) | <id> perf CPU
  (LightGBM, LinearModel, RandomForest; ...) | <id> stability+validity (%<N>) [local: PID <pid>, log <path>]
- agreed resources: <C> CPUs / <G> GPUs on <partitions/queue>; estimated finish <date range>
- next: check-jobs while it runs; then eval / collect / leaderboard preview (see the skill's Step 7)
```

## Step 7: Report and next steps

Tell the user, in a few lines: what was submitted (arrays, concurrency, resources), the estimated finish, and:
- progress, failures, finish estimates, more/less parallelism: `/check-jobs`;
- when done: `python scripts/run_benchmark.py eval --config configs/benchmark_<module>.yaml` (→ `results.csv`)
  and `python scripts/run_extra_benchmark.py collect --config configs/benchmark_<module>.yaml`;
- compare with the published methods: `python scripts/run_generate_website_artifacts.py --with-published-results
  --results <run dir>/results.csv --extra-dir <run dir>/extra_benchmark`, then `python scripts/preview_leaderboard.py`;
- to get on the public leaderboard: open a pull request with the method and point to the results.

Commit nothing (the configs are git-ignored by design).
