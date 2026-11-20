---
name: check-jobs
description: Status report of the user's cluster jobs (SLURM or OAR; e.g. SelectArena benchmark runs). Reads the local handoff notes (HANDOFF.md) if present, checks every job array of the user (completions, failures, running, pending, concurrency limits), the cluster's CPU/GPU usage and the account's caps, estimates when each array finishes, and suggests what to do about failures and whether to run more or fewer tasks in parallel (with the effect on the runtime). Use when the user asks "how are my jobs doing", "check the jobs", "job status", "can we use more CPUs/GPUs", "when will X be done".
---

# check-jobs

Read-only status check for SLURM and OAR clusters. **Never change anything (`scontrol update`, `sbatch`, `scancel`,
`scontrol release`, `oarsub`, `oardel`, `oarhold`, `oarresume`, reruns) without the user's OK**: this skill only
reports and suggests.

## 1. Context

If the repo has local handoff notes (`HANDOFF.md`, git-ignored), read them first: the header with the cluster rules
and the **last 2-3 sections** (`grep -n '^## ' HANDOFF.md` lists them). Also read the auto-memory notes about the
cluster. Extract:
- which array IDs belong to which run (method, performance vs. stability/validity, CPU/GPU, bundle size), so the
  report uses names instead of bare IDs;
- the user's standing rules (e.g. what to do on TIMEOUT/OOM vs. other failures, concurrency limits, how much of the
  cluster to leave to others, deadlines);
- failures already handled (reruns, fallbacks, watchers), so they are not reported as new problems;
- background watchers that should be running (check with `pgrep -af <script>` on the login node).

Without notes, ask the user only if a decision needs a rule they have not given; otherwise report and suggest.

## 2. Collect

```bash
python3 .claude/skills/check-jobs/scripts/collect.py              # since the last check (max 48 h; first run 24 h)
python3 .claude/skills/check-jobs/scripts/collect.py --since 24h --array <ids from the notes that may be missing>
```

The scheduler is detected from the commands on `PATH` (`squeue`/`sacct` -> SLURM, `oarstat` -> OAR); force it with
`--scheduler slurm|oar`. The first output line names the backend used. Read-only queries only; the time of the last run
is stored in `~/.cache/check-jobs/last_run`. The core (`scripts/collect.py`) is scheduler-independent; the queries are
in `scripts/backends/{slurm,oar}.py`. It prints:
- per array: state counts, concurrency limit (SLURM `throttle`; OAR has none), partitions/queues, CPUs/GPUs per task,
  runtimes, completions since the last check, every non-COMPLETED task (`NEW` = ended since the last check), failures
  clustered on one node, running tasks above 80 % of their time limit; finished arrays in one line;
- per partition/queue the user's jobs use: idle CPUs/GPUs (unavailable nodes excluded), the account's usage vs. its cap
  (SLURM: the partition QOS's `MaxTRESPA`; OAR: no cap, quotas are not queryable), usage by user, others' pending jobs,
  and the free room for the account;
- per active array: a finish-time estimate (bootstrap over the array's own completed and timed-out task durations,
  median and 90 % finish time) for the current concurrency and alternatives, with the CPUs/GPUs each option adds or
  frees, and the command to change it.

Normalized end states (both schedulers): COMPLETED, TIMEOUT, OUT_OF_MEMORY, FAILED, NODE_FAIL, LAUNCH_FAIL, CANCELLED;
anything else is the scheduler's own name.

### OAR specifics (backend is EXPERIMENTAL until confirmed on an OAR cluster)

- The OAR backend was written against OAR's documented `oarstat -f -J` / `oarnodes -J` output and tested only with
  mock data. On the first run on an OAR cluster, also run `collect.py --scheduler oar --raw` (one raw job and one raw
  resource record) and check that states, times, resources, GPUs and the walltime come out right; if field names
  differ, fix the lookups in `backends/oar.py` (each `_get(...)` lists the accepted spellings), then tell the user and
  remove the EXPERIMENTAL notes.
- Task ids print as `<array>_<index> (job <id>)`; OAR commands take the job id.
- End states come from `Terminated`/`Error` + events + exit code: `TIMEOUT` = `WALLTIME` event; `CANCELLED` =
  `FRAG_JOB_REQUEST`; `NODE_FAIL` = node suspected / switched to error; `LAUNCH_FAIL` = prologue/cpuset errors;
  `KILLED_9(OOM?)` = killed by SIGKILL, **only a guess** for OOM (also a manual kill): confirm in the job's stderr
  (the path in its `#OAR -E` directive, else `OAR.<id>.stderr` or `OAR.<name>.<id>.stderr` in the submit
  directory); `COMPLETED?` = terminated without an exit code.
  The raw state/events are shown in `[...]` after each failed task.
- No per-array concurrency limit: OAR starts whatever fits, so "current" = running + what fits in the free resources.
  Fewer in parallel: `oarhold <job ids>` on waiting subjobs (`oarresume` later); more in parallel is only possible by
  freeing resources, another queue, or `besteffort` jobs (killable; resubmission needed).
- Resource numbers are cluster-wide (queues are not tied to nodes); usage by user covers all users. Waiting subjobs
  may show a scheduled start time in their pending reason: compare it with the estimate.

## 3. Interpret (what the script cannot)

- **Failures**: for every `NEW` non-COMPLETED task not already handled, read its log (SLURM: `scontrol show job <id>`
  for `StdErr`, or the array's `slurm-<id>_<idx>.err`; OAR: the stderr file, see above) and classify: TIMEOUT /
  OUT_OF_MEMORY -> the user's rule (if none: suggest a longer limit / more memory or a fallback); NODE_FAIL, launch
  failure, CUDA/ECC errors, file-system errors -> infrastructure, suggest a rerun (and excluding the node if it
  repeats: SLURM `ExcNodeList`, OAR `-p "network_address != '<node>'"`); a Python exception in the code -> a bug, show
  the traceback, do not suggest a blind rerun. A "NOTE: n/m failures on <node>" can be a broken node **or** just the
  node where the first (slowest) tasks started: compare with the completions on that node.
- **Odd pending reasons** (SLURM: `launch failed requeued held`, `JobHeldUser`, `MaxGRESPerAccount`,
  `QOSMaxCpuPerAccount`, `ReqNodeNotAvail`; OAR: `Hold`, waiting far past its scheduled start): explain them and
  suggest the fix (`scontrol release` / `oarresume`, wait, other partition or queue, exclude nodes).
- **Estimates** assume pending tasks look like the finished ones. Correct them with what the notes say about the task
  order (e.g. long tasks first, shorter evaluation tasks later, bundled tasks longer) and say so. The free room on a
  partition/queue is **shared** between arrays: when suggesting more parallelism for several, split it.
- **Concurrency suggestions**: per active array, the current finish time and 1-2 alternatives with their finish times
  and the CPUs/GPUs they add or free; recommend one. Respect the user's rules and the account cap, and leave headroom
  for others unless the user said otherwise (e.g. overnight). Say when more is useless (capped by free resources, few
  tasks pending, or a few long tasks dominate). Suggest fewer when others on the account wait (SLURM:
  `QOSMax...PerAccount` pending reasons of other users; OAR: others' waiting jobs in the same queue).
  SLURM: `scontrol update JobId=<array id> ArrayTaskThrottle=<N>`; OAR: `oarhold` / `oarresume` (see above).
- Next steps for finished runs (evaluation, result collection, cleanup) as listed in the notes.

## 4. Report (concise, in the terminal)

1. One-line overall status (and the scheduler, if not the usual one).
2. Table per run (name, array, done/running/pending, finish time at the current concurrency).
3. Completions since the last check and what to do next.
4. Failures: new ones with classification and suggested action; handled ones in one line.
5. Cluster usage per partition/queue: idle CPUs/GPUs, account usage vs. cap, who else uses it.
6. Suggestions: concurrency changes with the resulting finish times and resource cost, other actions. Ask before doing
   any.

If the user approves changes, apply them and, if the notes exist, append a short dated entry (what changed, why,
the user's decision).
