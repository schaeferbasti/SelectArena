---
name: check-jobs
description: Status report of the user's SLURM jobs (e.g. SelectArena benchmark runs). Reads the local handoff notes (HANDOFF.md) if present, checks every job array of the user (completions, failures, running, pending, throttles), the cluster's CPU/GPU usage and the account's QOS caps, estimates when each array finishes, and suggests what to do about failures and whether to raise or lower throttles (with the effect on the runtime). Use when the user asks "how are my jobs doing", "check the jobs", "job status", "can we use more CPUs/GPUs", "when will X be done".
---

# check-jobs

Read-only status check for SLURM clusters. **Never change anything (`scontrol update`, `sbatch`, `scancel`,
`scontrol release`, reruns) without the user's OK**: this skill only reports and suggests.

## 1. Context

If the repo has local handoff notes (`HANDOFF.md`, git-ignored), read them first: the header with the cluster rules
and the **last 2-3 sections** (`grep -n '^## ' HANDOFF.md` lists them). Also read the auto-memory notes about the
cluster. Extract:
- which array IDs belong to which run (method, performance vs. stability/validity, CPU/GPU, bundle size), so the
  report uses names instead of bare IDs;
- the user's standing rules (e.g. what to do on TIMEOUT/OOM vs. other failures, throttle limits, how much of the
  cluster to leave to others, deadlines);
- failures already handled (reruns, fallbacks, watchers), so they are not reported as new problems;
- background watchers that should be running (check with `pgrep -af <script>` on the login node).

Without notes, ask the user only if a decision needs a rule they have not given; otherwise report and suggest.

## 2. Collect

```bash
python3 .claude/skills/check-jobs/scripts/collect.py              # since the last check (max 48 h; first run 24 h)
python3 .claude/skills/check-jobs/scripts/collect.py --since 24h --array <ids from the notes that may be missing>
```

Read-only (`squeue`, `sacct`, `scontrol show`, `sinfo`, `sacctmgr show`); stores the time of the last run in
`~/.cache/check-jobs/last_run`. It prints:
- per array: state counts, throttle, partitions, CPUs/GPUs per task, runtimes, completions since the last check,
  every non-COMPLETED task (`NEW` = ended since the last check), failures clustered on one node, running tasks above
  80 % of their time limit; finished arrays in one line;
- per partition the user's jobs use: cluster-wide idle CPUs/GPUs (down/drained nodes excluded; powered-down idle
  nodes count as available), the account's usage vs. the partition QOS's per-account cap (`MaxTRESPA`), usage by user
  on the account, others' pending jobs, and the free room for the account;
- per active array: a finish-time estimate (bootstrap over the array's own completed and timed-out task durations,
  median and 90 % finish time) for the current throttle, half of it, and the most the free room allows, with the
  CPUs/GPUs each option adds or frees.

## 3. Interpret (what the script cannot)

- **Failures**: for every `NEW` non-COMPLETED task not already handled, read its log (`scontrol show job <id>` for
  `StdErr`, or the array's `slurm-<id>_<idx>.err`) and classify: TIMEOUT / OUT_OF_MEMORY -> the user's rule (if none:
  suggest a longer limit / more memory or a fallback); NODE_FAIL, launch failure, CUDA/ECC errors, file-system errors ->
  infrastructure, suggest a rerun (and excluding the node if it repeats); a Python exception in the code -> a bug, show
  the traceback, do not suggest a blind rerun. A "NOTE: n/m failures on <node>" can be a broken node **or** just the
  node where the first (slowest) tasks started: compare with the completions on that node
  (`sacct -j <id> -X -n -P -o NodeList,State | sort | uniq -c`).
- **Odd pending reasons** (`launch failed requeued held`, `JobHeldUser`, `MaxGRESPerAccount`, `QOSMaxCpuPerAccount`,
  `ReqNodeNotAvail`): explain them and suggest the fix (`scontrol release`, wait, other partition, `ExcNodeList`).
- **Estimates** assume pending tasks look like the finished ones. Correct them with what the notes say about the task
  order (e.g. long tasks first, shorter evaluation tasks later, bundled tasks longer) and say so. The free room on a
  partition is **shared** between arrays: when suggesting raises for several, split it.
- **Throttle suggestions**: per active array, the current finish time and 1-2 alternatives with their finish times
  and the CPUs/GPUs they add or free; recommend one. Respect the user's rules and the account cap, and leave headroom
  for others on the account unless the user said otherwise (e.g. overnight). Say when a raise is useless (capped by
  free resources, few tasks pending, or a few long tasks dominate). Suggest lowering when others on the account wait
  (`QOSMax...PerAccount` pending reasons of other users). Command:
  `scontrol update JobId=<array id> ArrayTaskThrottle=<N>`.
- Next steps for finished runs (evaluation, result collection, cleanup) as listed in the notes.

## 4. Report (concise, in the terminal)

1. One-line overall status.
2. Table per run (name, array, done/running/pending, finish time at the current throttle).
3. Completions since the last check and what to do next.
4. Failures: new ones with classification and suggested action; handled ones in one line.
5. Cluster usage per partition: idle CPUs/GPUs, account usage vs. cap, who else uses the account.
6. Suggestions: throttle changes with the resulting finish times and resource cost, other actions. Ask before doing any.

If the user approves changes, apply them and, if the notes exist, append a short dated entry (what changed, why,
the user's decision).
