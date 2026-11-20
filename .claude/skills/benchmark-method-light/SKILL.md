---
name: benchmark-method-light
description: Start a light SelectArena benchmark of one registered feature selection method — performance only, on 1 split (r0f0) x 5 downstream models x all datasets x 5 budgets, no stability/validity — on the cluster (SLURM or OAR) or, after a warning, locally. A third of the full run's fits; for a first look at a new method before the full benchmark. Use when someone wants a quick / light / first / cheap benchmark of a method, e.g. "light benchmark of Boruta", "quick run for my method on one split". Checks the method is registered (else points to add-method), checks the free CPUs/GPUs, suggests how many to use, and submits only after the user's OK. For the complete benchmark use benchmark-method.
argument-hint: <MethodName>
user-invocable: true
---

# Benchmark a method (light run)

Follow **`.claude/skills/benchmark-method/SKILL.md`** (read it now) step by step, with these differences:

| | full (benchmark-method) | light (this skill) |
|---|---|---|
| splits | `r0f0`, `r0f1`, `r0f2` | `r0f0` |
| models, datasets, budgets, time limits | 5 models, all 71 datasets, 5 budgets, 1 h + 1 h | same |
| stability / validity | yes | **no**: skip every `run_extra_benchmark.py` command |
| fits | 5,325 | 1,775 |
| `plan` flag | `--variant full` | `--variant light` |
| config written | `configs/benchmark_<module>.yaml` | `configs/benchmark_<module>_light.yaml` |
| `benchmark_name` | `selectarena_<module>` | `selectarena_<module>` (same) |

Notes for the light run:
- **Resources**: with no stability/validity array, the CPUs only serve the GPU tasks and the CPU-model array;
  suggest a smaller allocation than for the full run (the plan's estimates show where the CPU array stops being
  the bottleneck).
- **Local**: still days to weeks on one machine (the plan prints the estimate); warn as in Step 3b.
- **Leaderboard**: the light results rank against the published methods on split `r0f0` only. Say so in the
  report; stability and validity will be missing on the leaderboard preview.
- **Upgrade later**: because the run shares `benchmark_name` with the full run, `/benchmark-method <name>`
  afterwards only adds splits `r0f1`, `r0f2` and stability/validity; the `r0f0` fits are reused.
- HANDOFF.md entry title: `benchmark-method-light: <paper name> (<class>), light run (r0f0, no stability/validity)`.
