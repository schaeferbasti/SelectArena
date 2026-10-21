# Running SelectArena on calculco (OAR)

calculco schedules with OAR, not SLURM. Set `scheduler: oar` in the cluster config and the usual
scripts write and print `oarsub` commands instead of `sbatch` ones. The jobs, their bundling,
caching and results are the same as on SLURM. Only the submission changes:

| | SLURM | OAR |
|---|---|---|
| Main benchmark | `sbatch --array` + TabArena's `submit_template.sh` | `oarsub -S <json>.oar --array-param-file <json>.params`; each subjob runs `python -m selectarena.benchmark.oar <json> <index>` |
| Stability / validity | `sbatch array.sh` | `oarsub -S array.oar --array-param-file job_commands.txt` |
| Logs | `<workspace>/slurm_out/` | `<workspace>/oar_out/<benchmark>/OAR.<jobid>.out` (extra benchmark: next to its `array.oar`) |

The front end is for editing and short tests only. Anything heavy goes through `scripts/oar/run.oar`,
which runs one Python command on a compute node.

## Once

```bash
bash scripts/oar/install_env.sh                    # venv in ~/venvs/selectarena (front end, a few minutes)
cp configs/cluster.calculco.example.yaml configs/cluster.yaml
# edit configs/cluster.yaml: workspace, caches, data_foundry_warehouse, python_path
oarsub -S "./scripts/oar/run.oar scripts/prepare_datasets.py"
```

Submit from the repository root (OAR jobs start in the submission directory). `install_env.sh`
creates `logs/oar/`, where `run.oar` writes its logs.

## A benchmark run

```bash
# 1. Build the jobs (skips the ones with results) and print the oarsub commands.
oarsub -S "./scripts/oar/run.oar scripts/run_benchmark.py setup"
cat logs/oar/run.<jobid>.out            # ends with the oarsub command(s)
# 2. Submit them from the front end (or rerun setup with --submit on the front end).
oarsub -S /.../setup_out/<benchmark>/oar_run_data_<benchmark>_gpu.oar --array-param-file /.../oar_run_data_<benchmark>_gpu.params
# 3. Follow them.
oarstat -u $USER
# 4. When they are done, collect the results into <workspace>/output/<benchmark>/results.csv.
oarsub -S "./scripts/oar/run.oar scripts/run_benchmark.py eval"
```

Rerunning `setup` after failed or killed subjobs only resubmits the missing items.

Stability / validity:

```bash
oarsub -S "./scripts/oar/run.oar scripts/run_extra_benchmark.py setup"     # prints the oarsub commands
oarsub -S "./scripts/oar/run.oar scripts/run_extra_benchmark.py collect"   # when they are done
```

## Resources

- **GPU models** (TabICLv2, RealMLP in `configs/benchmark.yaml`) request `oar.gpu_resources`
  (`/nodes=1/gpu=1`) restricted by `oar.gpu_properties` to A100/H100 nodes. The GH200 nodes are
  ARM and cannot run the x86 venv.
- **CPU jobs** request `oar.cpu_resources` (`/nodes=1/core=<num_cpus>`).
- **Walltime** per subjob = items in the bundle × (fit time limit + feature selection time limit)
  + `oar.time_limit_overhead` hours, as on SLURM. With the default config (1 h fit + 1 h selection,
  5 items per bundle) that is 11 h. Lower `bundle_size` (a `scheduler` override in a model job of
  `configs/benchmark.yaml`, e.g. `scheduler: {bundle_size: 1}`) if the queue's maximum walltime
  is shorter.
- **Memory** is not reserved by OAR. `resources.memory_limit` is the budget AutoGluon and Ray are
  told to stay within.
- **besteffort**: set `oar.besteffort: true` and `oar.queue: besteffort` to use idle nodes. Killed
  subjobs are simply rerun by the next `setup`.
- OAR has no limit on concurrent subjobs (SLURM's `%N`). All subjobs of an array are queued at once.

## Compute nodes and the internet

`setup` downloads the foundation-model weights (e.g. TabICL) into the shared Hugging Face cache
(`caches.huggingface`) before writing the jobs, so the subjobs read them offline. If the compute
nodes cannot reach the internet, run that first `setup` on the front end
(`python scripts/run_benchmark.py setup`).
