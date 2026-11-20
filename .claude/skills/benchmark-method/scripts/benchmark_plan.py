#!/usr/bin/env python3
"""Plan a SelectArena benchmark run of one method, for the benchmark-method(-light) skills.

    benchmark_plan.py resolve   <name>                    # registered? class, module, paper name
    benchmark_plan.py resources [--cluster-config PATH]   # scheduler, free CPUs/GPUs, account caps
    benchmark_plan.py plan      <name> --variant full|light --cpus C --gpus G [--fs-seconds S] [--write]

``plan`` prints the job counts, how the C CPUs / G GPUs are split between the job arrays, and
finish-time estimates; with ``--write`` it writes ``configs/benchmark_<module>[_light].yaml``
(git-ignored). The config always carries the published settings (datasets, splits, budgets, time
limits, models), never the ones of ``configs/benchmark.yaml``, so the results can be ranked
against the leaderboard. Full and light share ``benchmark_name``, so a full run after a light one
reuses the light run's results. Read-only except for ``--write``. Run it from the repository root
with the SelectArena environment's Python.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
CHECK_JOBS_SCRIPTS = REPO / ".claude" / "skills" / "check-jobs" / "scripts"

# ---- The published setup (configs of the leaderboard's results) ------------------------------
PUBLISHED_SPLITS = ["r0f0", "r0f1", "r0f2"]
LIGHT_SPLITS = ["r0f0"]
TOTAL_BUDGET = 5
FS_TIME_LIMIT = 3600
MODEL_TIME_LIMIT = 3600
GPU_MODELS = ["TabICLv2", "RealMLP"]
CPU_MODELS = ["LightGBM", "LinearModel", "RandomForest"]
EXTRA_MODES = ["stability", "validity"]
#: Mean model fit + inference time per item in seconds, without the feature selection (published
#: results, Nov 2026), plus a fixed per-item overhead (process start, data loading, Ray).
MODEL_SECONDS = {"TabICLv2": 59, "RealMLP": 572, "LightGBM": 9, "LinearModel": 8, "RandomForest": 9}
ITEM_OVERHEAD_S = 60
#: One stability / validity job runs the selection about this many times (published results:
#: stability ~20-30x the per-fit selection time, validity ~1-3x).
EXTRA_FS_FACTOR = {"stability": 25, "validity": 3}
#: Selection times (s per fit) to show when the method's own is unknown: fast filter, typical
#: model-based method, slow search method (the published methods span 0 s to the 3600 s limit).
FS_SCENARIOS = {"fast": 10, "typical": 300, "slow": 1800}


# ---- resolve ------------------------------------------------------------------------------------
def resolve(name: str) -> dict | None:
    from selectarena.methods.registry import LEGACY_NAMES, NAME_TO_MODULE_MAP, PAPER_NAMES

    keys: dict[str, str] = {}
    for cls, module in NAME_TO_MODULE_MAP.items():
        for alias in (cls, module, PAPER_NAMES.get(cls, ""), cls.removesuffix("FeatureSelector")):
            if alias:
                keys[alias.lower()] = cls
    for old, cls in LEGACY_NAMES.items():
        keys[old.lower()] = cls
    cls = keys.get(name.lower()) or keys.get(f"{name}FeatureSelector".lower())
    if cls is None:
        return None
    from selectarena.methods.registry import FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL, METHOD_SUBGROUPS

    return {
        "class": cls,
        "module": NAME_TO_MODULE_MAP[cls],
        "paper_name": PAPER_NAMES.get(cls),
        "subgroup": METHOD_SUBGROUPS.get(cls),
        "proxy_model": cls in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL,
    }


def cmd_resolve(args: argparse.Namespace) -> int:
    info = resolve(args.name)
    if info is None:
        from selectarena.methods.registry import NAME_TO_MODULE_MAP, PAPER_NAMES

        candidates = [*NAME_TO_MODULE_MAP, *NAME_TO_MODULE_MAP.values(), *PAPER_NAMES.values()]
        close = difflib.get_close_matches(args.name, candidates, n=5, cutoff=0.4)
        print(f"NOT_REGISTERED {args.name}")
        print(f"close matches: {', '.join(close) if close else 'none'}")
        print(f"registered: {', '.join(NAME_TO_MODULE_MAP)}")
        return 2
    print(json.dumps(info))
    return 0


# ---- resources ----------------------------------------------------------------------------------
def load_cluster(path: str | None) -> tuple[dict, Path | None]:
    path = path or os.environ.get("SELECTARENA_CLUSTER_CONFIG") or "configs/cluster.yaml"
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        return {}, None
    return yaml.safe_load(p.read_text()) or {}, p


def detect_scheduler() -> str:
    if shutil.which("sbatch") and shutil.which("squeue"):
        return "slurm"
    if shutil.which("oarsub") and shutil.which("oarstat"):
        return "oar"
    return "local"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def local_machine() -> dict:
    cpus = os.cpu_count() or 0
    mem_gb = None
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        out = _run(["sysctl", "-n", "hw.memsize"]).strip()
        mem_gb = int(out) / 1e9 if out.isdigit() else None
    gpus = []
    if shutil.which("nvidia-smi"):
        gpus = [
            g for g in _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).splitlines() if g
        ]
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        gpus = ["Apple MPS (not used by the benchmark's GPU models: they run on CPU)"]
    return {"cpus": cpus, "memory_gb": round(mem_gb) if mem_gb else None, "gpus": gpus}


def cmd_resources(args: argparse.Namespace) -> int:
    cluster, path = load_cluster(args.cluster_config)
    found = detect_scheduler()
    configured = cluster.get("scheduler", "slurm") if cluster else None
    print(f"scheduler on PATH: {found}")
    print(f"cluster config: {path or 'MISSING'}" + (f" (scheduler: {configured})" if cluster else ""))
    if cluster and configured != found:
        print(
            f"WARNING: the cluster config says {configured!r} but this machine has {found!r}"
            " (wrong machine, or the config needs updating)."
        )
    for key in ("workspace", "python_path"):
        if cluster.get(key):
            exists = Path(cluster[key]).expanduser().exists()
            print(f"{key}: {cluster[key]}" + ("" if exists else "  <-- does not exist on this machine"))
    res = cluster.get("resources", {})
    print(
        f"per task: {res.get('num_cpus', 8)} CPUs, {res.get('memory_limit', 32)} GB (resources in the cluster config)"
    )

    if found == "local":
        m = local_machine()
        print(f"\n# This machine\nCPUs: {m['cpus']}, memory: {m['memory_gb']} GB, GPUs: {m['gpus'] or 'none'}")
        return 0

    sys.path.insert(0, str(CHECK_JOBS_SCRIPTS))
    from backends import detect  # the check-jobs skill's read-only scheduler queries

    be = detect(found)
    user = os.environ.get("USER", "")
    if found == "slurm":
        accts = sorted(
            {a for a in _run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={user}", "format=account"]).split() if a}
        )
        sl, extra = cluster.get("slurm", {}), cluster.get("extra_benchmark_slurm", {})
        parts = {
            "gpu": sl.get("gpu_partition"),
            "cpu": sl.get("cpu_partition"),
            "stability/validity": extra.get("partition"),
        }
        mine = _run(["squeue", "-u", user, "-h", "-o", "%T"]).split()
    else:
        accts = []
        queue = cluster.get("oar", {}).get("queue") or "default"
        parts = {"all (OAR queues share the nodes)": queue}
        mine = [ln for ln in _run(["oarstat", "-u", user]).splitlines()[2:] if ln.strip()]
    print(f"accounts: {', '.join(accts) or '?'}; your queued/running jobs: {len(mine)}")
    seen = {}
    for role, p in parts.items():
        if not p or p.startswith("<"):
            print(f"\n## {role}: no partition/queue set in the cluster config")
            continue
        if p in seen:
            print(f"\n## {role}: same as {seen[p]}")
            continue
        seen[p] = role
        pi = be.partition_info(p, accts)
        idle_cpu, idle_gpu = pi["tot_cpu"] - pi["alloc_cpu"], pi["tot_gpu"] - pi["alloc_gpu"]
        room_cpu = max(0, min(idle_cpu, (pi["cap_cpu"] - pi["acct_cpu"]) if pi["cap_cpu"] else idle_cpu))
        room_gpu = max(0, min(idle_gpu, (pi["cap_gpu"] - pi["acct_gpu"]) if pi["cap_gpu"] else idle_gpu))
        print(f"\n## {role}: {pi['header']}")
        print(f"total: {pi['tot_cpu']} CPUs, {pi['tot_gpu']} GPUs; idle now: {idle_cpu} CPUs, {idle_gpu} GPUs")
        print(f"{pi['acct_label']} running: {pi['acct_cpu']} CPUs, {pi['acct_gpu']} GPUs; by user: {pi['by_user']}")
        if pi["pend_others"]:
            print(f"others' pending jobs by reason: {pi['pend_others']}")
        if pi["down"]:
            print(f"unavailable nodes: {len(pi['down'])}")
        print(f"FREE ROOM for you now: {room_cpu} CPUs, {room_gpu} GPUs (idle, within the account cap)")
        if found == "slurm" and pi["tot_gpu"]:
            gres = sorted(set(_run(["sinfo", "-p", p, "-h", "-o", "%G"]).split()))
            print(f"GPU types: {', '.join(gres)}")
    return 0


# ---- plan ---------------------------------------------------------------------------------------
def n_datasets() -> int:
    from selectarena.datasets import BENCHMARK_DATA_FOUNDRY_TASKS

    return len(BENCHMARK_DATA_FOUNDRY_TASKS)


def _hours(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:.0f} h ({h / 24:.1f} days)" if h >= 48 else f"{h:.1f} h"


def build_config(info: dict, variant: str, cpu_slots: int, gpu_slots: int, local: bool) -> dict:
    """The benchmark config of the run (published settings + the concurrency of each array)."""
    base_path = REPO / "configs" / "benchmark.yaml"
    base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}
    gpu_resources = {"num_gpus": 1}
    for job in base.get("model_jobs", []):  # keep the site's GPU memory hint (machine-specific)
        fake = job.get("resources", {}).get("fake_memory_for_estimates")
        if job.get("resources", {}).get("num_gpus") and fake:
            gpu_resources["fake_memory_for_estimates"] = fake
    if local and gpu_slots == 0:
        gpu_resources = {"num_gpus": 0}
    gpu_job = {"name": "gpu", "models": [[m, 0] for m in GPU_MODELS], "resources": gpu_resources}
    cpu_job = {"name": "cpu", "models": [[m, 0] for m in CPU_MODELS]}
    if not local:
        gpu_job["scheduler"] = {"array_job_limit": max(1, gpu_slots)}
        cpu_job["scheduler"] = {"array_job_limit": max(1, cpu_slots)}
    return {
        "benchmark_name": f"selectarena_{info['module']}",
        "feature_selection": {
            "methods": [info["class"]],
            "total_budget": TOTAL_BUDGET,
            "proxy_models": ["lgbm"],
            "fs_time_limits": [FS_TIME_LIMIT],
            "include_default": False,
            "no_fallback": False,
            "encoding_permutation_seeds": [],
        },
        "time_limit": MODEL_TIME_LIMIT,
        "experiment": {"n_random_configs": 0},
        "task_subset": {"split_indices": PUBLISHED_SPLITS if variant == "full" else LIGHT_SPLITS},
        "model_jobs": [gpu_job, cpu_job],
        "extra_benchmark": {"modes": EXTRA_MODES if variant == "full" else [], "noise_type": "gaussian"},
    }


def cmd_plan(args: argparse.Namespace) -> int:
    info = resolve(args.name)
    if info is None:
        print(f"NOT_REGISTERED {args.name}: run the add-method skill first.")
        return 2
    cluster, _ = load_cluster(args.cluster_config)
    scheduler = cluster.get("scheduler", "slurm") if cluster else detect_scheduler()
    local = scheduler == "local"
    cpt = int(cluster.get("resources", {}).get("num_cpus") or 8)
    extra_cpt = int(cluster.get("extra_benchmark_slurm", {}).get("cpus_per_task") or cpt)
    walltime_extra = cluster.get("extra_benchmark_slurm", {}).get("time") or cluster.get("extra_benchmark_oar", {}).get(
        "walltime", "10:00:00"
    )

    n_ds = n_datasets()
    splits = PUBLISHED_SPLITS if args.variant == "full" else LIGHT_SPLITS
    modes = EXTRA_MODES if args.variant == "full" else []
    per_model = n_ds * len(splits) * TOTAL_BUDGET
    gpu_items, cpu_items = per_model * len(GPU_MODELS), per_model * len(CPU_MODELS)
    extra_jobs = {m: n_ds * TOTAL_BUDGET for m in modes}

    if local:
        # Without a CUDA GPU the GPU models fit on CPU (num_gpus: 0 in the written config).
        gpu_slots, cpu_slots = (1 if shutil.which("nvidia-smi") else 0), 1
    else:
        gpu_slots = max(0, args.gpus)
        cpu_left = args.cpus - gpu_slots * cpt
        if gpu_slots == 0:
            print("ERROR: the GPU models (TabICLv2, RealMLP) need at least 1 GPU on a cluster.")
            return 1
        if cpu_left < cpt * (2 if modes else 1):
            need = gpu_slots * cpt + cpt * (2 if modes else 1)
            print(f"ERROR: {args.cpus} CPUs leave too few for the CPU models; need at least {need} ({cpt} per task,")
            print(f"       {cpt} for each of the {gpu_slots} GPU tasks).")
            return 1
        cpu_slots_total = cpu_left // cpt

    print(f"# {info['paper_name'] or info['class']} ({info['class']}), {args.variant} run, scheduler {scheduler}")
    print(
        f"performance: {n_ds} datasets x {len(splits)} split(s) x {TOTAL_BUDGET} budgets x 5 models"
        f" = {gpu_items + cpu_items} fits ({gpu_items} GPU: {', '.join(GPU_MODELS)}; {cpu_items} CPU)"
    )
    if modes:
        print(f"stability/validity: {' + '.join(f'{v} {k}' for k, v in extra_jobs.items())} jobs (CPU only)")
    else:
        print("stability/validity: not part of this run")

    def work(fs: float) -> tuple[float, float, float]:
        """Seconds of work of the GPU array, the CPU array and the stability/validity array."""
        w_gpu = per_model * sum(MODEL_SECONDS[m] + fs + ITEM_OVERHEAD_S for m in GPU_MODELS)
        w_cpu = per_model * sum(MODEL_SECONDS[m] + fs + ITEM_OVERHEAD_S for m in CPU_MODELS)
        w_extra = sum(n * (EXTRA_FS_FACTOR[m] * fs + ITEM_OVERHEAD_S) for m, n in extra_jobs.items())
        return w_gpu, w_cpu, w_extra

    extra_slots = 0
    if not local:
        # Split the CPU tasks between the performance CPU array and the extra array by their work.
        fs_split = args.fs_seconds if args.fs_seconds is not None else FS_SCENARIOS["typical"]
        _, w_cpu, w_extra = work(fs_split)
        cpu_slots = cpu_slots_total
        if modes:
            ratio = extra_cpt / cpt
            share = w_extra * ratio / (w_cpu + w_extra * ratio)
            extra_slots = min(sum(extra_jobs.values()), max(1, round(cpu_slots_total * share / ratio)))
            cpu_slots = max(1, cpu_slots_total - math.ceil(extra_slots * ratio))

    scenarios = {"given": args.fs_seconds} if args.fs_seconds is not None else FS_SCENARIOS
    print("\nEstimates (mean times of the published models; the selection time per fit is the unknown):")
    for label, fs in scenarios.items():
        w_gpu, w_cpu, w_extra = work(fs)
        if local:
            print(
                f"  {label:8s} ({fs:>5.0f} s/selection): {_hours(w_gpu + w_cpu + w_extra)} on this machine, one at a time"
            )
            continue
        t_gpu, t_cpu = w_gpu / gpu_slots, w_cpu / cpu_slots
        t_extra = w_extra / extra_slots if extra_slots else 0
        print(
            f"  {label:8s} ({fs:>5.0f} s/selection): GPU array {_hours(t_gpu)} | CPU array {_hours(t_cpu)}"
            + (f" | stability/validity {_hours(t_extra)}" if modes else "")
            + f"  -> about {_hours(max(t_gpu, t_cpu, t_extra))} if the slots stay busy"
        )
        if modes and EXTRA_FS_FACTOR["stability"] * fs > _secs(walltime_extra):
            print(
                f"           WARNING: a stability job (~{EXTRA_FS_FACTOR['stability']} selections) would exceed the"
                f" {walltime_extra} walltime of the stability/validity jobs; raise it in the cluster config."
            )
    if not local:
        print(
            f"\nConcurrency for {args.cpus} CPUs / {args.gpus} GPUs ({cpt} CPUs per task;"
            f" split for {fs_split:.0f} s/selection):"
            f"\n  GPU array: {gpu_slots} tasks at once ({gpu_slots} GPUs + {gpu_slots * cpt} CPUs)"
            f"\n  CPU array: {cpu_slots} tasks at once ({cpu_slots * cpt} CPUs)"
            + (
                f"\n  stability/validity array: {extra_slots} tasks at once ({extra_slots * extra_cpt} CPUs)"
                if modes
                else ""
            )
        )
        if scheduler == "oar":
            print("  OAR has no per-array limit: every subjob that fits starts; the numbers above are a target only.")
        print(
            "Bounds: a fit stops at its time limits (selection 1 h + model 1 h), so no fit takes over 2 h;"
            " the largest datasets make a tail of a few hours."
        )
    else:
        print("\nLocal: without a CUDA GPU the GPU models run on CPU, much slower than estimated above.")

    config = build_config(info, args.variant, cpu_slots, gpu_slots, local)
    suffix = "" if args.variant == "full" else "_light"
    out = REPO / "configs" / f"benchmark_{info['module']}{suffix}.yaml"
    if args.write:
        header = (
            f"# {args.variant} benchmark run of {info['class']} (written by the benchmark-method skill).\n"
            "# Published settings; only the array concurrency (array_job_limit) is machine-specific.\n"
        )
        out.write_text(header + yaml.safe_dump(config, sort_keys=False))
        print(f"\nwrote {out.relative_to(REPO)} (benchmark_name {config['benchmark_name']})")
        if modes:
            print(f"stability/validity concurrency: pass --parallel {extra_slots} to run_extra_benchmark.py setup")
    else:
        print(f"\n(dry run; --write writes {out.relative_to(REPO)})")
    return 0


def _secs(hms: str) -> int:
    parts = [int(x) for x in str(hms).replace("-", ":").split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    days = parts[0] if len(parts) == 4 else 0
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("resolve")
    p.add_argument("name")
    p = sub.add_parser("resources")
    p.add_argument("--cluster-config", default=None)
    p = sub.add_parser("plan")
    p.add_argument("name")
    p.add_argument("--variant", choices=["full", "light"], required=True)
    p.add_argument("--cpus", type=int, default=0, help="CPUs to use in total (all arrays together).")
    p.add_argument("--gpus", type=int, default=0, help="GPUs to use in total.")
    p.add_argument("--fs-seconds", type=float, default=None, help="Selection time per fit, if known.")
    p.add_argument("--cluster-config", default=None)
    p.add_argument("--write", action="store_true", help="Write configs/benchmark_<module>[_light].yaml.")
    args = ap.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    return {"resolve": cmd_resolve, "resources": cmd_resources, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
