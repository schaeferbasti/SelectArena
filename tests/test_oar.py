"""OAR scheduling: the batch scripts, the oarsub commands and the array task runner."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import yaml
from tabflow_slurm import PathSetup, TabArenaV0pt1ResourcesSetup

from selectarena.benchmark.oar import OarSetup, run_array_task
from selectarena.config import ClusterConfig
from selectarena.extra_benchmark.oar import ExtraBenchmarkOarSettings, write_oar_arrays
from selectarena.extra_benchmark.slurm import build_job_commands
from selectarena.oar import format_walltime

REPO_ROOT = Path(__file__).parent.parent
PIPELINE = "FSBench__MIFeatureSelector__2__0__lgbm__60"


def _item(i: int) -> dict:
    return {"experiment": f"exp{i}", "dataset": "toy", "fold": i, "repeat": 0}


def test_format_walltime():
    assert format_walltime(3600) == "1:00:00"
    assert format_walltime(11 * 3600 + 1) == "11:01:00"


def test_oar_setup_writes_one_array_per_bundle_size(tmp_path):
    path_setup = PathSetup(workspace=tmp_path / "ws", python_path="/venv/bin/python")
    path_setup.get_setup_out_path("bench").mkdir(parents=True)
    resources = TabArenaV0pt1ResourcesSetup(
        time_limit=3600, num_gpus=1, time_limit_for_model_agnostic_preprocessing=600
    )
    scheduler = OarSetup(gpu_properties="gpumodel IN ('A100','H100')", modules=["cuda/cuda-12.1"])
    jobs = [
        {"bundle_size": 2, "items": [_item(0), _item(1)]},
        {"bundle_size": 2, "items": [_item(2)]},
        {"bundle_size": 1, "items": [_item(3)]},
    ]
    commands = scheduler.get_run_commands(
        jobs_dict={"defaults": {"python": "/venv/bin/python"}, "jobs": jobs},
        path_setup=path_setup,
        benchmark_name="bench",
        parallel_safe_benchmark_name="bench_gpu",
        resources_setup=resources,
        print_summary=False,
    )
    assert len(commands) == 2  # bundle sizes 1 and 2
    args = shlex.split(commands[1])
    assert args[:2] == ["oarsub", "-S"]
    script, params = args[2], args[4]
    assert params.endswith("oar_run_data_bench_gpu_size2.params")

    text = open(script).read()
    assert "#OAR -l /nodes=1/gpu=1,walltime=3:20:00" in text  # 2 x (1 h + 10 min) + 1 h
    assert "#OAR -p \"gpumodel IN ('A100','H100')\"" in text
    assert "module load cuda/cuda-12.1" in text
    assert "-m selectarena.benchmark.oar" in text

    lines = open(params).read().splitlines()
    assert [line.split()[1] for line in lines] == ["0", "1"]
    shipped = json.loads(open(lines[0].split()[0]).read())
    assert shipped["jobs"] == [{"items": [_item(0), _item(1)]}, {"items": [_item(2)]}]  # no bundle_size hint


def test_run_array_task_runs_every_item(tmp_path, monkeypatch):
    # A stand-in for run_tabarena_experiment.py that records its arguments and fails on fold 1.
    run_script = tmp_path / "fake_run.py"
    run_script.write_text(
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['FAKE_LOG'], 'a') as f: f.write(' '.join(args) + '\\n')\n"
        "sys.exit(int(args[args.index('--fold') + 1] == '1'))\n"
    )
    defaults = {
        "python": sys.executable,
        "run_script": str(run_script),
        "job_batch_dir": "batch",
        "output_dir": "out",
        "num_cpus": 2,
        "num_gpus": 0,
        "memory_limit": 4,
        "setup_ray_for_slurm_shared_resources_environment": True,
        "ignore_cache": False,
        "scratch_root": str(tmp_path / "scratch"),
    }
    json_path = tmp_path / "jobs.json"
    json_path.write_text(json.dumps({"defaults": defaults, "jobs": [{"items": [_item(0), _item(1), _item(2)]}]}))
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LOG", str(log))

    assert run_array_task(json_path, 0) == 1
    calls = log.read_text().splitlines()
    assert len(calls) == 3  # a failed item does not stop its siblings
    assert "--experiment exp2" in calls[2] and "--num_cpus 2" in calls[2]
    assert list((tmp_path / "scratch").iterdir()) == []  # scratch removed


def test_extra_benchmark_oar_arrays(tmp_path):
    commands = build_job_commands(
        modes=["stability", "validity"],
        method_names=[PIPELINE],
        task_ids=["UserTask|1|a/1", "UserTask|2|b/2"],
        results_dir=tmp_path / "results",
    )
    oarsub_commands = write_oar_arrays(
        commands,
        out_dir=tmp_path / "setup",
        python_path="/venv/bin/python",
        settings=ExtraBenchmarkOarSettings(max_array_size=3),
    )
    assert len(oarsub_commands) == 2
    script = shlex.split(oarsub_commands[0])[2]
    text = open(script).read()
    assert "#OAR -l /nodes=1/core=8,walltime=10:00:00" in text
    assert "-m selectarena.extra_benchmark.runner" in text


def test_calculco_example_config_selects_oar():
    cluster = ClusterConfig(**yaml.safe_load(open(REPO_ROOT / "configs" / "cluster.calculco.example.yaml")))
    assert cluster.scheduler == "oar"
    OarSetup(**cluster.oar)
    ExtraBenchmarkOarSettings(**cluster.extra_benchmark_oar)
