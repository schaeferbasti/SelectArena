#!/usr/bin/env python3
"""Snapshot of the user's cluster jobs and the cluster, for the check-jobs skill (SLURM or OAR).

Prints (plain text, for Claude to read):
  1. per array job: state counts, concurrency limit, partitions/queues, CPUs/GPUs per task, runtimes, completions
     since the last check, running tasks near their time limit, and every task that ended in a non-COMPLETED state;
  2. per partition/queue the user's jobs use: cluster idle CPUs/GPUs, the account's usage vs. its cap (if the
     scheduler exposes one), usage by user, pending jobs of others, free room for the account;
  3. per active array a finish-time estimate (bootstrap over its own task durations) for the current concurrency
     and alternatives.

Usage: collect.py [--scheduler auto|slurm|oar] [--since 24h] [--array ID ...] [--raw]
  --since  also report arrays that finished in this window (default: since the last run, at most 48h; first run 24h)
  --array  extra array IDs to include even if they have no tasks queued
  --raw    (OAR) print one raw job and resource record, to check the field mapping on a new cluster
Read-only: only status queries. Stores the time of the last run in ~/.cache/check-jobs/last_run.
"""

from __future__ import annotations

import argparse
import heapq
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends import detect
from backends.common import USER

STATE_FILE = Path.home() / ".cache" / "check-jobs" / "last_run"
ACTIVE = {"PENDING", "RUNNING"}
SLOT_STATES = {"COMPLETED", "TIMEOUT"}  # durations used for the estimate (a timed-out task held its slot that long)


def hfmt(s: float) -> str:
    return f"{s / 3600:.1f}h" if s >= 3600 else f"{s / 60:.0f}m"


def makespan(running_left: list[float], pending: list[float], slots: int) -> float:
    """Greedy list scheduling: running tasks finish after running_left, pending tasks (in order) take their durations."""
    slots = max(1, slots)
    rl = sorted(running_left)
    heap = rl[:slots] + [0.0] * max(0, slots - len(rl))
    heapq.heapify(heap)
    for d in rl[slots:] + list(pending):
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + d)
    return max(heap) if heap else 0.0


def simulate(
    durations: list[float], elapsed: list[float], n_pending: int, slots: int, n: int = 300
) -> tuple[float, float]:
    """Bootstrap: pending tasks draw from the observed durations, running tasks from those longer than their
    elapsed time (residual). Returns the median and 90th percentile of the makespan in seconds."""
    rng = random.Random(0)
    res = []
    for _ in range(n):
        left = []
        for e in elapsed:
            longer = [d for d in durations if d > e]
            left.append((rng.choice(longer) - e) if longer else 0.05 * max(durations))
        res.append(makespan(left, [rng.choice(durations) for _ in range(n_pending)], slots))
    res.sort()
    return res[len(res) // 2], res[int(0.9 * (len(res) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", default="auto", choices=["auto", "slurm", "oar"])
    ap.add_argument("--since", default=None)
    ap.add_argument("--array", nargs="*", default=[])
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    be = detect(args.scheduler)
    if args.raw:
        print(be.raw_samples() if hasattr(be, "raw_samples") else "--raw is only implemented for OAR")
        return

    now = datetime.now()
    if args.since:
        m = re.fullmatch(r"(\d+)([hd])", args.since)
        since = now - timedelta(hours=int(m.group(1)) * (24 if m.group(2) == "d" else 1))
    elif STATE_FILE.exists():
        since = max(datetime.fromtimestamp(float(STATE_FILE.read_text().strip())), now - timedelta(hours=48))
    else:
        since = now - timedelta(hours=24)
    since_s = since.strftime("%Y-%m-%dT%H:%M")
    print(
        f"# check-jobs snapshot {now:%Y-%m-%d %H:%M} (scheduler {be.name}; user {USER};"
        f" 'since' = {since:%Y-%m-%d %H:%M})"
    )

    arrays = be.collect(since, args.array)
    report = []
    for aid in sorted(arrays, key=lambda x: int(x) if x.isdigit() else 0):
        a = arrays[aid]
        tasks = a["tasks"]
        if not tasks:
            continue
        running = [dict(t, id=k) for k, t in tasks.items() if t["state"] == "RUNNING"]
        pending = [dict(t, id=k) for k, t in tasks.items() if t["state"] == "PENDING"]
        states = Counter(t["state"] for t in tasks.values() if t["state"] not in ACTIVE)
        states.update({"RUNNING": len(running), "PENDING": len(pending)})
        done = [t["elapsed"] for t in tasks.values() if t["state"] == "COMPLETED" and t["elapsed"] > 60]
        durs = [t["elapsed"] for t in tasks.values() if t["state"] in SLOT_STATES and t["elapsed"] > 0]
        bad = {k: t for k, t in tasks.items() if t["state"] not in ACTIVE | {"COMPLETED"}}
        if not running and not pending:
            new_bad = {k: t for k, t in bad.items() if t["end"] >= since_s}
            last_end = max((t["end"] for t in tasks.values()), default="?")
            print(f"\n## Array {aid} FINISHED (last end {last_end})  name={a['name']}  states={dict(states)}")
            for k, t in sorted(new_bad.items(), key=lambda kv: kv[1]["end"]):
                print(f"  {k} {t['state']} {hfmt(t['elapsed'])} {t['end']} {t['node']} {t['exit']} NEW")
            continue
        info = be.array_info(aid, tasks)
        sample = (running or pending or list(tasks.values()))[0]
        cpt, gpt, limit = sample["cpus"], sample["gpus"], sample["limit"]
        print(f"\n## Array {aid}  name={a['name']}  account={a['acct']}")
        print(
            f"states: {dict(states)}  | {be.throttle_word}={info['throttle'] or 'none'}"
            f" | partitions={','.join(info['partitions'])}"
            f" | per task {cpt} CPU, {gpt} GPU, limit {hfmt(limit) if limit else '?'}"
            + (f" | {info['notes']}" if info["notes"] else "")
        )
        if running:
            print("running on: " + ", ".join(f"{p}x{n}" for p, n in Counter(t["part"] for t in running).items()))
        if pending:
            print("pending reasons: " + ", ".join(f"{r}x{n}" for r, n in Counter(t["reason"] for t in pending).items()))
        if done:
            q90 = statistics.quantiles(done, n=10)[-1] if len(done) >= 10 else max(done)
            print(
                f"completed runtimes (n={len(done)}): median {hfmt(statistics.median(done))}, mean "
                f"{hfmt(statistics.mean(done))}, p90 {hfmt(q90)}, max {hfmt(max(done))}"
            )
        recent_ok = [k for k, t in tasks.items() if t["state"] == "COMPLETED" and t["end"] >= since_s]
        print(f"completed since {since:%m-%d %H:%M}: {len(recent_ok)}")
        if bad:
            print("NON-COMPLETED tasks (id state elapsed end node exit):")
            for k, t in sorted(bad.items(), key=lambda kv: kv[1]["end"]):
                new = " NEW" if t["end"] >= since_s else ""
                raw = f" [{t['raw']}]" if be.name != "slurm" and t["raw"] else ""
                print(f"  {k} {t['state']} {hfmt(t['elapsed'])} {t['end']} {t['node']} {t['exit']}{new}{raw}")
        nodes = Counter(t["node"] for t in bad.values() if t["state"] != "CANCELLED")
        if len(bad) >= 3 and nodes and nodes.most_common(1)[0][1] >= 0.6 * len(bad):
            print(f"  NOTE: {nodes.most_common(1)[0][1]}/{len(bad)} failures on {nodes.most_common(1)[0][0]}")
        near = [t for t in running if t["limit"] and t["elapsed"] > 0.8 * t["limit"]]
        if near:
            print(f"running > 80% of time limit: {', '.join(t['id'] + ' (' + hfmt(t['elapsed']) + ')' for t in near)}")
        report.append(
            dict(
                aid=aid,
                name=a["name"],
                acct=a["acct"],
                running=running,
                pending=len(pending),
                throttle=info["throttle"],
                cpt=cpt,
                gpt=gpt,
                durs=durs,
                parts=info["partitions"],
                gpu_parts={t["part"] for t in tasks.values() if t["gpus"]},
            )
        )

    # ---- partitions / queues ----
    parts_seen = sorted({p for r in report for p in r["parts"] if p})
    gpu_parts = set().union(*(r["gpu_parts"] for r in report)) if report else set()
    cpu_parts = {p for r in report if not r["gpt"] for p in r["parts"]}
    accts = sorted({r["acct"] for r in report if r["acct"] and r["acct"] != "?"})
    free = {}
    print("\n# Partitions" if be.name == "slurm" else "\n# Queues")
    for p in parts_seen:
        pi = be.partition_info(p, accts)
        print(f"\n## {pi['header']}")
        print(
            f"cluster: CPUs {pi['alloc_cpu']}/{pi['tot_cpu']} allocated ({pi['tot_cpu'] - pi['alloc_cpu']} idle)"
            + (
                f", GPUs {pi['alloc_gpu']}/{pi['tot_gpu']} allocated ({pi['tot_gpu'] - pi['alloc_gpu']} idle)"
                if pi["tot_gpu"]
                else ""
            )
            + (f"; unavailable nodes: {' '.join(pi['down'])}" if pi["down"] else "")
        )
        print(
            f"{pi['acct_label']}: {pi['acct_cpu']} CPUs, {pi['acct_gpu']} GPUs running;"
            f" by user ({pi['by_user_unit']}): {pi['by_user']}"
        )
        if pi["pend_others"]:
            print(f"pending jobs of others by reason: {pi['pend_others']}")
        idle_cpu = pi["tot_cpu"] - pi["alloc_cpu"]
        idle_gpu = pi["tot_gpu"] - pi["alloc_gpu"]
        room_cpu = max(0, min(idle_cpu, (pi["cap_cpu"] - pi["acct_cpu"]) if pi["cap_cpu"] else idle_cpu))
        room_gpu = max(0, min(idle_gpu, (pi["cap_gpu"] - pi["acct_gpu"]) if pi["cap_gpu"] else idle_gpu))
        free[p] = (room_cpu, room_gpu)
        what = "account cap headroom" if (pi["cap_cpu"] or pi["cap_gpu"]) else "no cap known"
        if p in gpu_parts:
            print(f"=> more GPU tasks possible here now: {room_gpu} (min of idle GPUs and {what})")
        if p in cpu_parts or p not in gpu_parts:
            print(f"=> free CPUs for the account now: {room_cpu} (min of idle CPUs and {what})")

    # ---- estimates ----
    print(
        "\n# Estimates (bootstrap over the array's own completed/timed-out task durations; median / 90th pct of"
        " the finish time; free capacity is SHARED between arrays on the same partition)"
    )
    word = be.throttle_word
    for r in report:
        if not r["running"] and not r["pending"]:
            continue
        if len(r["durs"]) < 3:
            print(
                f"\n## {r['aid']} {r['name']}: <3 finished tasks, no estimate "
                f"({len(r['running'])} running, {r['pending']} pending)"
            )
            continue
        units_free = 0
        for p in r["parts"]:
            rc, rg = free.get(p, (0, 0))
            fits = [rc // r["cpt"]] if r["cpt"] else []
            if r["gpt"]:
                fits.append(rg // r["gpt"])
            units_free += min(fits) if fits else 0
        max_now = len(r["running"]) + units_free
        total = len(r["running"]) + r["pending"]
        # without a per-array limit the scheduler runs what fits: currently that is max_now
        cur = r["throttle"] or min(total, max_now)
        opts = {max(1, cur // 2), cur}
        if r["throttle"]:
            opts |= {max(cur, min(max_now, total)), min(total, cur + max(1, cur // 2))}
        opts = sorted(opts - {0})
        print(
            f"\n## {r['aid']} {r['name']}: {len(r['running'])} running, {r['pending']} pending; durations "
            f"median {hfmt(statistics.median(r['durs']))}, mean {hfmt(statistics.mean(r['durs']))}, "
            f"max {hfmt(max(r['durs']))}; slots that can run right now ~{max_now}"
        )
        unit = "GPUs" if r["gpt"] else "CPUs"
        per = r["gpt"] or r["cpt"]
        elapsed = [t["elapsed"] for t in r["running"]]
        for o in opts:
            eff = min(o, max_now) if o > cur else o
            med, p90 = simulate(r["durs"], elapsed, r["pending"], eff)
            tag = " (current)" if o == cur else ""
            if eff < o:
                tag += f" (only ~{eff} can run now)"
            delta = (eff - min(cur, total)) * per
            res = f" {'+' if delta > 0 else ''}{delta} {unit} vs. now" if delta else ""
            print(
                f"  {word} {o}{tag}: done ~{(now + timedelta(seconds=med)):%a %H:%M} (+{hfmt(med)}), "
                f"90%: {(now + timedelta(seconds=p90)):%a %H:%M} (+{hfmt(p90)}){res}"
            )
    print(f"\n(change concurrency: {be.throttle_hint})")

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(time.time()))


if __name__ == "__main__":
    main()
