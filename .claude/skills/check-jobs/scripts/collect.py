#!/usr/bin/env python3
"""Snapshot of the user's SLURM jobs and the cluster, for the check-jobs skill.

Prints (plain text, for Claude to read):
  1. per array job: state counts, throttle, partitions, CPUs/GPUs per task, runtimes of completed tasks,
     running tasks near their time limit, and every task that ended in a non-COMPLETED state;
  2. per partition the user's jobs touch: account usage vs. the QOS per-account cap, cluster-wide free CPUs/GPUs,
     pending jobs of others;
  3. a runtime estimate per array for the current throttle and alternatives (list scheduling with the
     median runtime of the completed tasks; capped by what the account cap / free resources allow).

Usage: collect.py [--since 24h] [--array ID ...]
  --since  also report arrays that finished in this window (default: since the last run, at most 48h; first run 24h)
  --array  extra array IDs to include even if they have no tasks queued (e.g. ones named in the HANDOFF)
Read-only: only squeue/sacct/scontrol/sinfo/sacctmgr queries. Stores the time of the last run in
~/.cache/check-jobs/last_run.
"""

from __future__ import annotations

import argparse
import heapq
import os
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

USER = os.environ.get("USER", "")
STATE_FILE = Path.home() / ".cache" / "check-jobs" / "last_run"
TERMINAL_OK = {"COMPLETED"}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        return f"__ERROR__ {e}"


def secs(t: str) -> int:
    """SLURM time ([D-]HH:MM:SS, MM:SS, UNLIMITED) -> seconds."""
    if not t or t in ("UNLIMITED", "INVALID", "Partition_Limit", "NOT_SET"):
        return 0
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [int(float(p)) for p in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def hfmt(s: float) -> str:
    return f"{s / 3600:.1f}h" if s >= 3600 else f"{s / 60:.0f}m"


def gpus_of(tres: str) -> int:
    """'gres/gpu:1,gres/localtmp:10' or 'gres/gpu=2' or 'gpu:l40s:1' -> 1/2/1."""
    n = 0
    for m in re.finditer(r"gpu(?::[A-Za-z0-9_-]+)?[:=](\d+)", tres or ""):
        n += int(m.group(1))
    return n


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
    ap.add_argument("--since", default=None)
    ap.add_argument("--array", nargs="*", default=[])
    args = ap.parse_args()

    now = datetime.now()
    if args.since:
        m = re.fullmatch(r"(\d+)([hd])", args.since)
        since = now - timedelta(hours=int(m.group(1)) * (24 if m.group(2) == "d" else 1))
    elif STATE_FILE.exists():
        since = max(datetime.fromtimestamp(float(STATE_FILE.read_text().strip())), now - timedelta(hours=48))
    else:
        since = now - timedelta(hours=24)
    print(f"# check-jobs snapshot {now:%Y-%m-%d %H:%M} (user {USER}; 'since' = {since:%Y-%m-%d %H:%M})")

    # ---- queued tasks (expanded per task) ----
    q = run(["squeue", "-u", USER, "-r", "-h", "-o", "%i|%T|%P|%C|%b|%M|%l|%R|%a|%j|%N"])
    arrays: dict[str, dict] = defaultdict(lambda: {"queued": [], "acct": None, "name": None})
    for line in q.strip().splitlines():
        jid, st, part, cpus, tres, elapsed, limit, reason, acct, name, node = line.split("|")
        aid = jid.split("_")[0]
        a = arrays[aid]
        a["queued"].append(
            dict(
                id=jid,
                state=st,
                part=part,
                cpus=int(cpus),
                gpus=gpus_of(tres),
                elapsed=secs(elapsed),
                limit=secs(limit),
                reason=reason,
                node=node,
            )
        )
        a["acct"], a["name"] = acct, name

    # ---- arrays that ended recently / named explicitly ----
    s = run(["sacct", "-u", USER, "-X", "-n", "-P", "-S", since.strftime("%Y-%m-%dT%H:%M"), "-o", "JobID,State"])
    for line in s.strip().splitlines():
        jid = line.split("|")[0]
        if re.match(r"^\d+(_\d+)?$", jid):
            arrays[jid.split("_")[0]]
    for aid in args.array:
        arrays[aid]

    part_usage_hint = defaultdict(lambda: {"cpus": 0, "gpus": 0})
    report = []
    for aid in sorted(arrays, key=int):
        a = arrays[aid]
        acc = run(
            [
                "sacct",
                "-j",
                aid,
                "-X",
                "-n",
                "-P",
                "-o",
                "JobID,State,Elapsed,End,NodeList,Partition,AllocCPUS,AllocTRES,Timelimit,Account,JobName,ExitCode",
            ]
        )
        tasks = {}
        for line in acc.strip().splitlines():
            f = line.split("|")
            if len(f) < 12 or "[" in f[0]:
                continue
            tasks[f[0]] = dict(
                state=f[1].split()[0],
                elapsed=secs(f[2]),
                end=f[3],
                node=f[4],
                part=f[5],
                cpus=int(f[6] or 0),
                gpus=gpus_of(f[7]),
                limit=secs(f[8]),
                acct=f[9],
                name=f[10],
                exit=f[11],
            )
        if not tasks and not a["queued"]:
            continue
        a["acct"] = a["acct"] or next((t["acct"] for t in tasks.values()), "?")
        a["name"] = a["name"] or next((t["name"] for t in tasks.values()), "?")
        states = Counter(t["state"] for t in tasks.values() if t["state"] not in ("PENDING", "RUNNING"))
        running = [t for t in a["queued"] if t["state"] in ("RUNNING", "CONFIGURING", "COMPLETING")]
        pending = [t for t in a["queued"] if t["state"] == "PENDING"]
        states.update({"RUNNING": len(running), "PENDING": len(pending)})
        done = [t["elapsed"] for t in tasks.values() if t["state"] == "COMPLETED" and t["elapsed"] > 60]
        # durations for the estimate: completed (also fast ones) + tasks that hit the limit (they occupied a slot that long)
        durs = [t["elapsed"] for t in tasks.values() if t["state"] in ("COMPLETED", "TIMEOUT") and t["elapsed"] > 0]
        sc = run(["scontrol", "show", "job", aid])
        throttle = re.search(r"ArrayTaskThrottle=(\d+)", sc)
        throttle = int(throttle.group(1)) if throttle else 0
        parts = sorted(set(re.findall(r"Partition=(\S+)", sc)) | {t["part"] for t in a["queued"]})
        exc = re.search(r"ExcNodeList=(\S+)", sc)
        sample = (running or pending or list(tasks.values()) or [{}])[0]
        cpt, gpt = sample.get("cpus", 0), sample.get("gpus", 0)
        limit = sample.get("limit", 0)
        for t in running:
            part_usage_hint[t["part"]]["cpus"] += t["cpus"]
            part_usage_hint[t["part"]]["gpus"] += t["gpus"]

        bad = {k: t for k, t in tasks.items() if t["state"] not in TERMINAL_OK | {"PENDING", "RUNNING"}}
        if not running and not pending:
            new_bad = {k: t for k, t in bad.items() if t["end"] >= since.strftime("%Y-%m-%dT%H:%M")}
            last_end = max((t["end"] for t in tasks.values()), default="?")
            print(f"\n## Array {aid} FINISHED (last end {last_end})  name={a['name']}  states={dict(states)}")
            for k, t in sorted(new_bad.items(), key=lambda kv: kv[1]["end"]):
                print(f"  {k} {t['state']} {hfmt(t['elapsed'])} {t['end']} {t['node']} {t['exit']} NEW")
            continue
        print(f"\n## Array {aid}  name={a['name']}  account={a['acct']}")
        print(
            f"states: {dict(states)}  | throttle={throttle or 'none'} | partitions={','.join(p for p in parts if p)}"
            f" | per task {cpt} CPU, {gpt} GPU, limit {hfmt(limit) if limit else '?'}"
            + (f" | ExcNodeList={exc.group(1)}" if exc and exc.group(1) != "(null)" else "")
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
        recent_ok = [
            k for k, t in tasks.items() if t["state"] == "COMPLETED" and t["end"] >= since.strftime("%Y-%m-%dT%H:%M")
        ]
        print(f"completed since {since:%m-%d %H:%M}: {len(recent_ok)}")
        if bad:
            print("NON-COMPLETED tasks (id state elapsed end node exit):")
            for k, t in sorted(bad.items(), key=lambda kv: kv[1]["end"]):
                new = " NEW" if t["end"] >= since.strftime("%Y-%m-%dT%H:%M") else ""
                print(f"  {k} {t['state']} {hfmt(t['elapsed'])} {t['end']} {t['node']} {t['exit']}{new}")
        nodes = Counter(t["node"] for t in bad.values() if t["state"] not in ("CANCELLED",))
        if len(bad) >= 3 and nodes.most_common(1)[0][1] >= 0.6 * len(bad):
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
                throttle=throttle,
                cpt=cpt,
                gpt=gpt,
                done=done,
                durs=durs,
                parts=parts,
                limit=limit,
            )
        )

    # ---- partitions ----
    parts_seen = sorted({p for r in report for p in r["parts"] if p})
    accts = sorted({r["acct"] for r in report if r["acct"] and r["acct"] != "?"})
    free = {}
    print("\n# Partitions")
    for p in parts_seen:
        sp = run(["scontrol", "show", "partition", p])
        qos = re.search(r"\bQoS=(\S+)", sp)
        qos = qos.group(1) if qos and qos.group(1) != "N/A" else None
        cap_cpu = cap_gpu = None
        if qos:
            line = run(["sacctmgr", "-n", "-P", "show", "qos", qos, "format=MaxTRESPA,GrpTRES"]).strip()
            m = re.search(r"cpu=(\d+)", line)
            cap_cpu = int(m.group(1)) if m else None
            m = re.search(r"gres/gpu=(\d+)", line)
            cap_gpu = int(m.group(1)) if m else None
        # cluster-wide
        nodes = run(["sinfo", "-p", p, "-h", "-N", "-o", "%N"]).split()
        tot_cpu = alloc_cpu = tot_gpu = alloc_gpu = 0
        down = []
        if nodes:
            for ln in run(["scontrol", "show", "node", "-o", ",".join(sorted(set(nodes)))]).splitlines():
                name = re.search(r"NodeName=(\S+)", ln).group(1)
                st = re.search(r"State=(\S+)", ln).group(1)
                if set(st.split("+")) & {
                    "DOWN",
                    "DRAIN",
                    "DRAINED",
                    "DRAINING",
                    "FAIL",
                    "FAILING",
                    "MAINT",
                    "NOT_RESPONDING",
                }:
                    down.append(f"{name}:{st}")
                    continue
                tot_cpu += int(re.search(r"CPUTot=(\d+)", ln).group(1))
                alloc_cpu += int(re.search(r"CPUAlloc=(\d+)", ln).group(1))
                g = re.search(r"Gres=\S*?gpu(?::[A-Za-z0-9_-]+)?:(\d+)", ln)
                tot_gpu += int(g.group(1)) if g else 0
                ga = re.search(r"AllocTRES=\S*gres/gpu=(\d+)", ln)
                alloc_gpu += int(ga.group(1)) if ga else 0
        # account usage
        acct_cpu = acct_gpu = 0
        by_user = Counter()
        for acct in accts:
            for ln in run(["squeue", "-A", acct, "-p", p, "-t", "R", "-h", "-o", "%u|%C|%b"]).splitlines():
                u, c, b = ln.split("|")
                acct_cpu += int(c)
                acct_gpu += gpus_of(b)
                by_user[u] += gpus_of(b) if "gpu" in p else int(c)
        pend_others = Counter(
            ln.split("|")[1]
            for ln in run(["squeue", "-p", p, "-t", "PD", "-h", "-o", "%u|%r"]).splitlines()
            if not ln.startswith(USER + "|")
        )
        unit = "GPUs" if "gpu" in p else "CPUs"
        print(f"\n## {p} (QOS {qos}; per-account cap cpu={cap_cpu} gpu={cap_gpu})")
        print(
            f"cluster: CPUs {alloc_cpu}/{tot_cpu} allocated ({tot_cpu - alloc_cpu} idle)"
            + (f", GPUs {alloc_gpu}/{tot_gpu} allocated ({tot_gpu - alloc_gpu} idle)" if tot_gpu else "")
            + (f"; unavailable nodes: {' '.join(down)}" if down else "")
        )
        print(f"account {','.join(accts)}: {acct_cpu} CPUs, {acct_gpu} GPUs running; by user ({unit}): {dict(by_user)}")
        if pend_others:
            print(f"pending jobs of others by reason: {dict(pend_others)}")
        idle_gpu = tot_gpu - alloc_gpu
        idle_cpu = tot_cpu - alloc_cpu
        if "gpu" in p or tot_gpu:
            room = min(idle_gpu, (cap_gpu - acct_gpu) if cap_gpu else idle_gpu)
            free[p] = ("gpu", max(0, room))
            print(f"=> more GPU tasks possible here now: {max(0, room)} (min of idle GPUs and account cap headroom)")
        else:
            room = min(idle_cpu, (cap_cpu - acct_cpu) if cap_cpu else idle_cpu)
            free[p] = ("cpu", max(0, room))
            print(f"=> free CPUs for the account now: {max(0, room)} (min of idle CPUs and account cap headroom)")

    # ---- estimates ----
    print(
        "\n# Estimates (bootstrap over the array's own completed/timed-out task durations; median / 90th pct of"
        " the finish time; free capacity is SHARED between arrays on the same partition)"
    )
    for r in report:
        if not r["running"] and not r["pending"]:
            continue
        if len(r["durs"]) < 3:
            print(
                f"\n## {r['aid']} {r['name']}: <3 finished tasks, no estimate "
                f"({len(r['running'])} running, {r['pending']} pending)"
            )
            continue
        cur = r["throttle"] or (len(r["running"]) + r["pending"])
        unit_free = 0
        for p in r["parts"]:
            kind, room = free.get(p, ("cpu", 0))
            per = r["gpt"] if kind == "gpu" else r["cpt"]
            if per:
                unit_free += room // per
        max_now = len(r["running"]) + unit_free
        total = len(r["running"]) + r["pending"]
        opts = sorted({max(1, cur // 2), cur, max(cur, min(max_now, total)), min(total, cur + max(1, cur // 2))} - {0})
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
                f"  throttle {o}{tag}: done ~{(now + timedelta(seconds=med)):%a %H:%M} (+{hfmt(med)}), "
                f"90%: {(now + timedelta(seconds=p90)):%a %H:%M} (+{hfmt(p90)}){res}"
            )

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(time.time()))


if __name__ == "__main__":
    main()
