"""SLURM backend: squeue (queued tasks), sacct (history), scontrol/sinfo (nodes), sacctmgr (QOS caps)."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from backends.common import USER, gpus_of, run, task

DOWN_STATES = {"DOWN", "DRAIN", "DRAINED", "DRAINING", "FAIL", "FAILING", "MAINT", "NOT_RESPONDING"}
STATE_MAP = {"OUT_OF_ME": "OUT_OF_MEMORY", "OUT_OF_MEMORY": "OUT_OF_MEMORY", "BOOT_FAIL": "LAUNCH_FAIL"}


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


class SlurmBackend:
    name = "slurm"
    throttle_word = "throttle"
    throttle_hint = "scontrol update JobId=<array id> ArrayTaskThrottle=<N>"

    def collect(self, since: datetime, extra_arrays: list[str]) -> dict[str, dict]:
        arrays: dict[str, dict] = {}

        def arr(aid: str) -> dict:
            return arrays.setdefault(aid, {"name": None, "acct": None, "tasks": {}, "queued": {}})

        # queued tasks (expanded per task)
        q = run(["squeue", "-u", USER, "-r", "-h", "-o", "%i|%T|%P|%C|%b|%M|%l|%R|%a|%j|%N"])
        for line in q.strip().splitlines():
            jid, st, part, cpus, tres, elapsed, limit, reason, acct, name, node = line.split("|")
            if st in ("CONFIGURING", "COMPLETING"):
                st = "RUNNING"
            a = arr(jid.split("_")[0])
            a["queued"][jid] = task(
                state=st,
                raw=st,
                part=part,
                cpus=int(cpus),
                gpus=gpus_of(tres),
                elapsed=secs(elapsed),
                limit=secs(limit),
                reason=reason,
                node=node,
            )
            a["acct"], a["name"] = acct, name
        # arrays that ended recently / named explicitly
        s = run(["sacct", "-u", USER, "-X", "-n", "-P", "-S", since.strftime("%Y-%m-%dT%H:%M"), "-o", "JobID,State"])
        for line in s.strip().splitlines():
            jid = line.split("|")[0]
            if re.match(r"^\d+(_\d+)?$", jid):
                arr(jid.split("_")[0])
        for aid in extra_arrays:
            arr(aid)
        # full history of each array
        fields = "JobID,State,Elapsed,End,NodeList,Partition,AllocCPUS,AllocTRES,Timelimit,Account,JobName,ExitCode"
        for aid, a in arrays.items():
            for line in run(["sacct", "-j", aid, "-X", "-n", "-P", "-o", fields]).strip().splitlines():
                f = line.split("|")
                if len(f) < 12 or "[" in f[0]:
                    continue
                st = f[1].split()[0]
                if st in ("PENDING", "RUNNING"):
                    continue  # the queued view is authoritative for those
                a["tasks"][f[0]] = task(
                    state=STATE_MAP.get(st, st),
                    raw=f[1],
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
            first = next(iter(a["tasks"].values()), {})
            a["acct"] = a["acct"] or first.get("acct") or "?"
            a["name"] = a["name"] or first.get("name") or "?"
            a["tasks"].update(a.pop("queued"))
        return arrays

    def array_info(self, aid: str, tasks: dict) -> dict:
        sc = run(["scontrol", "show", "job", aid])
        m = re.search(r"ArrayTaskThrottle=(\d+)", sc)
        exc = re.search(r"ExcNodeList=(\S+)", sc)
        parts = set(re.findall(r"Partition=(\S+)", sc)) | {
            t["part"] for t in tasks.values() if t["state"] in ("RUNNING", "PENDING")
        }
        return dict(
            throttle=int(m.group(1)) if m and int(m.group(1)) else None,
            partitions=sorted(p for p in parts if p),
            notes=f"ExcNodeList={exc.group(1)}" if exc and exc.group(1) != "(null)" else "",
        )

    def partition_info(self, p: str, accts: list[str]) -> dict:
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
        nodes = run(["sinfo", "-p", p, "-h", "-N", "-o", "%N"]).split()
        tot_cpu = alloc_cpu = tot_gpu = alloc_gpu = 0
        down = []
        if nodes:
            for ln in run(["scontrol", "show", "node", "-o", ",".join(sorted(set(nodes)))]).splitlines():
                name = re.search(r"NodeName=(\S+)", ln).group(1)
                st = re.search(r"State=(\S+)", ln).group(1)
                if set(st.split("+")) & DOWN_STATES:  # POWERED_DOWN idle nodes boot on demand: available
                    down.append(f"{name}:{st}")
                    continue
                tot_cpu += int(re.search(r"CPUTot=(\d+)", ln).group(1))
                alloc_cpu += int(re.search(r"CPUAlloc=(\d+)", ln).group(1))
                g = re.search(r"Gres=\S*?gpu(?::[A-Za-z0-9_-]+)?:(\d+)", ln)
                tot_gpu += int(g.group(1)) if g else 0
                ga = re.search(r"AllocTRES=\S*gres/gpu=(\d+)", ln)
                alloc_gpu += int(ga.group(1)) if ga else 0
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
        return dict(
            header=f"{p} (QOS {qos}; per-account cap cpu={cap_cpu} gpu={cap_gpu})",
            tot_cpu=tot_cpu,
            alloc_cpu=alloc_cpu,
            tot_gpu=tot_gpu,
            alloc_gpu=alloc_gpu,
            down=down,
            cap_cpu=cap_cpu,
            cap_gpu=cap_gpu,
            acct_label=f"account {','.join(accts)}",
            acct_cpu=acct_cpu,
            acct_gpu=acct_gpu,
            by_user=dict(by_user),
            by_user_unit="GPUs" if "gpu" in p else "CPUs",
            pend_others=dict(pend_others),
        )
