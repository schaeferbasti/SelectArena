"""OAR backend (EXPERIMENTAL: written against OAR's documented JSON output, not yet run on an OAR cluster).

Queries: ``oarstat -f -J`` (jobs; ``--sql`` for history, falling back to ``-u``/``--array``) and ``oarnodes -J``
(resources). Field names differ between OAR versions, so every lookup tries several spellings and missing values
stay unknown instead of failing. ``collect.py --raw`` prints one raw job and one raw resource record, to fix the
mapping on a new cluster.

Differences to SLURM the report has to live with:
- no per-array concurrency limit (``throttle`` is None): all subjobs are queued at once, the scheduler starts what fits;
  fewer in parallel = ``oarhold`` some waiting subjobs (``oarresume`` later);
- quotas (admission rules / ``quotas.json``) are site-specific and usually not queryable: no account cap;
- queues are not tied to nodes: resource numbers are cluster-wide;
- end states are Terminated / Error plus events and exit code; TIMEOUT comes from a WALLTIME event, OOM is only a
  guess (killed by signal 9) and must be confirmed in the log.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime

from backends.common import USER, run, task

PENDING = {"Waiting", "Hold", "toLaunch", "toAckReservation", "Launching"}
RUNNING = {"Running", "Finishing", "Suspended", "Resuming"}


def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _epoch(v) -> float:
    """Epoch seconds from an int/str epoch or 'YYYY-MM-DD HH:MM:SS'; 0 if unknown."""
    if v in (None, "", 0, "0"):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(v), fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def _secs(v) -> int:
    """Seconds from an int or 'H:MM:SS' / 'D:HH:MM:SS'."""
    if v in (None, ""):
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        pass
    parts = [int(p) for p in re.findall(r"\d+", str(v))]
    total = 0
    for p, mult in zip(reversed(parts), (1, 60, 3600, 86400)):
        total += p * mult
    return total


def _load(out: str) -> list[dict]:
    """oarstat/oarnodes JSON: a dict keyed by id or a list of records."""
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        recs = []
        for k, v in data.items():
            if isinstance(v, dict):
                v.setdefault("_key", k)
                recs.append(v)
        return recs
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _exit(v) -> tuple[int | None, int | None]:
    """(exit code, signal) from OAR's exit_code: the raw wait status (256 = exit 1, 9 = killed by SIGKILL),
    also printed as '256 (1,0,0)' = (exit, signal, core dump)."""
    if v in (None, ""):
        return None, None
    m = re.search(r"\((\d+),\s*(\d+)", str(v))
    if m:
        return int(m.group(1)), (int(m.group(2)) or None)
    m = re.match(r"\s*(\d+)", str(v))
    if not m:
        return None, None
    n = int(m.group(1))
    return n >> 8, (n & 127) or None


def _events(job: dict) -> list[str]:
    ev = _get(job, "events", "Events", default=[]) or []
    if isinstance(ev, dict):
        ev = list(ev.values())
    out = []
    for e in ev:
        out.append(str(e.get("type", e.get("Type", ""))) if isinstance(e, dict) else str(e))
    return out


def _normalize(state: str, exit_code, events: list[str], message: str) -> str:
    if state in PENDING:
        return "PENDING"
    if state in RUNNING:
        return "RUNNING"
    evs = " ".join(events).upper() + " " + (message or "").upper()
    code, sig = _exit(exit_code)
    if "WALLTIME" in evs:
        return "TIMEOUT"
    if "FRAG_JOB_REQUEST" in evs or "USER_DEL" in evs or "DELETED" in evs:
        return "CANCELLED"
    if state == "Terminated" and code == 0 and sig is None:
        return "COMPLETED"
    if any(x in evs for x in ("NODE_SUSPECTED", "PING_CHECKER", "SWITCH_INTO_ERROR_STATE", "NODE_FAIL")):
        return "NODE_FAIL"
    if any(x in evs for x in ("PROLOGUE_ERROR", "CPUSET_ERROR", "LAUNCHING_OAREXEC_TIMEOUT", "CANNOT_CREATE_TMP")):
        return "LAUNCH_FAIL"
    if sig == 9 or code == 137:  # SIGKILL: the OOM killer, or a kill by the user/admin
        return "KILLED_9(OOM?)"
    if state == "Terminated" and code is None:
        return "COMPLETED?"  # no exit code reported
    return "FAILED"


def _gpus_wanted(wanted: str) -> int:
    m = re.search(r"\bgpu(?:_?device|_count)?\s*=\s*(\d+)", wanted or "")
    return int(m.group(1)) if m else 0


def _walltime_wanted(wanted: str) -> str:
    m = re.search(r"walltime=([\d:]+)", wanted or "")
    return m.group(1) if m else ""


def _cores_wanted(wanted: str) -> int:
    """Product of the hierarchy counts after the last host/node level, e.g. /host=1/core=8 -> 8 (best effort)."""
    m = re.findall(r"/(\w+)=(\d+)", wanted or "")
    n = 1
    seen = False
    for level, count in m:
        if level in ("core", "cpu", "nodes", "host", "network_address", "resource_id"):
            n *= int(count)
            seen = True
    return n if seen else 0


class OarBackend:
    name = "oar"
    throttle_word = "parallel"
    throttle_hint = "no per-array limit in OAR: run fewer with `oarhold <job ids>` (later `oarresume <job ids>`)"

    def __init__(self) -> None:
        self._resources: list[dict] | None = None
        self._all_jobs: list[dict] | None = None

    # ---------- jobs ----------
    def _jobs(self, sql: str | None = None, extra: list[str] | None = None) -> list[dict]:
        cmd = ["oarstat", "-f", "-J"] + (["--sql", sql] if sql else []) + (extra or [])
        return _load(run(cmd))

    def _to_task(self, job: dict, now: float) -> tuple[str, str, dict]:
        jid = str(_get(job, "Job_Id", "job_id", "id", "_key"))
        aid = str(_get(job, "array_id", "arrayId", default=jid))
        state = str(_get(job, "state", "State", default="?"))
        start = _epoch(_get(job, "startTime", "start_time"))
        stop = _epoch(_get(job, "stopTime", "stop_time"))
        events = _events(job)
        message = str(_get(job, "message", default=""))
        exit_code = _get(job, "exit_code", "exitCode")
        norm = _normalize(state, exit_code, events, message)
        elapsed = (stop - start) if (stop and start) else ((now - start) if (start and norm == "RUNNING") else 0)
        wanted = str(_get(job, "wanted_resources", "wantedResources", default=""))
        assigned = _get(job, "assigned_resources", "assignedResources", default=[]) or []
        if isinstance(assigned, str):
            assigned = re.findall(r"\d+", assigned)
        nodes = _get(job, "assigned_network_address", "assignedHostnames", default=[]) or []
        if isinstance(nodes, str):
            nodes = nodes.split()
        sched = _epoch(_get(job, "scheduledStart", "scheduled_start"))
        reason = state + (f" (scheduled start {datetime.fromtimestamp(sched):%a %H:%M})" if sched else "")
        types = _get(job, "types", "jobType", default="")
        if "besteffort" in str(types):
            reason += " besteffort"
        t = task(
            state=norm,
            raw=f"{state} exit={exit_code} events={','.join(events)}".strip(),
            elapsed=int(max(0, elapsed)),
            end=datetime.fromtimestamp(stop).strftime("%Y-%m-%dT%H:%M:%S") if stop else "",
            node=",".join(sorted(set(map(str, nodes)))),
            part=str(_get(job, "queue", "queue_name", default="")),
            cpus=len(assigned) or _cores_wanted(wanted),
            gpus=self._gpus_of_resources(assigned) or _gpus_wanted(wanted),
            limit=_secs(_get(job, "walltime", "Walltime")) or _secs(_walltime_wanted(wanted)),
            exit=str(exit_code if exit_code is not None else ""),
            reason=reason if norm == "PENDING" else "",
            name=str(_get(job, "name", "Name", default="")),
            acct=str(_get(job, "project", "Project", default="")),
        )
        idx = _get(job, "array_index", "arrayIndex")
        tid = f"{aid}_{idx}" if (idx is not None and aid != jid) else jid
        return aid, f"{tid} (job {jid})" if tid != jid else jid, t

    def collect(self, since: datetime, extra_arrays: list[str]) -> dict[str, dict]:
        now = time.time()
        ep = int(since.timestamp())
        sql_user = f"job_user = '{USER}' and (stop_time = 0 or stop_time >= {ep})"
        jobs = self._jobs(sql_user)
        use_sql = bool(jobs) or bool(run(["oarstat", "-J", "--sql", "job_id = -1"]).strip())
        if not jobs:
            jobs = self._jobs(extra=["-u", USER])
        aids = {str(_get(j, "array_id", "arrayId", default=_get(j, "Job_Id", "job_id", "id", "_key"))) for j in jobs}
        aids |= set(extra_arrays)
        arrays: dict[str, dict] = {}
        for aid in sorted(aids, key=lambda x: int(x) if x.isdigit() else 0):
            recs = self._jobs(f"array_id = {int(aid)}") if (use_sql and aid.isdigit()) else []
            if not recs:
                recs = self._jobs(extra=["--array", aid]) or [
                    j for j in jobs if str(_get(j, "array_id", "arrayId", default="")) == aid
                ]
            a = arrays.setdefault(aid, {"name": None, "acct": None, "tasks": {}})
            for j in recs:
                _, tid, t = self._to_task(j, now)
                a["tasks"][tid] = t
                a["name"] = a["name"] or t["name"] or "?"
                a["acct"] = a["acct"] or t["acct"] or "?"
        return arrays

    def array_info(self, aid: str, tasks: dict) -> dict:
        parts = sorted({t["part"] for t in tasks.values() if t["part"]})
        besteffort = any("besteffort" in t["reason"] for t in tasks.values())
        return dict(throttle=None, partitions=parts, notes="besteffort" if besteffort else "")

    # ---------- resources ----------
    def _res(self) -> list[dict]:
        if self._resources is None:
            self._resources = _load(run(["oarnodes", "-J"]))
        return self._resources

    @staticmethod
    def _props(r: dict) -> dict:
        p = r.get("properties")
        return p if isinstance(p, dict) else r

    def _gpu_key(self, r: dict):
        p = self._props(r)
        g = _get(p, "gpu", "gpudevice", "gpu_device")
        if g in (None, "", "NO", "no", "0", 0) and not str(_get(p, "gpu_model", default="")).strip():
            return None
        host = _get(p, "host", "network_address", default=r.get("network_address", ""))
        return (host, g)

    def _gpus_of_resources(self, ids: list) -> int:
        if not ids or not self._res():
            return 0
        want = {str(i) for i in ids}
        keys = {
            self._gpu_key(r)
            for r in self._res()
            if str(_get(r, "resource_id", "id", "_key")) in want and self._gpu_key(r) is not None
        }
        return len(keys)

    def _all(self) -> list[dict]:
        if self._all_jobs is None:
            self._all_jobs = self._jobs()
        return self._all_jobs

    def partition_info(self, p: str, accts: list[str]) -> dict:
        now = time.time()
        tot_cpu = alloc_cpu = 0
        gpus_all, gpus_busy, down = set(), set(), set()
        for r in self._res():
            st = str(_get(r, "state", "State", default=""))
            host = str(_get(self._props(r), "network_address", "host", default=r.get("network_address", "")))
            upto = _epoch(_get(self._props(r), "available_upto", default=0))
            if st in ("Dead", "Suspected") or (st == "Absent" and upto <= now):
                down.add(f"{host}:{st}")
                continue
            busy = bool(_get(r, "jobs", "Jobs", default=None))
            tot_cpu += 1
            alloc_cpu += busy
            k = self._gpu_key(r)
            if k is not None:
                gpus_all.add(k)
                if busy:
                    gpus_busy.add(k)
        acct_cpu = acct_gpu = 0
        by_user = Counter()  # running jobs per user, to rank them
        usage: dict[str, list[int]] = {}  # user -> [cores, GPUs]
        pend_others = Counter()
        for j in self._all():
            st = str(_get(j, "state", "State", default=""))
            owner = str(_get(j, "owner", "job_user", "user", default=""))
            queue = str(_get(j, "queue", "queue_name", default=""))
            if st in PENDING and owner != USER and queue == p:
                pend_others[st] += 1
            if st not in RUNNING:
                continue
            assigned = _get(j, "assigned_resources", "assignedResources", default=[]) or []
            if isinstance(assigned, str):
                assigned = re.findall(r"\d+", assigned)
            g = self._gpus_of_resources(assigned)
            if str(_get(j, "project", "Project", default="")) in accts:
                acct_cpu += len(assigned)
                acct_gpu += g
            by_user[owner] += 1
            usage.setdefault(owner, [0, 0])
            usage[owner][0] += len(assigned)
            usage[owner][1] += g
        return dict(
            header=f"queue {p} (OAR: cluster-wide numbers, queues are not tied to nodes; quotas not queryable,"
            " check the site's documentation / admission rules)",
            tot_cpu=tot_cpu,
            alloc_cpu=alloc_cpu,
            tot_gpu=len(gpus_all),
            alloc_gpu=len(gpus_busy),
            down=sorted(down),
            cap_cpu=None,
            cap_gpu=None,
            acct_label=f"project {','.join(accts)}",
            acct_cpu=acct_cpu,
            acct_gpu=acct_gpu,
            by_user={
                u: f"{usage[u][0]} cores/{usage[u][1]} GPUs"
                for u, _ in Counter({u: c[0] + 100 * c[1] for u, c in usage.items()}).most_common(10)
            },
            by_user_unit="cluster-wide, top 10",
            pend_others=dict(pend_others),
        )

    def raw_samples(self) -> str:
        jobs = self._jobs(extra=["-u", USER])
        res = self._res()
        return (
            "## raw oarstat -f -J (first job)\n"
            + (json.dumps(jobs[0], indent=1, default=str) if jobs else "(none)")
            + "\n## raw oarnodes -J (first resource)\n"
            + (json.dumps(res[0], indent=1, default=str) if res else "(none)")
        )
