"""Process boundary for the independently installed AGPL solver."""

from dataclasses import dataclass
from hashlib import sha256
from functools import cached_property
import json
import os
from pathlib import Path
import subprocess
from time import monotonic
import platform
from uuid import uuid4

import numpy as np

from src.blueprint.abstraction import Choice
from src.blueprint.hu20_turn_tree import line_key
from src.game.types import Action, ActionKind


class SolveFailure(ValueError):
    def __init__(self, cause, message):
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True)
class PolicyMatrix:
    menu: tuple
    holdings: tuple
    probabilities: np.ndarray

    @cached_property
    def holding_indices(self):
        return {h: i for i, h in enumerate(self.holdings)}

    def row(self, holding):
        try:
            index = self.holding_indices[tuple(sorted(holding))]
        except KeyError as exc:
            raise SolveFailure("unsupported_holding", "Holding has zero root support") from exc
        return tuple(float(x) for x in self.probabilities[index])


def parse_profiles(request, path):
    nodes = {line_key(n["line"]): n for n in request["nodes"]
             if not n["terminal"] and n["street"] == request["initial_street"]}
    profiles = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = line_key(row["line"])
            node = nodes.get(key)
            if (node is None or key in profiles or row["board"] != request["board"]
                    or row["actions"] != node["actions"] or row["player"] != node["player"]):
                raise SolveFailure("invalid_response", "Profile node/action identity differs")
            holdings = tuple(tuple(sorted(h)) for h in row["holdings"])
            expected = {tuple(sorted(r["hand"])) for r in request["ranges"][node["player"]]
                        if r["weight"] > 0}
            if len(set(holdings)) != len(holdings) or set(holdings) != expected:
                raise SolveFailure("invalid_response", "Profile holdings differ from supported range")
            p = np.asarray(row["strategy"], dtype=np.float64)
            if (p.size != len(holdings) * len(node["actions"])
                    or not np.isfinite(p).all() or (p < 0).any()):
                raise SolveFailure("invalid_response", "Invalid strategy matrix")
            p = p.reshape(len(node["actions"]), len(holdings)).T.copy()
            if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=2e-5):
                raise SolveFailure("invalid_response", "Strategy is not normalized")
            p /= p.sum(axis=1, keepdims=True)
            p.setflags(write=False)
            menu = tuple(Choice(name, Action(ActionKind(a["kind"]), a["raise_to"]))
                         for name, a in zip(node["names"], node["native_actions"], strict=True))
            profiles[key] = PolicyMatrix(menu, holdings, p)
    if set(profiles) != set(nodes):
        raise SolveFailure("invalid_response", "Incomplete current-round profile")
    for lock in request.get("locks",[]):
        key=line_key(lock["line"])
        matrix=profiles.get(key)
        if matrix is None:continue
        node=nodes[key]
        if lock["actions"]!=node["actions"] or lock["player"]!=node["player"]:
            raise SolveFailure("invalid_response","Hero lock identity differs")
        lookup={tuple(sorted(h)):i for i,h in enumerate(lock["holdings"])}
        values=np.asarray(lock["strategy"]).reshape(len(node["actions"]),-1).T
        expected=np.asarray([values[lookup[h]] for h in matrix.holdings])
        if not np.allclose(matrix.probabilities,expected,atol=2e-5,rtol=0):
            raise SolveFailure("invalid_response","Exported strategy violates a prior hero lock")
    return profiles


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


FIXED_WORK_PROTOCOL = "hu20-fixed50-no-fallback-v1"


PROFILE_RETENTION_RULE = "request-sha256-lowest-1pct-v1"


def sampled_profile(request_sha256):
    """A fixed one-percent hash-space predicate, independent of solve outcomes."""
    if len(request_sha256)!=64:
        raise ValueError("Request SHA-256 must contain 64 hex digits")
    return int(request_sha256,16)<2**256//100


def host_telemetry(threads):
    """Descriptive host metadata never participates in policy selection."""
    model=platform.processor() or platform.machine()
    cpu=Path('/proc/cpuinfo')
    if cpu.exists():
        model=next((line.split(':',1)[1].strip() for line in cpu.read_text().splitlines()
                    if line.startswith('model name')),model)
    elif platform.system()=='Darwin':
        model=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    quota=None
    for q,p in ((Path('/sys/fs/cgroup/cpu.max'),None),
                (Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')),
                (Path('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us'),Path('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us'))):
        if not q.exists():continue
        parts=q.read_text().split() if p is None else [q.read_text().strip(),p.read_text().strip()]
        quota=None if parts[0] in ('max','-1') else int(parts[0])/int(parts[1]);break
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    return {'cpu_model':model,'cgroup_quota_cpus':quota,'affinity_cpus':affinity,
            'threads_per_worker':threads,'host':platform.node(),'architecture':platform.machine()}


class ExternalTurnSolver:
    def __init__(self, executable, evidence_dir, *, expected_sha256=None,
                 resource_check=lambda: None, allocation_budget=None, profile_retention=None,
                 hang_attention_seconds=None):
        if profile_retention not in (None,PROFILE_RETENTION_RULE):
            raise ValueError("Unknown profile retention rule")
        self.executable = Path(executable).expanduser().resolve()
        self.evidence_dir = Path(evidence_dir).expanduser().resolve()
        self.expected_sha256 = expected_sha256 or (
            file_hash(self.executable) if self.executable.is_file() else None)
        self.records = []
        self.resource_check = resource_check
        self.allocation_budget = allocation_budget
        self.profile_retention = profile_retention
        if hang_attention_seconds is not None and hang_attention_seconds<=0:
            raise ValueError("Positive attention threshold required")
        self.hang_attention_seconds=hang_attention_seconds

    def solve(self, request, deadline, *, mode="play"):
        if mode not in ("play", "quality"):
            raise ValueError("Invalid process mode")
        started = monotonic()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        out = self.evidence_dir / uuid4().hex
        out.mkdir()
        request = dict(request, mode=mode, dump_path=str(out / "profile.jsonl"))
        fixed_work=request.get("work_protocol")==FIXED_WORK_PROTOCOL
        if fixed_work and (deadline is not None or request["max_iterations"]!=50 or request["threads"]!=6):
            raise ValueError("Fixed-work request has a timer or different iteration/thread count")
        if deadline is None and not fixed_work:raise ValueError("No-timer solve needs the fixed-work protocol")
        request["seconds"] = None if fixed_work else max(0, deadline-monotonic())
        (out / "request.json").write_text(json.dumps(request, allow_nan=False, sort_keys=True) + "\n")
        record = {"status": "failure", "path": str(out), "spot": request["spot"]}
        process = None
        try:
            if not self.executable.is_file():
                raise SolveFailure("solver_unavailable", "External solver is missing")
            identity = file_hash(self.executable)
            if self.expected_sha256 and identity != self.expected_sha256:
                raise SolveFailure("solver_identity", "Executable SHA-256 differs")
            record["executable_sha256"] = identity
            if self.allocation_budget is not None:
                requested = request["memory_budget_bytes"]
                admitted = self.allocation_budget(requested)
                if type(admitted) is not int or not 0 <= admitted <= requested:
                    raise ValueError("Invalid native allocation admission")
                request.update(memory_budget_bytes=admitted, requested_memory_budget_bytes=requested)
                record["memory_admission"] = {"configured_bytes": requested, "admitted_bytes": admitted}
                (out / "request.json").write_text(json.dumps(request, allow_nan=False, sort_keys=True)+"\n")
                if admitted == 0:
                    raise SolveFailure("memory_refusal", "No native allocation headroom in owned family")
            if deadline is not None and monotonic() >= deadline:
                raise SolveFailure("timeout", "No startup budget remains")
            record["host_telemetry"]=host_telemetry(request["threads"])
            with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
                env = dict(os.environ, RAYON_NUM_THREADS=str(request["threads"]))
                if self.profile_retention:
                    # The final request bytes are frozen before process launch.
                    record["request_sha256"] = file_hash(out / "request.json")
                try:
                    process = subprocess.Popen([str(self.executable), str(out / "request.json"),
                        str(out / "response.jsonl")], stdout=stdout, stderr=stderr, env=env)
                except OSError as exc:
                    raise SolveFailure("solver_unavailable", str(exc)) from exc
                while process.poll() is None:
                    self.resource_check()
                    if (self.hang_attention_seconds is not None and not record.get("hang_attention")
                            and monotonic()-started>self.hang_attention_seconds):
                        record["hang_attention"]={"elapsed_seconds":monotonic()-started,
                            "threshold_seconds":self.hang_attention_seconds,
                            "action":"owner attention only; solve continues; no substituted action"}
                        with (out/'hang-attention.json').open('w') as attention:
                            json.dump(record["hang_attention"],attention);attention.flush();os.fsync(attention.fileno())
                    remaining = None if fixed_work else deadline-monotonic()
                    if remaining is not None and remaining <= 0:
                        raise SolveFailure("timeout", "External solve deadline")
                    try:
                        process.wait(timeout=.1 if fixed_work else min(.1,remaining))
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise SolveFailure("solver_exit", f"Solver exited {process.returncode}")
            events = [json.loads(line) for line in (out / "response.jsonl").read_text().splitlines()]
            completion = [e for e in events if e.get("event") == "completion"]
            if not completion or completion[-1].get("status") != "play_complete":
                cause = "memory_refusal" if completion and "oversize" in completion[-1].get("status", "") else "invalid_response"
                raise SolveFailure(cause, "Solver did not complete fixed play work")
            if completion[-1]["iterations"] != request["max_iterations"]:
                raise SolveFailure("invalid_response" if fixed_work else "timeout", "Partial solver iterations")
            if fixed_work and completion[-1].get("work_protocol")!=FIXED_WORK_PROTOCOL:
                raise SolveFailure("invalid_response","Native harness did not acknowledge fixed work")
            if (completion[-1].get("solver_commit") != request["solver_commit"]
                    or completion[-1].get("compressed") != request["compress"]):
                raise SolveFailure("invalid_response", "Solver source/compression identity differs")
            if max(e.get("solver_peak_rss_bytes", 0) for e in events) > request["memory_budget_bytes"]:
                raise SolveFailure("memory_refusal", "Solver RSS exceeds request budget")
            profiles = parse_profiles(request, out / "profile.jsonl")
            if deadline is not None and monotonic() >= deadline:
                raise SolveFailure("timeout", "Profile parse exceeded decision deadline")
            record.update(status="completed", nodes=len(profiles), completion=completion[-1])
            record["quality"] = [e for e in events if e.get("event") == "quality"]
            return profiles
        except SolveFailure as exc:
            record.update(cause=exc.cause, reason=str(exc))
            raise
        except (ValueError, KeyError, OSError, TypeError, MemoryError, TimeoutError) as exc:
            cause = "memory_refusal" if isinstance(exc, MemoryError) else "timeout" if isinstance(exc, TimeoutError) else "invalid_response"
            record.update(cause=cause, reason=str(exc))
            raise SolveFailure(cause, str(exc)) from exc
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            record["seconds"] = monotonic() - started
            files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)}
                     for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
            if self.profile_retention:
                request_hash=files["request.json"]["sha256"]
                if record.get("request_sha256",request_hash)!=request_hash:
                    raise ValueError("Request changed after sample selection")
                selected=sampled_profile(request_hash)
                profile=files.get("profile.jsonl")
                record["profile_retention"]={"rule":self.profile_retention,
                    "request_sha256":request_hash,"sample_selected":selected,
                    "profile_sha256":profile["sha256"] if profile else None,
                    "profile_bytes":profile["bytes"] if profile else 0,
                    "profile_generated":profile is not None,
                    "body_retained":selected and profile is not None}
            self.records.append(record)
            receipt=out / "receipt.json"
            with receipt.open("w") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                if self.profile_retention:
                    stream.flush();os.fsync(stream.fileno())
            if self.profile_retention:
                directory=os.open(out,os.O_RDONLY)
                try:os.fsync(directory)
                finally:os.close(directory)
                # Live matrices have already been parsed into memory. Persist
                # the body hash in the receipt before removing only this dump.
                if profile is not None and not selected:
                    (out / "profile.jsonl").unlink()
                    files.pop("profile.jsonl")
            files["receipt.json"]={"bytes":receipt.stat().st_size,"sha256":file_hash(receipt)}
            (out / "manifest.json").write_text(json.dumps(files, sort_keys=True) + "\n")
