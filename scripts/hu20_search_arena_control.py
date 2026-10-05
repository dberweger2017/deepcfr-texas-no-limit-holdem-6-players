"""Outcome-blind, fail-closed admission for the owner-approved paid arena.

An authenticated pod-hosted controller survives the operator laptop sleeping.
Losing the controller halts dispatch; existing journals prohibit restarts.
"""

import argparse
import json
import os
from pathlib import Path
import threading
from time import time, sleep
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request


def durable_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArenaControl:
    def __init__(self, journal, ledger, *, clock=time):
        if journal.exists():
            raise ValueError("Preserve prior controller journal; automatic restart is prohibited")
        self.journal, self.ledger, self.clock = journal, ledger, clock
        self.lock = threading.Lock()
        self.state = {"status": "preflight", "reason": None, "issued": {},
                      "completed": {}, "events": [], "checkpoints": []}
        self.event_path = journal.with_suffix(".events.jsonl")
        if self.event_path.exists():
            raise ValueError("Preserve prior decision events")
        self.persisted_events = 0
        self.persist()

    def persist(self):
        with self.event_path.open("a") as stream:
            for row in self.state["events"][self.persisted_events:]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.persisted_events = len(self.state["events"])
        durable_json(self.journal, {
            **{k: v for k, v in self.state.items() if k not in ("events", "completed")},
            "decisions": len(self.state["events"]),
            "fallbacks": sum(r["fallback"] for r in self.state["events"]),
            "event_file": str(self.event_path)})

    def stop(self, reason):
        if self.state["status"] != "stopped":
            self.state.update(status="stopped", reason=reason)

    def charge(self):
        ledger = json.loads(self.ledger.read_text())
        now = self.clock()
        return ledger.get("historical_cap_charge_usd", 0) + ledger.get("pilot_reserve_usd", .5) + ledger.get("storage_contingency_usd", 1) + sum(
            max(0, p.get("terminated_at", now) - p["created_at"]) / 3600 * p["hourly_usd"]
            for p in ledger["pods"]) - ledger.get("owner_excluded_charge_usd", 0)

    def guard(self):
        ledger = json.loads(self.ledger.read_text())
        if self.charge() >= 21:
            self.stop("Conservative campaign charge reached $21 dispatch stop")
        for pod in ledger["pods"]:
            if (ledger.get("work_protocol")!="hu20-fixed50-no-fallback-v1"
                    and not pod.get("terminated_at") and self.clock()>=pod["production_stop_at"]):
                self.stop("Quoted pod clock reached the reserved closeout boundary")
        self.state["conservative_charge_usd"] = self.charge()

    def rows(self, pod=None):
        return [row for row in self.state["events"] if pod is None or row["pod"] == pod]

    @staticmethod
    def boundary(n):
        return 500 if n < 500 else 500 + ((n - 500) // 100 + 1) * 100

    def checkpoint(self, pod=None):
        if json.loads(self.ledger.read_text()).get("work_protocol")=="hu20-fixed50-no-fallback-v1":
            if any(row["fallback"] for row in self.rows(pod)):
                self.stop("Search fallback is a defect under fixed-work protocol")
            return
        rows = self.rows(pod)
        n = len(rows)
        if n < 500 or (n - 500) % 100:
            return
        bad = sum(r["fallback"] for r in rows)
        trailing = sum(r["fallback"] for r in rows[-500:])
        self.state["checkpoints"].append({"pod": pod, "decisions": n,
                                          "fallbacks": bad, "trailing_500_fallbacks": trailing})
        if bad * 20 > n or trailing > 25:
            self.stop(f"Fallback rate exceeds 5% at {pod or 'global'} checkpoint {n}")

    def request(self, body):
        with self.lock:
            self.guard()
            op = body["op"]
            if op == "start":
                if self.state["status"] != "preflight":
                    raise ValueError("Arena cannot restart")
                ledger = json.loads(self.ledger.read_text())
                if len([p for p in ledger["pods"] if not p.get("terminated_at")]) != ledger.get("expected_active_pods", 3):
                    raise ValueError("All owner-approved active pods required")
                required=ledger.get("start_required_pods",[p["id"] for p in ledger["pods"] if not p.get("terminated_at")])
                if not required or not all(p.get("parity_retention_passed") for p in ledger["pods"] if p["id"] in required):
                    raise ValueError("Every initial host must pass parity and retention")
                self.state["status"] = "running"
            elif op == "stop":
                self.stop(body["reason"])
            elif op == "acquire":
                key, pod = body["event"], body["pod"]
                ledger = json.loads(self.ledger.read_text())
                matching = [p for p in ledger["pods"] if p["id"] == pod and not p.get("terminated_at")]
                if (len(matching) != 1 or not matching[0].get("parity_retention_passed")
                        or body["worker"] not in matching[0]["workers"]):
                    raise ValueError("Unknown owned pod/worker")
                if key in self.state["completed"]:
                    raise ValueError("Decision already completed")
                if self.state["status"] == "running" and key not in self.state["issued"]:
                    live = list(self.state["issued"].values())
                    if ledger.get("work_protocol")!="hu20-fixed50-no-fallback-v1" and (len(self.rows()) + len(live) >= self.boundary(len(self.rows()))
                            or len(self.rows(pod)) + sum(r["pod"] == pod for r in live)
                            >= self.boundary(len(self.rows(pod)))):
                        self.persist()
                        return {"status": "wait"}
                    self.state["issued"][key] = {"pod": pod, "worker": body["worker"]}
            elif op == "complete":
                key = body["event"]
                row = {"event": key, "pod": body["pod"], "worker": body["worker"],
                       "fallback": body["fallback"], "cause": body.get("cause"),
                       "completed_at": self.clock()}
                if key in self.state["completed"]:
                    prior = self.state["completed"][key]
                    if any(prior[k] != row[k] for k in ("pod", "worker", "fallback", "cause")):
                        raise ValueError("Duplicate completion differs")
                else:
                    issued = self.state["issued"].pop(key)
                    if issued != {"pod": body["pod"], "worker": body["worker"]} or type(body["fallback"]) is not bool:
                        raise ValueError("Completion identity differs")
                    self.state["completed"][key] = row
                    self.state["events"].append(row)
                    self.checkpoint()
                    self.checkpoint(body["pod"])
            elif op != "check":
                raise ValueError("Unknown controller operation")
            self.persist()
            return {"status": self.state["status"], "reason": self.state["reason"],
                    "decisions": len(self.state["events"]), "charge_usd": self.charge()}


class ControlClient:
    def __init__(self, url, pod, worker, token=None):
        self.url, self.pod, self.worker = url, pod, worker
        self.sequence = 0
        self.token = token

    def request(self, op, **kwargs):
        body = {"op": op, "pod": self.pod, "worker": self.worker, **kwargs}
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (compatible; HU20Arena/1.0)"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.load(response)
        except Exception as exc:
            raise RuntimeError("Arena controller unavailable; halt owned work") from exc
        if result["status"] == "stopped":
            raise RuntimeError("Arena stopped: " + str(result["reason"]))
        return result

    def acquire(self, guard):
        self.sequence += 1
        event = f"{self.worker}/{self.sequence}"
        while True:
            guard()
            result = self.request("acquire", event=event)
            if result["status"] == "running":
                return event
            if result["status"] != "wait":
                raise RuntimeError("Arena has not passed preflight")
            sleep(.1)


def serve(control, port, *, bind="127.0.0.1", token=None):
    import hmac
    if bind != "127.0.0.1" and not token:
        raise ValueError("A public controller requires authentication")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer "+token):
                self.send_response(403)
                self.end_headers()
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n < 4096:
                    raise ValueError("Invalid control request size")
                result = control.request(json.loads(self.rfile.read(n)))
                encoded = json.dumps(result).encode()
                self.send_response(200)
            except Exception as exc:
                with control.lock:
                    control.stop(f"Controller failure: {type(exc).__name__}: {exc}")
                    control.persist()
                encoded = json.dumps({"status": "stopped", "reason": control.state["reason"]}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(encoded)
    ThreadingHTTPServer((bind, port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18066)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    serve(ArenaControl(args.journal, args.ledger), args.port, bind=args.bind,
          token=args.token_file.read_text().strip() if args.token_file else None)


if __name__ == "__main__":
    main()
