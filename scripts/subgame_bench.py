"""Train production CFR rules on #149 turn roots and score them exactly on held-out boards.

    train     one lineage, one held-out fold: train on the other half, export checkpoints
    evaluate  one native lock-only pass per held-out root, scoring every exported policy
    report    paired comparison with #149's blueprint (B), per-root (L) and held-out (P) witnesses
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import signal
from statistics import mean
from time import time

from src.diagnostics.board_pooling_results import completion
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.pooling_runtime import run_owned_tool
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.subgame_bench import STRATEGIES, FrozenRoot, SubgameTrainer

MIN_FREE_DISK = 20 * 1024**3


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def frozen_inputs(plan_path, prepared, lineage_index):
    plan = json.loads(plan_path.read_text())
    corpus = json.loads(Path(plan["corpus"]["path"]).read_text())
    folds = json.loads(Path(plan["crossfit"]["path"]).read_text())["folds"]
    if file_hash(plan["corpus"]["path"]) != plan["corpus"]["sha256"] or file_hash(plan["crossfit"]["path"]) != plan["crossfit"]["sha256"]:
        raise ValueError("Frozen #149 corpus or split differs")
    records = {r["spot"]: r for r in corpus["roots"]}
    jobs = [j for j in json.loads((prepared / "manifest.json").read_text())["jobs"] if j["policy_index"] == lineage_index]
    if len(jobs) != len(records):
        raise ValueError("Every corpus root needs one prepared job for the lineage")
    for job in jobs:
        if file_hash(job["request"]) != job["request_sha256"]:
            raise ValueError("Prepared request hash differs")
    return plan, records, folds, sorted(jobs, key=lambda j: j["spot"])


def train(a):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Bench training stopped")))
    plan, records, folds, jobs = frozen_inputs(a.plan, a.prepared, a.lineage_index)
    training = [j for j in jobs if folds[j["spot"]] != a.evaluation_fold]
    roots = [FrozenRoot.from_request(records[j["spot"]], json.loads(Path(j["request"]).read_text())) for j in training]
    lineage = training[0]["lineage"]
    out = a.out / f"fold-{a.evaluation_fold}"
    state = out / "state.pickle"
    if state.exists():
        trainer = pickle.loads(state.read_bytes())
        if trainer.seed != a.seed or [r.spot for r in trainer.roots] != [r.spot for r in roots]:
            raise ValueError("Saved bench state belongs to another run")
    else:
        if out.exists():
            out.rename(out.with_name(out.name + ".partial-" + str(int(time()))))
        out.mkdir(parents=True)
        trainer = SubgameTrainer(roots, seed=a.seed)
        atomic_json(out / "run.json", {"lineage": lineage, "evaluation_fold": a.evaluation_fold, "seed": a.seed,
                    "training_spots": [r.spot for r in roots], "iterations": a.iterations,
                    "checkpoints": a.checkpoints, "started": time()})
    checkpoints = sorted(set(a.checkpoints) | {a.iterations})
    started, last = time(), 0
    while trainer.iteration < a.iterations:
        trainer.step()
        if trainer.iteration in checkpoints:
            folder = out / f"iteration-{trainer.iteration}"
            folder.mkdir(exist_ok=True)
            for strategy in STRATEGIES:
                atomic_json(folder / f"{strategy}.json", trainer.export(lineage, strategy))
            temporary = state.with_suffix(".tmp")
            temporary.write_bytes(pickle.dumps(trainer)); temporary.replace(state)
        if time() - last >= 30:
            atomic_json(out / "status.json", {"iteration": trainer.iteration, "target": a.iterations,
                        "keys": len(trainer.table), "nodes": trainer.nodes, "timestamp": time(),
                        "iterations_per_second": trainer.iteration / max(time() - started, 1e-9)})
            last = time()
    atomic_json(out / "status.json", {"iteration": trainer.iteration, "target": a.iterations, "keys": len(trainer.table),
                "nodes": trainer.nodes, "timestamp": time(), "complete": True})


def policies(folder):
    found = sorted(folder.glob("iteration-*/*.json"), key=lambda p: (int(p.parent.name.split("-")[1]), p.stem))
    return [(f"{p.parent.name}/{p.stem}", p) for p in found]


def evaluate(a):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Bench evaluation stopped")))
    plan, records, folds, jobs = frozen_inputs(a.plan, a.prepared, a.lineage_index)
    a.out.mkdir(parents=True, exist_ok=True)
    selected = [j for j in jobs if a.evaluation_fold is None or folds[j["spot"]] == a.evaluation_fold]
    for job in selected[:a.limit]:
        fold = folds[job["spot"]]
        destination = a.out / job["job"]
        if (destination / "result.json").exists():
            continue
        if destination.exists():
            destination.rename(destination.with_name(destination.name + ".partial-" + str(int(time()))))
        if shutil.disk_usage(a.out).free < MIN_FREE_DISK:
            raise RuntimeError("Free disk below 20 GiB")
        trained = policies(a.runs / f"fold-{fold}")
        if not trained:
            raise ValueError(f"No exported policies for held-out fold {fold}")
        reference = a.main / "collect" / job["job"] / "solver/response.jsonl"
        request = json.loads(Path(job["request"]).read_text())
        request.update(pooling_phase="lock-only", max_iterations=0,
                       reference_equilibrium_ev_chips=completion(rows(reference))["current_ev_chips"],
                       reference_response_sha256=file_hash(reference),
                       pooling_measurements=[{"metric": name, "projection_metric": "v1",
                                              "policy_path": str(path.resolve()), "allow_missing": True}
                                             for name, path in trained])
        destination.mkdir(parents=True)
        atomic_json(destination / "request.json", request)
        runtime = run_owned_tool(a.binary, destination / "request.json", destination / "solver",
                                 memory_bytes=request["memory_budget_bytes"], threads=6,
                                 seconds=request["seconds"] + 300, job_memory_bytes=7 * 1024**3)
        if runtime["status"] != "completed":
            raise RuntimeError(runtime["failure"])
        metrics = [r for r in rows(destination / "solver/response.jsonl") if r["event"] == "pooling_metric"]
        if {(m["metric"], m["target_solver_seat"]) for m in metrics} != {(n, s) for n, _ in trained for s in (0, 1)}:
            raise ValueError("Missing or duplicate bench measurements")
        atomic_json(destination / "result.json", {"job": job, "fold": fold, "metrics": metrics, "runtime": runtime,
                    "policies": {name: file_hash(path) for name, path in trained}})


def seat_mean(metrics, name):
    values = [m["gain_bb"] for m in metrics if m["metric"] == name]
    if len(values) != 2:
        raise ValueError(f"Expected two seats for {name}")
    return mean(values)


def report(a):
    plan, records, folds, jobs = frozen_inputs(a.plan, a.prepared, a.lineage_index)
    rows_by_board = []
    for job in jobs:
        bench = json.loads((a.out / job["job"] / "result.json").read_text())
        collect = json.loads((a.main / "collect" / job["job"] / "result.json").read_text())["metrics"]
        relock = json.loads((a.main / "relock" / job["job"] / "result.json").read_text())["metrics"]
        row = {"spot": job["spot"], "fold": folds[job["spot"]], "B": seat_mean(collect, "e_bp"),
               "L": seat_mean(collect, "e_root_v1"), "P": seat_mean(relock, "e_cross_v1")}
        for name in sorted({m["metric"] for m in bench["metrics"]}):
            row[name] = seat_mean(bench["metrics"], name)
        rows_by_board.append(row)
    names = [k for k in rows_by_board[0] if k not in ("spot", "fold")]
    if any(set(r) != set(rows_by_board[0]) for r in rows_by_board):
        raise ValueError("Boards were scored with different policy sets")
    rng = random.Random(a.bootstrap_seed)
    draws = [[rng.randrange(len(rows_by_board)) for _ in rows_by_board] for _ in range(2000)]

    def summary(stat):
        point = stat(rows_by_board)
        values = sorted(stat([rows_by_board[i] for i in draw]) for draw in draws)
        return {"mean": point, "low": values[49], "high": values[1949]}

    estimates = {name: summary(lambda rs, n=name: mean(r[n] for r in rs)) for name in names}
    placements = {}
    for name in names:
        if name in ("B", "L", "P"):
            continue
        q = summary(lambda rs, n=name: (mean(r[n] for r in rs) - mean(r["P"] for r in rs))
                    / (mean(r["B"] for r in rs) - mean(r["P"] for r in rs)))
        label = ("near held-out witness" if q["mean"] <= .3 else "between" if q["mean"] < .7
                 else "near blueprint" if q["mean"] <= 1 else "worse than blueprint")
        placements[name] = dict(q, classification=label)
    atomic_json(a.out / "summary.json", {"boards": len(rows_by_board), "estimates_bb": estimates,
                "placement_Q": placements, "rows": rows_by_board,
                "rule": "Q=(E-P)/(B-P) on paired held-out boards; <=0.3 near witness, >=0.7 near blueprint, >1 worse than blueprint"})
    print(json.dumps({"estimates_bb": {k: round(v["mean"], 4) for k, v in estimates.items()},
                      "Q": {k: (round(v["mean"], 3), v["classification"]) for k, v in placements.items()}}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "evaluate", "report"))
    p.add_argument("--plan", type=Path, default=Path("configs/diagnostics/hu20-board-pooling.json"))
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--lineage-index", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--evaluation-fold", type=int, choices=(0, 1))
    p.add_argument("--iterations", type=int)
    p.add_argument("--checkpoints", type=lambda s: [int(x) for x in s.split(",")], default=[])
    p.add_argument("--seed", type=int, default=202610050001)
    p.add_argument("--runs", type=Path, help="training output folder holding fold-0 and fold-1")
    p.add_argument("--main", type=Path, help="#149 main-06 campaign folder")
    p.add_argument("--binary", type=Path)
    p.add_argument("--bootstrap-seed", type=int, default=202610050002)
    p.add_argument("--limit", type=int, help="evaluate only the first N held-out roots (smoke tests)")
    a = p.parse_args()
    {"train": train, "evaluate": evaluate, "report": report}[a.command](a)


if __name__ == "__main__":
    main()
