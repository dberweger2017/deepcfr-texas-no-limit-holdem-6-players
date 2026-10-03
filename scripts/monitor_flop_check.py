"""Incrementally stream the external flop diagnostic to TensorBoard."""

import argparse
import json
from pathlib import Path
import time

from src.diagnostics.flop_check_analysis import bootstrap_spots


METRICS=("e_bp", "e_v1proj", "e_v1proj_line", "e_eq50", "e_eq200", "alias_cost")


class Monitor:
    def __init__(self, run, writer):
        self.run = Path(run); self.writer = writer; self.offset = 0
        self.spots = {}; self.events = 0
        if hasattr(writer, "add_custom_scalars"):
            writer.add_custom_scalars({
                "Monitoring only": {
                    f"{group}/{strategy}/{metric}": ["Margin", [
                        f"results/monitoring_only/{group}/{strategy}/{metric}/{x}"
                        for x in ("mean", "ci95_lower", "ci95_upper")]]
                    for group in ("A", "B") for strategy in ("current", "stored-average")
                    for metric in METRICS},
                "Overfold": {"Blueprint and equilibrium": ["Multiline", [
                    "overfold/fold_bp", "overfold/fold_eq"]]},
                "Resources": {"GiB": ["Multiline", [
                    "resources/rss_gib", "resources/memory_estimate_gib"]]}})

    def poll(self):
        path = self.run / "progress.jsonl"
        if path.exists():
            with path.open("rb") as source:
                source.seek(self.offset)
                while True:
                    before = source.tell(); line = source.readline()
                    if not line or not line.endswith(b"\n"):
                        self.offset = before; break
                    row = json.loads(line); self.offset = source.tell()
                    self.events += 1; self.emit(row)
        self.writer.flush()
        return (self.run / "result.json").exists() or (self.run / "failure.json").exists()

    def scalar(self, tag, value, step):
        if value is not None:
            self.writer.add_scalar(tag, value, step)

    def emit(self, row):
        step = row.get("iteration", self.events)
        if "exploitability_pct_pot" in row:
            self.scalar(f'solver/exploitability_pct_pot/{row.get("series", row.get("spot", "fixture"))}',
                        row["exploitability_pct_pot"], step)
        self.scalar("resources/rss_gib", row.get("rss_bytes", 0) / 1024**3, self.events)
        if "solver_peak_rss_bytes" in row:
            self.scalar("resources/solver_peak_rss_gib", row["solver_peak_rss_bytes"] / 1024**3, self.events)
        estimate = row.get("compressed_bytes", row.get("uncompressed_bytes"))
        if estimate is not None:
            self.scalar("resources/memory_estimate_gib", estimate / 1024**3, self.events)
        if row.get("event") == "gate":
            self.scalar(f'validation/gates_passed/{row["gate"]}', int(row["passed"]), self.events)
        if row.get("event") == "run_counter":
            for group, count in row["spots_completed_by_set"].items():
                self.scalar("run/spots_completed_by_set/" + group, count, self.events)
        if row.get('event')=='run_counter':
            for name in ('jobs_completed_by_set','roots_fully_completed_by_set'):
                for group,count in row.get(name,{}).items():self.scalar('run/'+name+'/'+group,count,self.events)
        if row.get("event") != "spot_complete":
            return
        key = (row["set"], row["spot"], row["lineage"], row["strategy"])
        self.spots[key] = row
        for name in ("fold_bp", "fold_eq"):
            self.scalar("overfold/" + name, row.get(name), len(self.spots))
        for group in sorted({r["set"] for r in self.spots.values()}):
            count = len({r["spot"] for r in self.spots.values() if r["set"] == group})
            self.scalar("run/spots_completed_by_set/" + group, count, len(self.spots))
        groups = sorted({(r["set"], r["strategy"]) for r in self.spots.values()})
        for group, strategy in groups:
            rows = [r for r in self.spots.values()
                    if (r["set"], r["strategy"]) == (group, strategy)]
            for metric in METRICS:
                summary = bootstrap_spots(rows, metric, resamples=200)
                prefix = f"results/monitoring_only/{group}/{strategy}/{metric}"
                self.scalar(prefix + "/mean", summary["mean"], len(rows))
                if summary["ci95"]:
                    self.scalar(prefix + "/ci95_lower", summary["ci95"][0], len(rows))
                    self.scalar(prefix + "/ci95_upper", summary["ci95"][1], len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.logdir)); monitor = Monitor(args.run, writer)
    try:
        while True:
            if monitor.poll() or args.once:
                break
            time.sleep(5)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
