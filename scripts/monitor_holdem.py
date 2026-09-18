"""Stream saved Hold'em measurements to TensorBoard without touching the trainer."""

import argparse
import json
import math
import time
from pathlib import Path

STREAMS = {
    "training-timing.jsonl": "timing",
    "iteration-reports.jsonl": "training",
    "checkpoint-timing.jsonl": "checkpoint",
}


def scalars(value, prefix):
    if isinstance(value, dict):
        for name, child in value.items():
            yield from scalars(child, f"{prefix}/{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from scalars(child, f"{prefix}/{index}")
    elif type(value) in (int, float) and math.isfinite(value):
        yield prefix, value


class Monitor:
    def __init__(self, roots, logdir, writer_factory, *, simple=False):
        self.simple = simple
        self.roots = roots
        self.logdir = logdir
        self.writer_factory = writer_factory
        self.writers = {}
        self.seen = set()

    def emit(self, job, source, index, row, prefix):
        identity = (job, source, index, json.dumps(row, sort_keys=True))
        if identity in self.seen:
            return
        if self.simple:
            if source == "training-timing.jsonl":
                values = {"progress/completed_iterations": row["iteration"],
                          "speed/seconds_per_iteration": row["total_seconds"]}
            elif source == "curve" and row["benchmark"] in ("random", "styles"):
                estimate = row["comparison"]["candidate"]
                benchmark = row["benchmark"]
                values = {f"poker/{benchmark}_bb_per_100": estimate["bb_per_100"]}
                if estimate["ci95"] is not None:
                    values.update(zip((f"poker/{benchmark}_ci95_lower", f"poker/{benchmark}_ci95_upper"), estimate["ci95"]))
            else:
                self.seen.add(identity)
                return
        if job not in self.writers:
            self.writers[job] = self.writer_factory(str(self.logdir / job))
            if self.simple:
                self.writers[job].add_custom_scalars({
                    "Training": {
                        "Completed iterations": ["Multiline", ["progress/completed_iterations"]],
                        "Seconds per iteration": ["Multiline", ["speed/seconds_per_iteration"]],
                    },
                    "Poker validation": {
                        "Profit vs scripted opponents (BB per 100 hands, 95% interval)": [
                            "Margin", ["poker/styles_bb_per_100", "poker/styles_ci95_lower", "poker/styles_ci95_upper"]
                        ],
                        "Profit vs random (BB per 100 hands, 95% interval)": [
                            "Margin", ["poker/random_bb_per_100", "poker/random_ci95_lower", "poker/random_ci95_upper"]
                        ],
                    },
                })
        writer = self.writers[job]
        step = row["iteration"]
        measurements = values.items() if self.simple else scalars(row, prefix)
        for tag, value in measurements:
            writer.add_scalar(tag, value, step)
        if not self.simple and "status" in row:
            writer.add_scalar(f"{prefix}/failed", row["status"] != "complete", step)
        writer.flush()
        self.seen.add(identity)

    def poll(self):
        for root in self.roots:
            for directory in sorted(root.glob("scenario-*-seed-*")):
                job = f"{root.name}/{directory.name}"
                for filename, prefix in STREAMS.items():
                    path = directory / filename
                    if not path.exists():
                        continue
                    for index, line in enumerate(path.read_text().splitlines(True)):
                        # The trainer may still be appending the final record.
                        if line.endswith("\n"):
                            self.emit(job, filename, index, json.loads(line), prefix)
                curve = directory / "learning-curve.json"
                if curve.exists():
                    try:
                        rows = json.loads(curve.read_text())
                    except json.JSONDecodeError:
                        continue
                    for row in rows:
                        prefix = f"poker/{row['benchmark']}/{row['scenario']}"
                        self.emit(job, "curve", (row["iteration"], prefix), row, prefix)
        return all(
            (r / "result.json").exists() or (r / "failure.json").exists()
            for r in self.roots
        )

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--simple", action="store_true", help="Four charts: progress, speed, and random/scripted validation with uncertainty")
    args = parser.parse_args()
    if len({r.name for r in args.runs}) != len(args.runs):
        parser.error("Run directories must have distinct names")
    from torch.utils.tensorboard import SummaryWriter

    # A restart rebuilds from source records into a new directory, without duplicate events.
    args.logdir.mkdir(parents=True, exist_ok=False)
    monitor = Monitor(args.runs, args.logdir, SummaryWriter, simple=args.simple)
    try:
        while True:
            finished = monitor.poll()
            if finished or args.once:
                break
            time.sleep(5)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
