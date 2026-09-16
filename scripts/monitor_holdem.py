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
    def __init__(self, roots, logdir, writer_factory):
        self.roots = roots
        self.logdir = logdir
        self.writer_factory = writer_factory
        self.writers = {}
        self.seen = set()

    def emit(self, job, source, index, row, prefix):
        identity = (job, source, index, json.dumps(row, sort_keys=True))
        if identity in self.seen:
            return
        if job not in self.writers:
            self.writers[job] = self.writer_factory(str(self.logdir / job))
        writer = self.writers[job]
        step = row["iteration"]
        for tag, value in scalars(row, prefix):
            writer.add_scalar(tag, value, step)
        if "status" in row:
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
    args = parser.parse_args()
    if len({r.name for r in args.runs}) != len(args.runs):
        parser.error("Run directories must have distinct names")
    from torch.utils.tensorboard import SummaryWriter

    # A restart rebuilds from source records into a new directory, without duplicate events.
    args.logdir.mkdir(parents=True, exist_ok=False)
    monitor = Monitor(args.runs, args.logdir, SummaryWriter)
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
