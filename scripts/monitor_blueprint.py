"""Stream blueprint campaign progress and poker validation to TensorBoard."""

import argparse
import json
import time
from pathlib import Path


class Monitor:
    def __init__(self, run: Path, writer):
        self.run = run
        self.writer = writer
        self.seen = {"progress.jsonl": 0, "evaluation.jsonl": 0}
        if hasattr(writer, "add_custom_scalars"):
            writer.add_custom_scalars({
                "Poker profit": {
                    "Random BB/100 with 95% interval": ["Margin", [
                        "poker/random/candidate_bb_per_100",
                        "poker/random/candidate_ci95_lower",
                        "poker/random/candidate_ci95_upper",
                    ]],
                    "Scripted BB/100 with 95% interval": ["Margin", [
                        "poker/styles/candidate_bb_per_100",
                        "poker/styles/candidate_ci95_lower",
                        "poker/styles/candidate_ci95_upper",
                    ]],
                },
                "Trained decision coverage": {
                    "Random preflop and flop": ["Multiline", [
                        "coverage/random/preflop/trained_fraction",
                        "coverage/random/flop/trained_fraction",
                    ]],
                    "Scripted preflop and flop": ["Multiline", [
                        "coverage/styles/preflop/trained_fraction",
                        "coverage/styles/flop/trained_fraction",
                    ]],
                },
            })

    def poll(self):
        for filename in self.seen:
            path = self.run / filename
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            for line in lines[self.seen[filename]:]:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                self._emit_progress(row) if filename == "progress.jsonl" else self._emit_evaluation(row)
                self.seen[filename] += 1
        self.writer.flush()
        return (self.run / "result.json").exists() or (self.run / "failure.json").exists()

    def _emit_progress(self, row):
        step = row["iteration"]
        values = {
            "training/table_entries": row["entries"],
            "training/traversal_nodes": row["nodes"],
            "training/nodes_per_step_second": row["nodes_per_step_second"],
            "training/peak_rss_gib": row["peak_rss_bytes"] / 1024**3,
            "training/conservative_rss_gib": row["conservative_rss_bytes"] / 1024**3,
            "training/wall_hours": row["wall_seconds"] / 3600,
            "checkpoint/size_gib": row["checkpoint_bytes"] / 1024**3,
            "checkpoint/save_seconds": row["checkpoint_seconds"],
        }
        for tag, value in values.items():
            if value is not None:
                self.writer.add_scalar(tag, value, step)

    def _emit_evaluation(self, row):
        if row["status"] != "valid":
            self.writer.add_scalar(f"poker/{row['benchmark']}/invalid", 1, row["iteration"])
            return
        step = row["iteration"]
        prefix = f"poker/{row['benchmark']}"
        comparison = row["comparison"]
        for name in ("candidate", "baseline", "paired_difference"):
            estimate = comparison[name]
            self.writer.add_scalar(f"{prefix}/{name}_bb_per_100", estimate["bb_per_100"], step)
            if estimate["ci95"] is not None:
                self.writer.add_scalar(f"{prefix}/{name}_ci95_lower", estimate["ci95"][0], step)
                self.writer.add_scalar(f"{prefix}/{name}_ci95_upper", estimate["ci95"][1], step)
        self.writer.add_scalar(f"{prefix}/completed_hands", row["completed_hands"], step)
        for street, values in row["coverage"].items():
            if values["trained_fraction"] is not None:
                self.writer.add_scalar(f"coverage/{row['benchmark']}/{street}/trained_fraction", values["trained_fraction"], step)
            self.writer.add_scalar(f"coverage/{row['benchmark']}/{street}/decisions", values["decisions"], step)
            if values["decisions"]:
                for action, count in values["actions"].items():
                    self.writer.add_scalar(
                        f"actions/{row['benchmark']}/{street}/{action}_fraction",
                        count / values["decisions"], step,
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    from torch.utils.tensorboard import SummaryWriter

    args.logdir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(args.logdir))
    monitor = Monitor(args.run, writer)
    try:
        while True:
            if monitor.poll() or args.once:
                break
            time.sleep(5)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
