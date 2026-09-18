import json

import pytest

from scripts.monitor_holdem import Monitor


class Writer:
    def __init__(self, directory):
        self.values = []
        self.closed = False
        self.flushed = 0

    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


def test_monitor_tails_partial_records_and_keeps_seeds_separate(tmp_path):
    roots = [tmp_path / f"run-{seed}" for seed in (307, 311)]
    for root in roots:
        job = root / f"scenario-0-seed-{root.name[-3:]}"
        job.mkdir(parents=True)
        (job / "training-timing.jsonl").write_text(
            '{"iteration":1,"total_seconds":4,"status":"complete"}\n{"iteration":2'
        )
    monitor = Monitor(roots, tmp_path / "events", Writer)
    assert not monitor.poll()
    assert len(monitor.writers) == 2
    first = monitor.writers["run-307/scenario-0-seed-307"]
    original = list(first.values)
    monitor.poll()
    assert first.values == original
    with (roots[0] / "scenario-0-seed-307/training-timing.jsonl").open("a") as f:
        f.write(',"status":"failed"}\n')
    monitor.poll()
    assert ("timing/failed", True, 2) in first.values
    for root in roots:
        (root / "failure.json").write_text("{}")
    assert monitor.poll()
    monitor.close()
    assert all(w.closed for w in monitor.writers.values())
    assert first.flushed == 2


def test_monitor_exports_evaluation_intervals_and_real_events(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    root = tmp_path / "run"
    job = root / "scenario-0-seed-307"
    job.mkdir(parents=True)
    curve = job / "learning-curve.json"
    monitor = Monitor([root], tmp_path / "events", SummaryWriter)
    curve.write_text("[")
    assert not monitor.poll()
    curve.write_text(
        json.dumps(
            [
                {
                    "iteration": 128,
                    "benchmark": "random",
                    "scenario": "six-100bb",
                    "comparison": {"candidate": {"bb_per_100": 42, "ci95": [-10, 94]}},
                }
            ]
        )
    )
    monitor.poll()
    monitor.poll()
    monitor.close()
    events = EventAccumulator(str(tmp_path / "events/run/scenario-0-seed-307")).Reload()
    points = events.Scalars("poker/random/six-100bb/comparison/candidate/bb_per_100")
    assert [(p.step, p.value) for p in points] == [(128, 42)]
    assert (
        events.Scalars("poker/random/six-100bb/comparison/candidate/ci95/0")[0].value
        == -10
    )


def test_simple_monitor_has_four_charts_and_retains_uncertainty(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    monitor = Monitor([], tmp_path, SummaryWriter, simple=True)
    monitor.emit("seed", "training-timing.jsonl", 0,
                 {"iteration": 64, "total_seconds": 20, "nodes": 100}, "timing")
    monitor.emit("seed", "curve", 64,
                 {"iteration": 64, "benchmark": "random", "comparison": {
                     "candidate": {"bb_per_100": 42, "ci95": [-10, 94]}}}, "poker")
    monitor.emit("seed", "curve", 64,
                 {"iteration": 64, "benchmark": "styles", "comparison": {
                     "candidate": {"bb_per_100": -1100, "ci95": [-1300, -900]}}}, "poker")
    monitor.emit("seed", "iteration-reports.jsonl", 0,
                 {"iteration": 64, "irrelevant": 1000}, "training")
    monitor.close()
    events = EventAccumulator(str(tmp_path / "seed")).Reload()
    assert set(events.Tags()["scalars"]) == {
        "progress/completed_iterations", "speed/seconds_per_iteration",
        "poker/random_bb_per_100", "poker/random_ci95_lower", "poker/random_ci95_upper",
        "poker/styles_bb_per_100", "poker/styles_ci95_lower", "poker/styles_ci95_upper",
    }
    assert events.Scalars("poker/random_ci95_lower")[0].value == -10
    assert events.Scalars("poker/styles_bb_per_100")[0].value == -1100
    assert events.Scalars("poker/styles_ci95_upper")[0].value == -900
    from tensorboard.plugins.custom_scalar import layout_pb2
    layout = layout_pb2.Layout.FromString(
        events.Tensors("custom_scalars__config__")[0].tensor_proto.string_val[0])
    assert sum(len(category.chart) for category in layout.category) == 4
