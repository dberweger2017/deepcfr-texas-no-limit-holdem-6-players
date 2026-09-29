"""Exercise the existing complete artifact path on a declared synthetic entry bound."""

import argparse
import gc
from pathlib import Path
from time import perf_counter, time

from scripts.evaluate_hu20_reopening import Target
from scripts.hu20_scaling_common import acquire, identity, specification
from scripts.train_hu20 import rss, write_json
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, Node, PilotConfig
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.game.hand import Table


def run(out, entries):
    acquire(out)
    r = {"status": "incomplete", "synthetic_entries": entries, "identity": identity(),
         "scope": "Allocation/save/export/reload bound; not trained poker probabilities", "started": time()}
    write_json(out / "manifest.json", r)
    trainer = BlueprintTrainer(Table(("player-0", "player-1"), (2000, 2000)),
        PilotConfig(seed=2026093001, raise_cap=None, max_entries=entries,
                    abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME))
    # Five actions and separately allocated names/lists/floats are conservative
    # relative to the actual menu (fold/check or call, min, pot, conditional jam).
    for i in range(entries):
        names = tuple(["fold", "call", "min", "pot", "jam"])
        trainer.nodes[f"{i:032x}"] = Node(names, [float(i+j+.731) for j in range(5)],
                                         [float(i+j+.319) for j in range(5)], i+1)
    r["table_peak_rss_bytes"] = rss()
    cp = out / "synthetic-checkpoint.json.gz"; policy = out / "synthetic-current.json.gz"
    t = perf_counter(); h = save_training(trainer, cp); r["checkpoint_seconds"] = perf_counter()-t
    t = perf_counter(); ph = export_policy(trainer, policy); r["export_seconds"] = perf_counter()-t
    r["export_peak_rss_bytes"] = rss()
    del trainer; gc.collect()
    spec = specification(2026093001, 0, 0, cp, policy, h, ph)
    t = perf_counter(); source = Target(spec); r["verified_reload_seconds"] = perf_counter()-t
    r.update(status="complete", verified_entries=len(source.source.entries), peak_rss_bytes=rss(),
             checkpoint_sha256=h, policy_sha256=ph, finished=time())
    write_json(out / "result.json", r)
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--entries", type=int, default=3000000); a = p.parse_args()
    print(run(a.out, a.entries))


if __name__ == "__main__": main()
