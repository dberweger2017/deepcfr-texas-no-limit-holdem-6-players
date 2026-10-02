"""Observation-only stored-average HU20 play, with optional external late-street search."""

import argparse
import json
from pathlib import Path

from scripts.evaluate_hu20_turn_search import load
from scripts.play_hu20_native import play
from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy, TurnSearchConfig
from src.blueprint.hu20_turn_solver import ExternalTurnSolver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,default=Path("configs/diagnostics/hu20-turn-search-part-a.json"))
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--history",type=Path,required=True)
    p.add_argument("--lineage",type=int,default=2026093001)
    p.add_argument("--strategy",choices=("average","current"),default="average")
    p.add_argument("--search-config",type=Path);p.add_argument("--binary",type=Path)
    p.add_argument("--binary-sha256");p.add_argument("--evidence",type=Path)
    p.add_argument("--seed",type=int,default=202610020804);p.add_argument("--hands",type=int)
    a=p.parse_args()
    spec=next(s for s in json.loads(a.artifacts.read_text())["models"]
              if s["strategy"]==a.strategy and s["seed"]==a.lineage)
    base=load(spec,a.inputs);source=base
    if a.search_config:
        if not a.binary or not a.binary_sha256 or not a.evidence:
            p.error("Search requires fingerprinted executable and retained evidence directory")
        source=HU20TurnSearchPolicy(base,ExternalTurnSolver(a.binary,a.evidence,
            expected_sha256=a.binary_sha256),TurnSearchConfig(**json.loads(a.search_config.read_text())))
    result=play(a.inputs/spec["path"],spec["sha256"],a.history,source=source,
                seed=a.seed,max_hands=a.hands)
    if isinstance(source,HU20TurnSearchPolicy):
        result["search_counts"]=dict(source.stats)
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":main()
