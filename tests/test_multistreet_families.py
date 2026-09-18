import json
from pathlib import Path

from src.holdem.multistreet_families import (
    SPLIT_COUNTS,
    excluded_flops,
    family_summary,
    generate_families,
)
from src.holdem.multistreet_reference import compatible_visible_deals, flop_key
from src.holdem.representation_reference import range_support

PLAN_PATH = Path("configs/holdem/multistreet-campaign.json")


def test_frozen_families_reproduce_and_keep_every_descendant_compatible():
    plan = json.loads(PLAN_PATH.read_text())
    families = generate_families(plan)
    assert families == plan["families"]
    support = range_support(plan["range_templates"])
    forbidden = excluded_flops(plan["forbidden_flop_plans"])
    keys = [flop_key(family["flop"]) for family in families]
    assert len(set(keys)) == 48
    assert not set(keys) & forbidden
    assert {split: sum(row["split"] == split for row in families) for split in SPLIT_COUNTS} == SPLIT_COUNTS
    for family in families:
        board = family["flop"] + family["continuation"]
        assert len({tuple(sorted(holding)) for holding in family["holdings"]}) == 4
        for holding in family["holdings"]:
            for street_cards in (3, 4, 5):
                assert compatible_visible_deals(support, board[:street_cards], holding)
    summary = family_summary(families)
    assert set(summary["train"]["flop_suits"]) == {"1", "2", "3"}
    assert set(summary["train"]["flop_distinct_ranks"]) == {"2", "3"}
    assert summary["train"]["distinct_holdings"] > 24
    assert len({tuple(row["continuation"]) for row in families}) > 40


def test_exclusions_include_pilot_contexts_and_suit_equivalents(tmp_path):
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"contexts": [{"board": ["Ac", "Kd", "7h", "2s"]}]}))
    keys = excluded_flops(["pilot.json"], base_dir=tmp_path)
    assert flop_key(("Ah", "Ks", "7c")) in keys
    assert flop_key(("Ac", "Kd", "7h", "3s", "4s")) in keys
