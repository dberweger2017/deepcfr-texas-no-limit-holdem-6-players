"""Tally only individual hand results already rendered to the browser player."""

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

from scripts.audit_luna_browser import _metadata, read


def text_blocks(output):
    if isinstance(output, list):
        for item in output:
            yield from text_blocks(item)
    elif isinstance(output, dict):
        yield from text_blocks(output.get("text", ""))
    elif isinstance(output, str):
        yield output


def tally(records):
    ledger = {}
    for record in records:
        payload = record.get("payload", {})
        if record.get("type") != "response_item" or payload.get("type") != "function_call_output":
            continue
        text = "\n".join(text_blocks(payload.get("output", "")))
        amounts = set(re.findall(r"This hand:\s*([+−-][0-9.]+) BB", text))
        ordinals = {item["handOrdinal"] for item in _metadata(text)
                    if not item.get("attemptedButtonLabel", "").lower().startswith("deal hand")}
        # Standalone rereads have no decision correlation. Avoid guessing which
        # hand they belong to; correlated duplicate observations are deduplicated.
        if len(amounts) != 1 or len(ordinals) != 1:
            continue
        ordinal = ordinals.pop()
        chips = int(Decimal(amounts.pop().replace("−", "-")) * 100)
        if ordinal in ledger and ledger[ordinal] != chips:
            raise ValueError(f"Conflicting rendered results for hand {ordinal}")
        ledger[ordinal] = chips
    if not ledger:
        raise ValueError("No correlated rendered results")
    missing = sorted(set(range(1, max(ledger) + 1)) - ledger.keys())
    if missing:
        raise ValueError(f"Missing rendered hand results: {missing}")
    net = sum(ledger.values())
    return {"scope": "Public rendered hand results; provisional until final native/server validation",
            "throughHand": max(ledger), "netChips": net, "netBB": net / 100,
            "bbPer100": net / len(ledger), "wins": sum(x > 0 for x in ledger.values()),
            "losses": sum(x < 0 for x in ledger.values()), "ties": sum(x == 0 for x in ledger.values()),
            "buttonSBChips": sum(x for i, x in ledger.items() if i % 2 == 1),
            "bigBlindChips": sum(x for i, x in ledger.items() if i % 2 == 0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    print(json.dumps(tally(read(parser.parse_args().rollout)), indent=2))
