"""Independently audit every retained spectator decision, including an active hand."""

import argparse
import json
from pathlib import Path
import sqlite3

from src.play_api.o_candidate import ASSET_NAME, load_o_candidate
from src.play_api.releases import release_identity
from src.play_api.service import load_b100m
from src.play_api.spectator_audit import audit_states
from scripts.verify_v04_model import EXPECTED_NAME, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    verify(args.models_dir / EXPECTED_NAME)
    policies = {'v0.4.0': load_b100m(args.models_dir / EXPECTED_NAME),
                'v0.4.1': load_o_candidate(args.models_dir / ASSET_NAME)}
    identities = {version: release_identity(version, policy) for version, policy in policies.items()}
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        states = [json.loads(row[0]) for row in db.execute('SELECT state FROM sessions ORDER BY id')]
    result = audit_states(states, policies, identities)
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
