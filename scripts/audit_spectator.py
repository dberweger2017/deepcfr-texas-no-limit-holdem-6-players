"""Independently audit every retained spectator decision, including an active hand."""

import argparse
import json
from pathlib import Path
import sqlite3

from src.play_api.versions import RELEASES
from src.play_api.spectator_audit import audit_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    for release in RELEASES:
        release.verify(args.models_dir / release.asset_name)
    policies = {release.version: release.load(args.models_dir / release.asset_name) for release in RELEASES}
    identities = {release.version: release.identity(policies[release.version]) for release in RELEASES}
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        states = [json.loads(row[0]) for row in db.execute('SELECT state FROM sessions ORDER BY id')]
    result = audit_states(states, policies, identities)
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
