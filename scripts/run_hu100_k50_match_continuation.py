"""Owner-authorized match-only continuation; original training and stops stay immutable."""
import argparse

from scripts import run_hu100_k50_2b as campaign
from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash

ORIGINAL = campaign.BASE
ATTEMPT = ORIGINAL / 'match-continuation-warning-20261012'
AUTHORIZATION = campaign.ROOT / 'docs/reports/hu100-k50-2b-artifacts/continuation-authorization.json'
RESERVATION = campaign.ROOT / 'docs/reports/hu100-k50-2b-artifacts/continuation-root-reservation.json'
ROOTS = {'pilot': 20261012101104, 'primary': 20261012101102, 'secondary': 20261012101103}
ORIGINAL_LIMITS = campaign.limits
SWAP_GROWTH_BYTES = 2_000_000_000
STAGES = ('pilot', 'final', 'readout')


def limits(sample, rss, baseline, now):
    # Darwin pressure values are normal=1, warning=2, critical=4. Unknown values fail closed.
    if sample['pressure_level'] not in (1, 2):
        return 'critical or unknown macOS memory pressure'
    return ORIGINAL_LIMITS(sample, rss, baseline, now,
        allowed_pressure=(1, 2), swap_growth_bytes=SWAP_GROWTH_BYTES)


def configure():
    authorization = campaign.read(AUTHORIZATION)
    reservation = campaign.read(RESERVATION)
    if authorization['status'] != 'owner-authorized-match-only' or reservation['roots'] != ROOTS:
        raise ValueError('Owner authorization and reserved continuation roots required')
    if file_hash(ORIGINAL / 'campaign-failure.json') != authorization['original_stop_sha256']:
        raise ValueError('Original stop changed')
    for name, expected in authorization['immutable_metadata_sha256'].items():
        if file_hash(ORIGINAL / name) != expected:
            raise ValueError('Completed training/input metadata changed: ' + name)
        if file_hash(ATTEMPT / name) != expected:
            raise ValueError('Continuation metadata differs: ' + name)
    campaign.BASE = ATTEMPT
    campaign.ROOTS = dict(ROOTS)
    campaign.limits = limits


def initialize():
    authorization = campaign.read(AUTHORIZATION)
    if authorization['status'] != 'owner-authorized-match-only':
        raise ValueError('Explicit owner amendment required')
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    for name, expected in authorization['immutable_metadata_sha256'].items():
        source = ORIGINAL / name
        if file_hash(source) != expected:
            raise ValueError('Pinned metadata changed: ' + name)
        destination = ATTEMPT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    put(ATTEMPT / 'authorization.json', authorization)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('initialize', 'guard', 'freeze', *STAGES))
    parser.add_argument('--worker', choices=STAGES)
    parser.add_argument('--pr', type=int)
    args = parser.parse_args()
    if args.stage == 'initialize':
        initialize()
        return
    configure()
    if args.stage == 'guard':
        if args.worker is None:
            raise ValueError('Explicit match-only worker required')
        campaign.guard(args.worker, ['.venv/bin/python', '-m',
            'scripts.run_hu100_k50_match_continuation', args.worker])
    elif args.stage == 'freeze':
        campaign.freeze(args.pr, reservation_path=RESERVATION, fixed_counts=True)
    else:
        {'pilot': campaign.pilot, 'final': campaign.final_play, 'readout': campaign.readout}[args.stage]()


if __name__ == '__main__':
    main()
