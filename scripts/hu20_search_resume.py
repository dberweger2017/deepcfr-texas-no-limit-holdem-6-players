"""Hash-bound retained screens and exact, successful unlocked-seat solve reuse."""

from copy import deepcopy
import json
from pathlib import Path

from scripts.forecast_hu20_search_final import requests_shareable
from src.arena.schedule import digest
from src.blueprint.hu20_turn_solver import file_hash


def retained_screen(spec, native_configs, targets):
    path = Path(spec['path'])
    if file_hash(path) != spec['sha256']:
        raise ValueError('Retained screen hash differs')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != spec['rows'] or any(r['stage'] != 'screen' for r in rows):
        raise ValueError('Retained screen count/stage differs')
    allowed = {f"{i['root']['spot']}/{i['policy']['seed']}/{b}" for i, b in targets}
    coordinates = [(r['configuration_id'], r['root']) for r in rows]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError('Duplicate retained screen coordinate')
    for row in rows:
        if (row['root'] not in allowed or row['configuration_id'] != digest(row['config'])
                or row['config']['iterations'] != 100
                or row['config']['decision_seconds'] != 30):
            raise ValueError('Retained screen coordinate/configuration differs')
    actual = {(r['configuration_id'], r['root']) for r in rows if r['config']['menu'] == 'native'}
    expected = {(digest(c), root) for c in native_configs for root in allowed}
    if actual != expected:
        raise ValueError('Complete native retained screen required')
    return rows


class SharedRootSolver:
    """Reuse only successful identical requests; never cache failed native work."""

    def __init__(self, native):
        self.native = native
        self.expected_sha256 = native.expected_sha256
        self.cache = {}
        self.records = []
        self.reused_play_seconds = 0
        self.source_cold_seconds = None

    def start_coordinate(self):
        self.records = []
        self.reused_play_seconds = 0

    def solve(self, request, deadline, *, mode='play'):
        cached = self.cache.get(mode)
        if cached and requests_shareable(cached[0], request):
            _, profiles, receipt = cached
            record = deepcopy(receipt)
            record.update(seconds=0, shared_request_sha256=digest(request),
                          shared_native_seconds=receipt['seconds'],
                          shared_receipt_path=receipt.get('path'), shared=True)
            self.records.append(record)
            if mode == 'play':
                self.reused_play_seconds += receipt['seconds']
            return profiles
        before = len(self.native.records)
        try:
            profiles = self.native.solve(request, deadline, mode=mode)
        finally:
            # Include failed receipts even though failures are never shared.
            if len(self.native.records)>before:
                self.records.append(deepcopy(self.native.records[-1]))
        if not request.get('locks'):
            self.cache[mode] = (deepcopy(request), profiles, deepcopy(self.records[-1]))
        return profiles
