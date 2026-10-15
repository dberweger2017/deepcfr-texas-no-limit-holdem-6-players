import gzip
import hashlib
import json

import pytest

from scripts import run_hu200_v055_confirmation as campaign


def test_clock_reserves_and_stage1_recipe(tmp_path):
    seconds = campaign.remaining_seconds(50_000, 10_000)
    assert seconds == 31_900
    args = campaign.train_args(2026101421, 3_000_000_000, tmp_path, 85_373_862, seconds)
    flags = dict(zip(args[2::2], args[3::2]))
    assert flags['--seed'] == 2026101421
    assert flags['--max-entries'] == 85_373_862
    assert flags['--roots-per-seat'] == 1
    assert flags['--average-rule'] == 'opponent-sampled'
    assert flags['--nodes'] == 3_000_000_000
    assert flags['--max-seconds'] == seconds
    assert '--freeze-at-cap' not in args and '--resume' not in args
    with pytest.raises(ValueError, match='No training time'):
        campaign.train_args(2026101422, 3_000_000_000, tmp_path, 85_373_862, 0)


def test_serialized_key_hash_checks_all_keys_and_order(tmp_path):
    path = tmp_path/'checkpoint.gz'
    key1, key2 = '0'*32, '1'*32
    def write(keys):
        with gzip.open(path, 'wt') as stream:
            stream.write(json.dumps({'iteration': 7})+'\n')
            for key in keys:
                stream.write(json.dumps([key, [], [], [], 0, 0], separators=(',', ':'))+'\n')
    write([key1, key2])
    result = campaign.keys(path, 2)
    assert result['sha256'] == hashlib.sha256((key1+key2).encode()).hexdigest()
    with pytest.raises(ValueError, match='count mismatch'):
        campaign.keys(path, 3)
    write([key2, key1])
    with pytest.raises(ValueError, match='Unsorted'):
        campaign.keys(path, 2)
