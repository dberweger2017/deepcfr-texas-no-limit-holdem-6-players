"""Pinned average loading, rejection, and durable native play."""

import gzip
import hashlib
import json

import pytest

from src.diagnostics.cfr_average import DiagnosticAverage, extract
from src.play_api import o_candidate as candidate
from src.play_api.service import PlayService
from tests.diagnostics.test_cfr_average import fixture
from tests.play_ui.test_service import create, hand_start, human_action, bot_action


def model(tmp_path, monkeypatch):
    _, view, checkpoint, _, spec = fixture(tmp_path)
    lines = gzip.decompress(checkpoint.read_bytes()).decode().splitlines()
    header = json.loads(lines[0])
    header['average_rule'] = 'opponent-sampled'
    lines[0] = json.dumps(header)
    checkpoint.write_bytes(gzip.compress(('\n'.join(lines) + '\n').encode(), mtime=0))
    spec['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path = tmp_path / 'O.average.jsonl.gz'
    result = extract(checkpoint, spec, path)
    for name, value in {'MODEL_BYTES': path.stat().st_size, 'MODEL_SHA256': result['sha256'],
                        'CHECKPOINT_SHA256': spec['checkpoint_sha256'],
                        'SEED': spec['seed'], 'ITERATION': spec['iteration']}.items():
        monkeypatch.setattr(candidate, name, value)
    return path, view


def test_pinned_reader_preserves_average_probabilities_and_replays(tmp_path, monkeypatch):
    path, view = model(tmp_path, monkeypatch)
    loaded = candidate.load_o_candidate(path)
    original = DiagnosticAverage(path, candidate.MODEL_SHA256)
    assert loaded.distribution(view) == original.distribution(view)
    assert loaded.distribution(view)[1][0] == 0
    service = PlayService(tmp_path / 'play.sqlite', loaded)
    try:
        assert service.model_info()['sha256'] == candidate.MODEL_SHA256
        state = hand_start(service, create(service, 'free'))
        state = human_action(service, state, 'raise', 201)
        for step in range(100):
            if state['phase'] == 'finished':
                break
            if state['hand']['actor'] == 1:
                state = bot_action(service, state, f'bot-step-{step:016d}')
            else:
                kind = 'check' if 'check' in state['hand']['legal']['kinds'] else 'call'
                state = human_action(service, state, kind, key=f'human-step-{step:014d}')
        assert state['phase'] == 'finished'
        assert service.verify_replay(state['sessionId']) == 1
        assert service._load(state['sessionId'])['history'][0]['actions'][0]['raiseTo'] == 201
    finally:
        service.close()
    restarted = PlayService(tmp_path / 'play.sqlite', candidate.load_o_candidate(path))
    try:
        assert restarted.state(state['sessionId'])['model']['sha256'] == candidate.MODEL_SHA256
        assert restarted.verify_replay(state['sessionId']) == 1
    finally:
        restarted.close()


def test_wrong_lineage_and_changed_bytes_are_rejected(tmp_path, monkeypatch):
    path, _ = model(tmp_path, monkeypatch)
    monkeypatch.setattr(candidate, 'SEED', 999)
    with pytest.raises(ValueError, match='lineage'):
        candidate.load_o_candidate(path)
    data = path.read_bytes()
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='byte count'):
        candidate.verify(path)
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match='SHA-256'):
        candidate.verify(path)
    linked = tmp_path / 'link.gz'
    linked.symlink_to(path)
    with pytest.raises(ValueError, match='regular file'):
        candidate.verify(linked)


def test_bundle_retains_bytes_hashes_and_publication_hold(tmp_path, monkeypatch):
    from scripts import build_v041_bundle as bundle
    path, _ = model(tmp_path, monkeypatch)
    destination = tmp_path / 'unpublished'
    bundle.build(path, destination, 'a' * 40)
    assert (destination / bundle.ASSET_NAME).read_bytes() == path.read_bytes()
    manifest = json.loads((destination / 'release-manifest.json').read_text())
    assert manifest['approved_release_source_commit'] is None
    assert manifest['owner_publication_approval'] is False
    assert manifest['status'] == 'unpublished-owner-review'
    for line in (destination / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ')
        assert bundle.sha(destination / name) == expected
    with pytest.raises(ValueError, match='Preserve'):
        bundle.build(path, destination, 'a' * 40)
