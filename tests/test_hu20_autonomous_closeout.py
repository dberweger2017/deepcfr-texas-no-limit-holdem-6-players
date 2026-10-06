import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts.closeout_hu20_autonomous_arena import retrieve_manifest, terminate_all, OWNED
from scripts.hu20_search_evidence import pack, verify_archives, file_hash
from scripts.report_hu20_autonomous_arena import paired_analysis, defect_affected


def paired_rows(blocks=40):
    plan = {'models': [{'seed': s} for s in (1, 2, 3)], 'panels': [{'name': 'panel', 'blocks': blocks}]}
    rows = []
    for seed in (1, 2, 3):
        for b in range(blocks):
            for arm in ('base', 'search'):
                for r in (0, 1):
                    rows.append({'seed': seed, 'strategy': 'average', 'panel': 'panel', 'block': b,
                        'rotation': r, 'button': b % 2, 'arm': arm, 'pod_id': 'pod-' + str(b % 3),
                        'host': 'host', 'target_chips': 0 if arm == 'base' else b * seed,
                        'tails': {'counts': {}, 'first_large_raise_response': 'none'},
                        'coverage': {}, 'actions': [], 'defect_affected': False})
    return plan, rows


def test_primary_matches_frozen_estimator_and_sensitivity_flags_whole_pair():
    from scripts.evaluate_hu20_turn_search import summarize_phase
    plan, rows = paired_rows()
    primary = paired_analysis(plan, rows)
    assert primary['three_lineage_changes'][0]['search_minus_base'] == summarize_phase(rows, 'arena')['three_lineage_changes'][0]['search_minus_base']
    rows[0]['defect_affected'] = True
    sensitivity = paired_analysis(plan, rows, exclude_defects=True)
    counts = sensitivity['paired_blocks_per_panel']['panel']
    assert counts['excluded_defect'] == 1
    assert sensitivity['included_hands'] == len(rows) - 4
    assert sensitivity['three_lineage_changes'][0]['blocks'] == list(range(1, 40))
    assert primary['included_hands'] == len(rows)


def test_crash_missing_hand_excludes_pair_and_aligns_aggregate_by_block_id():
    plan, rows = paired_rows()
    rows = [r for r in rows if not (r['seed'] == 2 and r['block'] == 3 and r['arm'] == 'search' and r['rotation'] == 1)]
    for mode in (False, True):
        result = paired_analysis(plan, rows, exclude_defects=mode)
        assert result['included_hands'] == 4 * (120 - 1)
        assert result['paired_blocks_per_panel']['panel']['incomplete'] == 1
        value = result['three_lineage_changes'][0]
        expected = [b for b in range(40) if b != 3]
        assert value['blocks'] == expected
        assert value['search_minus_base']['bb_per_100'] == sum(2 * b for b in expected) / len(expected)


def test_fallback_probe_and_conditioning_gap_flag_the_block():
    for counts, records in (({'probe:fallback:zero_support': 1}, []),
                            ({'range:turn_conditioning_fallback:no_lock': 1}, []),
                            ({}, [{'status': 'defect'}])):
        assert defect_affected({'search_counts': counts, 'search_records': records})
    assert not defect_affected({'search_counts': {'play:decision': 3}, 'search_records': [{'status': 'decision'}]})


def test_retrieval_verifies_hashes_and_checks_disk_before_any_download(tmp_path, monkeypatch):
    evidence = tmp_path / 'evidence'; evidence.mkdir(); (evidence / 'raw').write_bytes(b'important raw bytes')
    source = tmp_path / 'source'; manifest = pack(evidence, source)
    destination = tmp_path / 'retrieved'; destination.mkdir()
    shutil.copyfile(source / 'manifest.json', destination / 'manifest.json')
    fetched = []
    def fetch(name, path):
        fetched.append(name); shutil.copyfile(source / name, path)
    ledger = {'retrieval_limit_bytes': 10**9, 'retrieval_free_reserve_bytes': 100}
    monkeypatch.setattr('scripts.closeout_hu20_autonomous_arena.shutil.disk_usage', lambda p: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='real free disk'):
        retrieve_manifest(manifest, destination, fetch, tmp_path, ledger)
    assert not fetched
    monkeypatch.setattr('scripts.closeout_hu20_autonomous_arena.shutil.disk_usage', lambda p: SimpleNamespace(free=10**10))
    assert retrieve_manifest(manifest, destination, fetch, tmp_path, ledger)['verified']
    assert len(fetched) == 1
    retrieve_manifest(manifest, destination, fetch, tmp_path, ledger)
    assert len(fetched) == 1
    (destination / manifest['archives'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        verify_archives(destination)


def test_no_termination_until_every_pods_members_verify(tmp_path):
    pods = [{'id': p} for p in OWNED]
    ledger = {'pods': pods, 'attempt': 2, 'relay_pod_id': pods[0]['id']}
    calls = []
    with pytest.raises(FileNotFoundError):
        terminate_all(tmp_path, ledger, lambda *args: calls.append(args))
    assert calls == []


def test_termination_confirms_identity_404_and_all_list_pages(tmp_path):
    ids = sorted(OWNED)
    ledger = {'attempt': 2, 'pods': [{'id': p, 'name': p + '-name', 'gpu_id': 'gpu', 'owner_handoff': 'owner-approved'} for p in ids], 'relay_pod_id': ids[0]}
    for p in ledger['pods']:
        raw = tmp_path / (p['id'] + '-raw'); raw.mkdir(); (raw / 'data').write_text(p['id'])
        folder = tmp_path / 'retrieved' / p['id']; folder.parent.mkdir(exist_ok=True)
        pack(raw, folder)
        (folder / 'retrieval-verified.json').write_text(json.dumps({**verify_archives(folder), 'archive_manifest_sha256': file_hash(folder / 'manifest.json')}))
    deleted = set(); calls = []
    def call(method, params, ident):
        name, args = params['name'], params['arguments']; calls.append((name, args))
        if name == 'get-pod':
            pid = args['id']
            if pid in deleted:
                return {'result': {'isError': True, 'content': [{'type': 'text', 'text': '404 Not Found'}]}}
            value = {'id': pid, 'name': pid + '-name', 'gpu': {'id': 'gpu'}}
        elif name == 'delete-pod':
            deleted.add(args['id']); return {'result': {'isError': False}}
        elif not args:
            value = {'pods': [{'id': 'unrelated'}], 'pagination': {'hasNextPage': True, 'nextCursor': 'page-two'}}
        else:
            assert args == {'cursor': 'page-two'}
            value = {'pods': [], 'pagination': {'hasNextPage': False}}
        return {'result': {'content': [{'type': 'text', 'text': json.dumps(value)}]}}
    terminate_all(tmp_path, ledger, call)
    assert deleted == OWNED
    assert [a['id'] for n, a in calls if n == 'delete-pod'][-1] == ids[0]
    assert (tmp_path / 'CLOSEOUT_COMPLETE.json').exists()
    assert len(json.loads((tmp_path / 'final-list-pods.json').read_text())) == 2


def test_research_zip_retains_originals_and_refuses_overwrite(tmp_path):
    from scripts.archive_hu20_autonomous_arena import archive_research
    import zipfile
    root = tmp_path / 'research'; root.mkdir(); (root / 'result').write_text('raw science')
    (root / 'control-token').write_text('never export credentials')
    output = tmp_path / 'drive' / 'attempt.zip'
    proof = archive_research(root, output)
    assert proof['verified'] and proof['members'] == 1
    assert (root / 'result').read_text() == 'raw science'
    with zipfile.ZipFile(output) as archive:
        assert 'control-token' not in archive.namelist()
        assert 'RESEARCH_MEMBER_HASHES.json' in archive.namelist()
    with pytest.raises(ValueError, match='new archive'):
        archive_research(root, output)
