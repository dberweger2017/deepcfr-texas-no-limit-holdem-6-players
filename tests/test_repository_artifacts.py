"""Storage checks inspect staged bytes and keep legacy exemptions exact."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_repository_artifacts as storage


class StoragePolicyTests(unittest.TestCase):
    def test_readable_reports_source_and_small_fixtures_are_allowed(self):
        for path in ('docs/reports/summary.json', 'tests/fixtures/decision.json',
                     'docs/reports/chart.svg', 'native/hu20-trainer/src/game.rs',
                     'docs/archive.zip.sha256', 'docs/trace.jsonl.manifest.json'):
            self.assertEqual(storage.reasons(storage.Entry(path, 'a' * 40, 100)), [])

    def test_payloads_are_detected_even_when_small_or_interrupted(self):
        for path in ('models/readme.md', 'planning/frozen.json', 'results/manifest.json',
                     'docs/reports/model.npz', 'docs/reports/hand.jsonl',
                     'docs/reports/hand.jsonl.gz.interrupted', 'docs/new/weights.PT',
                     'private.sqlite', 'docs/events.out.tfevents.1', 'evidence.tar.zst'):
            self.assertTrue(storage.reasons(storage.Entry(path, 'a' * 40, 10)), path)
        self.assertTrue(storage.reasons(storage.Entry('docs/raw.json', 'a' * 40, storage.MAX_BYTES + 1)))

    def test_legacy_exemption_is_bound_to_path_bytes_and_blob(self):
        original = storage.Entry('docs/old.jsonl.gz', 'a' * 40, 100)
        policy = {'version': 1, 'retained': {original.path: {
            'git_blob': original.blob, 'bytes': original.size, 'reason': 'Retained evidence'}}}
        self.assertEqual(storage.violations([original], policy), [])
        for changed in (storage.Entry(original.path, 'b' * 40, 100),
                        storage.Entry(original.path, original.blob, 101),
                        storage.Entry('docs/copied.jsonl.gz', original.blob, 100)):
            self.assertEqual(len(storage.violations([changed], policy)), 1)
        # Removing a verified migrated file does not require retaining the payload.
        self.assertEqual(storage.violations([], policy), [])
        policy['retained'][original.path]['reason'] = ''
        self.assertEqual(len(storage.violations([original], policy)), 1)

    def test_staged_snapshot_ignores_working_bytes_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args, input=None):
                return subprocess.check_output(['git', '-C', str(root), *args], input=input)
            git('init', '-q')
            payload = root / 'trace.jsonl'
            payload.write_bytes(b'original\n')
            (root / 'reference.md').symlink_to('trace.jsonl')
            git('add', 'trace.jsonl', 'reference.md')
            payload.write_bytes(b'changed after staging\n')
            with patch.object(storage, 'git', git):
                indexed = {e.path: e for e in storage.entries()}
                self.assertEqual(indexed['trace.jsonl'].size, len(b'original\n'))
                self.assertEqual(indexed['reference.md'].size, len('trace.jsonl'))
                tree = git('write-tree').decode().strip()
                self.assertEqual(storage.entries(tree), storage.entries())
                with self.assertRaises(subprocess.CalledProcessError):
                    storage.entries('--help')

    def test_inventory_does_not_claim_archival_acceptance(self):
        report = storage.inventory([storage.Entry('docs/raw.gz', 'a' * 40, 300),
                                    storage.Entry('src/a.py', 'b' * 40, 100)])
        self.assertEqual(report['tracked_bytes'], 400)
        self.assertEqual(report['directories']['docs']['files'], 1)
        self.assertEqual(len(report['research_payload_candidates']), 1)
        self.assertIn('neither Drive upload nor deletion eligibility', report['scope'])
        json.dumps(report)


if __name__ == '__main__':
    unittest.main()
