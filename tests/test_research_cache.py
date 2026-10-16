"""Offline cache lifecycle tests: only temporary files and fake command-line tools."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error
import zipfile
import zlib

from scripts import research_cache as cache


DIGEST = cache.digest
NOW = 2_000_000_000
DAY = 86400
FAKE_RCLONE = r'''
import hashlib,json,os,pathlib,shutil,sys
args=sys.argv[1:]
with open(os.environ['CACHE_TEST_CALLS'],'a') as f:f.write(json.dumps(args)+'\n')
if args[0] not in {'copyto','lsjson'} or any(a.startswith('--delete-') for a in args):
 raise SystemExit('forbidden rclone operation')
root=pathlib.Path(os.environ['CACHE_TEST_DRIVE'])
if args[0]=='copyto':
 assert args[1]=='--immutable'
 source=pathlib.Path(args[2]);target=root/args[3].removeprefix('gdrive:')
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and target.read_bytes()!=source.read_bytes():raise SystemExit(4)
 if not target.exists():shutil.copyfile(source,target)
else:
 folder=root/args[-1].removeprefix('gdrive:');rows=[]
 for p in sorted(folder.rglob('*')):
  if p.is_file():rows.append({'Path':p.relative_to(folder).as_posix(),'Size':p.stat().st_size,'ID':'fake-'+p.name,'Hashes':{'md5':hashlib.md5(p.read_bytes()).hexdigest()}})
 print(json.dumps(rows))
'''
FAKE_LSOF = r'''
import os,sys
mode=os.environ.get('CACHE_TEST_LSOF','closed')
if mode=='open':print('a process has this folder open');sys.exit(0)
if mode=='denied':print('lsof: permission denied',file=sys.stderr)
if mode=='error':sys.exit(2)
sys.exit(1)
'''


class Response(io.BytesIO):
    def __init__(self, rows):
        super().__init__(json.dumps(rows).encode())


class ResearchCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'cache'
        self.root.mkdir()
        self.drive = self.base / 'drive'
        self.drive.mkdir()
        self.calls = self.base / 'rclone-calls.jsonl'
        binary = self.base / 'bin'
        binary.mkdir()
        for name, body in [('rclone', FAKE_RCLONE), ('lsof', FAKE_LSOF)]:
            p = binary / name
            p.write_text('#!' + sys.executable + '\n' + body)
            p.chmod(0o755)
        self.enter(patch.dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                              CACHE_TEST_DRIVE=str(self.drive), CACHE_TEST_CALLS=str(self.calls),
                              CACHE_TEST_LSOF='closed'))
        for name, value in [('ROOT', self.root), ('TRASH', self.root / '.evicted'),
                            ('LOG', self.root / '.eviction-log.jsonl')]:
            self.enter(patch.object(cache, name, value))
        self.enter(patch.object(cache.time, 'time', return_value=NOW))
        real_gmtime = time.gmtime
        self.enter(patch.object(cache.time, 'gmtime', side_effect=lambda seconds=None:
                               real_gmtime(cache.time.time() if seconds is None else seconds)))
        self.enter(patch.object(cache.urllib.request, 'urlopen', side_effect=lambda *a, **k: Response([])))
        self.enter(contextlib.redirect_stdout(io.StringIO()))
        if not hasattr(os, 'O_NOATIME'):
            # Emulate Linux no-atime I/O for lifecycle tests on the Mac coordinator.
            # A separate test exercises the actual flag on Linux CI.
            self.enter(patch.object(cache, 'digest', side_effect=self.noatime_digest))

    def enter(self, manager):
        result = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return result

    @staticmethod
    def noatime_digest(path, kind):
        before = path.stat()
        payload = path.read_bytes()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return hashlib.md5(payload).hexdigest() if kind == 'md5' else zlib.crc32(payload)

    def seeded(self, number=900):
        folder = self.root / f'PR-{number}-offline'
        (folder / 'research').mkdir(parents=True)
        (folder / 'research/result.txt').write_bytes(b'frozen-research-result')
        (folder / 'ARCHIVE-MANIFEST.json').write_text('{"test":"manifest"}\n')
        archive = folder / 'evidence.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.write(folder / 'research/result.txt', 'research/result.txt')
            z.write(folder / 'ARCHIVE-MANIFEST.json', 'ARCHIVE-MANIFEST.json')
        for p in folder.rglob('*'):
            os.utime(p, (NOW - 9 * DAY, NOW - 9 * DAY))
        cache.upload(folder.name, [str(archive)])
        self.age_access(folder)
        return folder

    def age_access(self, folder):
        for p in folder.rglob('*'):
            st = p.stat()
            os.utime(p, ns=(int((NOW - 9 * DAY) * 1e9), st.st_mtime_ns))
        for p in [*folder.rglob('*'), folder]:
            if p.is_dir():
                os.utime(p, (NOW - 9 * DAY, NOW - 9 * DAY))

    def events(self):
        return [json.loads(x) for x in cache.LOG.read_text().splitlines()]

    def test_receipt_covers_zip_members_and_archive(self):
        folder = self.seeded()
        data = cache.load_receipt(folder)
        self.assertEqual(set(data['files']), {'research/result.txt', 'ARCHIVE-MANIFEST.json', 'evidence.zip'})
        self.assertEqual(data['files']['research/result.txt']['covered_by'], 'evidence.zip')
        self.assertIn('crc32', data['files']['research/result.txt'])
        self.assertIn('md5', data['files']['evidence.zip'])
        self.assertIsNone(cache.why_kept(folder, set(), NOW))

    def test_uncovered_file_keeps_folder(self):
        folder = self.seeded()
        p = folder / 'missing.bin'
        p.write_bytes(b'unarchived')
        os.utime(p, (NOW - 8 * DAY, NOW - 8 * DAY))
        self.age_access(folder)
        self.assertIn('not covered', cache.why_kept(folder, set(), NOW))
        cache.evict(False)
        self.assertTrue(folder.exists())

    def test_same_size_changed_content_with_preserved_mtime_keeps_folder(self):
        folder = self.seeded()
        p = folder / 'research/result.txt'
        st = p.stat()
        p.write_bytes(b'X' * st.st_size)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertIn('content differs', cache.why_kept(folder, set(), NOW))

    def test_changed_size_or_mtime_keeps_folder(self):
        folder = self.seeded()
        p = folder / 'research/result.txt'
        os.utime(p, (NOW - 8 * DAY, NOW - 8 * DAY))
        self.assertIn('changed since', cache.why_kept(folder, set(), NOW))

    def test_keep_file_protects_folder(self):
        folder = self.seeded()
        (folder / 'KEEP').touch()
        cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertEqual(self.events()[-1]['reason'], 'KEEP file')

    def test_open_pr_protects_folder(self):
        folder = self.seeded()
        with patch.object(cache.urllib.request, 'urlopen', return_value=Response([{'number': 900}])):
            cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertIn('is open', self.events()[-1]['reason'])

    def test_github_pagination_protects_pr_after_first_hundred(self):
        folder = self.seeded()
        pages = [Response([{'number': n} for n in range(100)]), Response([{'number': 900}])]
        with patch.object(cache.urllib.request, 'urlopen', side_effect=pages):
            cache.evict(False)
        self.assertTrue(folder.exists())

    def test_unavailable_github_blocks_eviction_and_purge(self):
        folder = self.seeded()
        cache.TRASH.mkdir()
        held = cache.TRASH / 'PR-901-old--evicted-20000101T000000Z'
        held.mkdir()
        with patch.object(cache.urllib.request, 'urlopen', side_effect=urllib.error.URLError('offline')):
            cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertTrue(held.exists())
        self.assertEqual(self.events()[-1]['event'], 'aborted')

    def test_lsof_open_or_inspection_error_keeps_folder(self):
        folder = self.seeded()
        for mode in ['open', 'denied', 'error']:
            with self.subTest(mode=mode), patch.dict(os.environ, CACHE_TEST_LSOF=mode):
                cache.evict(False)
                self.assertTrue(folder.exists())

    def test_missing_lsof_keeps_folder(self):
        folder = self.seeded()
        (self.base / 'bin/lsof').unlink()
        with patch.object(cache, 'process_protection', side_effect=FileNotFoundError('lsof unavailable')):
            cache.evict(False)
        self.assertTrue(folder.exists())

    def test_corrupt_receipt_keeps_folder(self):
        folder = self.seeded()
        (folder / cache.RECEIPT).write_text('not JSON')
        cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertIn('check failed', self.events()[-1]['reason'])

    def test_missing_drive_object_keeps_folder(self):
        folder = self.seeded()
        (self.drive / folder.name / 'evidence.zip').unlink()
        self.assertIn('Drive copy missing', cache.why_kept(folder, set(), NOW))

    def test_changed_drive_content_keeps_folder(self):
        folder = self.seeded()
        p = self.drive / folder.name / 'evidence.zip'
        p.write_bytes(b'X' * p.stat().st_size)
        self.assertIn('Drive copy missing or different', cache.why_kept(folder, set(), NOW))

    def test_missing_covering_object_in_receipt_keeps_folder(self):
        folder = self.seeded()
        data = cache.load_receipt(folder)
        data['files']['research/result.txt']['covered_by'] = 'absent.zip'
        (folder / cache.RECEIPT).write_text(json.dumps(data))
        self.age_access(folder)
        self.assertIsNotNone(cache.why_kept(folder, set(), NOW))

    def test_replaced_zip_revokes_old_member_coverage(self):
        folder = self.seeded()
        archive = folder / 'evidence.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.write(folder / 'ARCHIVE-MANIFEST.json', 'ARCHIVE-MANIFEST.json')
        # Model an independently replaced remote object, not an immutable upload.
        (self.drive / folder.name / archive.name).write_bytes(archive.read_bytes())
        cache.receipt(folder.name, [str(archive)])
        self.age_access(folder)
        self.assertNotIn('research/result.txt', cache.load_receipt(folder)['files'])
        self.assertIn('not covered', cache.why_kept(folder, set(), NOW))

    def test_duplicate_zip_members_cannot_prove_coverage(self):
        folder = self.root / 'PR-900-offline'
        (folder / 'research').mkdir(parents=True)
        (folder / 'research/result.txt').write_bytes(b'original')
        archive = folder / 'evidence.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('research/result.txt', b'original')
            z.writestr('research/./result.txt', b'different')
        with self.assertRaises(ValueError):
            cache.upload(folder.name, [str(archive)])
        self.assertFalse((folder / cache.RECEIPT).exists())

    def test_receipt_and_folder_listing_do_not_count_as_use(self):
        folder = self.seeded()
        before = cache.last_use(folder)
        cache.load_receipt(folder)
        list(folder.iterdir())
        os.utime(folder / cache.RECEIPT, (NOW, NOW))
        os.utime(folder, (NOW, NOW))
        self.assertEqual(cache.last_use(folder), before)

    @unittest.skipUnless(hasattr(os, 'O_NOATIME'), 'Linux kernel flag; exercised in CI')
    def test_real_noatime_digest_leaves_access_time_unchanged(self):
        p = self.root / 'probe'
        p.write_bytes(b'no-atime' * 1000)
        old = time.time_ns() - 10 * DAY * 1_000_000_000
        os.utime(p, ns=(old, old))
        for kind in ['md5', 'crc']:
            before = p.stat().st_atime_ns
            DIGEST(p, kind)
            self.assertEqual(p.stat().st_atime_ns, before)

    def test_digest_refuses_missing_noatime_support(self):
        p = self.root / 'probe'
        p.write_bytes(b'data')
        with patch.dict(os.__dict__):
            os.__dict__.pop('O_NOATIME', None)
            with self.assertRaises(OSError):
                DIGEST(p, 'md5')

    def test_symlinked_research_entry_prevents_eviction(self):
        folder = self.seeded()
        outside = self.base / 'outside'
        outside.write_bytes(b'must stay')
        (folder / 'uncovered-link').symlink_to(outside)
        self.age_access(folder)
        cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertTrue(outside.exists())

    def test_symlinked_trash_never_moves_data_outside_cache(self):
        folder = self.seeded()
        outside = self.base / 'outside-trash'
        outside.mkdir()
        cache.TRASH.symlink_to(outside, target_is_directory=True)
        cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_log_cannot_be_written_or_allow_mutation(self):
        folder = self.seeded()
        outside = self.base / 'outside-log'
        outside.write_text('preserve')
        cache.LOG.symlink_to(outside)
        with self.assertRaises(OSError):
            cache.evict(False)
        self.assertTrue(folder.exists())
        self.assertEqual(outside.read_text(), 'preserve')

    def test_recent_read_during_validation_prevents_eviction(self):
        folder = self.seeded()
        p = folder / 'research/result.txt'
        original = cache.digest
        def read_then_use(path, kind):
            result = original(path, kind)
            if path == p:
                os.utime(p, ns=(NOW * 1_000_000_000, p.stat().st_mtime_ns))
            return result
        with patch.object(cache, 'digest', side_effect=read_then_use):
            cache.evict(False)
        self.assertTrue(folder.exists())

    def test_trash_stamp_not_old_directory_mtime_controls_purge(self):
        folder = self.seeded()
        os.utime(folder, (NOW - 100 * DAY, NOW - 100 * DAY))
        cache.evict(False)
        held, = cache.TRASH.iterdir()
        self.assertFalse(folder.exists())
        cache.evict(False)
        self.assertTrue(held.exists())
        with patch.object(cache.time, 'time', return_value=NOW + 7 * DAY - 1):
            cache.evict(False)
        self.assertTrue(held.exists())
        with patch.object(cache.time, 'time', return_value=NOW + 7 * DAY):
            cache.evict(False)
        self.assertFalse(held.exists())
        event = next(x for x in self.events() if x['event'] == 'evicted-to-trash')
        self.assertIn('fake-evidence.zip', event['drive'].values())
        self.assertIn('rclone copy gdrive:PR-900-offline/', event['restore'])

    def test_keep_and_open_pr_still_protect_old_trash(self):
        folder = self.seeded()
        cache.evict(False)
        held, = cache.TRASH.iterdir()
        (held / 'KEEP').touch()
        with patch.object(cache.time, 'time', return_value=NOW + 8 * DAY):
            cache.evict(False)
        self.assertTrue(held.exists())
        (held / 'KEEP').unlink()
        with patch.object(cache.time, 'time', return_value=NOW + 8 * DAY), \
             patch.object(cache.urllib.request, 'urlopen', return_value=Response([{'number': 900}])):
            cache.evict(False)
        self.assertTrue(held.exists())

    def test_drive_loss_blocks_permanent_purge(self):
        folder = self.seeded()
        cache.evict(False)
        held, = cache.TRASH.iterdir()
        (self.drive / folder.name / 'evidence.zip').unlink()
        with patch.object(cache.time, 'time', return_value=NOW + 8 * DAY):
            cache.evict(False)
        self.assertTrue(held.exists())
        self.assertIn('Drive copy missing', self.events()[-1]['reason'])

    def test_process_open_in_trash_blocks_permanent_purge(self):
        folder = self.seeded()
        cache.evict(False)
        held, = cache.TRASH.iterdir()
        with patch.object(cache.time, 'time', return_value=NOW + 8 * DAY), \
             patch.dict(os.environ, CACHE_TEST_LSOF='open'):
            cache.evict(False)
        self.assertTrue(held.exists())

    def test_dry_run_never_moves_or_purges(self):
        folder = self.seeded()
        cache.evict(True)
        self.assertTrue(folder.exists())
        self.assertEqual(self.events()[-1]['event'], 'would-evict')

    def test_path_escape_is_refused(self):
        outside = self.base / 'outside'
        outside.mkdir()
        with self.assertRaises((ValueError, SystemExit)):
            cache.upload('../outside', [])
        self.assertFalse((outside / cache.RECEIPT).exists())

    def test_zip_member_cannot_cover_outside_cache(self):
        folder = self.root / 'PR-900-offline'
        folder.mkdir()
        outside = self.root / 'outside'
        outside.write_bytes(b'outside')
        archive = folder / 'evidence.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../outside', b'outside')
        with self.assertRaises((ValueError, SystemExit)):
            cache.upload(folder.name, [str(archive)])
        self.assertFalse((folder / cache.RECEIPT).exists())

    def test_rclone_only_uses_immutable_copy_and_readonly_listing(self):
        folder = self.seeded()
        cache.receipt(folder.name, [str(folder / 'evidence.zip')])
        cache.status()
        self.age_access(folder)
        cache.evict(False)
        with patch.object(cache.time, 'time', return_value=NOW + 8 * DAY):
            cache.evict(False)
        commands = [json.loads(x) for x in self.calls.read_text().splitlines()]
        self.assertTrue(commands)
        self.assertTrue(all(x[0] in {'copyto', 'lsjson'} for x in commands))
        self.assertTrue(all('--immutable' in x for x in commands if x[0] == 'copyto'))
        self.assertFalse(any(a.startswith('--delete-') for x in commands for a in x))


if __name__ == '__main__':
    unittest.main()
