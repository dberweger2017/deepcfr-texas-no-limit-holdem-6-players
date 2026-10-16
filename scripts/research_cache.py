#!/usr/bin/env python3
"""bb128's research cache: /research (= /mnt/data2tb/research-cloud), one folder per PR.

Google Drive (rclone remote `gdrive:`, the deepcfr-research-results folder) is the long-term
store. This disk is a fast copy in front of it. A PR folder is deleted locally once nothing in
it has been read or written for 7 days, it has a receipt proving Drive holds everything in it,
it isn't an open PR's folder, it has no KEEP file and no process has it open.
Right before eviction every local file is re-read and compared with its receipt. An evicted
folder first moves to /research/.evicted/ and is removed for good only 7 days later.

    research_cache.py upload PR-FOLDER FILE...   upload to gdrive:PR-FOLDER/ (one-way), then receipt
    research_cache.py receipt PR-FOLDER FILE...  record files already on Drive (verifies them)
    research_cache.py status                     every folder: last use, receipt, protection
    research_cache.py evict [--dry-run]          the daily cron job

A FILE is either a ZIP, which covers the local files at its member paths when their size and
CRC32 match, or a file inside PR-FOLDER, which covers itself. Receipts are made once, when
files are already being read. The daily check compares Drive's MD5 and local metadata,
then re-reads local content with O_NOATIME before eviction or permanent deletion.
These verification reads never count as use. Nothing here deletes or changes Drive.
"""

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

ROOT = Path('/mnt/data2tb/research-cloud')
REMOTE = 'gdrive:'
RECEIPT = '.drive-receipt.json'
LOG = ROOT / '.eviction-log.jsonl'
TRASH = ROOT / '.evicted'
IDLE_SECONDS = 7 * 86400
FULL_WARNING = 0.85
OPEN_PRS_URL = 'https://api.github.com/repos/dberweger2017/CFR-NLHE-6max/pulls?state=open&per_page=100'
IGNORED = {RECEIPT, 'KEEP'}


def now_utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(record):
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(LOG, flags, 0o600), 'a') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError('eviction log must be a regular file')
        stream.write(json.dumps({'utc': now_utc(), **record}, sort_keys=True) + '\n')


def digest(path, kind):
    """MD5 hex or CRC32. Opened with O_NOATIME, so checking a file never counts as using it."""
    h = hashlib.md5() if kind == 'md5' else None
    crc = 0
    if not hasattr(os, 'O_NOATIME'):
        raise OSError('O_NOATIME is required for cache verification')
    flags = os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError('cache verification requires a regular file')
        while chunk := stream.read(8 << 20):
            if h:
                h.update(chunk)
            else:
                crc = zlib.crc32(chunk, crc)
    return h.hexdigest() if h else crc


def drive_listing(folder):
    out = subprocess.run(['rclone', 'lsjson', '-R', '--hash', '--files-only', REMOTE + folder + '/'],
                         capture_output=True, text=True, timeout=600)
    if out.returncode:
        raise RuntimeError('rclone lsjson failed: ' + out.stderr.strip()[-300:])
    return {e['Path']: e for e in json.loads(out.stdout or '[]')}


def local_files(folder):
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in research folder: ' + path.relative_to(folder).as_posix())
        if not path.is_dir() and not path.is_file():
            raise ValueError('non-regular research entry: ' + path.relative_to(folder).as_posix())
        if path.is_file() and path.relative_to(folder).as_posix() not in IGNORED:
            yield path


def cache_folder(name):
    if not re.fullmatch(r'PR-\d+-[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('expected one PR-<number>-<name> folder')
    folder = ROOT / name
    if folder.is_symlink() or folder.resolve().parent != ROOT.resolve():
        raise ValueError('cache folder escapes the research root')
    return folder


def member_path(folder, name):
    key = PurePosixPath(name)
    if key.is_absolute() or '..' in key.parts or '\\' in name or not key.parts:
        raise ValueError('unsafe archive/receipt member')
    local = folder.joinpath(*key.parts)
    if any(folder.joinpath(*key.parts[:i]).is_symlink() for i in range(1, len(key.parts) + 1)):
        raise ValueError('symlink in archive/receipt member')
    return local


def load_receipt(folder):
    path = folder / RECEIPT
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('receipt must be a regular file')
    return json.loads(path.read_text()) if path.exists() else {'folder': folder.name, 'drive': {}, 'files': {}}


def restore_record(folder, name):
    try:
        data = load_receipt(folder)
        ids = {k: v['id'] for k, v in data['drive'].items()}
    except (OSError, ValueError, KeyError, TypeError):
        ids = {}
    return {'folder': name, 'drive': ids,
            'restore': f'within 7 days: move it back from {TRASH}; later: rclone copy '
                       f'{REMOTE}{name}/ {ROOT / name}, then unzip any ZIP there'}


def receipt(name, files):
    folder = cache_folder(name)
    if not folder.is_dir():
        raise SystemExit(f'{folder} is not a folder')
    data, listing = load_receipt(folder), drive_listing(name)
    for file in map(Path, files):
        remote = file.resolve().relative_to(folder.resolve()).as_posix() if file.resolve().is_relative_to(folder.resolve()) else file.name
        entry = listing.get(remote)
        if entry is None:
            raise SystemExit(f'not on Drive: {REMOTE}{name}/{remote}')
        md5 = digest(file, 'md5')
        if entry['Size'] != file.stat().st_size or entry.get('Hashes', {}).get('md5') != md5:
            raise SystemExit(f'Drive copy differs from {file}')
        # Re-receipting a replaced ZIP must revoke its old member coverage.
        data['files'] = {rel: item for rel, item in data['files'].items()
                         if item.get('covered_by') != remote}
        data['drive'][remote] = {'id': entry.get('ID'), 'size': entry['Size'], 'md5': md5}
        covered = []
        if zipfile.is_zipfile(file):
            with zipfile.ZipFile(file) as archive:
                names = [member_path(folder, m.filename).relative_to(folder).as_posix()
                         for m in archive.infolist() if not m.is_dir()]
                if len(names) != len(set(names)):
                    raise ValueError('duplicate archive member path')
                for member in archive.infolist():
                    local = member_path(folder, member.filename)
                    if member.is_dir() or not local.is_file():
                        continue
                    if local.stat().st_size == member.file_size and digest(local, 'crc') == member.CRC:
                        covered.append((local, {'crc32': member.CRC}))
        if file.resolve().is_relative_to(folder.resolve()):
            covered.append((file, {'md5': md5}))
        for local, content in covered:
            st = local.stat()
            data['files'][local.relative_to(folder).as_posix()] = {
                'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'covered_by': remote, **content}
        print(f'{remote}: on Drive (md5 {md5}), covers {len(covered)} local files')
    data['updated_utc'] = now_utc()
    (folder / RECEIPT).write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    uncovered = [p.relative_to(folder).as_posix() for p in local_files(folder)
                 if p.relative_to(folder).as_posix() not in data['files']]
    if uncovered:
        print(f'not yet covered by Drive ({len(uncovered)}): ' + ', '.join(uncovered[:10]))


def upload(name, files):
    folder = cache_folder(name)
    folder.mkdir(exist_ok=True)
    for file in map(Path, files):
        remote = file.resolve().relative_to(folder.resolve()).as_posix() if file.resolve().is_relative_to(folder.resolve()) else file.name
        subprocess.run(['rclone', 'copyto', '--immutable', str(file), f'{REMOTE}{name}/{remote}'], check=True)
    receipt(name, files)


def last_use(folder):
    """Latest read or write of any research file. Folder listings and the receipt don't count:
    this job reads both itself every day."""
    latest = 0
    for path in local_files(folder):
        st = path.lstat()
        latest = max(latest, st.st_atime, st.st_mtime)
    return latest


def open_prs():
    numbers = set()
    for page in range(1, 1001):
        with urllib.request.urlopen(OPEN_PRS_URL + f'&page={page}', timeout=30) as response:
            rows = json.load(response)
        if not isinstance(rows, list) or any(not isinstance(row.get('number'), int) for row in rows):
            raise ValueError('invalid GitHub open-PR list')
        numbers.update(row['number'] for row in rows)
        if len(rows) < 100:
            return numbers
    raise RuntimeError('GitHub open-PR pagination did not finish')


def process_protection(folder):
    result = subprocess.run(['lsof', '+D', str(folder)], capture_output=True)
    if result.returncode == 0:
        return 'a process has it open'
    # lsof's empty exit 1 means no matches; warnings/other exits are not proof.
    if result.returncode != 1 or result.stdout or result.stderr:
        return 'process inspection unavailable'
    return None


def why_kept(folder, open_numbers, now, idle_seconds=IDLE_SECONDS, original_name=None):
    """None if the folder may be deleted, else the reason it stays."""
    if (folder / 'KEEP').exists():
        return 'KEEP file'
    name = original_name or folder.name
    cache_folder(name)
    match = re.match(r'PR-(\d+)-', name)
    if match and int(match.group(1)) in open_numbers:
        return f'PR {match.group(1)} is open'
    idle = now - last_use(folder)
    if idle < idle_seconds:
        return f'used {idle / 86400:.1f} days ago'
    reason = process_protection(folder)
    if reason:
        return reason
    data = load_receipt(folder)
    if data['folder'] != name:
        return 'receipt belongs to a different folder'
    if not data['drive']:
        return 'no receipt'
    paths = list(local_files(folder))
    for path in paths:
        rel, st = path.relative_to(folder).as_posix(), path.stat()
        entry = data['files'].get(rel)
        if entry is None:
            return f'not covered by Drive: {rel}'
        if entry.get('covered_by') not in data['drive']:
            return f'not covered by a recorded Drive object: {rel}'
        if (entry['size'], entry['mtime_ns']) != (st.st_size, st.st_mtime_ns):
            return f'changed since its receipt: {rel}'
    listing = drive_listing(name)
    for remote, entry in data['drive'].items():
        found = listing.get(remote)
        if found is None or found['Size'] != entry['size'] or found.get('Hashes', {}).get('md5') != entry['md5']:
            return f'Drive copy missing or different: {remote}'
    for rel, entry in data['files'].items():
        path = member_path(folder, rel)
        if not path.is_file():
            continue
        if ('crc32' in entry and digest(path, 'crc') != entry['crc32']) or ('md5' in entry and digest(path, 'md5') != entry['md5']) \
                or not ({'crc32', 'md5'} & entry.keys()):
            return f'content differs from its Drive copy: {rel}'
    # Reads and writes during verification must not be mistaken for idle data.
    if list(local_files(folder)) != paths:
        return 'research files changed during verification'
    for path in paths:
        st = path.stat()
        entry = data['files'][path.relative_to(folder).as_posix()]
        if (st.st_size, st.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            return 'research files changed during verification'
    if now - last_use(folder) < idle_seconds:
        return 'research files used during verification'
    if (folder / 'KEEP').exists():
        return 'KEEP file'
    reason = process_protection(folder)
    if reason:
        return reason
    return None


def folders():
    return [p for p in sorted(ROOT.iterdir()) if p.is_dir() and not p.is_symlink() and not p.name.startswith('.')]


def evict(dry_run, idle_seconds=IDLE_SECONDS):
    now = time.time()
    try:
        open_numbers = open_prs()
    except Exception as error:  # Without the open-PR list nothing is safe to delete.
        log({'event': 'aborted', 'reason': f'open PR list unavailable: {error}'})
        return
    # Establish a writable audit trail before any rename or permanent deletion.
    log({'event': 'scan-started', 'dry_run': dry_run})
    if TRASH.is_symlink() or (TRASH.exists() and not TRASH.is_dir()):
        log({'event': 'aborted', 'reason': 'trash must be an ordinary cache directory'})
        return
    usage = shutil.disk_usage(ROOT)
    if usage.used / usage.total > FULL_WARNING:
        log({'event': 'disk-warning', 'used_fraction': round(usage.used / usage.total, 3)})
    for folder in folders():
        try:
            reason = why_kept(folder, open_numbers, now, idle_seconds)
        except Exception as error:
            reason = f'check failed: {error}'
        if reason:
            log({'event': 'kept', **restore_record(folder, folder.name), 'reason': reason})
            continue
        size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        record = {'folder': folder.name, 'bytes': size, 'last_use_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(last_use(folder))),
                  **restore_record(folder, folder.name)}
        if dry_run:
            log({'event': 'would-evict', **record})
        else:
            TRASH.mkdir(exist_ok=True)
            target = TRASH / f'{folder.name}--evicted-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
            folder.rename(target)
            log({'event': 'evicted-to-trash', 'trash': str(target), **record})
    # Trash is removed for good only after another 7 days.
    # Moving keeps a folder's old modification time, so the eviction time comes from the name.
    for held in sorted(TRASH.iterdir()) if TRASH.exists() else []:
        stamp = re.search(r'--evicted-(\d{8}T\d{6}Z)$', held.name)
        if stamp is None:
            continue
        try:
            age = now - calendar.timegm(time.strptime(stamp.group(1), '%Y%m%dT%H%M%SZ'))
        except ValueError:
            log({'event': 'kept-in-trash', 'trash': str(held), 'reason': 'invalid eviction stamp'})
            continue
        if age >= IDLE_SECONDS:
            name = held.name[:stamp.start()]
            record = restore_record(held, name) if not held.is_symlink() else {'folder': name, 'drive': {}}
            # Trash is still research data: reopening a PR/adding KEEP or using
            # a file must also block permanent deletion, not just the first move.
            try:
                if held.is_symlink() or not held.is_dir():
                    raise ValueError('trash entry must be an ordinary directory')
                reason = why_kept(held, open_numbers, now, original_name=name)
            except Exception as error:
                reason = f'check failed: {error}'
            if reason:
                log({'event': 'kept-in-trash', **record, 'trash': str(held), 'reason': reason})
                continue
            if dry_run:
                log({'event': 'would-purge', **record, 'trash': str(held)})
            else:
                shutil.rmtree(held)
                log({'event': 'purged', **record, 'trash': str(held)})


def status():
    now = time.time()
    try:
        open_numbers = open_prs()
    except Exception:
        open_numbers = set()
    for folder in folders():
        data, match = load_receipt(folder), re.match(r'PR-(\d+)-', folder.name)
        flags = ['KEEP'] * (folder / 'KEEP').exists() + ['open PR'] * bool(match and int(match.group(1)) in open_numbers)
        print(f'{folder.name:56s} idle {(now - last_use(folder)) / 86400:5.1f} d  '
              f'drive files {len(data["drive"]):3d}  covered {len(data["files"]):5d}  {" ".join(flags)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('upload', 'receipt'):
        p = sub.add_parser(command)
        p.add_argument('folder')
        p.add_argument('files', nargs='+')
    sub.add_parser('status')
    p = sub.add_parser('evict')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--idle-days', type=float, default=IDLE_SECONDS / 86400, help='for testing only')
    args = parser.parse_args()
    if args.command == 'upload':
        upload(args.folder, args.files)
    elif args.command == 'receipt':
        receipt(args.folder, args.files)
    elif args.command == 'status':
        status()
    else:
        evict(args.dry_run, args.idle_days * 86400)


if __name__ == '__main__':
    main()
