"""Durable block transport: verify every arrival before exact local retirement.

This module is also a standalone, standard-library-only storage verifier on the
file server. It never runs policies or deletes remote files.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sqlite3
import stat
import subprocess
import time

MANIFEST = 'transfer-manifest.json'
ARRIVAL = 'arrival-receipt.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def pin(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1 << 20):
            h.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing: '+str(path))
    return dict(bytes=after.st_size, sha256=h.hexdigest())


def durable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(canonical(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def inventory(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Regular block directory required')
    result = {}
    for path in directory.rglob('*'):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('Unexpected nonregular evidence: '+str(path))
        if stat.S_ISREG(mode):
            result[path.relative_to(directory).as_posix()] = path
    return result


def member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(p in ('.','..') for p in path.parts) or not path.parts:
        raise ValueError('Unsafe evidence member')
    return path


def seal(directory, identity):
    path = directory/MANIFEST
    if path.exists():
        value = json.loads(path.read_text())
        if value['identity'] != identity:
            raise ValueError('Foreign sealed block')
        verify(directory, pin(path)['sha256'])
        return value
    files = inventory(directory)
    if ARRIVAL in files or MANIFEST in files:
        raise ValueError('Unexpected storage metadata in source')
    value = dict(version=1, identity=identity, files={name:pin(p) for name,p in sorted(files.items())})
    durable(path, value)
    return value


def verify(directory, manifest_sha256):
    files = inventory(directory)
    if MANIFEST not in files or pin(files[MANIFEST])['sha256'] != manifest_sha256:
        raise ValueError('Transfer manifest checksum differs')
    value = json.loads(files[MANIFEST].read_text())
    expected = value['files']
    if set(files)-{MANIFEST,ARRIVAL} != set(expected):
        raise ValueError('Evidence inventory differs')
    for name,wanted in expected.items():
        member(name)
        if name in (MANIFEST,ARRIVAL) or pin(files[name]) != wanted:
            raise ValueError('Evidence checksum/size differs: '+name)
    return value


def receive(directory, manifest_sha256):
    value = verify(directory, manifest_sha256)
    # Arrival proof must survive server restart before the source is retired.
    for path in inventory(directory).values():
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    for path in [*sorted((p for p in directory.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True),directory]:
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    result = dict(status='every-file-size-sha256-verified', manifest_sha256=manifest_sha256,
        manifest_bytes=(directory/MANIFEST).stat().st_size, identity=value['identity'],
        files=len(value['files'])+1, bytes=sum(v['bytes'] for v in value['files'].values())+(directory/MANIFEST).stat().st_size,
        at=time.time(), verifier_sha256=pin(Path(__file__))['sha256'])
    path = directory/ARRIVAL
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('status','manifest_sha256','identity','files','bytes'):
            if old[key] != result[key]: raise ValueError('Existing arrival receipt differs')
        return old
    durable(path, result)
    return result


def pack(value): return gzip.compress(canonical(value).encode(), compresslevel=1, mtime=0)
def unpack(value): return json.loads(gzip.decompress(value))


class Journal:
    """One local writer, compressed full receipts, durable before each retirement."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS blocks (panel TEXT, number INTEGER, phase TEXT, manifest BLOB, arrival BLOB, retirement BLOB, summary BLOB, PRIMARY KEY(panel,number))')
        self.connection.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at REAL, panel TEXT, number INTEGER, detail BLOB)')
        self.connection.commit()

    def get(self, panel, number):
        row = self.connection.execute('SELECT phase,manifest,arrival,retirement,summary FROM blocks WHERE panel=? AND number=?',(panel,number)).fetchone()
        if row is None: return None
        return dict(zip(('phase','manifest','arrival','retirement','summary'),[row[0],*[unpack(v) if v is not None else None for v in row[1:]]]))

    def prepare(self, identity, manifest, summary):
        self.connection.execute('INSERT INTO blocks VALUES (?,?,?,?,?,?,?)',
            (identity['panel'],identity['block'],'prepared',pack(manifest),None,None,pack(summary)))
        self.connection.commit()

    def update(self, panel, number, phase, **values):
        if set(values)-{'arrival','retirement'}:raise ValueError('Unexpected journal field')
        fields=['phase=?']+[k+'=?' for k in values]
        args=[phase]+[pack(v) for v in values.values()]+[panel,number]
        self.connection.execute('UPDATE blocks SET '+','.join(fields)+' WHERE panel=? AND number=?',args)
        self.connection.commit()

    def event(self, panel, number, detail):
        self.connection.execute('INSERT INTO events(at,panel,number,detail) VALUES (?,?,?,?)',(time.time(),panel,number,pack(detail)))
        self.connection.commit()

    def checkpoint(self):self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')


def retire(directory, manifest, arrival, journal, *, after_unlink=lambda name:None):
    """Never use recursive deletion: extra or changed files remain untouched."""
    identity=manifest['identity'];panel=identity['panel'];number=identity['block']
    sha=hashlib.sha256((canonical(manifest)+'\n').encode()).hexdigest()
    if arrival['status']!='every-file-size-sha256-verified' or arrival['manifest_sha256']!=sha or arrival['identity']!=identity:
        raise ValueError('Retirement lacks matching verified arrival')
    state=journal.get(panel,number)
    if state is None or state['manifest']!=manifest or state['phase'] not in ('verified','retiring'):
        raise ValueError('Retirement lacks durable prepared/verified evidence')
    already_started=state['phase']=='retiring'
    if not already_started:
        verify(directory,sha)
        journal.update(panel,number,'retiring',arrival=arrival)
    expected=dict(manifest['files']);expected[MANIFEST]=dict(bytes=len((canonical(manifest)+'\n').encode()),sha256=sha)
    files=inventory(directory) if directory.exists() else {}
    if set(files)-set(expected):raise ValueError('Unverified files added before retirement')
    # Check all remaining files before deleting any on this invocation.
    for name,path in files.items():
        if pin(path)!=expected[name]:raise ValueError('Local evidence changed before retirement: '+name)
    missing=set(expected)-set(files)
    if missing and not already_started:raise ValueError('Local evidence disappeared before retirement')
    for name in sorted(files,key=lambda n:n==MANIFEST):
        path=files[name]
        if pin(path)!=expected[name]:raise ValueError('Local evidence changed during retirement')
        path.unlink()
        after_unlink(name)
    if directory.exists():
        for path in sorted(directory.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if path.is_dir() and not path.is_symlink():path.rmdir()
        directory.rmdir()
    receipt=dict(status='verified-owned-block-retired',manifest_sha256=sha,
        retired_files=sorted(expected),already_absent_after_verified_interruption=sorted(missing),
        files=len(expected),bytes=sum(v['bytes'] for v in expected.values()),at=time.time())
    journal.update(panel,number,'retired',arrival=arrival,retirement=receipt)
    return receipt


class CableTransfer:
    def __init__(self, host, remote_root, verifier, *, socket='/tmp/pr317-final-%C', expected_code_sha=None):
        self.host=host;self.root=remote_root.rstrip('/');self.verifier=verifier;self.expected_code_sha=expected_code_sha
        self.ssh=['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+socket]

    def arrival(self, relative, sha):
        remote=self.root+'/'+relative
        command=['nice','-n','19','ionice','-c','3','python3',self.verifier,'--receive',remote,'--manifest-sha256',sha]
        result=subprocess.run([*self.ssh,self.host,shlex.join(command)],text=True,capture_output=True,timeout=600,check=True)
        value=json.loads(result.stdout)
        if self.expected_code_sha is not None and value['verifier_sha256']!=self.expected_code_sha:
            raise ValueError('Remote storage verifier source differs')
        return value

    def copy(self, directory, relative):
        remote=self.root+'/'+relative
        subprocess.run([*self.ssh,self.host,shlex.join(['mkdir','-p',remote])],capture_output=True,timeout=30,check=True)
        subprocess.run(['rsync','-a','--ignore-existing','-e',shlex.join(self.ssh),str(directory)+'/',self.host+':'+remote+'/'],capture_output=True,timeout=600,check=True)

    def transfer(self, directory, identity, summary, journal):
        panel=identity['panel'];number=identity['block'];relative=f'blocks/{panel}/{number}'
        state=journal.get(panel,number)
        if state is None:
            manifest=seal(directory,identity);journal.prepare(identity,manifest,summary)
            state=journal.get(panel,number)
        manifest=state['manifest'];sha=hashlib.sha256((canonical(manifest)+'\n').encode()).hexdigest()
        if state['phase']=='retired':return state['retirement']
        journal.event(panel,number,dict(operation='transfer-or-reverify',relative=relative,manifest_sha256=sha,phase=state['phase']))
        if state['phase'] in ('prepared','verified'):
            self.copy(directory,relative)
        arrival=self.arrival(relative,sha)
        if state['phase']!='retiring':journal.update(panel,number,'verified',arrival=arrival)
        result=retire(directory,manifest,arrival,journal)
        journal.event(panel,number,dict(operation='retired',relative=relative,files=result['files']))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receive',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    allowed=Path('/research/PR-317-HU200-river-search').resolve()
    if not args.receive.resolve().is_relative_to(allowed) or args.receive.is_symlink():
        raise ValueError('Receiver outside authorized PR317 storage')
    print(canonical(receive(args.receive,args.manifest_sha256)),flush=True)
