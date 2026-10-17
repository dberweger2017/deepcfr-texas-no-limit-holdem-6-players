"""Owner-approved remaining PR301 work after its terminal storage stop."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from scripts import recover_hu200_capacity as old
from scripts import schedule_hu200_capacity as q
from scripts.run_hu200_capacity_training import BINARY_SHA, QUALIFIED, SEEDS, pin, put
from scripts.audit_hu200_capacity_sorted import audit_sorted
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.policies.files import file_hash

BASE = q.BASE
PHASE = 'continuation-20261017'
OUTPUT = 'results-continuation-20261017'
GUARD = 'continuation-20261017'
APPROVAL = 'storage-stop-restart-approval-20261017.json'
PINS = old.PINS
FAILURE_SHA = 'cdb3827480bdff73c57b768d224773173dba45f87d128916f74c9396213551fe'
PRIOR_LEDGER_SHA = 'b7bf856475cbab6ab09b433b0e52c7f1758a41bb85fe6e5ba341a10ab4a60134'
PRIOR_DEADLINE = 1792380557.996109
STOPPED_UTC = '2026-10-17T09:21:22.269998+00:00'
STOPPED_EPOCH = dt.datetime.fromisoformat(STOPPED_UTC).timestamp()
METADATA_RESERVE = 64*1024**2
HDD = Path('/mnt/data2tb')
COLD = HDD/'research/PR-301-hu200-capacity-cygnus/inactive-models-after-20b-20261017'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def phase(base):
    return base/PHASE


def model_folder(base, seed):
    return base/(OUTPUT if seed == SEEDS[0] else old.OUTPUT)/str(seed)/'20000000000'


def audit_folder(base, seed):
    return model_folder(base, seed) if seed == SEEDS[0] else phase(base)/'audits'/str(seed)


def approval(base):
    value = q.read(PINS/APPROVAL)
    if value['status'] != 'owner-approved' or value['future_automatic_retries']:
        raise ValueError('Explicit bounded owner restart approval required')
    if file_hash(base/'training-failed.json') != FAILURE_SHA:
        raise ValueError('Historical storage failure changed')
    if q.read(base/'training-failed.json')['utc'] != STOPPED_UTC:
        raise ValueError('Stopped interval differs from recorded failure')
    if (file_hash(base/'training-clock.json') != old.CLOCK_SHA
            or file_hash(base/old.LEDGER) != PRIOR_LEDGER_SHA):
        raise ValueError('Prior immutable clock/ledger changed')
    return value


def adjusted_deadline(stopped_epoch, resumed_epoch):
    if stopped_epoch != STOPPED_EPOCH or not STOPPED_EPOCH <= resumed_epoch < PRIOR_DEADLINE:
        raise ValueError('Only the recorded stopped interval can be credited')
    return PRIOR_DEADLINE + (resumed_epoch-stopped_epoch)


def checked_ledger(base):
    approval(base)
    value = q.read(phase(base)/'deadline.json')
    if (value['prior_ledger_sha256'] != PRIOR_LEDGER_SHA
            or value['approval_sha256'] != file_hash(PINS/APPROVAL)
            or value['stopped_epoch'] != STOPPED_EPOCH
            or value['extension_seconds'] != value['resumed_epoch']-STOPPED_EPOCH
            or value['deadline_epoch'] != adjusted_deadline(STOPPED_EPOCH, value['resumed_epoch'])):
        raise ValueError('Sealed stopped-interval ledger changed')
    return value


def fresh_stat(item):
    p = Path(item['path'])
    old.intact(item, p)
    return p


def reused_inputs(base, seed):
    if seed not in SEEDS[1:]:
        raise ValueError('Only two completed20B saves may reuse exports')
    folder = model_folder(base, seed)
    record = q.read(folder/'native-record.json')
    expected = q.read(PINS/f'failure-20261017/results-recovery-20261016/{seed}/20000000000/native-record.json')
    checked = q.read(folder/'saved-bytes.json')
    if (record != expected or not checked['full_gzip_eof_crc_read']
            or checked['file']['sha256'] != record['checkpoint_sha256']
            or checked['entries'] != record['diagnostics']['entries']):
        raise ValueError('Retained native/save receipt differs')
    fresh_stat(checked['file'])
    prior = q.read(PINS/'training-failure-20261017.json')['live_stop_verification']['save_files']
    by_name = {x['name']:x for x in prior[str(folder.relative_to(base))]}
    for name in ['checkpoint.current.gz', 'checkpoint.average.gz']:
        p = folder/name
        if p.is_symlink() or (p.stat().st_size, p.stat().st_mtime_ns) != (by_name[name]['bytes'], by_name[name]['mtime_ns']):
            raise ValueError('Retained extraction changed')
    return record, checked


def validate_inputs(base, *, rehash):
    approval(base)
    admitted = phase(base)/'admission.json'
    if admitted.exists() and old.source_pin() != q.read(admitted)['source']:
        raise ValueError('Source differs from admission')
    if file_hash(QUALIFIED/'trainer-first') != BINARY_SHA:
        raise ValueError('Qualified trainer changed')
    admission = q.read(base/'admission.json')
    for street, sha in admission['card_table_hashes'].items():
        files = [p for p in (QUALIFIED/'tables').glob(street+'*') if p.is_file()]
        if len(files) != 1 or file_hash(files[0]) != sha:
            raise ValueError('K50 table changed')
    parents = {}
    for seed in SEEDS:
        folder = base/old.OUTPUT/str(seed)/'10000000000'
        audit = q.checked_audit(folder)
        if audit != q.read(PINS/f'save-10b/{seed}/audit.json'):
            raise ValueError('Audited10B parent differs from committed receipt')
        item = audit['files']['checkpoint'];fresh_stat(item)
        if rehash and file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('Retained10B parent bytes differ')
        parents[str(seed)] = item
    for name in ['resume-gates.json','twin-gates.json','prefix-memory-admission.json','twins-restored.json']:
        if file_hash(base/name) != file_hash(PINS/name):
            raise ValueError('Adopted original gate changed')
    storage = q.read(PINS/'inactive-model-storage-20261017.json')['manifest']
    mount_checked()
    for row in storage['verified']:
        p = Path(row['destination'])
        if (p.stat().st_size, p.stat().st_mtime_ns) != (row['destination_bytes'],row['source_identity']['mtime_ns']):
            raise ValueError('Relocated model/evidence metadata changed')
    for seed in SEEDS[1:]:reused_inputs(base,seed)
    return parents


def sample(base):
    vrm, fan = q.motherboard_sensors()
    r = q.sensors.sample(q.sensors.package_sensor())
    r.update(vrm_c=q.sensors.read_int(vrm)/1000,fan1_rpm=q.sensors.read_int(fan),free_disk_bytes=shutil.disk_usage(base).free)
    return r


def admission_check(base, row):
    required = 301153954*130+1384031032+8*q.GIB
    if (row['package_c'] >= 68 or row['vrm_c'] >= 83 or row['available_bytes'] < required
            or row['swap_used_bytes']-1048576 > 256*1024**2
            or not q.fits(q.owned_bytes(base),row['free_disk_bytes'],0,20_000_000_000+2*METADATA_RESERVE)):
        raise ValueError('Fresh single-trainer/two-stream-audit resource admission failed')
    return required


def prepare(base):
    if phase(base).exists() or (base/'guards'/GUARD).exists() or (base/OUTPUT).exists():
        raise ValueError('Continuation already prepared/attempted; no automatic retry')
    parents = validate_inputs(base,rehash=True)
    r=sample(base);required=admission_check(base,r)
    put(phase(base)/'admission.json',dict(status='admitted',source=old.source_pin(),parents=parents,
        resources=r,memory_required_bytes=required,owner_approval_sha256=file_hash(PINS/APPROVAL),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),utc=now(),later_automatic_retries=False))


def validate_existing_exports(folder, receipt_folder, *, expected_schema):
    record = q.read(folder/'native-record.json');checked=q.read(folder/'saved-bytes.json')
    checkpoint=fresh_stat(checked['file'])
    current, average=folder/'checkpoint.current.gz',folder/'checkpoint.average.gz'
    files=dict(checkpoint=pin(checkpoint),current=pin(current),average=pin(average))
    if files['checkpoint']['sha256'] != record['checkpoint_sha256']:
        raise ValueError('Retained checkpoint hash differs')
    put(receipt_folder/'adopted-inputs.json',dict(files=files,old_saved_byte_receipt=pin(folder/'saved-bytes.json'),export_invoked=False))
    spec=dict(seed=checked['header']['config']['seed'],iteration=record['iteration'],
        checkpoint_sha256=files['checkpoint']['sha256'],sha256=files['current']['sha256'])
    result=audit_sorted(checkpoint,current,average,spec,files['average']['sha256'],expected_schema=expected_schema)
    if result['all_nodes_verified'] != record['diagnostics']['entries']:
        raise ValueError('Final every-row audit incomplete')
    for item in files.values():fresh_stat(item)
    receipt=dict(status='verified',native=record,audit=result,files=files)
    put(receipt_folder/'audit.json',receipt)
    # The old failed attempt never published this file; append successful proof
    # without replacing any original stop, input, export or failure bytes.
    put(folder/'audit.json',receipt)
    return receipt


def mount_checked():
    fs=json.loads(subprocess.check_output(['findmnt','-J','-T',str(HDD)],text=True))['filesystems']
    if len(fs)!=1 or fs[0]['target']!=str(HDD) or fs[0]['source']!='/dev/sda1' or fs[0]['fstype']!='ext4' or 'rw' not in fs[0]['options'].split(','):
        raise ValueError('Expected mounted2TB drive unavailable')
    for p in [COLD,*COLD.parents]:
        if p==HDD.parent:break
        if p.is_symlink():raise ValueError('Refuse symlink storage destination')


def sync_dir(p):
    fd=os.open(p,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def relocate_one(src,dst,expected,stop,journal):
    mount_checked();fresh_stat(expected)
    if src != Path(expected['path']) or src.stat().st_nlink != 1 or dst.exists():
        raise ValueError('Unexpected linked/source/destination storage state')
    identity=(src.stat().st_dev,src.stat().st_ino)
    dst.parent.mkdir(parents=True,exist_ok=True);temporary=dst.with_name(dst.name+'.copying')
    def check():
        if stop.exists():raise RuntimeError('Storage worker stop requested')
        mount_checked()
        if shutil.disk_usage(HDD).free < 50_000_000_000:raise RuntimeError('HDD free floor')
    h=hashlib.sha256();tick=0
    with src.open('rb') as reader,temporary.open('xb') as writer:
        for block in iter(lambda:reader.read(4*1024*1024),b''):
            writer.write(block);h.update(block)
            if time.monotonic()-tick>1:check();tick=time.monotonic()
        writer.flush();os.fsync(writer.fileno())
    if h.hexdigest()!=expected['sha256']:raise ValueError('Storage source SHA mismatch')
    os.utime(temporary,ns=(src.stat().st_atime_ns,expected['mtime_ns']))
    h=hashlib.sha256()
    with temporary.open('rb') as reader:
        for block in iter(lambda:reader.read(4*1024*1024),b''):
            h.update(block)
            if time.monotonic()-tick>1:check();tick=time.monotonic()
    if temporary.stat().st_size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:
        raise ValueError('Independent HDD readback differs')
    check();fresh_stat(expected)
    if (src.stat().st_dev,src.stat().st_ino)!=identity:raise ValueError('Source inode changed')
    os.rename(temporary,dst);sync_dir(dst.parent)
    row=dict(source=expected,destination=str(dst),sha256=h.hexdigest(),verified_utc=now())
    with journal.open('a') as f:f.write(json.dumps(dict(**row,state='verified-before-unlink'))+'\n');f.flush();os.fsync(f.fileno())
    check();fresh_stat(expected);src.unlink();sync_dir(src.parent)
    with journal.open('a') as f:f.write(json.dumps(dict(**row,state='relocated-source-unlinked'))+'\n');f.flush();os.fsync(f.fileno())
    return row


def offload(base,seed):
    folder=model_folder(base,seed);audit=q.checked_audit(folder)
    parent=q.checked_audit(base/old.OUTPUT/str(seed)/'10000000000')
    fresh_stat(audit['files']['checkpoint']);fresh_stat(audit['files']['average'])
    out=phase(base)/'offloads'/str(seed);items=[]
    for item in [parent['files']['checkpoint'],audit['files']['current']]:
        src=Path(item['path']);dst=COLD/src.relative_to(base)
        items.append(relocate_one(src,dst,item,out/'retire.stop',out/'journal.jsonl'))
    put(out/'receipt.json',dict(status='verified',relocated=items,retained_checkpoint=audit['files']['checkpoint'],utc=now()))


class Continuation(q.Queue):
    def __init__(self,base):
        self.base=base;self.source=old.source_pin();approval(base)
        admitted=q.read(phase(base)/'admission.json')
        if (admitted['source']!=self.source or admitted['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                or (base/'guards'/GUARD).exists() or (base/OUTPUT).exists() or (phase(base)/'deadline.json').exists()):
            raise ValueError('Source/boot/attempt identity differs; no automatic retry')
        validate_inputs(base,rehash=False);r=sample(base);admission_check(base,r)
        resumed=time.time();self.deadline=adjusted_deadline(STOPPED_EPOCH,resumed)
        put(phase(base)/'deadline.json',dict(status='owner-approved-storage-stop-credit',stopped_epoch=STOPPED_EPOCH,
            resumed_epoch=resumed,extension_seconds=resumed-STOPPED_EPOCH,deadline_epoch=self.deadline,
            original_clock_sha256=old.CLOCK_SHA,prior_ledger_sha256=PRIOR_LEDGER_SHA,
            approval_sha256=file_hash(PINS/APPROVAL),restart_admitted_utc=now()))
        self.root=base/'guards'/GUARD;self.root.mkdir(exist_ok=False)
        (base/OUTPUT).mkdir(exist_ok=False)
        self.clock=q.read(base/'training-clock.json');self.admission=q.read(base/'admission.json')
        self.active,self.done,self.tasks={},set(),{}
        self.audit_limit,self.batch_decided=3,True;self.batch,self.batch_started=[],None
        self.next_counter,self.last_counter=0,None;self.cpu_deltas,self.disk_busy_ms,self.last_disk={},0,None
        self.resource_summary=dict(peak_cpu_c=0,peak_vrm_c=0,peak_helper_rss_bytes=0,minimum_available_bytes=2**63,minimum_free_disk_bytes=2**63)
        self.build()

    def build(self):
        for seed in SEEDS:
            train=f'train-{seed}-20000000000';audit=f'audit-{seed}-20000000000'
            if seed==SEEDS[0]:self.task(train,'train',[],seed=seed,label='20000000000')
            self.task(audit,'audit',[train] if seed==SEEDS[0] else [],seed=seed,label='20000000000')
            self.task(f'offload-{seed}-20000000000','retire',[audit],seed=seed,label='20000000000')

    def folder(self,task):
        seed=task['seed']
        if task['kind']=='retire':return phase(self.base)/'offloads'/str(seed)
        return audit_folder(self.base,seed) if task['kind']=='audit' else model_folder(self.base,seed)

    def output_reservation(self,task):
        if task['kind']=='retire' or (task['kind']=='audit' and task['seed'] in SEEDS[1:]):return METADATA_RESERVE
        return super().output_reservation(task)

    def command(self,task):
        seed=task['seed'];folder=self.folder(task)
        if task['kind']=='retire' or (task['kind']=='audit' and seed in SEEDS[1:]):
            folder.mkdir(parents=True,exist_ok=False)
            return [sys.executable,'-m','scripts.continue_hu200_capacity','offload' if task['kind']=='retire' else 'audit-existing','--base',str(self.base),'--seed',str(seed)]
        if task['kind']=='audit':return [sys.executable,'-m','scripts.run_hu200_capacity_training','audit','--base',str(folder)]
        if folder.exists():raise ValueError('Refuse repeated native attempt')
        parent=q.checked_audit(self.base/old.OUTPUT/str(seed)/'10000000000')['files']['checkpoint'];fresh_stat(parent)
        folder.mkdir(parents=True,exist_ok=False)
        return [str(QUALIFIED/'trainer-first'),'train','--stack-bb','200','--menu','v1','--seed',str(seed),
            '--roots-per-seat','1','--average-rule','opponent-sampled','--card-buckets',str(QUALIFIED/'tables'),
            '--max-entries','301153954','--nodes','20000000000','--recovery',
            '--max-seconds',str(max(1,self.deadline-time.time()-3600)),'--out',str(folder/'checkpoint.gz'),
            '--telemetry',str(folder/'telemetry.jsonl'),'--stop-file',str(folder/'stop.request'),
            '--freeze-at-cap','--resume',parent['path'],'--resume-sha256',parent['sha256']]

    def finish(self,name):
        job=self.active[name];task=self.tasks[name]
        if job['proc'].poll()!=0:raise RuntimeError(f'{name} failed; no automatic retry')
        if task['kind']=='train':old.validate_native(self.folder(task),task['seed'],20000000000)
        elif task['kind']=='audit':q.checked_audit(self.folder(task))
        elif q.read(self.folder(task)/'receipt.json')['status']!='verified':raise ValueError('Storage receipt incomplete')
        put(self.root/(name+'.result.json'),dict(status='passed',exit_codes=[0],elapsed_seconds=time.monotonic()-job['started'],stop_reason=None,hard_stop_reason=None,cpus=job['cpus']))
        job['log'].close();del self.active[name];self.done.add(name)
        print(json.dumps(dict(event='passed',job=name,utc=now())),flush=True)

    def decide_batch(self,row,totals):pass

    def publish(self):
        path=self.base/'milestone-20000000000.json'
        if not path.exists() and all(f'audit-{s}-20000000000' in self.done for s in SEEDS):
            put(path,{s:q.read(model_folder(self.base,s)/'native-record.json') for s in SEEDS})

    def complete(self):
        result=dict(status='all-three-20B-audited',terminal=q.read(self.base/'milestone-20000000000.json'),
            finished_utc=now(),scheduling_source=self.source,deadline_amendment=checked_ledger(self.base),
            superseded_failures=[pin(self.base/'training-failed.json')],
            storage_receipts=[pin(phase(self.base)/'offloads'/str(s)/'receipt.json') for s in SEEDS])
        put(phase(self.base)/'training-complete.json',result);put(self.base/'training-complete.json',result)


def observe(base):
    ledger=checked_ledger(base);sent=phase(base)/'event-delivery.json'
    url=(base/'campaign-events-url.txt').read_text().strip();target=urllib.parse.urlsplit(url)
    if target.scheme!='https' or target.hostname!='relay.t3.codes' or not target.path.startswith('/v1/hooks/'):
        raise ValueError('Unexpected existing event endpoint')
    while time.time()<ledger['deadline_epoch']+3600:
        event='training_failed' if (phase(base)/'training-failed.json').exists() else 'training_complete' if (phase(base)/'training-complete.json').exists() else None
        if event:
            if sent.exists():return
            payload=dict(event=event,experiment='PR301',attempt=PHASE,observed_utc=now())
            request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'T3Code-HU200Capacity-Observer/1.0'},method='POST')
            try:
                with urllib.request.urlopen(request,timeout=10) as response:
                    if 200<=response.status<300:
                        receipt=dict(payload=payload,http_status=response.status);put(sent,receipt);print(json.dumps(receipt),flush=True);return
            except Exception as exc:print(json.dumps(dict(delivery='pending',error_type=type(exc).__name__)),flush=True)
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','run','audit-existing','offload','observe'])
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args();base=args.base
    if os.uname().nodename!='cygnus-x1' or base.resolve()!=BASE:raise ValueError('Only owned bb128 PR301 root')
    if args.phase=='prepare':prepare(base)
    elif args.phase=='audit-existing':
        reused_inputs(base,args.seed);validate_existing_exports(model_folder(base,args.seed),audit_folder(base,args.seed),expected_schema=HU200_EQUITY_SCHEMA)
    elif args.phase=='offload':offload(base,args.seed)
    elif args.phase=='observe':observe(base)
    else:
        with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:Continuation(base).run()
            except BaseException as exc:
                path=phase(base)/'training-failed.json'
                if not path.exists():put(path,dict(status='failed',error=repr(exc),utc=now(),automatic_retry=False,attempt=PHASE))
                raise


if __name__=='__main__':main()
