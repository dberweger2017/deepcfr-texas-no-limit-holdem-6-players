"""Exact owner-comment-bound opt-in envelope; original campaign limits stay intact."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

APPROVAL_URL = 'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197#issuecomment-6054725121'
APPROVAL_SHA = '7184f0049aaac9beb416e2425698050dacb1af015ba3876ec224303b86b0c18e'


def envelope(approval=None):
    if approval is None:
        return {'rss_gib':5.5,'serialization_gib':1.5,'system_memory_guard':False}
    a=json.loads(Path(approval).read_text())
    if (a.get('approval_url')!=APPROVAL_URL or a.get('author')!='dberweger2017'
        or a.get('author_association')!='OWNER'
        or hashlib.sha256(a.get('body','').encode()).hexdigest()!=APPROVAL_SHA
        or a.get('hard_deadline')!='2026-10-08T10:00:00Z'
        or (a.get('rss_gib'),a.get('swap_gib'),a.get('disk_gib'),a.get('require_ac'),a.get('verification_attempts'))!=(10,.5,15.5,True,1)):
        raise ValueError('Exact owner follow-up approval required')
    return {'rss_gib':10,'serialization_gib':8,'system_memory_guard':True}


def memory_snapshot():
    level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True,timeout=10))
    raw=subprocess.check_output(['memory_pressure'],text=True,timeout=10)
    m=re.search(r'System-wide memory free percentage:\s*(\d+)%',raw)
    if m is None: raise ValueError('Unreadable system memory pressure')
    physical=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True,timeout=10))
    return {'pressure_level':level,'free_percent':int(m[1]),'physical_bytes':physical,'raw':raw}


def unsafe_memory(sample, admission=False, rss_gib=10):
    if sample['physical_bytes']!=16*1024**3 or sample['pressure_level']!=1 or sample['free_percent']<15:
        return True
    # At admission reserve the entire family ceiling plus two GiB for the system.
    return admission and sample['physical_bytes']*sample['free_percent']/100 < (rss_gib+2)*1024**3


def family_rss(processes, roots):
    owned=set(roots)
    while True:
        added={pid for pid,parent,_ in processes if parent in owned}-owned
        if not added: break
        owned.update(added)
    return [size*1024 for pid,_,size in processes if pid in owned]
