"""Three guarded workers on one Mac; the caller supplies a frozen model assignment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import shutil
import signal
import subprocess
import threading
import time

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True);p.add_argument('--python',required=True)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--policies',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--models',nargs='+',required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
plan=json.loads(a.plan.read_text()); processes={}; stop=threading.Event()
def worker(name):
 if stop.is_set():raise RuntimeError("Launcher stopped")
 log=a.out/(name+'.launch.log')
 if log.exists():raise FileExistsError('Preserve earlier launches')
 with log.open('x') as f:
  child=subprocess.Popen([a.python,'-m','scripts.evaluate_hu20_v041_arena','play','--plan',str(a.plan),'--policies',str(a.policies),'--model',name,'--out',str(a.out),'--rss-limit-gib','6'],cwd=a.repo,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  processes[name]=child
  if stop.is_set():os.killpg(child.pid,signal.SIGTERM)
  code=child.wait()
  if code:raise RuntimeError(f'{name} failed with {code}; see {log}')
  return name
pool=ThreadPoolExecutor(max_workers=3)
futures=[pool.submit(worker,name) for name in a.models]
try:
 while not all(f.done() for f in futures):
  if shutil.disk_usage(a.out).free<15*1024**3 or time.time()>plan['started_at']+plan['max_seconds']:
   raise RuntimeError('Arena disk/deadline guard')
  for f in futures:
   if f.done():f.result()
  status={'time':time.time(),'running':{n:x.pid for n,x in list(processes.items()) if x.poll() is None},'completed':[f.result() for f in futures if f.done()]}
  (a.out/'launcher-status.json').write_text(json.dumps(status,indent=2)+'\n')
  time.sleep(5)
 for f in futures:f.result()
 (a.out/'launcher-complete.json').write_text(json.dumps({'models':a.models,'completed_at':time.time()})+'\n')
finally:
 stop.set()
 pool.shutdown(wait=False,cancel_futures=True)
 for child in list(processes.values()):
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 pool.shutdown(wait=True,cancel_futures=True)
