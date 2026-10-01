"""Independent exact-name cutoff for the approved three-pod card experiment."""
import argparse
import json
import os
from pathlib import Path
import time
from scripts.mature_cpu_rental_guard import api,owned_pods,write


def validate(state):
    names=state['names']
    if (len(names)!=3 or len(set(names))!=3 or any(not n.startswith('new-guy-hu20-card-v2-') for n in names)
        or not 0<state['deadline']-state['started']<=18000
        or state['max_hourly_per_pod']>.57 or state['max_hourly_per_pod']<=0
        or state.get('prior_cost_upper_usd',0)<0
        or state.get('prior_cost_upper_usd',0)+len(names)*state['max_hourly_per_pod']*(state['deadline']-state['started'])/3600>10):
        raise ValueError('Lease differs from approved three-pod $10/5h plan')


def watch(lease,key):
    state=json.loads(lease.read_text());validate(state);names=set(state['names']);out=lease.with_name('watchdog.json')
    if owned_pods(api(key,'/v2/pods')['pods'],names):raise ValueError('Owned names existed before arming')
    while time.time()<state['deadline']:
        write(out,{'status':'armed','heartbeat':time.time(),'pid':os.getpid(),'deadline':state['deadline']})
        if lease.with_name('operator-finished.json').exists() and not owned_pods(api(key,'/v2/pods')['pods'],names):
            write(out,{'status':'operator-finished','no_owned_pods':True,'heartbeat':time.time()});return
        time.sleep(min(15,max(0,state['deadline']-time.time())))
    # Retry exact-owned teardown until the provider confirms absence. A network
    # outage cannot guarantee billing stops at the deadline; record it explicitly.
    while True:
        try:
            for pod in owned_pods(api(key,'/v2/pods')['pods'],names):api(key,'/v2/pods/'+pod['id'],'DELETE')
            if not owned_pods(api(key,'/v2/pods')['pods'],names):
                write(out,{'status':'cutoff-verified','heartbeat':time.time()});return
        except Exception as error:write(out,{'status':'cutoff-error','heartbeat':time.time(),'error_type':type(error).__name__})
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lease',type=Path,required=True);p.add_argument('--key',type=Path,required=True);a=p.parse_args();watch(a.lease,a.key)
