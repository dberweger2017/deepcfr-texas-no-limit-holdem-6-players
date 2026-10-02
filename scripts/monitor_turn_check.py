"""Bridge owned nested solver JSONL to the existing TensorBoard sidecar."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append
from scripts.monitor_flop_check import Monitor

SKIP={'compact_aggregation','compact_metric','compact_secondary_metric','both_blueprint_ev','payoff_query'}


class Bridge:
    def __init__(self,run):
        self.run=Path(run);self.path=self.run/'bridge-offsets.json'
        self.offsets=json.loads(self.path.read_text()) if self.path.exists() else {}

    def poll(self):
        for path in sorted(self.run.glob('spots/**/progress.jsonl')):
            relative=str(path.relative_to(self.run));offset=self.offsets.get(relative,0)
            with path.open('rb') as source:
                source.seek(offset)
                while True:
                    before=source.tell();line=source.readline()
                    if not line or not line.endswith(b'\n'):
                        self.offsets[relative]=before;break
                    row=json.loads(line)
                    if row['event'] in SKIP:continue
                    base=path
                    while base.parent!=self.run and not (base/'job.json').exists():base=base.parent
                    metadata=json.loads((base/'job.json').read_text()) if (base/'job.json').exists() else None
                    if metadata:
                        row['spot']=metadata['job']['root']['spot']
                        row['series']=metadata['policy']['name']+'/'+row['spot']
                        row['set']=metadata['job']['set']
                    append(self.run/'progress.jsonl',row)
        atomic_json(self.path,self.offsets)
        status=self.run/'status.json'
        if status.exists():
            state=json.loads(status.read_text());eta=state.get('eta_seconds')
            text=(f"Stage: {state['stage']}\n\nJobs: {state['jobs_done']}/{state['jobs_total']}\n\n"
                  f"Current job: {state.get('job') or 'none'}\n\n"
                  f"ETA: {eta/3600:.2f} hours" if eta is not None else
                  f"Stage: {state['stage']}\n\nJobs: {state['jobs_done']}/{state['jobs_total']}\n\nETA: pending timings")
            text+=f"\n\nLast error: {state.get('last_error') or 'none'}\n\nHeartbeat: {datetime.now(timezone.utc).isoformat()}\n"
            target=self.run/'status.tmp';target.write_text(text);target.replace(self.run/'status.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--logdir',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(a.logdir));bridge=Bridge(a.run);monitor=Monitor(a.run,writer)
    try:
        while True:
            bridge.poll()
            if monitor.poll() or a.once:break
            time.sleep(5)
    finally:writer.close()
if __name__=='__main__':main()
