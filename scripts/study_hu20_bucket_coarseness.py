"""Run the frozen model-free bucket sample once on M1."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from src.diagnostics.bucket_coarseness import ROOT, dashboard, sample_rows, summarize
from src.diagnostics.saved_hu20 import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=perf_counter();rows=[]
    with (a.out/'holdings.jsonl').open('w') as f:
        for row in sample_rows():
            rows.append(row);f.write(json.dumps(row,sort_keys=True)+'\n')
            if row['holding_index']==15:f.flush()
    result=summarize(rows)
    result.update(root=ROOT,boards_per_street=64,holdings_per_board=16,chance_samples=512,
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        schema_source_sha256=file_hash('src/blueprint/abstraction.py'),
        ranker_source_sha256=file_hash('src/diagnostics/exact_ranker.py'),
        seconds=perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        total_worlds=sum(r['worlds'] for r in rows))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (a.out/'dashboard.md').write_text(dashboard(result))
    (a.out/'manifest.json').write_text(json.dumps({f.name:{'bytes':f.stat().st_size,'sha256':file_hash(f)} for f in a.out.iterdir()},indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('street_totals','total_worlds','seconds','peak_rss_bytes')}))


if __name__=='__main__':main()
