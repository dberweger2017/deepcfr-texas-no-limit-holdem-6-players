"""Render the committed fixed-size study as a standalone SVG figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
root=Path('docs/reports/nplayer-size-smoke-artifacts')
results=json.loads((root/'size-results.json').read_text())
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'nplayer-size-20261015'})
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
colors={'n2-20bb':'#1967b3','n2-100bb':'#6c55a3','n3-20bb':'#d36b18','n3-100bb':'#168064'}
for result in results[:6]:
    name=result['name'];base,scope=name.rsplit('-scope',1)
    rows=[r for r in result['records'] if 'table_requested_bytes' in r]
    nodes=[r['nodes']/1e6 for r in rows]
    label=base.replace('n2-','HU ').replace('n3-','3-player ').replace('bb',' BB')
    if scope=='3':axes[0].plot(nodes,[r['reached_keys']/1e6 for r in rows],marker='o',color=colors[base],label=label)
    axes[1].plot(nodes,[r['table_requested_bytes']/2**20 for r in rows],marker='o',linestyle='-' if scope=='3' else '--',color=colors[base],label=label+(' full' if scope=='3' else ' preflop'))
for ax in axes:
    ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.18)
    ax.set_xlabel('Training nodes (millions)');ax.set_xticks([.1,1,2,5,10,20]);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(fontsize=8,loc='upper left')
axes[0].set(ylabel='Distinct reached keys (millions)',title='Same traversal: keys reached')
axes[1].set(ylabel='Requested table storage (MiB)',title='Retained table: averaging scope')
fig.suptitle('Fixed seed 2026101521 · K50 · v1 menu · measured endpoints only',fontsize=12)
fig.savefig(root/'size-growth.svg',metadata={'Date':None,'Creator':'Matplotlib; scripts/plot_nplayer_sizes.py'})
