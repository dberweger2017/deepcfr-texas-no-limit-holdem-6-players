# HU200 preparation and bounded M1 feasibility

This task adds a distinct 200BB game for v0.5.5/Slumbot. v0.5.0 stays HU100.
No release, live match, paid compute or 1B/10B campaign is authorized here.
Current-main base: `6e18043`. Independent correctness review blocks pilot training.
The M4 independent-seed work and all other research roots remain untouched.

Fresh seed **2026100905**, linear CFR, one root/seat, opponent-sampled averaging,
v1 abstraction and unchanged min/pot/conditional-jam menu; uniform zero/missing
fallback, translation off. Start HU200 fresh; only this pilot's HU200 checkpoints
may resume. Provisionally retain **20M and 100M** endpoints, plus a **1M admission
measurement**. Save at the first complete iteration at/above each target, with
full current/average exports and independent accumulator audit before continuation.
A clean capacity stop is a valid partial and its actual node count is authoritative.

One 16GiB Apple M1 worker. Check current process ownership, acquire exclusive
`/tmp/deepcfr-m1-research.lock`, and preserve the ownership/admission receipt.
Require AC, normal pressure, >=15% memory-pressure free, >=4GiB psutil available
at initial admission; 3GiB soft /4GiB hard entire owned-family RSS, <=512MiB swap
growth from the original baseline and <=3,000,000,000 bytes total swap. Retain
15.5GiB disk floor. These limits are chosen for this M1, not inherited M4 quotes.
Sample the whole family and host approximately every half second; timestamps,
sampling gaps and kernel command peaks bound the claims.

The single **60-minute monotonic cap starts at first pilot admission** and includes
training, saving, export/audit, optional smoke, local archive/readback and closeout.
Reserve 600s for local closeout plus 600s for a final save/tool set. Each next
endpoint requires a measured 2x-time quote, `110 bytes/entry +100MB` forecast,
1GiB additional available-memory margin, and 400 bytes/forecast entry plus 1GiB
working/archive disk budget above the floor. Entries are conservatively scaled linearly with nodes and capped by the soft
memory capacity, even if that stops short of the provisional node endpoint;
no direct HU100-to-HU200 capacity claim. Resource/time breach latches science off;
never raise a limit, retry training, or infer successful auditing from file presence.
Hard breach terminates only the owned family, retaining accepted checkpoints and
partials. Soft/time request lets native finish an iteration and atomic save.

Optional fixed smoke: terminal/latest audited average, five unchanged scripted
opponents (`random`, `check_call`, `tight_aggressive`, `loose_aggressive`,
`pot_pressure`), **32 swapped-seat blocks/opponent =320 hands**, reset HU200,
root family `20261009050000 + opponent_index*1000 + block`. Each fixed policy gets
an independent private action stream; no checkpoint selection or pooling. Inspect
all actions/settlements with an independent replay and deterministically reproduce
selected actions. Report descriptive 95% block intervals, coverage and visits.
This is feasibility evidence, not a strength gate. Omit if <780s remain.

Before archival check this owning PR's live status. This user explicitly authorizes
archiving this PR's own open evidence; no other open roots may be moved/copied.
Archive models, raw resources, failures, source/runtime and manifests in its actual
Research-Cloud folder; verify member sizes/hashes locally, separately confirm
native upload status and independent Drive metadata, index exact restoration.
Retain originals. Network upload/PR review are administrative, with local pilot
closeout complete inside the compute cap; disclose pending upload rather than
claiming acceptance. Source/tests/review preparation are outside pilot computation.
