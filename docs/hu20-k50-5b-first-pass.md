# HU20 K50 first pass at 5B nodes

Owner-authorized short first pass, replacing the proposed 28B ladder for this task.
Question: at equal training node work, is K50 stronger than the v1 blueprint in direct play?
This is evidence for a next research decision, not adoption or release authorization.

## Admission and source

No campaign preparation preceded PR236's merge. The initial ETA wake at 02:02 Madrid
was superseded by the owner's 06:00 wake, then 30-minute admission checks. At 06:30 on
October 12, `gh pr view 236` reported MERGED at 04:29:05 UTC; M4 process enumeration
with `psutil.process_iter` found no campaign processes, with no access-denied entries.
Only this thread's admission timer was disabled. Other PR processes, work roots and
schedules remain protected. All computation is on the 10-core, 16 GiB Apple M4;
M1 is only the lightweight coordinator.

Base is current main `045c793a4abd2e371d4d77de663fd3dd2f00316b`, in isolated
`~/Local/hu20-k50-5b-first-pass-20261012`, branch `feature/hu20-k50-5b-first-pass`.
Pin the built binary and source before use. Native trainer source stays unchanged.
The CLI's supported node-limit spelling is `--nodes`, not `--max-nodes`;
`--nodes 5000000000` means the requested complete-iteration 5B stop.
The saved configuration's historical max_nodes field remains 1B; command, telemetry,
actual nodes and complete iterations establish the actual terminal budget.

## Fixed treatment, controls and gates

Sequential seeds **2026100601, 2026100602, 2026100603**, one trainer at a time.
Fresh HU20 20BB training uses the production linear CFR / opponent-sampled-average
recipe: one root per seat, unchanged ordered history and menus, no regret floor,
`--average-rule opponent-sampled --nodes 5000000000 --max-entries 1000000000`.
Add only `--card-buckets results/hu20-k50-5b/tables` to the corresponding v1 recipe.
Do not use recoverable checkpoint formatting for a fresh treatment unless the
byte-identity gate proves it belongs to the reference recipe.
Export terminal current and average together with `export --zero-mass uniform`.
Schema: `hu20-native-reopening-ordered-history-equity-k50-v1`.

Restore #163's indexed TAR into an ignored nonsynced local directory. Before use,
verify its whole hash, manifest and all three required members:

| Table | SHA256 |
| --- | --- |
| flop-k50.bin | `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111` |
| turn-k50.bin | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |
| river-k50.bin | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |

Live PR163/182/185 checks at admission confirm MERGED. Required v1 5B checkpoints
are restored from the indexed milestone archives, verifying whole/manifest/member
hashes before use. Full member hashes:

- seed601, #182: `2af7f04e094cc5a8975c22c230677ee65d4effff2b7dca3cfc7387b3bae14d0f`;
- seed602, #185: `14651b481fb44c968f1039f69710344973f5b54a0a75fe5395f4ee319afd85ad`;
- seed603, #185: `bb08472a95b11e2dc5c06565d59f6e04df0d4dc082d0d839528ee495c769d050`.

Use verified averages of those exact checkpoints if already archived; otherwise
export current-main uniform averages and record hashes. Secondary reference is
v0.4.2's `O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz`, SHA256
`15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`.
Every model also records source checkpoint, bytes, iteration, seed, schema and
explicit external table path. Research binaries stay outside Git.

Dependent main work requires all applicable gates:

1. Current-main binary reproduces an archived v1 opponent-sampled checkpoint
   byte for byte. Seed601 1B reference is `4ee91d3977d98c0b6b462ed310396c6a7cbba48ef51716bb0fdbf6609dfbc825`.
   Read and reuse its max_entries; header writes max(requested, actual entries).
2. A fresh short K50 run equals a resume from its own midpoint, every row with
   exact IEEE numbers, allowing only documented native recovery metadata.
   Both checkpoint identities record the three table hashes.
3. Each terminal checkpoint passes the existing complete native export auditor
   against both its current and average, with the K50 schema explicitly supplied.

No mismatch, failed hand, guard failure or scientific attempt is retried.

## Counts, proposed roots and decision

Proposed reservations, to be confirmed against configs, reports, manifests and
readable retained plans before freeze: cost-only pilot **20261012051101**,
primary **20261012051102**, secondary **20261012051103**. If used, reserve fresh
replacement roots before seeing any outcomes. Never use 28B proposal roots
20261012023501–05 or run that protocol.

Each contrast has three equally weighted seed pairs, **32,768 duplicate blocks per
seed**, **196,608 hands per contrast**. Primary: K50 5B minus same-seed v1 5B.
Secondary: each K50 5B versus the common v0.4.2 reference, descriptive.
Use the existing HU20 direct harness, explicit abstraction/card_buckets fields,
each policy's own observation/schema, independent policy RNGs and rotated seats.
Average both seats and the three seeds equally inside each shared independent
block. Report aggregate, each seed, each seat and seed-by-seat nominal paired
Student-t 95% intervals, conditional on these trained policies. No population
training-seed uncertainty or exploitability claim.

Freeze canonical counts, roots, input hashes and decision rule on the PR before
any final play. No outcome-driven extension, checkpoint/seed selection or reruns.
Primary aggregate decision, BB/100:

- lower bound >0: K50 stronger at 5B; recommend the longer protocol;
- upper bound <0: K50 weaker at 5B; recommend stopping full-game K50;
- otherwise: inconclusive; report without extending.

Owner decides next. No LBR, native pressure or scripted panels in this pass.

## Measurement, cost and guards

For all six 5B checkpoints: actual nodes, iterations, keys, mean traverser visits
per key, zero-visit fraction, and per-street min/p10/p25/median/p75/p90/p99/max.
K50 is expected to have about 1.4 times the keys and fewer visits at equal nodes;
disclose measured density, do not adjust training work.

Expected training is about 36 minutes per seed, about 2 hours total; provisional
all-work estimate 3–4 hours. **Six-hour aggregate ceiling**, no paid compute.
Cost-only pilot: at most 2,048 blocks per seed per contrast; inspect only costs,
completeness, memory and paired SDs. If projected primary aggregate half-width
exceeds 3 BB/100 or total projection exceeds six hours, stop for a revised owner
quote. Never weaken gates or resize outcomes to fit. Account for setup, gates,
restoration, training, export, full audit, play, replay and archive work; report
actual resource measurements and forecast assumptions.

Continuously enforce whole-family RSS <7 GiB, normal pressure, free memory >=15%,
fixed-baseline swap growth <=512 MiB, free disk >=16 GiB, and AC power. Sensors
failing closed stop the campaign. Enumerate processes with psutil and never call
memory_info on unrelated processes; only identified owned descendants may be
stopped. Preserve all failures and partials. Budget and swap baseline remain fixed
across operations; this task's guard writes unique operation receipts and refuses
terminal failures or duplicate stage names.

Before each dependent stage, check projected temporary and retained disk use in
addition to the live floor. Initial M4 snapshot: normal pressure, 89% free memory,
AC, 1,347,818,618 baseline swap bytes and 23,628,259,328 free disk bytes (22.01 GiB).
This live floor pass alone does not admit all retained outputs plus archive/scratch.
Scientific gates and training have not started. The initial conservative storage
forecast requires 35.72 GiB free, rounded to a 38 GiB entry target; see the
[capacity receipt](reports/hu20-k50-5b-first-pass-artifacts/storage-admission.json).
Until that target is available, keep only this task's 30-minute resource-admission
schedule active. This is a preventive refusal, not a guard failure. The six-hour
guarded clock has not started. After admission, replace the waiting timer with
checks of already-running owned work; never launch duplicate stages.

## Evidence and completion

Archive every action/decision, public event digest and deal seed. Independently
replay every final hand and verify legal bounds, public observations, event digests,
chip settlement/accounting and raw arithmetic before numerical readout. Keep all
partials/failures/provenance. Existing direct harness/auditor supply the core paths;
any adaptation must be tested and reviewed before dependent use.

Create a member-hashed ZIP in the actual Research-Cloud PR folder, verify every
member and whole hash, then record separate native uploaded/not-uploading/no-conflicts
and independent cloud ID/name/size/parent acceptance. RESULTS_INDEX includes Drive
IDs, exact input archive members/hashes/retrieval commands and source/binary pins.
Never delete inside synced folders, force offload or touch other PR work roots.
Any own-original cleanup requires current owning PR state, accepted upload and
unchanged path/size/mtime plus dependency review under AGENTS.md.

Final report: `docs/reports/hu20-k50-5b-first-pass.md`. Update ROADMAP Current position
when this work lands. Commit/push completed subtasks; stage intended files and run
`python -m scripts.check_repository_artifacts` before every commit. Review findings
and all green final checks precede merge; no bypass, release/default/tag/publication.

## Owner storage amendment

The owner answered the preventive storage quote: **“on it, you can start though.”**
Start individually admitted stages while storage is provisioned; the 38 GiB
full-campaign entry target is no longer a blanket hold. Each stage still reserves
its inputs, retained outputs and scratch above the unchanged 16 GiB floor. All
other guards, gates, samples and the six-hour aggregate limit remain binding.
No deletion, offloading, paid compute, guard relaxation or extra training recipe
is authorized by this amendment. Preserve the earlier refusal as provenance.
