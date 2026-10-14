# HU200 frozen-table playing comparison

Owner-authorized October 13, 2026. Does continuing #250's HU200 K50
752,593,175-node lineage on the same 85,373,907-key table improve play?
Only cumulative training changes: HU200, cards, menu v1, seed 2026101302,
R=1, linear CFR and opponent-sampled lifetime averaging stay fixed.
No merge, release or default change is authorized by this campaign.

## Admission and fixed clock

Prepare source and commands before compute. The next window is **October 14,
2026, Europe/Madrid (UTC+02:00)**. T3 resumes this thread at 01:00 and checks
every 15 minutes until admission or 05:00. An ETA does not admit compute.
Require an explicit affirmative M4 release/free-for-compute statement by
#265's agent in its current description or comments, no live #265 campaign,
export, audit, retention or closeout process, and free locks. Record the live
GitHub body/comments and process identities. Never signal #265 processes.
Probe `~/Local/.hu20-m4-research.lock` and #265's
`results/hu200-k50-frozen/supervisor.lock` with nonblocking `flock`; acquire
the shared lock for this campaign before starting. No start at or after 05:00.

All compute runs on M4, detached in a dedicated tmux session. No worker,
sampler, time bound or archive depends on an M1 SSH connection. The fixed
deadline is **10:00 Madrid /08:00 UTC**; final play/replay/reproduction must
fit by **09:30 Madrid /07:30 UTC**, leaving 30 minutes for closeout. Stop and
preserve incomplete work on a guard, integrity or deadline failure; no failed
hand rerun, outcome-dependent extension or late start.

## Input pins and commands

Baseline: #250 terminal average SHA256
`60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c`,
2,221,370,011 bytes, 752,593,175 actual nodes. Restore from its accepted
average-only archive/index into fresh ignored unsynced storage if needed.
Candidates: #265's fully audited 5B and final averages; take actual nodes,
full hashes, paths, checkpoint parents and archive locators from its updated
report/index. Use its actual final if below 20B, explicitly labeling it.
Verify the audit covers every retained row, source checkpoint parent, header,
file size/full SHA256, fixed key count and lineage; do not infer actual nodes
from target filenames. #163's flop/turn/river K50 tables must match the full
hashes in #265 and the accepted #250/#163 index. Keep each spec's own schema
and `card_buckets` directory. Uniform missing-key and zero-mass behavior stays
unchanged. Do not modify the source inputs or their existing caches.

Use merged #263's `scripts.evaluate_hu100_direct` execution/audit functions
and its seven-opponent panel, with #259's verified policy cache at an explicit
unsynced `results/hu200-frozen-evaluation/policy-cache`. First loads build
caches; later loads verify and map them. Run each contrast in a separate
process, serially, so at most two distinct model caches are mapped at once.
Warm-cache times cannot substitute for measuring initial cache construction.

The checked command shapes are:

```sh
python -m scripts.evaluate_hu100_direct --stack-bb 200 \
  --specs planning/final-752m.json --policy-cache results/policy-cache \
  --blocks 1024 --root PILOT_ROOT --rung frozen-lineage \
  --source SOURCE_SHA --out results/pilot-direct --pilot
python -m scripts.evaluate_hu200_panel \
  --specs planning/final-752m.json --policy-cache results/policy-cache \
  --blocks 64 --root PANEL_PILOT_ROOT --source SOURCE_SHA \
  --out results/pilot-panel --pilot
```

The campaign worker uses these same library functions but delays all final
means and intervals until **all** final executions have independently replayed
and exactly reproduced. The standard CLI emits its report before reproduction
and therefore is only used directly for cost pilots here.

## Canonical plan skeleton and reduction

Requested final counts are primary **262,144** direct duplicate blocks
(final minus 752M); learning curve **131,072** duplicate blocks (5B minus
752M); panel **16,384** blocks per opponent (final minus 752M against each
script). Direct blocks have two seat-swapped hands; panel blocks have four
hands, pairing both model arms on shared deals. Every primary hand is also
independently replayed and exactly reproduced.

Cost pilots: 1,024 direct blocks for final minus 752M, 64 blocks per script
for final minus 752M. Expose only time, memory and paired SDs to sizing, not
means, intervals, directions or pilot coverage. Keep raw pilots archived,
separate from final data. Measure fixed setup plus play/replay/reproduction,
use a 1.25 planning multiplier and the remaining fixed clock. Cache construction
has already consumed clock before sizing. Build all three caches serially
without playing hands before the pilot. Use the measured primary direct cost
as the declared planning proxy for the 5B pair. Include trace/archive disk headroom.

Before any final outcome, select once along this ordered ladder:

1. Requested counts: 262,144 /131,072 /16,384.
2. Learning curve reduced to 32,768, then dropped if still too costly.
3. Primary reduced to 131,072 if still too costly.
4. Panel reduced to 8,192 per script if still too costly.

If the minimum configuration cannot fit, stop without final play. No other
count reduction, scientific retry or budget extension. Post the full canonical
JSON and its SHA256 in **this PR description**, read it back exactly, then
start final hands. The plan pins actual models/hashes/parents/tables/audits,
source, cache path, roots, counts, pilot times and SDs, projection ladder,
deadline, decision rule and guards. PR comments are reserved for issues, each
with SOMA; no delegation or process diary on GitHub.

Fresh root skeleton: direct pilot 20261014091501, panel pilot 20261014091502,
final direct 20261014091503, final panel 20261014091504. Reserve against
current main and every other open PR before pilots. Both direct final pairs
intentionally share the final direct root and rung, so their overlapping
blocks use the same deals/action streams. The descriptive final-minus-5B
learning-curve contrast is the paired difference of their returns against the
fixed 752M opponent on those overlapping blocks; it is **not** a direct
final-versus-5B duel. If the curve is dropped, this comparison is unavailable.

## Readout and decision

Nominal paired Student-t 95% intervals over independent blocks. Positive
direct values favor the first model; positive panel differences favor final
over 752M against that script. Recommend Slumbot confirmation only when the
primary lower bound is strictly above zero **and** no panel difference has
an upper bound strictly below **−20 BB/100**. Report every other contrast
descriptively, including the paired learning curve. Do not select a different
candidate based on outcomes. This conditions on one training lineage; no
unrestricted convergence, multi-seed or external-strength claim.

Scripts: `random`, `check_call`, `tight_aggressive`, `loose_aggressive`,
`pot_pressure` (1.5× pot), `half_pot_pressure` and
`matched_pot_pressure` (1× pot), exactly #263/#236's styles. Report absolute
arm scores as well as paired differences. For every model and street report
missing-key/zero-mass decision shares and known reached-key visit bands
0, 1–9, 10–99, 100+, following #260; exclude missing keys from known-key
band denominators and also retain original decision-weighted counts.

## Guards, optional handoff and evidence

Sample every five seconds and preserve every reading. RSS >13 GiB, critical
pressure, free memory <8% or swap growth >4 GB stops only after 60 continuous
observed seconds. Disk <16 GiB stops immediately. Reset sustained timers after
observation gaps; warning pressure is allowed. Use one campaign swap baseline.
Check PID creation times before any signal; signal only this campaign's family.
Guard stops or scientific failures prohibit later final work.

If and only if the decision rule passes, prepare on db-ocean the hash-verified
final average under `~/deepcfr-slumbot/inputs/model-frozen/`, build its cache
with `~/deepcfr-slumbot/venv` and checkout `~/deepcfr-slumbot/repo`, and write
a 5,000-hand launcher matching `~/deepcfr-slumbot/run-200m.sh`, a fresh seed,
and account `deepcfr-k50-frozen-20b` (or actual-final-node name). Measure and
report model backing/residency and host RAM/swap with the existing 752M player;
two mapped models may exceed free RAM during network-bound play. Never start
the launcher, create an account, handle credentials, touch existing
`slumbot-*` tmux sessions or write in `~/deepcfr-slumbot/runs`. The owner
creates the named account and starts it.

Seal evidence with an embedded member manifest and full local readback into
`~/Local/Research-Cloud/PR-N-hu200-frozen-evaluation/`; index source, raw traces,
pilots, guards, failures, plan/readback, input retrieval provenance, archive
hash/bytes and separate native/independent cloud acceptance. Do not delete or
evict synced files. Report outcomes, coverage, rule, guards and evidence under
`docs/reports/hu200-frozen-evaluation.md`, update ROADMAP, commit staged storage
checks and relevant verification, and leave a green-CI PR to main for the
owner's merge decision.
