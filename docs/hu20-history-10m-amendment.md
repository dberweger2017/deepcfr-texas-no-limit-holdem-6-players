# Owner-approved history-density pivot: 10M total nodes

Approved October1 after the frozen2M result and ad hoc review. This is a
separate prospective follow-up in PR144, not a replacement result. The original
2M gate remains failed. No card/history schema or training recipe changes.

## Work and acceptance goal

Continue the same six retained2M full/compressed states to **10M total
completed nodes each** (about8M additional per state /48M additional overall).
Keep all three original seeds, completed iteration/RNG lineage, keys, regrets,
averages, visits, uncapped menu and K1. Hash each parent before loading; do not
rewrite it. The [frozen plan](../configs/blueprint/dr2x2-history-10m-followup.json)
pins each parent, completed work/iteration and the original common corpus.

Save descriptive5M/7M milestones and assess admission at10M only. Apply the
unchanged primary river requirements to the same uniform-menu corpus:
compressed median ≥max(2,2×full median), and both <10 and <100 fractions at
least5 percentage points lower. Require the aggregate and at least two of
three complete seed pairs to pass, plus correctness/recovery checks. No early
stopping or threshold relaxation based on intermediate results. A failed10M
result remains failed; no automatic further extension.

The changed goal is to assess whether the measured density improvement reaches
the original materiality threshold at a larger declared work budget. This does
not measure poker strength or prove that sparse keys cause LBR losses. Linear
growth of existing2M visits projects a6.85-point <100 improvement at10M,
versus1.65 at5M; strategy drift makes this a planning assumption, not a promise.

## Secondary policy-reach diagnostic

Before continuation, freeze two public-decision corpora per seed using its
retained2M full and compressed current policies, each against uniform native
menu play. Use512 paired deal blocks /both target positions /root202610110002,
with the same deals across reference families. Record target keys under both
schemas, actions and fallback; do not record or inspect terminal payoffs.
Report visits0/<10/<100 and median by street at2M/5M/7M/10M separately for each
reference family. These policies are fixed at2M, not selected using strength.
This describes baseline-policy reach, not final10M policy occupancy; it does
not replace the common-corpus gate or create a strength test.

Original2M checkpoints do not retain a key-to-street map, and keys are hashes.
Classify retained keys using known corpus decisions and new training visits.
Report unclassified retained keys explicitly; per-street stored-key statistics
cover classified keys only. Common-corpus and policy-reach measurements have
an exact street for every decision and include missing keys as zero. Do not
rerun the completed2M training merely to reconstruct metadata.

## Resources, recovery and next admission

Run sequentially on M1, one worker, Python3.11.14 /engine5db20e3. Keep6GiB
process RSS,8GiB disk headroom,4M entries,250k nodes/300seconds per iteration
and30minutes per worker. Bound this unpaid follow-up at two hours overall;
expected continuation time is approximately50–70minutes plus probing/saves
from the measured2M throughput. New output root uses `dr2x2-`; preserve every
failure/partial and do not blindly retry. Atomic recovery each1M total boundary;
retain5/7/10M resumable checkpoints and10M current exports. Verify deterministic
continuation/save/load in a fresh process before launching these workers.

After a pass, proceed with the already requested exact#143 D integration,
measured C/D resource prefixes and Linux parity. Publish a live quote and
conservative all-in hard cap for owner approval before any paid C/D launch.
No100M rental,300M extension, model promotion or merge is authorized here.
New Guy retains B/#143 ownership. M4/#136 source, work and ACTIVE90-minute
supervision remain unchanged; do not allocate M4 to this follow-up.
