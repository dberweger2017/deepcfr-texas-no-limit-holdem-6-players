# PR #107: conditional local CFR compute scaling

This extends the [48-case conditional diagnostic](blueprint-local-cfr-conditional.md).
It measures whether more search work reaches the actual first flop decision
and the public range. It does not measure whole-game playing strength.

## Frozen inputs

- The same 48 eligible observations in `configs/blueprint/local-cfr-conditional-cases.json`.
- The saved 12M-entry checkpoint with SHA-256
  `c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845`.
- `configs/blueprint/local-cfr-scaling-m4.json` fixes five independent
  traversal seeds per case, derived from its namespace, case ID and repetition.
  Targeting on/off share each repetition's seed; their random streams can
  diverge after targeting consumes draws. Repetitions are independent seeds,
  not independent poker deals.
- `configs/blueprint/local-cfr-scaling-ranges.json.gz` records the public
  root-range samples and weights. Each case has one distinct range seed,
  separate from every traversal seed. Both modes and all five repetitions use
  the identical case range. The run verifies this file and the source-case
  SHA-256 before loading the checkpoint.

The solver's regret updates, continuation styles, target traversal, final
iteration policy and first-hero-decision eligibility remain unchanged. Only
the range RNG separation, optional resource guard and diagnostics are added.

## Budget and output

Run sequentially on the M4. Load the checkpoint once per process. There are
48 cases × 5 repetitions × 2 targeting modes = 480 planned attempts. Each
attempt is limited to 60 seconds, 4,096 complete cycles and 2,000,000 sampled
nodes. The process stops before starting a solve that cannot fit within the
10-hour wall budget. Its peak RSS must remain below 10.5 GiB, and free disk
must remain at least 30 GiB. No paid host or new blueprint training is used.

Before the overnight run, execute a distinct four-attempt preflight on the
first two frozen cases, both modes, one repetition, with a 20-second decision
limit and 2,048-cycle limit. Verify that a solve passes 128 cycles, the
5/15-second time snapshots appear, ranges load, outputs persist and resource
guards work. A preflight result is never pooled with the overnight result.

Every attempt is appended and fsynced to `attempts.jsonl`, including errors,
timeouts and partial completed cycles. The manifest records code revision,
input hashes and environment. TensorBoard is written every ten attempts.
Results retain a checksum listing. The final report should show status and
cycle distribution by mode; target visits; target holdings and public-prior
mass visited; information-set work; policy movement at 128/256/512/1024 and
across five repetitions; continuation trained-lookup fraction; time
snapshots around 5/15/30 seconds and the last fully completed cycle near
60 seconds; and wall/RSS/disk use. A deadline can interrupt an incomplete
cycle, so the last snapshot is explicitly a completed-cycle state rather
than a fabricated 60-second update.

The run is a conditional solver diagnostic. These situations were selected
for eligibility and cannot support a BB/100 or playing-strength claim.
