# PR197 HU100 playing baseline

The owner separately authorizes this bounded evaluation before merge. Existing
training, recovery and capacity evidence stays unchanged. No additional training,
checkpoint selection, release gate, merge or general-strength claim follows.

## Frozen inputs and comparison

Use only the fully audited **11,042,440-node average**, iteration 7,722,
3,255,387 entries, **79,195,090 bytes**, SHA256
`ffd53decdd4af5bffc0ae34e98144d43e27eef92a49033a7d4008615576a93be`.
[Indexed model and restoration](reports/native-recovery-hu100-artifacts/followup-model-index.json).
Its accepted science archive is Drive `1kMXJIUUB6YkYphhHSKsRp3Xacz_Oxno2`,
member `research/followup-01/final-capacity-audit/average-11042440.jsonl.gz`.
Retained M4 bytes are rehashed by the existing average loader; the evaluation
snapshots them once. No new extraction or model choice occurs.

Heads-up, fixed **10,000/10,000-chip stacks reset every hand**, blinds **50/100**,
chip unit `0.01`, no rake. Opponents are the unchanged registry implementations:
`random`, `check_call`, `tight_aggressive`, `loose_aggressive`, `pot_pressure`.
HU100 fixture tests and the separately seeded pilot verify legality/completeness.
Random chooses legal action kinds uniformly; a raise splits its probability
equally between distinct minimum-raise and maximum/all-in targets, collapsing
them if equal. It is **not** uniform over the trained native menu and is never
projected into that menu. The three styles keep their original hand scores,
thresholds, two private draws and clamped pot-fraction raises.

Compare the average and `native_hu100_uniform`: uniform probabilities over the
**same uncapped native HU100 choices**, free-fold disabled, min/pot/deduplicated
jam rules unchanged. Deep-stack speculative open jams remain absent from this
reference menu. Both use the existing weighted `Random.choices` sampler.

The existing arena registry, schedule, runner and block reporter are reused;
a small adapter fixes each opponent panel. Each duplicate block reuses the same
physical deal across two target seat rotations and both arms: **four hands**.
Action streams are independently hash-derived per root/opponent/block/rotation/
arm/logical-player. Policies receive only their entitled immutable observations,
never seeds or schedule. The original runner's default streams remain unchanged.
Manifests pin source/environment, exact model and opponent implementation hashes;
saved schedules and explicit private-action-seed coordinates pin pairing.

## Pilot, final count and budget

[Configuration](../configs/arena/hu100-playing-baseline-v1.json) pins pilot root
`2026100819711`, final root `2026100819712`, and **16 pilot blocks/opponent**.
Inspect pilot cost and completeness only; keep its outcomes excluded from final
inference. Proposed final count is **2,048 blocks/opponent /40,960 total hands**.
Before final play, freeze a hashed quote/count receipt from measured model loads,
play, independent replay and full deterministic reproduction. Use a 2× measured
cost allowance plus fixed closeout reserve; reduce counts if necessary before
seeing final outcomes. No outcome-driven extension, retry or checkpoint selection.
The quote reserves 180 seconds for closeout, doubles fixed model-loading/snapshot
cost and measured play/replay/reproduction cost, and freezes a common count in
multiples of 32 (at most 2,048). If fewer than 32 blocks/opponent fit, retain the
pilot and report no final baseline. The source-qualified launcher uses one fresh
exclusive root and the existing `hu20_scaling_supervise` guard; an uncertain or
failed root cannot be relaunched. All heavy stages execute sequentially in the
guard's owned session, with fresh headroom/swap/disk/AC/deadline admission.

Free isolated M4 only, one evaluation worker at a time. A **fresh 1,800-second
execution cap** starts before pilot model loading and includes pilot, final model
loading/play, replay and reproduction. The previous morning deadline neither
extends nor blocks this new authorization. Continuous external guards cover the
entire owned family: **10-GiB RSS**, normal system memory pressure/full admission
headroom, original **448.81-MiB swap baseline /≤0.5-GiB growth**, **≥15.5-GiB disk**
and AC. Warning/critical/unknown pressure or free percentage below 15 hard-stops.
Never stop another job. Stop on correctness/resource failure, preserve partials,
and mark interrupted output/incomplete panels honestly. Old timers stay disabled.

## Evidence and interpretation

Report each arm's BB/100 and paired average-minus-uniform differences per opponent.
Student-t 95% intervals use independent duplicate-block means; seat rotations
and arms are not independent samples. Intervals are unadjusted for five opponents;
one small training seed and scripted opponents support only this development
baseline. Losing or inconclusive results do not block a correct engineering PR.

Decision telemetry records positive-mass known-key, zero-mass and missing-key
rates by arm/street/opponent, action frequencies and actual chooser latency.
For the uniform arm, coverage means lookup exposure relative to the fixed trained
model, **not** use of trained probabilities. Opponent lookup coverage is inapplicable.
Telemetry runs after the original action draw, consumes no RNG, and is checked
against an uninstrumented sampler's exact actions and RNG state. Timing excludes
the subsequent telemetry lookup; total wall cost includes all instrumentation.

The independent auditor replays **all final actions and settlements**, verifies
menus/keys/streams and recounts rates, and independently recomputes block arithmetic.
A fresh, sequential full reproduction verifies every hand and deterministic
decision trace, including lookup classifications and probabilities; only measured
latency is excluded from equality. Raw inputs, traces, failures, resource samples,
manifests and restoration provenance go to the existing PR197 Research-Cloud folder.
Obtain independent review and green final-head CI, then hand back **without merge**.
