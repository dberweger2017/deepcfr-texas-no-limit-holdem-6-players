# Three-player flop local CFR: M4 pilot result

The [frozen protocol](../blueprint-local-cfr.md) ran against the saved 12M-entry
blueprint on the M4. The final code completed **97/97 eligible local solves**,
each with 128 full per-player cycles, and all **2,304 paired arena hands**
without an invalid action. Peak process RSS was **7.27 GiB** under the 10.5-GiB
guard; maximum solver time was **4.39 seconds** under the five-second decision
limit. The predeclared correctness and feasibility gate passed.

The experiment did **not** establish better poker play. Local CFR minus
corrected rollout search was **+41.57 BB/100 [95% CI −90.40, +173.54]** against
the scripted pool and **+29.04 [−19.68, +77.75]** against random opponents.
Both intervals include zero. Absolute scripted profit for the local-CFR arm
was **−252.67 BB/100 [−532.30, +26.96]**. No player is promoted and v0.5
remains unqualified.

## Paired results and resource use

| Frozen schedule | Blocks / hands | Local CFR BB/100 | Corrected search BB/100 | Paired difference BB/100 | Eligible solves | Search p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scripted styles | 128 / 1,536 | −252.67 [−532.30, +26.96] | −294.24 [−574.54, −13.93] | +41.57 [−90.40, +173.54] | 78/78 | 3.82 / 4.39 s |
| Random | 64 / 768 | +513.67 [−239.15, +1,266.50] | +484.64 [−271.70, +1,240.97] | +29.04 [−19.68, +77.75] | 19/19 | 3.98 / 3.99 s |

All completed solves used the configured 128 cycles. The run recorded about
2.19 million sampled traversal nodes, 867,712 leaf choices, and 1,088 public
range updates across both schedules. The candidate delegated other decisions
to corrected search: 542 scripted and 105 random search attempts, with zero
delegated fallback. The 12M blueprint remained sparse in simulated
continuations: only 37,122 of 1,981,662 scripted continuation lookups used a
trained entry (about 1.9%); the rest used its uniform abstract fallback.

These intervals use independent seat-rotation blocks, not individual hands.
The local solver changed only 97 candidate decisions, so this pilot has
limited power to measure a full-game strength effect. Multiplayer CFR also
has no two-player zero-sum equilibrium guarantee. This pilot samples 96
holdings per seat and adds a targeted current-path traversal to reach the
hero's observed information set; those approximations preclude a Pluribus
fidelity or exploitability claim.

## Correction during the pilot

The first frozen-code run completed 97 of 98 eligible attempts. Its sole
fallback had already passed the **second flop raise**, the declared depth
limit, so it had no current action node to play. A diagnostic-only replay
reproduced both hand files byte for byte and identified that reason. The code
then delegated such decisions to corrected search, as required by the depth
limit. The repaired run on the **same** frozen schedules completed 97/97; its
scripted paired effect changed from +42.81 [−91.20, +176.81] to +41.57
[−90.40, +173.54]. A subsequent information-set review canonicalized the
order of private cards in leaf keys; it changed no hand outcome on these
schedules. Both fixes happened after the first result was inspected.
Treat the final paired intervals as exploratory descriptions, not fresh
confirmatory evidence. The prior runs and their artifacts are retained on
the M4.

## Verification and artifacts

The final M4 run and a second run at source revision
`999d5320659a19d493b812a30fa5ace60a4b2c7d` produced byte-identical
`styles-hands.jsonl` and `random-hands.jsonl` files. Their SHA-256 values are
`88c58f6e2ce5c5bed86a21a67fefabdf564360aab817e9b0963e7416771ff3f7`
and `a42a6c11d72ef02600117d3b319b795f9ce67cb426a8aa948851423859db1378`.
Every saved artifact matches its checksum manifest, every compact hand row
matches its own digest, and both schedules contain exactly the declared
blocks, arms, and six seat rotations. Final and replay checksum-manifest
SHA-256 values and the rates, telemetry, source and checkpoint hashes, and
artifact locations are in the [compact JSON result](blueprint-local-cfr-m4.json).

Raw hand rows, per-attempt solver records, manifests, and TensorBoard events
remain in `~/Local/blueprint-local-cfr/results/blueprint-local-cfr-m4-canonical-20260925/`
and its `-canonical-replay-20260925` companion on the M4. Hash-verified copies
are under the ignored `results/` directory of the local PR worktree. The
initial and diagnostic runs are retained on the M4 in their respective
`blueprint-local-cfr-m4-20260925` and `-replay-20260925` directories.

The focused local solver/search suite passed 20 tests. The complete M4 suite
passed 781 tests at the initial code revision, and PR CI passed on the final
solver code. One unrelated local-Mac full-suite test hit its existing
free-disk guard when that machine had about 4.9 GiB available; it passed on
the M4. The final run produced TensorBoard updates every eight completed
blocks or five minutes at the next block. No paid compute was used.

**Decision:** the Python implementation is feasible for this narrow M4
subgame. The sparse blueprint and inconclusive strength comparison do not
justify a 58M or paid-host campaign yet. The next investigation should
measure the targeted traversal's approximation error and the low trained
continuation coverage before spending on a larger blueprint or native solver.
