# Conditional three-player flop local CFR diagnostic

The [frozen protocol](../blueprint-local-cfr-conditional.md) selected 48
eligible first-hero-decision observations before solving: 12 deterministic
deals, each with four public flop-action patterns. On the M4, source revision
`5aa8cb89c223fc36a2a870f0e2c424d1195661ee` used the saved 12M blueprint
with SHA-256 `c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845`.
Each observation was solved with targeted traversal on and off, using the same
starting search seed and alternating condition order. These are correlated,
scripted conditional situations, not independent full hands or a BB/100
comparison.

## Results

| Measure | Targeting off | Targeting on |
| --- | ---: | ---: |
| Actual decision policy produced | 2/48 | **48/48** |
| Completed cycles in every attempt | 128 | 128 |
| Target information set absent after 128 cycles | 46/48 | 0/48 |
| Median visits to actual decision | 0 | 103.5 |
| Median distinct hero holdings visited at that public decision | 32 | 34 |
| Median public root-range prior mass of those holdings | 2.89% | 3.02% |
| Median solve time | 2.51 s | 3.24 s |
| Maximum solve time | 2.63 s | 3.63 s |
| Trained / untrained continuation lookup calls | 37,909 / 934,644 | 46,362 / 1,235,376 |

All 96 scheduled attempts completed their 128 full per-player cycles without
a timeout or unexpected search error. The run took **334.6 seconds**, under the
15-minute wall limit, and peak process RSS was **7.50 GiB**, under 10.5 GiB.
The paired median added latency with targeting was **0.72 seconds**. This is
descriptive: the two conditions start with the same seed but consume random
draws differently after their first targeted pass.

Targeting off reached the hero's actual information set in only two cases,
both with the hero first to act. The other 46 completed their cycles but
could not provide a policy for the actual decision. Targeting on provided a
policy in all 12 cases of each pattern. Its actual-decision visit count ranged
from 75 to 129; a targeted draw can contribute zero weight when compatible
opponent hands cannot complete the sampled deal or observed-path reach is
zero. The extra pass therefore solves an important *coverage of the actual
decision* problem at this 128-cycle budget. It remains a heuristic that
reweights the actual-hand stratum; this diagnostic does not establish better
poker play or an unbiased MCCFR estimator.

The public range remains sparsely explored. At the target public decision,
the targeted condition visited a median of **34 distinct hero holdings** and
**3.02% of their public root-range prior mass**. After two preceding checks,
the median prior mass was only **0.72%**; for the hero first to act it was
**10.46%**. These values omit collision-conditioned joint probabilities and
opponent-action reach. They show where regret work was recorded, not the
probability mass of all strategically relevant states solved.

The mean L1 movement of the *targeted* final action policy from cycle 64 to
128 was **0.845** on a 0–2 scale across 48 cases. It was **0.991** in the
12 raise-facing cases. The current policy was still moving appreciably; this
is not a convergence test. Only two untargeted cases produced a comparable
policy, so their mean on/off policy difference of 1.17 is too sparse to
interpret as a strategic effect.

The targeted condition found trained blueprint entries in **46,362 of
1,281,738** trained-plus-untrained continuation lookup calls (**3.62%**).
These are repeated calls, not distinct leaf states. The continuations still
apply fold/call/raise style biases when an entry is untrained, and this
coverage measurement alone does not isolate their contribution to strategy
quality. The earlier three-solve smoke had zero trained lookups; that small
observation did not generalize to this conditional set.

## Artifacts and reproducibility

The [result JSON](local-cfr-conditional-m4-result.json) and all
[96 attempt rows](local-cfr-conditional-m4-attempts.jsonl) are versioned in
this PR. The latter has SHA-256
`d494f5f658f51a8ced48b43f94c2454bfc1b2a88c82ceeb05df301d540441e0a`.
The frozen case file has SHA-256
`2eef4f80c1850608e31e5364a84096ae0f68c8fa4c145f71e70a6ac88c394fb5`.
The clean-source manifest, checksums, attempts, result and TensorBoard events
remain on the M4 under
`~/Local/local-cfr-conditional-diagnostic/results/local-cfr-conditional-m4/`;
all four listed artifact checksums verified. A same-source two-case replay
matched the first four full-run attempt rows on every deterministic field;
only latency and peak RSS were excluded from that equality check. The
48 frozen observation hashes reproduced before solving.

An earlier run of the identical cases was stopped by the operator after 20
rows because an unnecessary `gc.collect()` after every solve scanned the
12M-entry checkpoint and was on course to consume the wall budget. Its
[partial rows](local-cfr-conditional-aborted-gc-attempts.jsonl) and
[explicit stop record](local-cfr-conditional-aborted-gc-stop.json) are
versioned here and remain on the M4 in
`results/local-cfr-conditional-m4-aborted-gc/`. Revision `5aa8cb8` removed
that measurement overhead without changing the cases, search seeds, solver
settings or five-second decision cap. The stopped run is retained as a
resource failure, not included in the 48-pair result.

**Decision:** targeting is needed to produce the actual-hand policy reliably
at the present 128-cycle budget on these selected states. The resulting
conditional solves fit the M4 resource envelope, while sparse range work,
policy movement and low trained continuation coverage remain open strategic
questions. No model promotion, 58M comparison, paid compute or full-game
strength claim follows from this diagnostic.
