# River CFR reference and M4 resource preflight

The short preflight completed on the M4 using source `b361b80e495371138855ee211c381dad463711a7`
and the unchanged 12M-entry blueprint with SHA-256
`c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845`.
Its [frozen plan](../../configs/blueprint/river-preflight-m4.json),
[summary JSON](river-quality-m4.json), and
[raw manifest, rows, result, and checksums](river-quality-m4/) retain the
inputs, native-engine binary hash, environment and every case. All three
checksummed raw files passed SHA-256 verification after copying from the M4.
No attempt failed
or disappeared. No checkpoint was trained or changed.

| Measure | Result |
| --- | ---: |
| Reference cases | 12 two-player, four tiny three-player; 16/16 completed |
| Full-range case | `hu-dry`, 1,081 holdings per seat, 1,070,190 compatible joint deals |
| Checkpoint load, once | 57.91 s |
| Public-range / tree and payoff setup | 0.220 / 0.272 s |
| Full-range solve | 694 complete sweeps in 29.51 s; stopped at 30 s decision deadline |
| Total decision time, including setup | 30.0004 s |
| Total preflight wall time | 118.96 s, below 600 s limit |
| Peak process RSS | 6.50 GiB, below 10.5 GiB limit |
| System swap before / after | 777.38 / 777.38 MiB used; no measured increase |
| System free-memory percentage before / after | 78% / 59% |

The 30-second deadline includes range and game construction. The full-range
case uses one 53-node public tree and the declared
`active-marginals-ignore-folded-removal-v1` range approximation. Its
own-reach-weighted **average** profile has exploitability 0.01670 BB, or
0.008351 of the root pot. The final **played** iterate has exploitability
0.73280 BB, or 0.366401 of the root pot. Both are exact best-response values
*within this restricted river game and stipulated joint law*. The 969,288
zero-external-reach entry encounters are counted separately; the final
average profile has zero zero-denominator entries.

Across the 12 small two-player cases, median average-profile exploitability
in root-pot units falls from 0.6820 after one sweep to 0.0170 at 256,
0.00517 at 1,024, and 0.00245 at 4,096. At 4,096, the maximum is 0.00613
and only **5/12** cases are at or below the *proposed* `1e-3` threshold.
The development suite therefore does not pass that proposed all-case quality
gate at this work limit. Final-iterate quality remains much worse on several
cases; its median at 4,096 is 0.1824 root pots. Per-case current and average
values, native-oracle values, and timings are in `rows.jsonl`. Three-player
rows contain individual deviation gains and NashConv, with no multiway
convergence claim.

The local focused suite checks independent scalar one-sweep regret increments,
native terminal payoffs, odd-chip ties, legal all-ins, information-set best
responses, own-reach averaging, interruption atomicity, hidden-deal
indistinguishability and play delegation. The full repository suite result is
recorded in the PR validation section.

**Interpretation:** the vector solver and exact-value oracle run within the
short M4 resource budget. The average profile is substantially stronger than
the current iterate on these development fixtures, but the proposed tight
all-case threshold is unmet. No extraction rule or solve budget has been
selected for fresh confirmation. This conditional preflight says nothing
about full-hand BB/100 or improvement over corrected rollout search. The
separate conditional comparison must be frozen and run before either claim.
