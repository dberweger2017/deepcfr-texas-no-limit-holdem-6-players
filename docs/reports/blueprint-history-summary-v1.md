# Summary-history blueprint comparison

## Question and fixed work

The 5.83-million-entry original blueprint covered only 24.4% of preflop and 9.1% of flop decisions made against a held-out style pool. Its information key retained the ordered public action history. This comparison tests one shorter, player-visible history representation before deciding how to spend a large memory-backed training run. The protocol and frozen six-player, 100-BB plan are in [the check](../blueprint-history-summary-check.md).

The original and summary runs share seed `2026092402`, four roots per seat, raise cap two, one M4 worker, and an 8,733-iteration target. The summary key also includes coarse pot and effective-stack bands. It is therefore a different abstraction, not mathematically guaranteed to merge every pair of original keys. Report both entry growth and lookup rates; do not equate a shorter key with poker strength.

## Training-time evidence

The following matched windows each contain 500 iterations. Lookup fractions come from sampled trainer visits and are **not** held-out policy coverage. Node counts differ slightly because the policies visit different paths.

| Iterations | Key | Nodes | Entries at window end | Preflop trained | Flop trained | Turn trained | River trained |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,834–4,333 | Original | 2,577,730 | 2,941,988 | 54.0% | 26.7% | 4.2% | 1.6% |
| 3,834–4,333 | Summary | 2,596,602 | 2,756,478 | 58.2% | 40.6% | 12.9% | 5.4% |
| 5,950–6,449 | Original | 2,629,900 | 4,332,250 | 61.1% | 32.5% | 5.8% | 2.5% |
| 5,950–6,449 | Summary | 2,649,316 | 3,962,157 | 64.8% | 47.7% | 17.2% | 7.8% |
| 8,234–8,733 | Original | 2,653,656 | 5,834,622 | 66.1% | 36.6% | 7.4% | 3.2% |
| 8,234–8,733 | Summary | 2,686,257 | 5,227,955 | 70.0% | 52.8% | 20.5% | 9.8% |

Both runs reached iteration 8,733. The summary run traversed 45,609,620 nodes versus 45,157,550 for the original, a 1.0% increase, and used 5,227,955 entries versus 5,834,622, a 10.4% reduction. Summed step time was 3,693.4 versus 3,485.1 seconds, so the summary run's extra key computation cost about 6% more time in this comparison. The summary trainer's maximum reported training-process RSS was 3.34 GiB versus 4.22 GiB for the original. The final summary checkpoint is 229,689,846 bytes with SHA-256 `d0fe534ff6f92e7f6139b3b6a48b1bc3fc58c054c4c291fe5f595ec97ac443e1`; its built-in 24-hand check completed with no invalid actions. These training points show fewer entries and more repeated sampled states at essentially equal traversal work. The held-out result determines whether the new representation is useful for the next run.

## Fresh opponent decisions and play

The frozen 128-block arena uses 1,536 candidate hands against tight-passive, loose-aggressive and pot-pressure opponents, each paired with a uniform abstract-menu control. The summary current and average policies both made zero invalid actions. Exporting both after checkpoint load peaked at 7.24 GiB process RSS, below the 10-GiB check guard. The current and average compressed policies are 145,230,162 and 107,873,053 bytes. First-to-act preflop probes found a trained entry for every one of 169 hand classes. The current policy folds 72o and pot-raises AA in that probe; this is differentiated play, not a strength certificate.

| Current policy on held-out style pool | Original key | Summary key |
| --- | ---: | ---: |
| Preflop trained lookups | 236 / 968 (24.4%) | 232 / 966 (24.0%) |
| Flop trained lookups | 34 / 374 (9.1%) | 23 / 368 (6.3%) |
| Turn trained lookups | 3 / 140 (2.1%) | 1 / 143 (0.7%) |
| River trained lookups | 3 / 65 (4.6%) | 1 / 58 (1.7%) |
| Candidate BB/100 against pool | −770.7 | −762.6 |
| Paired gain over uniform, 95% CI | +33.5 [−88.1, +155.0] | +41.6 [−81.8, +165.0] |

Those policies reach different histories, so the table above cannot isolate the key. The fixed-decision probe made the **original** current policy act in exactly the same 1,536 hands while querying both tables for each observation. It reproduced the original arena's 236/968 preflop and 34/374 flop hits exactly, confirming that the old policy and schedule loaded consistently.

| Same held-out decisions | Original trained | Summary trained |
| --- | ---: | ---: |
| Preflop | 236 / 968 (24.4%) | 234 / 968 (24.2%) |
| Flop | 34 / 374 (9.1%) | 37 / 374 (9.9%) |
| Turn | 3 / 140 (2.1%) | 1 / 140 (0.7%) |
| River | 3 / 65 (4.6%) | 2 / 65 (3.1%) |

On the same decisions, there were 904 distinct original and summary preflop keys, 365 original and 366 summary flop keys, and identical distinct-key counts of 140 turn and 65 river. **No summary key merged two observed original keys** in this small held-out sample. One original flop key split into distinct summary keys, consistent with the new pot and stack bands. The probe peaked at 8.03 GiB RSS and completed with zero invalid actions.

## Decision

The summary representation clearly increased repeated lookups during self-play training and reduced table size, but it did **not** materially increase trained coverage against the held-out style pool at this work scale. The short arena comparisons are inconclusive and cannot establish poker strength. The main failure mode remains sparse generalization to new decisions, especially after the flop. Keep `blueprint-abstraction-v1` as the reference recipe; the summary schema remains an opt-in, versioned experiment and cannot silently load as v1. Do not spend a large rental on the assumption that history compression solved coverage.

The owner now prioritizes a sustained checkpoint-0.4 campaign that demonstrates continued improvement against random play and reveals practical memory or learning bounds, even if the table will not yet be strong against the scripted pool. Start that campaign with a fixed random-opponent schedule and sparse checkpoint evaluation, recording trained coverage, memory and BB/100 with uncertainty as training progresses. Choose a table ceiling that uses substantial rented RAM, and stop only on declared work, resource, cost or correctness limits. This is a new campaign decision; its host, cap and budget should be stated explicitly before launch. Avoid another short architecture sweep.

The training checkpoint, current and average exports, iteration logs, fixed schedule and arena outputs remain on the M4 in the ignored `results/blueprint-history-summary-v1/`, `results/blueprint-history-summary-learning/` and `results/blueprint-history-fixed-coverage/` directories of its isolated worktree. Copies are retained locally under corresponding ignored `results/*-m4/` directories: 15 training, 22 learning and 10 fixed-probe files. The local final checkpoint and both exported policies match their recorded SHA-256 hashes. No policy is promoted as a default by this PR.
