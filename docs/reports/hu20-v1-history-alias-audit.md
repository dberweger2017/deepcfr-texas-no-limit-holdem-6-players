# HU20 v1 native bet-size alias audit

2026-10-02, draft [PR #146](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/146), stacked on #145. No solver, training, promotion or rental was used. Enumeration and fresh play ran on the M1; other M4 work was untouched.

The size-band defect exists, including the 1 BB minimum / 2 BB pot bet in a limped pot. However, **a colliding history token does not always imply the same information key**: the current legal menu is also part of v1's key. Only **28/273 (10.26%)** selected flop decisions have a full native-key alias, versus **77/273 (28.21%)** with a history-token collision. Of the 28, 23 first diverge on the flop and five inherit a preflop alias. This is a representation issue worth testing, not an explanation established for the plateau.

## Exhaustive native enumeration

Replay every live path from the reset 20 BB HU root through the uncapped native-reopening menu. A fixed native deal is used only to verify the production keys; betting legality is independent of cards. Actors, street, folded/all-in flags, ordered history and current menu names remain in the card-free signature. The real `information_key` is evaluated at every node, with zero mismatches within each collision group.

| Street | Live decision nodes | Full-key alias groups | Aliased nodes |
| --- | ---: | ---: | ---: |
| Preflop | 156 | 3 | 6 |
| Flop | 3,076 | 206 | 412 |
| Turn | 36,932 | 3,362 | 6,724 |
| River | 331,708 | 34,350 | 68,700 |
| Total | 371,872 | 37,921 | 75,842 |

There are 669,458 terminal paths, 333,951 unique card-free signatures and exactly two concrete contexts in each full-key alias group. The complete tree took 103.35 seconds. Terminal paths have no live information key; they are counted but not treated as policy decisions. All 37,921 cases, full concrete action paths, amounts, menus and verified real key are in [`aliases.jsonl.gz`](hu20-v1-history-alias-artifacts/aliases.jsonl.gz), SHA-256 `360095bb2cc94cdb3b7b52b1398bd21652a338c04ea5ef094e3bf8da1753f9ae`. [`tree.json`](hu20-v1-history-alias-artifacts/tree.json) records the scope, seed and retained SQLite hash/path; [`patterns.json`](hu20-v1-history-alias-artifacts/patterns.json) gives the shortest witness for each first-divergence pattern.

All collisions first diverge between the native `min` and `pot` actions at a **200-chip pot**, paying **100 versus 200 chips**, both labelled `raise-1`. The first divergence is preflop in 31,191 groups, flop in 6,318, turn in 406 and river in six. Most later aliases inherit that initial collision and also differ in subsequent exact amounts. They are not 37,921 independent sizing mechanisms. Counting history collisions while ignoring the current menu yields 42,512 groups / 85,024 nodes; that larger count includes contexts whose production keys are separated by menu names.

## Stored LBR exposure

The three #143 hand files match the SHA-256 inventory from #145 before reading. Native replay verifies every actor and private holding before each recorded action. All 1,743 target decisions are present in the complete native index; no missing/off-menu context is silently classified as unaliased.

| Population | Decisions | Full-key alias | History-token collision |
| --- | ---: | ---: | ---: |
| All target decisions against LBR | 1,743 | 154 (8.84%) | 227 (13.02%) |
| All target decisions facing a bet | 1,376 | 126 (9.16%) | 187 (13.59%) |
| Preflop facing a bet | 1,007 | 89 (8.84%) | 89 (8.84%) |
| Selected flop / Set A | 273 | 28 (10.26%) | 77 (28.21%) |
| Turn facing a bet | 74 | 6 (8.11%) | 16 (21.62%) |
| River facing a bet | 22 | 3 (13.64%) | 5 (22.73%) |

The supplied first-bet count independently reproduces: LBR makes **227 flop first bets**, of which **113 are exactly pot-sized**. Only **26/227** resulting target decisions have a full-key alias, and only **2/113 pot-sized first bets** do. Thus “LBR bets pot half the time” does not imply that the bot uses the identical minimum-bet policy on all those pot bets. Changes in legal menu names separate most of them. No causal EV attribution is made.

[`stored-exposure.json`](hu20-v1-history-alias-artifacts/stored-exposure.json) lists every target decision with original source SHA/line/action reference, full-key and token status, alias origin and exact witness actions. These are exact observed-panel shares, not independent-hand estimates or a new playing-strength evaluation.

## Fresh B500M self-play

Use each hash-verified current B500M export for 3,000 fresh, independently seeded deals, alternating button. Sample the actual policy through native settlement, retaining all alias contexts and probabilities without selecting on outcomes. Seeds are 202610020102/112/122 for deals and +1 for actions.

| Lineage | All decisions | Aliased decisions | Shared card keys observed | Keys seen against both sizes | Trained key present | Minimum / pot context occurrences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026093001 | 15,706 | 2,508 (15.97%) | 879 | 224 | 878 | 1,294 / 1,214 |
| 2026093002 | 15,399 | 2,835 (18.41%) | 1,040 | 242 | 1,040 | 1,523 / 1,312 |
| 2026093003 | 15,478 | 2,464 (15.92%) | 962 | 208 | 962 | 1,194 / 1,270 |

Every observed shared card key has its concrete paths, first aliased action, menu size, per-context occurrence count and policy probabilities in the three [`selfplay-<lineage>.json`](hu20-v1-history-alias-artifacts/selfplay-2026093001.json) files ([seed 2](hu20-v1-history-alias-artifacts/selfplay-2026093002.json), [seed 3](hu20-v1-history-alias-artifacts/selfplay-2026093003.json)). Both sizes occur in many already-trained keys; the near-balanced pooled context counts conceal considerable per-key imbalance.

**Historical training attribution is unavailable.** Frozen exports contain one distribution per key; retained training state aggregates visits/regret/average mass under that key, without a counter per concrete public size. Fresh self-play identifies present occupancy and whether a key was trained, not which size contributed how many historical CFR updates. Counterfactual traversal also differs from realized self-play. The artifacts explicitly leave `training_per_size_counts` null. A future A/B should log that attribution rather than reconstruct it from pooled visits.

## Proposed schema and experiment — not implemented or launched

The repeated trained-key exposure justifies a bounded representation hypothesis. It does not justify assuming this fixes the negative LBR result, especially given the small direct exposure among pot-sized first bets.

Propose `hu20-native-reopening-menu-history-card-v1`: preserve v1's card descriptor, actors, all-street ordering, board-event labels and current menu. For each historical raise, reconstruct the **native uncapped menu from public legal bounds before that action**, match the exact executed raise-to amount and record its canonical menu name (`min`, `pot`, `jam`, preserving `choices()` deduplication). For off-menu amounts use a separately tagged `offmenu-raise-<paid/pot band>` and explicit all-in flag. Non-raise labels remain unchanged. Never infer the label from opponent cards, current hidden state or the caller's desired action. Save an explicit history-schema identity in checkpoints/exports; old keys must not be silently reinterpreted or resumed under the new schema.

Before any experiment, require exact amount/name replay checks, correct deduplicated all-ins, off-menu fallbacks, button rotation, public-information isolation and checkpoint recovery. Propose a **100M, three-lineage paired A/B from fresh trainer states**, original ordered v1 versus this sole history-label change, with identical seeds/deals/node budgets/menus/cards and common independent evaluation panels. Retain all three final policies, resource growth, visit density, LBR results, paired intervals and failures. Add per-key/per-concrete-history training counts so historical size exposure is measurable. Freeze seeds, evaluation size, thresholds, cost and resource budget in a separately approved launch protocol; this report authorizes none of that training.

#144 already completed the C-only100M comparison: compressed earlier-street history preserves the existing current-street band tokens. Its primary C−ownA100M bounded-LBR difference is −31.60 BB/100 [−52.42, −10.78], with D deferred. That is a separate density/history-compression intervention, not a test of menu-name separation. Do not reuse its negative result as evidence against this proposal or combine both changes in the first A/B. Any later compressed-history/menu-label factorial needs distinct schema identities and its own protocol; current-street aliases remain in C, while earlier-street compression deliberately discards information that ordered menu-name history would retain.

## Reproduction and limitations

```sh
python -m scripts.audit_history_aliases --stage tree --out <new-run>/tree
python -m scripts.audit_history_aliases --stage stored --tree <new-run>/tree/tree.sqlite --out <new-run>/stored.json
python -m scripts.audit_history_aliases --stage self-play --tree <new-run>/tree/tree.sqlite --inputs <verified-policy-directory> --policy-index 0 --out <new-run>/selfplay.json
```

The exhaustive tree uses seed 202610020101, the pinned native engine from `requirements.txt`, one fixed board/private deal and production keys; card-independent completeness follows from native betting legality and the already-validated v1 key factorization in #145. The first test setup omitted the required replay board, then was corrected; two native regression tests pass. Initial exploratory aggregation assumed only one differing exact action per alias pair; inherited pot/stack differences also change later amounts, so the published patterns use the **first divergence** and retain the complete paths. Initial exposure outputs were enriched with token-versus-full-key and first-divergence attribution without discarding their closed originals. All raw versions and the full SQLite index remain at `/Users/dberweger/Local/hu20-alias-audit-20261002`.

This audit supplies no exact equilibrium values, no causal decomposition of LBR losses and no historical per-size CFR counts. The turn-root follow-up remains separate in #145 and cannot answer the flop-root question.
