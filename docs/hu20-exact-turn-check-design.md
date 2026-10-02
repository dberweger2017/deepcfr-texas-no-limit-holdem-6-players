# HU20 exact turn check: implementation and admission

This is the owner-requested follow-up to draft #145. The flop memory failure
remains a failure; turn/river values cannot answer the flop question. Draft #146
contains the separate history-alias audit, completed before any turn solver work.
No training, promotion, rental, automatic merge, or change to the existing M4
jobs and TensorBoard server is authorized by this diagnostic.

The pinned AGPL solver and its independently built Rust integration remain in
`~/Local/hu20-exact-flop-tool`, outside this MIT repository. The hash inventory
includes every external Rust integration source, the binary, the pinned upstream
files, and the six B500M exports. Python communicates through files.

Native turn trees fit the measured M4 admission budget in all three memory
preflight spots. Compression is used to leave room for locks and feature tables.
The turn-only cap ladder is native → three → two, retaining jams. Capped exports
are deliberately refused until a removed-blueprint/equilibrium-reach audit exists;
cap two is not admitted for the original flop experiment. Stop if the three-bet
root cannot fit. A memory estimate alone does not admit a main run.

The compact exporter covers all 1,128 holdings not blocked by the four-card root
and all 48 river cards, with blocked holdings explicitly marked. It exports the
real v1 descriptor and a deduplicated public-template/descriptor policy table.
Exact uniform-opponent equities supply 20-bin turn equity histograms. Fixed
K=50/200 mean-centroid clustering assigns by cumulative L1 (EMD); river quantile
thresholds pool all river contexts of this turn. This assignment convention is
not an assertion that Euclidean k-means optimizes EMD.

The external tool accumulates equilibrium own-range × own-action-reach weights
for four projections: full v1 keys across public lines/runouts; v1 keys within
one public line across runouts; and full public templates with equity labels at
K=50/200. Street chance constants cancel within a group. Target-only locks leave
the respondent unlocked for MES. Action identities, rather than array positions,
map probabilities at the file boundary. The same equilibrium EV baseline is
retained for all ten target/metric comparisons. Both-policy EV is a separate lock.

`validate_turn_compact.py` checks the compact values against the independent
Python profile exporter/projector. Its deliberately large fixture profile uses
separate preparation, projection and comparison workers, so it is released
before locked solvers run. The full limped native turn fixture includes 412
aliased public nodes; all ten comparisons pass within 1.53e-7 BB. Artificial
four-holding ranges are validation evidence, not main playing-strength evidence.
The native-turn Gate K passes 100,000 real-key comparisons with zero mismatches.

`validate_turn_roots.py` is the real-export preflight. It checks native V1/V2,
locks both players to the actual hash-verified B500M policy, compares exact EV
with at least 20,000 independent native-policy deals from the same full ranges,
and times an equilibrium solve to 0.2% of pot. It does not measure main target-only
losses. River-only V3 must also pass against the pinned current binary. Failed
gates and all resource attempts remain retained. Main admission additionally
requires an outcome-blind frozen corpus, rule and measured ≤24-hour budget.

`select_turn_check_spots.py` selects the start of the turn before any turn
action. The retained LBR population has 67 unique selected roots behind 74
bet-facing decisions; 134 of 768 LBR-panel hands reach a live turn root. The fresh
population stops at the turn, samples no turn outcomes, and has 1,088 live roots
from 3,000 deals. Native replay reproduces every root identity. These are
populations, not a frozen main sample or permission to launch one.
