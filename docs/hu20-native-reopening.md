# Capped versus native-reopening HU20

This dependent draft tests one setting: capped A uses the existing HU20 game;
B makes the same min/pot/conditional-jam sizes available whenever native
reopening and stack bounds allow, using `raise_cap=None`. B is not a menu of
all integer bet sizes. Jam eligibility, no-free-fold behavior, exact terminal
payoffs, cards, ordered button-relative histories, K1, roots per seat and linear
iteration regret weights are unchanged. All main runs, if feasible, begin at
zero and extract final-current C. No old table is migrated.

B has separate game, abstraction/menu and inference-format identities. Its
checkpoint manifest declares native reopening without an artificial count cap;
loading it with a capped adapter or changing its cap configuration is rejected.
Capped artifact serialization remains byte-identical in a three-iteration
regression against reviewed #114. Existing HU20/TP20 checkpoints and demos stay
intact. This experiment changes both training exposure and target action
availability through that single cap setting.

## Resource gate

The committed initial plan is `configs/blueprint/hu20-native-reopening-m4.json`.
Three distinct resource-preflight seeds run sequential A then B, each for 1M
completed nodes. Uniform and short-trained profiles have separately recorded,
outcome-suppressed inference/LBR timing. Every complete outer iteration records
cost, nodes/updates by street and raise count, new/revisited keys beyond the
original cap, RSS and checkpoint/export overhead. Cancellation and byte-identical
last-completed-checkpoint recovery are tested without publishing partial regrets.

A failed traversal stops the preflight. Remaining seeds are explicitly unrun;
there is no retry, cap=4 substitute, pruning, fake leaf value or resampling of a
difficult deal. The iteration limits are the existing 250,000 nodes/300 seconds;
resource bounds remain 10.5 GiB RSS, 8 GiB free disk and 0.5 GiB swap growth.
One absolute ten-hour deadline begins with preflight. At least two hours are
reserved for evaluation/export/audit/report; measured LBR costs can increase
that reserve. The preflight does not inspect returns.

If feasible, freeze a common 5M/10M/20M completed-node budget (prefer 20M),
10%/25%/50%/100% checkpoints and a common 512/1,024/2,048 LBR block count from
resource evidence before main outcomes. Paired seeds acknowledge action-stream
divergence as trees differ. Overshoot, incomplete work and all failures remain
in the denominator/report. A smaller campaign is an early-learning comparison.

## Frozen statistical and attacker contracts

Primary B−A is against unchanged native pressure. The quality safeguard is B−A
against #114's original-cap2 LBR, with four future samples and five soft seconds.
Both use two-sided **97.5% rotation-block intervals**, averaging seed-paired
contrasts within each shared block. Non-inferiority requires the safeguard's
lower bound to exceed **−10 BB/100**; an interval containing zero does not prove
safety. Estimate precision from prior block variance before freeze; insufficient
precision remains inconclusive. Other comparisons are exploratory.

The pressure/minraise original-cap2 opponents always use the original capped
menu for BOTH targets; native rules and passive control are also unchanged.
Each target uses its own trained menu and fallback. LBR hypothetical queries use
the correct target-specific distribution, including B's later reraises, while
its own action set remains original-cap2. Telemetry records action membership
and preceding-history exposure under both original and target menus. No common
attacker comparison is labeled exact profile exploitability.

All evaluation schedules are fresh and separate from training/preflight and
#112–#114 opened deals. Cheap suggested counts are 4,096 paired blocks; the
six-opponent secondary panel is 512 blocks per opponent. Counts and independent
observation fixtures are frozen before outcomes. Absolute losses, each seed and
role, trained/fallback street exposure, completed/limited LBR, fixed-observation
and own-trajectory visit density must be retained. No card-bucket diagnosis,
model promotion, paid compute or automatic follow-on campaign is implied.

## Current state

The [completed M4 report](reports/hu20-native-reopening-m4.md) retains all six fresh 20M-node runs, all 24 milestone checkpoint/export pairs and 1,148,928 audited evaluation hands. Native-pressure B−A is +356.94 BB/100 [97.5% +337.01, +376.87], with B absolute +102.20. Original-cap2 LBR B−A is +34.39 [−5.55, +74.33], passing the fixed −10 margin but not establishing improvement; B still loses −95.88 BB/100 to LBR. Capped-minraise and passive-control regressions remain. Every attempt and inventory hash passed; the fixed-first-seed [candidate play command](hu20-native-reopening-model-card.md) completed and replayed its smoke. Draft #115 remains open without promotion or automatic follow-on work. Historical preflight and training snapshots are retained.
