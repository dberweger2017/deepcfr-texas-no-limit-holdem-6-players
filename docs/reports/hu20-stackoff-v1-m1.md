# Post-Luna HU20 selective-stackoff regression

I evaluated the twelve saved **B20M/B40M/B80M/B100M current-policy exports across three lineages** against one frozen selective-stackoff opponent and the existing scripted/LBR panel. All **55,296 fresh, position-balanced hands completed**, passed native replay and conserved chips. There was no training, policy change, action translation or model promotion.

**The aggregate stress-test mean improved, while tail behavior remained uneven across seeds.** This opponent was designed after the Luna observations in [#127/#130](luna-browser-big-pot-analysis.md). It is a post-Luna stress test, not independent confirmation of the suspected mechanism or evidence of general human strength.

## Freeze and execution

I committed the [opponent and schedule](../hu20-stackoff-protocol.md) in `1db8e0e` before loading policies or measuring outcomes. The [fixed plan](../../configs/diagnostics/hu20-stackoff-v1.json) pins twelve inference/checkpoint hash pairs from #116; its canonical digest is `bcf64187cb16b582f9503790abfaceee3ded9a9c6c4f9e0731f96960599750d2`.

The opponent sees only its own immutable observation. It uses the unchanged restricted native-reopening menu, calls large wagers with its prespecified strong-hand tier, and sometimes min-bets/min-raises strong hands. Its thresholds, 0.35 trap frequency, one RNG draw per decision and evaluation roots remain frozen as `selective-stackoff-hu20-v1` for future comparisons. I did not tune it to make B100M lose.

The game is two-player NLHE, 2,000 chips/20BB per seat reset every hand, 50/100 blinds, no rake or ante. Work milestones refer to **traversal nodes**, not poker hands or completed traversal iterations. I hash-verified unchanged compressed inputs and audited current probabilities against checkpoint regrets while streaming visits, without constructing resumable trainer state.

Gameplay source: `1e7fd1a1d68ce5f001b8000be4e385da8076f88e`. Inspection/reporting source: `3c2d9ee3a55c8d09c7b36968e047fbb30c61d348`. Later changes package and explain evidence; gameplay code stayed unchanged. The audited engine revision is `5db20e3d5d6862b32a7402035c1340b622d3b005`.

## Checkpoint trends

Each stress checkpoint has 1,024 independent paired deal blocks per lineage: 2,048 hands per policy and 6,144 hands across three lineages. I average positions and the three shared lineages **within each block** before calculating aggregate intervals. These are exploratory, unadjusted 95% Student-t intervals conditional on these three saved lineages; neither rotations nor common-deal lineages count as independent samples.

| Opponent | B20M | B40M | B80M | B100M | B100M − B20M [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| Selective-stackoff v1, 1,024 blocks | +27.24 | +34.21 | +37.08 | +40.34 | **+13.10 [6.79, 19.41]** |
| Native pressure, 256 blocks | +81.41 | +92.84 | +105.99 | +84.11 | +2.70 [−57.16, 62.57] |
| Original-cap2 bounded LBR, 128 blocks | −104.30 | −96.03 | −86.00 | −97.07 | +7.23 [−57.94, 72.39] |

Entries are target BB/100. B100M's absolute stress interval is **[32.19, 48.50]**. Its LBR interval is **[−157.54, −36.60]**: the policy still loses against this bounded attacker. The smaller pressure/LBR contrasts are imprecise and do not establish improvement in those panels. I retain all five reactive panels and all six secondary opponents in the [complete dashboard](hu20-stackoff-artifacts/dashboard.md) and [machine summary](hu20-stackoff-artifacts/summary.json).

The aggregate stress curve rises, but adjacent B40M→B80M and B80M→B100M intervals include zero. Individual lineages are not uniformly monotonic:

| Saved seed | Stress B20M | B40M | B80M | B100M | Own B100M − B20M [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026093001, distributed playable seed | +26.90 | +31.03 | +43.02 | +38.33 | +11.43 [−0.70, 23.55] |
| 2026093002 | +27.66 | +33.35 | +35.23 | +28.86 | +1.20 [−10.21, 12.60] |
| 2026093003 | +27.15 | +38.26 | +33.01 | +53.83 | +26.68 [16.37, 37.00] |

I report all seed/position estimates, adjacent contrasts and paired between-seed differences in the summary. The third seed contributes much of the final gain; these results do not justify selecting or promoting a different seed. I keep [#116's primary schedule](hu20-scaling-m4-recovery.md) and [#117's completed diagnostics](hu20-scaling-diagnostics.md) separate rather than pooling their estimates with this smaller fresh schedule.

## Tail dashboard

A large target wager means **at least 800 additional chips for the rival to call**, accounting for existing street commitments and the rival's remaining stack. An opportunity is a target decision whose concrete menu contains such an action. Exact all-in raises are tracked separately from all-in calls.

| Work | Stress hands | Large raises / opportunities | Rival folds / large raises | Rival continuations / large raises | Exact +20BB wins | Exact −20BB losses | Fallback / target decisions |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| B20M | 6,144 | 111/275 | 18/111 | 93/111 | 22 | 40 | 24/7,820 |
| B40M | 6,144 | 89/256 | 14/89 | 75/89 | 17 | 31 | 12/8,120 |
| B80M | 6,144 | 91/284 | 5/91 | 86/91 | 27 | 40 | 5/8,164 |
| B100M | 6,144 | 88/293 | 8/88 | 80/88 | 27 | 25 | 1/8,126 |

No large raise lacked an immediate rival response. Counts are descriptive; shared deals remain correlated. Both near-full losses and smaller losses stay in overall returns, but do not enter the exact ±20BB counters.

**B80M illustrates why the tail dashboard matters:** its average improved over B20M, but it still recorded 40 full-stack losses. B100M's loss count fell to 25, yet that improvement is concentrated in the third lineage:

| Seed | B20M full-stack wins / losses | B100M wins / losses | B100M stress hands |
| --- | --- | --- | ---: |
| 2026093001 | 8 / 15 | **7 / 12** | 2,048 |
| 2026093002 | 7 / 11 | **7 / 10** | 2,048 |
| 2026093003 | 7 / 14 | **13 / 3** | 2,048 |

The distributed first seed still loses more full stacks than it wins against this opponent despite positive average profit. I do not assign individual-bet EV from those outcomes. [Whole-hand return partitions](hu20-stackoff-artifacts/dashboard.md#whole-hand-return-partitions-stress) use the **first** large target raise's response, count each hand once and retain no-large-raise cases. Positional tails, jam opportunities/actions and street-specific lookup denominators are also retained.

## Matched policy-table inspection

The frozen rule selected 13 origins representing **11 distinct public river situations**. Five seed/position/small-raise strata were empty. I queried all 1,081 two-card holdings compatible with each public board for each saved policy: **142,692 queries**. I did not condition on actual unrevealed rival cards. Duplicate origins remain recorded, while each public situation is queried once. The [inspection summary](hu20-stackoff-artifacts/inspection-summary.json) and twelve raw holding tables retain concrete categories, keys, visits, ordered menus, probabilities and fallback.

There is an important coverage limit: **10 of the 11 situations cannot offer an 8BB raise in the restricted menu**. Their pot raises leave only 400 or 600 chips to call, and the conditional jam is absent. Their zero large/jam probability is structural, not evidence that training repaired weak-hand aggression. I retain these contexts and the five empty strata rather than broadening the selection rule after seeing results.

The sole qualifying large-wager situation, context `be761977c9187da150ccf8c836ba5bd1b28cc0e6160a5692f42f60f40c5985d2`, is:

- Board **Ad 7d 7c Ah 3s**; target on the button.
- Limp/check preflop; check/check flop; rival bets 100 on the turn, target calls; river rival checks, target bets 100, rival min-raises to 200.
- Pot 700; target owes 100, has 1,700 remaining chips and 100 committed on the river.
- Concrete five-action menu: fold, call, min raise-to 300, pot raise-to **1,000** (800 rival call), jam raise-to **1,800** (1,600 rival call).

This yields a concrete abstraction lead. **2c 2d plays the board; Qc Kc improves its kicker to a king. Both map to the same descriptor `(2,2,0,0,1)` and information key `872868ae2200030bf5ecc308d267d41f`.** Their distributions are therefore identical in this matched public state. Different concrete showdown values sharing a key warrant investigation; this does not establish a strategy error or the cause of Luna's results.

I abbreviate the three full training seeds as 3001, 3002 and 3003 below.

| Seed / checkpoint | Shared board-only / king-kicker node visits | Large-raise probability | Jam probability |
| --- | ---: | ---: | ---: |
| 3001 B20M | 32 | 11.81% | 0% |
| 3001 B40M | 89 | 67.70% | 31.48% |
| 3001 B80M | 204 | 0% | 0% |
| 3001 B100M | 229 | 0% | 0% |
| 3002 B20M | 30 | 56.59% | 10.02% |
| 3002 B40M | 68 | 94.62% | 42.56% |
| 3002 B80M | 131 | 0% | 0% |
| 3002 B100M | 138 | 33.55% | 0% |
| 3003 B20M | 110 | 0% | 0% |
| 3003 B40M | 141 | 0% | 0% |
| 3003 B80M | **347** | 18.51% | 18.51% |
| 3003 B100M | **358** | 0% | 0% |

Only the bold visit counts meet the frozen ≥300 high-visit threshold here. This local strategy is non-monotonic across saved checkpoints; it is not a population-wide aggression estimate or a paired bet-value experiment.

At the same B100M state, stronger holdings remain sparse:

| Seed | Ac Kc: aces full, visits; large / jam probability | 7h 7s: quads, visits; large / jam probability |
| --- | --- | --- |
| 3001 | 37; 56.21% / 49.58% | 3; 79.56% / 76.57% |
| 3002 | 6; 83.20% / 20.08% | 2; 100% / 100% |
| 3003 | 29; 100% / 100% | **0, fallback; 40% / 20%** |

There is no ≥300-visit strong node in this particular large-wager context. The missing quads node's uniform fallback includes 20% fold probability; that is adapter fallback, not a learned quads policy. Low actual-play fallback (1/8,126 decisions at B100M) does not guarantee coverage of rare compatible holdings. The [exact example rows](hu20-stackoff-artifacts/matched-river-examples.json) and all raw queries make these checks reproducible. Holdings sharing a key are not independent trained entries. Comparable aggression across weak/strong categories alone would not prove an error.

## Validation, resources and reproduction

All twelve policy/checkpoint pairs passed compressed-byte hashes, schema/game/current-extraction and seed/iteration audit. Every completed hand passed native legal action, replay/event digest and chip-settlement checks. The full local suite passed **1,054 tests plus 54 subtests** at the initial harness revision; subsequent affected checks passed **33 focused tests**. [Full CI at the inspected implementation](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/36774383354) passed both shards and the existing CI gates. Fixtures are generated tiny exports populated directly, without trainer steps; measured tables use the twelve real saved policies.

M4 supplied only a coordinated read-only transfer of 1,017,051,279 bytes; I released it to Doctor Research afterward. All loads, tests and measurements ran on **Apple M1**, Python 3.11.15, NumPy 1.26.4, SciPy 1.17.1 and the pinned native engine. The single sequential evaluator took **2,619.64 seconds / 43.7 minutes**, with peak process RSS **1,964,638,208 bytes / 1.83 GiB**. Swap ended below its starting use; memory, 0.5GiB swap-growth and 1GiB disk-floor guards did not trip. A background Drive upload ran during part of evaluation; actual LBR budget flags are retained rather than inferred from CPU use. Inspection took **161.42 seconds**, with peak child RSS **1.57 GiB**. No service was left running, no paid compute was used and no research job was interrupted.

The packaged-evidence check replayed all **55,296 exported hands**, rebuilt tail counters and the paired summary from **230,807 raw decision rows**, and verified all 27 initial packaged files. All **6,455 LBR decisions** completed their requested batches, with no soft-budget exceedance or partial decision; the longest took **2.06 seconds**. Inspection produced **142,692** raw holding queries. [Validation records](hu20-stackoff-artifacts/validation.json) retain these checks. The temporary audit wrapper first failed its import path before reading evidence; that reporting-only attempt is retained locally and was repaired without rerunning gameplay.

The [dashboard](hu20-stackoff-artifacts/dashboard.md), [raw decisions](hu20-stackoff-artifacts/decisions.csv.gz), [native replay records](hu20-stackoff-artifacts/generated-hands.jsonl.gz), [input manifest](hu20-stackoff-artifacts/input-manifest.json) and [evidence manifest](hu20-stackoff-artifacts/evidence-manifest.json) contain generated simulator evidence, not private human journals or tokens. Large model binaries and full observation snapshots remain outside Git. The [protocol](../hu20-stackoff-protocol.md#running-and-inspecting-the-regression) supplies evaluation/report/inspection commands.

Replay the compact hand records from a checkout with the pinned engine:

```sh
python - <<'PY'
import gzip, json
from scripts.play_robustness import replay_row
path = 'docs/reports/hu20-stackoff-artifacts/generated-hands.jsonl.gz'
count = 0
with gzip.open(path, 'rt') as source:
    for line in source:
        replay_row(json.loads(line))
        count += 1
print(f'Native replay verified: {count} hands')
PY
```

## Diagnostic leads and unresolved questions

1. **Average and tails can diverge.** B80M's better average coexists with unchanged full-stack loss count; B100M's tail gain is concentrated in one seed. Preserve the frozen opponent and denominators for the scaling comparison.
2. **The concrete bucket collision is real; its causal importance is unresolved.** Board-only and improved-kicker hands share a state and strategy. A future validated abstraction experiment or paired action-value study would be needed to attribute losses to it.
3. **High-visit large-river coverage is narrow.** Ten selected situations lack an 8BB action, most small-raise strata are empty, and strong holdings at the qualifying state have few visits or fallback. Do not relax thresholds retroactively or generalize this sample to the whole policy table.
4. **This restricted stress test does not test off-tree repair.** It neither reproduces Luna's entire decision process nor validates free-sizing strategy, posterior stability, or an action translator. The bounded LBR remains approximate and the policy still loses against it.

I retain the opponent unchanged for future checkpoint tests. No training, paid campaign, automatic merge, six-player claim or policy-strength promotion follows from this report.
