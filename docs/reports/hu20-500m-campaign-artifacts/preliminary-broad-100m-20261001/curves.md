# Preliminary HU20 100M→500M broad checkpoint curves

Snapshot: 3 closed broad tasks, **79,872 hands**. 15 broad tasks pending at capture.

**Exploratory only.** Unadjusted 95% intervals use the frozen paired deal blocks: LBR-original-cap2: 2048, Pressure-native: 4096, selective_stackoff: 4096, pot_pressure: 512, passive: 512, loose_passive: 512, loose_aggressive: 512, tight_passive: 512, tight_aggressive: 512. Three seed returns/contrasts are averaged inside each deal/rotation block, then both roles are averaged. Neither seats nor seeds multiply the sample count. All declared opponents and every available checkpoint are retained. Incomplete broad tasks and fresh final confirmation remain pending; no strength or promotion gate is inferred.

## Absolute three-lineage returns: BB/100 [95% interval]

| Nodes | LBR-original-cap2 | Pressure-native | selective_stackoff | pot_pressure | passive | loose_passive | loose_aggressive | tight_passive | tight_aggressive |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100M | -64.97 [-79.85, -50.10] | +114.04 [+100.78, +127.31] | +35.13 [+31.22, +39.04] | +16.18 [-11.55, +43.91] | +108.04 [+81.72, +134.37] | +29.82 [+10.86, +48.78] | +24.72 [+0.21, +49.24] | +52.69 [+45.07, +60.30] | +50.75 [+36.57, +64.92] |

## Paired change versus each lineage’s own B100M: BB/100 [95% interval]

| Nodes | LBR-original-cap2 | Pressure-native | selective_stackoff | pot_pressure | passive | loose_passive | loose_aggressive | tight_passive | tight_aggressive |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Selective-stackoff: individual seeds and positions

| Seed | Nodes | Overall BB/100 [95%] | Button | Big blind |
| --- | ---: | --- | --- | --- |
| 2026093001 | 100M | +38.68 [+33.75, +43.61] | +38.29 [+29.92, +46.67] | +39.06 [+33.21, +44.92] |
| 2026093002 | 100M | +28.50 [+23.18, +33.81] | +23.82 [+14.37, +33.26] | +33.18 [+26.86, +39.50] |
| 2026093003 | 100M | +38.21 [+33.18, +43.25] | +39.98 [+31.34, +48.62] | +36.45 [+30.00, +42.90] |

## Selective-stackoff tails: counts and denominators

Large raises/calls use the retained 800-chip threshold. Full stack is ±20 BB. These are descriptive correlated-hand counts; whole-hand partitions are not individual-bet EV.

| Seed | Nodes | −20BB / hands | +20BB / hands | Large raises / opportunities | Jams / opportunities | Large calls | All-in calls | Fallback / decisions |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 2026093001 | 100M | 36/8192 | 23/8192 | 116/371 | 73/384 | 0 | 0 | 3/10659 |
| 2026093002 | 100M | 49/8192 | 28/8192 | 137/413 | 93/421 | 0 | 0 | 2/11467 |
| 2026093003 | 100M | 49/8192 | 22/8192 | 146/413 | 95/436 | 0 | 0 | 2/10613 |

## Coverage and interpretation

The machine-readable summary and CSV retain all opponents, seeds, positions, checkpoint changes, fallback by street, tails and whole-hand return partitions. An aggregate requires all three seeds; a checkpoint available for only one or two seeds is shown individually, not substituted for a three-seed curve. Missing tasks stay pending. Model/checkpoint identities and raw closed-file hashes are in the input manifest. Each generated hand was natively replayed during evaluation; this reporting pass checks replay evidence, frozen coordinates, chip accounting, recomputed tails, pairing and independent sums. It does not rerun gameplay.

The selective-stackoff opponent was designed after Luna and remains a regression/stress opponent, not independent confirmation of Luna. Point estimates can fluctuate; these small panels do not replace any incomplete broader panels or fresh final schedules. Training, saving and evaluation settings remain unchanged.
