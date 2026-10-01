# Preliminary HU20 100M→500M broad checkpoint curves

Snapshot: 17 closed broad tasks, **452,608 hands**. 1 broad tasks pending at capture.

**Exploratory only.** Unadjusted 95% intervals use the frozen paired deal blocks: LBR-original-cap2: 2048, Pressure-native: 4096, selective_stackoff: 4096, pot_pressure: 512, passive: 512, loose_passive: 512, loose_aggressive: 512, tight_passive: 512, tight_aggressive: 512. Three seed returns/contrasts are averaged inside each deal/rotation block, then both roles are averaged. Neither seats nor seeds multiply the sample count. All declared opponents and every available checkpoint are retained. Incomplete broad tasks and fresh final confirmation remain pending; no strength or promotion gate is inferred.

## Absolute three-lineage returns: BB/100 [95% interval]

| Nodes | LBR-original-cap2 | Pressure-native | selective_stackoff | pot_pressure | passive | loose_passive | loose_aggressive | tight_passive | tight_aggressive |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100M | -64.97 [-79.85, -50.10] | +114.04 [+100.78, +127.31] | +35.13 [+31.22, +39.04] | +16.18 [-11.55, +43.91] | +108.04 [+81.72, +134.37] | +29.82 [+10.86, +48.78] | +24.72 [+0.21, +49.24] | +52.69 [+45.07, +60.30] | +50.75 [+36.57, +64.92] |
| 150M | -61.91 [-76.12, -47.70] | +123.82 [+110.44, +137.19] | +38.03 [+33.90, +42.17] | +26.30 [-2.19, +54.80] | +133.82 [+106.03, +161.61] | +32.81 [+13.07, +52.55] | +23.97 [-0.20, +48.15] | +57.44 [+49.48, +65.40] | +55.57 [+41.86, +69.28] |
| 200M | -63.36 [-77.89, -48.84] | +125.01 [+111.43, +138.60] | +39.56 [+35.55, +43.57] | +18.91 [-9.90, +47.72] | +98.44 [+70.01, +126.86] | +25.93 [+4.99, +46.87] | +26.45 [+1.68, +51.22] | +57.34 [+49.26, +65.42] | +53.03 [+39.66, +66.39] |
| 300M | -67.46 [-82.56, -52.35] | +121.65 [+108.08, +135.22] | +39.89 [+35.73, +44.05] | +29.79 [+0.30, +59.27] | +128.35 [+98.80, +157.91] | +30.19 [+9.22, +51.16] | +24.69 [-0.57, +49.96] | +58.63 [+51.19, +66.07] | +47.74 [+33.32, +62.16] |
| 500M | -72.48 [-88.56, -56.40] | +119.53 [+105.69, +133.37] | +40.71 [+36.61, +44.81] | +31.75 [+2.63, +60.88] | +126.68 [+97.33, +156.02] | +33.51 [+12.69, +54.34] | +32.45 [+7.59, +57.31] | +57.13 [+49.46, +64.80] | +53.40 [+39.29, +67.52] |

## Paired change versus each lineage’s own B100M: BB/100 [95% interval]

| Nodes | LBR-original-cap2 | Pressure-native | selective_stackoff | pot_pressure | passive | loose_passive | loose_aggressive | tight_passive | tight_aggressive |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 150M | +3.06 [-10.72, +16.85] | +9.77 [-1.38, +20.92] | +2.90 [-0.01, +5.81] | +10.12 [-1.72, +21.97] | +25.78 [+4.15, +47.41] | +2.99 [-10.07, +16.06] | -0.75 [-19.01, +17.52] | +4.75 [+0.25, +9.26] | +4.82 [-2.53, +12.17] |
| 200M | +1.61 [-13.06, +16.28] | +10.97 [-0.99, +22.93] | +4.43 [+1.50, +7.36] | +2.73 [-10.18, +15.64] | -9.60 [-32.10, +12.90] | -3.89 [-18.72, +10.94] | +1.73 [-17.34, +20.79] | +4.65 [+0.46, +8.84] | +2.28 [-5.44, +10.00] |
| 300M | -2.48 [-17.73, +12.77] | +7.60 [-4.85, +20.06] | +4.76 [+1.54, +7.97] | +13.61 [+0.06, +27.16] | +20.31 [-5.10, +45.73] | +0.37 [-14.19, +14.94] | -0.03 [-19.44, +19.38] | +5.94 [+1.26, +10.62] | -3.01 [-11.89, +5.87] |
| 500M | -7.51 [-22.72, +7.70] | +5.49 [-6.82, +17.80] | +5.58 [+2.31, +8.85] | +15.58 [+1.74, +29.41] | +18.64 [-6.87, +44.14] | +3.69 [-12.23, +19.62] | +7.73 [-11.23, +26.69] | +4.44 [-0.45, +9.34] | +2.65 [-5.70, +11.00] |

## Selective-stackoff: individual seeds and positions

| Seed | Nodes | Overall BB/100 [95%] | Button | Big blind |
| --- | ---: | --- | --- | --- |
| 2026093001 | 100M | +38.68 [+33.75, +43.61] | +38.29 [+29.92, +46.67] | +39.06 [+33.21, +44.92] |
| 2026093002 | 100M | +28.50 [+23.18, +33.81] | +23.82 [+14.37, +33.26] | +33.18 [+26.86, +39.50] |
| 2026093003 | 100M | +38.21 [+33.18, +43.25] | +39.98 [+31.34, +48.62] | +36.45 [+30.00, +42.90] |
| 2026093001 | 150M | +38.92 [+33.64, +44.20] | +42.30 [+33.21, +51.38] | +35.55 [+29.11, +41.98] |
| 2026093002 | 150M | +37.37 [+32.06, +42.68] | +39.03 [+29.75, +48.30] | +35.72 [+29.39, +42.05] |
| 2026093003 | 150M | +37.81 [+32.68, +42.93] | +35.25 [+26.27, +44.24] | +40.36 [+34.59, +46.13] |
| 2026093001 | 200M | +40.99 [+36.29, +45.68] | +43.71 [+35.35, +52.08] | +38.26 [+33.18, +43.33] |
| 2026093002 | 200M | +40.45 [+35.07, +45.83] | +42.29 [+33.25, +51.32] | +38.62 [+32.37, +44.88] |
| 2026093003 | 200M | +37.24 [+32.13, +42.34] | +34.66 [+25.33, +43.98] | +39.82 [+34.09, +45.54] |
| 2026093001 | 300M | +44.51 [+39.58, +49.43] | +49.68 [+41.04, +58.32] | +39.33 [+33.83, +44.84] |
| 2026093002 | 300M | +35.86 [+30.51, +41.21] | +34.05 [+24.36, +43.74] | +37.67 [+32.02, +43.33] |
| 2026093003 | 300M | +39.30 [+33.98, +44.62] | +42.15 [+32.66, +51.64] | +36.45 [+30.76, +42.15] |
| 2026093001 | 400M | +40.37 [+35.49, +45.25] | +45.43 [+36.64, +54.23] | +35.30 [+29.56, +41.05] |
| 2026093002 | 400M | +36.53 [+31.33, +41.73] | +34.34 [+24.95, +43.73] | +38.72 [+33.01, +44.43] |
| 2026093001 | 500M | +38.58 [+33.37, +43.79] | +41.81 [+32.51, +51.11] | +35.35 [+29.47, +41.23] |
| 2026093002 | 500M | +40.81 [+35.72, +45.89] | +43.43 [+34.37, +52.49] | +38.18 [+32.76, +43.60] |
| 2026093003 | 500M | +42.74 [+37.87, +47.61] | +48.80 [+40.02, +57.59] | +36.67 [+31.24, +42.10] |

## Selective-stackoff tails: counts and denominators

Large raises/calls use the retained 800-chip threshold. Full stack is ±20 BB. These are descriptive correlated-hand counts; whole-hand partitions are not individual-bet EV.

| Seed | Nodes | −20BB / hands | +20BB / hands | Large raises / opportunities | Jams / opportunities | Large calls | All-in calls | Fallback / decisions |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 2026093001 | 100M | 36/8192 | 23/8192 | 116/371 | 73/384 | 0 | 0 | 3/10659 |
| 2026093002 | 100M | 49/8192 | 28/8192 | 137/413 | 93/421 | 0 | 0 | 2/11467 |
| 2026093003 | 100M | 49/8192 | 22/8192 | 146/413 | 95/436 | 0 | 0 | 2/10613 |
| 2026093001 | 150M | 44/8192 | 35/8192 | 151/416 | 105/424 | 0 | 0 | 2/10849 |
| 2026093002 | 150M | 47/8192 | 31/8192 | 137/441 | 91/454 | 0 | 0 | 9/10776 |
| 2026093003 | 150M | 43/8192 | 27/8192 | 121/414 | 82/424 | 0 | 0 | 0/10794 |
| 2026093001 | 200M | 36/8192 | 16/8192 | 107/365 | 74/373 | 0 | 0 | 0/10531 |
| 2026093002 | 200M | 40/8192 | 31/8192 | 131/449 | 85/457 | 0 | 0 | 4/10819 |
| 2026093003 | 200M | 46/8192 | 29/8192 | 141/402 | 89/413 | 0 | 0 | 2/11041 |
| 2026093001 | 300M | 35/8192 | 28/8192 | 130/416 | 86/423 | 0 | 0 | 2/10827 |
| 2026093002 | 300M | 57/8192 | 27/8192 | 156/440 | 108/450 | 0 | 0 | 3/11201 |
| 2026093003 | 300M | 50/8192 | 29/8192 | 131/429 | 93/441 | 0 | 0 | 1/11091 |
| 2026093001 | 400M | 43/8192 | 26/8192 | 124/390 | 82/397 | 0 | 0 | 2/10620 |
| 2026093002 | 400M | 40/8192 | 31/8192 | 131/428 | 81/441 | 0 | 0 | 2/11427 |
| 2026093001 | 500M | 50/8192 | 27/8192 | 141/412 | 91/428 | 0 | 0 | 0/10849 |
| 2026093002 | 500M | 35/8192 | 29/8192 | 128/373 | 80/383 | 0 | 0 | 1/11081 |
| 2026093003 | 500M | 34/8192 | 29/8192 | 120/396 | 81/409 | 0 | 1 | 0/10955 |

## Coverage and interpretation

The machine-readable summary and CSV retain all opponents, seeds, positions, checkpoint changes, fallback by street, tails and whole-hand return partitions. An aggregate requires all three seeds; a checkpoint available for only one or two seeds is shown individually, not substituted for a three-seed curve. Missing tasks stay pending. Model/checkpoint identities and raw closed-file hashes are in the input manifest. Each generated hand was natively replayed during evaluation; this reporting pass checks replay evidence, frozen coordinates, chip accounting, recomputed tails, pairing and independent sums. It does not rerun gameplay.

The selective-stackoff opponent was designed after Luna and remains a regression/stress opponent, not independent confirmation of Luna. Point estimates can fluctuate; these small panels do not replace any incomplete broader panels or fresh final schedules. Training, saving and evaluation settings remain unchanged.
