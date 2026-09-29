# Saved 20BB robustness — M4 report

Status: **complete**. 947,200 hands; every retained hand replayed through native settlement. All estimates below are exploratory block-clustered 95% intervals. No model promotion.

| Game / work | Attack | Contract | Absolute target BB/100 [95% CI] | Trained − uniform BB/100 [95% CI] |
| --- | --- | --- | --- | --- |
| HU 2M | pressure | menu | -12.31 [-23.91, -0.72] | +119.97 [+105.13, +134.80] |
| HU 2M | pressure | native | -291.14 [-304.38, -277.90] | +20.65 [+4.65, +36.64] |
| HU 2M | minraise | menu | -39.13 [-53.42, -24.84] | +237.21 [+215.97, +258.45] |
| HU 2M | minraise | native | -292.12 [-306.43, -277.82] | +21.41 [+4.62, +38.21] |
| HU 2M | passive | menu | +93.76 [+84.31, +103.22] | +95.24 [+84.88, +105.60] |
| HU 2M | passive | native | +93.76 [+84.31, +103.22] | +95.24 [+84.88, +105.60] |
| HU 2M | lbr | menu | -181.62 [-214.25, -149.00] | +105.19 [+51.67, +158.71] |
| HU 5M | pressure | menu | +23.06 [+12.79, +33.33] | +155.34 [+140.43, +170.25] |
| HU 5M | pressure | native | -291.94 [-304.95, -278.92] | +19.85 [+4.09, +35.61] |
| HU 5M | minraise | menu | +8.93 [-4.56, +22.41] | +285.27 [+263.62, +306.92] |
| HU 5M | minraise | native | -295.34 [-309.54, -281.14] | +18.20 [+1.30, +35.09] |
| HU 5M | passive | menu | +93.31 [+84.98, +101.64] | +94.79 [+84.33, +105.24] |
| HU 5M | passive | native | +93.31 [+84.98, +101.64] | +94.79 [+84.33, +105.24] |
| HU 5M | lbr | menu | -168.23 [-200.48, -135.98] | +118.59 [+64.36, +172.81] |
| HU 10M | pressure | menu | +30.37 [+20.20, +40.53] | +162.65 [+147.58, +177.71] |
| HU 10M | pressure | native | -260.86 [-273.60, -248.11] | +50.93 [+34.87, +66.99] |
| HU 10M | minraise | menu | +49.25 [+35.54, +62.96] | +325.59 [+303.61, +347.58] |
| HU 10M | minraise | native | -267.48 [-281.68, -253.28] | +46.06 [+29.17, +62.94] |
| HU 10M | passive | menu | +98.87 [+90.13, +107.61] | +100.35 [+89.23, +111.47] |
| HU 10M | passive | native | +98.87 [+90.13, +107.61] | +100.35 [+89.23, +111.47] |
| HU 10M | lbr | menu | -112.92 [-141.44, -84.41] | +173.89 [+121.95, +225.84] |
| HU 20M | pressure | menu | +35.13 [+24.52, +45.75] | +167.41 [+151.83, +183.00] |
| HU 20M | pressure | native | -264.01 [-277.08, -250.95] | +47.77 [+31.35, +64.20] |
| HU 20M | minraise | menu | +71.82 [+58.24, +85.40] | +348.16 [+325.68, +370.64] |
| HU 20M | minraise | native | -252.25 [-266.76, -237.73] | +61.29 [+43.96, +78.62] |
| HU 20M | passive | menu | +101.40 [+92.40, +110.39] | +102.87 [+91.32, +114.43] |
| HU 20M | passive | native | +101.40 [+92.40, +110.39] | +102.87 [+91.32, +114.43] |
| HU 20M | lbr | menu | -111.96 [-143.48, -80.45] | +174.85 [+120.27, +229.44] |
| TP 20M | pressure / pressure | menu | -25.80 [-40.00, -11.61] | +98.41 [+78.03, +118.78] |
| TP 20M | pressure / pressure | native | -279.86 [-291.61, -268.11] | +57.97 [+43.16, +72.77] |
| TP 20M | minraise / minraise | menu | -79.93 [-102.91, -56.95] | +176.02 [+149.56, +202.48] |
| TP 20M | minraise / minraise | native | -304.65 [-316.40, -292.90] | +54.18 [+39.90, +68.46] |
| TP 20M | passive / passive | menu | +116.31 [+96.16, +136.47] | +123.38 [+101.87, +144.89] |
| TP 20M | passive / passive | native | +116.31 [+96.16, +136.47] | +123.38 [+101.87, +144.89] |
| TP 20M | pressure / minraise | menu | -46.23 [-65.57, -26.90] | +191.31 [+164.88, +217.73] |
| TP 20M | pressure / minraise | native | -285.22 [-296.27, -274.16] | +68.68 [+54.37, +83.00] |
| TP 20M | pressure / passive | menu | +23.08 [+1.23, +44.93] | +144.87 [+118.28, +171.47] |
| TP 20M | pressure / passive | native | -227.20 [-249.32, -205.09] | +98.42 [+72.62, +124.21] |
| TP 20M | minraise / passive | menu | -26.28 [-50.92, -1.64] | +223.16 [+191.27, +255.05] |
| TP 20M | minraise / passive | native | -226.13 [-248.90, -203.36] | +125.07 [+97.68, +152.46] |

BB/hand is BB/100 divided by 100. 20BB buy-ins/100 is BB/100 divided by 20. HU attacker returns negate target returns; role-specific and individual-seed results are retained in results.json. TP worst-case quality is unmeasured.

The local response integrates the full compatible range and samples four future boards per holding, uses a five-second soft batch deadline, and plays against the unchanged saved target. Internal maxima are never used as reported profit. A negative or imprecise attacker result is not a robustness certificate.

Trained/fallback and off-menu-history decision counts are separated by street in the JSON. These are exposure measurements; whole-hand profits are not attributed to individual streets. Selected river probes use a declared artificial range and cannot establish full-game exploitability.

Peak sampled RSS: 1.187 GiB; minimum disk: 44.10 GiB. Resource records and every attempt are retained.
