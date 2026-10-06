# CFR+ complete diagnostics

[Owner review packet](hu20-cfr-plus.md). O = CFR+ production average; C = CFR+ current; R = v0.4.0; T = #165 opponent-sampled average. All intervals are exploratory 95% paired deal-block intervals.

**Response-specific conditional losses.** Terminal hand loss is counted at each qualifying response; rows overlap and are not additive or causal. Immediate committed chips differ from final losses. Negative net loss means profit.

| Arm | Street | Facing | Response | Observed decisions | Frequency % [95%] | Terminal net chips lost [95%] | Gross chips lost [95%] | Immediate chips committed [95%] | Missing % [95%] | Zero-mass % [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O | flop | jam | all | 1432 | 100.00 [100.00, 100.00] | -410.82 [-535.72, -285.93] | 723.25 [660.92, 785.59] | 436.59 [416.52, 456.66] | 0.14 [0.00, 0.41] | 0.07 [0.00, 0.21] |
| O | flop | jam | call | 1316 | 91.90 [89.93, 93.87] | -565.35 [-695.67, -435.03] | 668.69 [602.41, 734.98] | 475.08 [456.05, 494.10] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| O | flop | jam | fold | 116 | 8.10 [6.13, 10.07] | 1342.24 [1283.81, 1400.68] | 1342.24 [1283.81, 1400.68] | 0.00 [0.00, 0.00] | 1.72 [0.00, 5.04] | 0.86 [0.00, 2.52] |
| O | flop | raise | all | 42822 | 100.00 [100.00, 100.00] | -85.06 [-108.50, -61.61] | 510.74 [497.61, 523.86] | 285.70 [279.00, 292.39] | 0.03 [0.01, 0.05] | 0.05 [0.02, 0.07] |
| O | flop | raise | call | 14789 | 34.54 [33.82, 35.26] | -136.37 [-167.94, -104.80] | 562.03 [544.96, 579.09] | 118.45 [116.36, 120.53] | 0.01 [0.00, 0.03] | 0.03 [0.00, 0.05] |
| O | flop | raise | fold | 10042 | 23.45 [22.74, 24.16] | 331.08 [325.16, 337.00] | 331.08 [325.16, 337.00] | 0.00 [0.00, 0.00] | 0.05 [0.01, 0.09] | 0.03 [0.00, 0.07] |
| O | flop | raise | raise | 17991 | 42.01 [41.17, 42.86] | -275.15 [-317.87, -232.43] | 568.85 [545.13, 592.57] | 582.65 [571.67, 593.62] | 0.04 [0.01, 0.07] | 0.07 [0.03, 0.12] |
| O | preflop | jam | all | 2285 | 100.00 [100.00, 100.00] | -311.25 [-414.51, -207.99] | 810.85 [759.19, 862.52] | 564.60 [554.56, 574.63] | 0.04 [0.00, 0.13] | 0.00 [0.00, 0.00] |
| O | preflop | jam | call | 2197 | 96.15 [95.18, 97.12] | -379.61 [-485.64, -273.57] | 787.44 [734.04, 840.84] | 587.21 [578.63, 595.79] | 0.05 [0.00, 0.13] | 0.00 [0.00, 0.00] |
| O | preflop | jam | fold | 88 | 3.85 [2.88, 4.82] | 1395.45 [1349.72, 1441.18] | 1395.45 [1349.72, 1441.18] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| O | preflop | raise | all | 84265 | 100.00 [100.00, 100.00] | -212.25 [-225.61, -198.89] | 386.29 [378.34, 394.24] | 384.08 [378.10, 390.05] | 0.02 [0.00, 0.03] | 0.00 [0.00, 0.01] |
| O | preflop | raise | call | 32781 | 38.90 [38.34, 39.47] | -66.64 [-84.54, -48.73] | 491.08 [480.67, 501.50] | 137.29 [135.64, 138.94] | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.00] |
| O | preflop | raise | fold | 16975 | 20.14 [19.68, 20.61] | 166.66 [162.55, 170.76] | 166.66 [162.55, 170.76] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] | 0.02 [0.00, 0.04] |
| O | preflop | raise | raise | 34509 | 40.95 [40.31, 41.59] | -536.95 [-562.32, -511.58] | 394.79 [379.99, 409.59] | 807.43 [801.21, 813.65] | 0.02 [0.00, 0.05] | 0.00 [0.00, 0.01] |
| O | river | jam | all | 540 | 100.00 [100.00, 100.00] | -346.11 [-554.36, -137.86] | 724.26 [623.03, 825.49] | 260.00 [237.41, 282.59] | 0.93 [0.11, 1.74] | 0.19 [0.00, 0.55] |
| O | river | jam | call | 415 | 76.85 [71.85, 81.86] | -925.30 [-1144.77, -705.83] | 467.47 [357.34, 577.60] | 338.31 [318.36, 358.27] | 0.72 [0.00, 1.54] | 0.00 [0.00, 0.00] |
| O | river | jam | fold | 125 | 23.15 [18.14, 28.15] | 1576.80 [1536.75, 1616.85] | 1576.80 [1536.75, 1616.85] | 0.00 [0.00, 0.00] | 1.60 [0.00, 3.82] | 0.80 [0.00, 2.37] |
| O | river | raise | all | 7519 | 100.00 [100.00, 100.00] | -72.27 [-121.88, -22.67] | 524.79 [498.60, 550.98] | 473.40 [455.26, 491.54] | 0.90 [0.64, 1.17] | 0.77 [0.56, 0.98] |
| O | river | raise | call | 3222 | 42.85 [41.18, 44.53] | 52.20 [18.09, 86.32] | 361.30 [343.67, 378.93] | 100.00 [100.00, 100.00] | 0.40 [0.18, 0.62] | 0.50 [0.25, 0.74] |
| O | river | raise | fold | 725 | 9.64 [8.71, 10.57] | 683.17 [656.12, 710.23] | 683.17 [656.12, 710.23] | 0.00 [0.00, 0.00] | 3.72 [2.23, 5.22] | 1.93 [0.92, 2.94] |
| O | river | raise | raise | 3572 | 47.51 [45.79, 49.22] | -337.88 [-431.70, -244.06] | 640.12 [590.92, 689.32] | 906.30 [880.08, 932.52] | 0.78 [0.42, 1.14] | 0.78 [0.50, 1.07] |
| O | turn | jam | all | 1397 | 100.00 [100.00, 100.00] | -436.01 [-563.44, -308.57] | 707.87 [646.11, 769.64] | 408.59 [388.23, 428.95] | 0.07 [0.00, 0.21] | 0.07 [0.00, 0.21] |
| O | turn | jam | call | 1216 | 87.04 [84.60, 89.49] | -707.24 [-841.35, -573.13] | 606.91 [539.45, 674.37] | 469.41 [450.81, 488.00] | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.24] |
| O | turn | jam | fold | 181 | 12.96 [10.51, 15.40] | 1386.19 [1343.81, 1428.56] | 1386.19 [1343.81, 1428.56] | 0.00 [0.00, 0.00] | 0.55 [0.00, 1.64] | 0.00 [0.00, 0.00] |
| O | turn | raise | all | 17655 | 100.00 [100.00, 100.00] | -127.96 [-166.19, -89.73] | 569.82 [549.39, 590.24] | 397.75 [386.65, 408.85] | 0.35 [0.23, 0.47] | 0.36 [0.25, 0.46] |
| O | turn | raise | call | 7122 | 40.34 [39.22, 41.46] | -94.73 [-137.07, -52.40] | 521.27 [498.70, 543.84] | 103.93 [102.57, 105.29] | 0.27 [0.13, 0.40] | 0.18 [0.08, 0.28] |
| O | turn | raise | fold | 1847 | 10.46 [9.78, 11.14] | 551.38 [535.83, 566.93] | 551.38 [535.83, 566.93] | 0.00 [0.00, 0.00] | 0.97 [0.44, 1.51] | 0.76 [0.36, 1.16] |
| O | turn | raise | raise | 8686 | 49.20 [48.03, 50.36] | -299.65 [-361.70, -237.60] | 613.54 [580.29, 646.78] | 723.24 [706.75, 739.74] | 0.29 [0.14, 0.43] | 0.41 [0.26, 0.57] |
| R | flop | jam | all | 1571 | 100.00 [100.00, 100.00] | -449.65 [-544.45, -354.85] | 703.76 [657.39, 750.12] | 400.70 [386.22, 415.18] | 0.76 [0.30, 1.23] | 0.00 [0.00, 0.00] |
| R | flop | jam | call | 1406 | 89.50 [87.98, 91.02] | -651.49 [-751.51, -551.48] | 637.27 [586.86, 687.67] | 447.72 [433.26, 462.19] | 0.43 [0.09, 0.77] | 0.00 [0.00, 0.00] |
| R | flop | jam | fold | 165 | 10.50 [8.98, 12.02] | 1270.30 [1244.21, 1296.39] | 1270.30 [1244.21, 1296.39] | 0.00 [0.00, 0.00] | 3.64 [0.78, 6.49] | 0.00 [0.00, 0.00] |
| R | flop | raise | all | 49253 | 100.00 [100.00, 100.00] | -143.32 [-160.67, -125.97] | 504.74 [495.08, 514.41] | 253.15 [248.90, 257.40] | 0.45 [0.38, 0.52] | 0.00 [0.00, 0.00] |
| R | flop | raise | call | 21873 | 44.41 [43.90, 44.92] | -139.35 [-159.08, -119.61] | 541.94 [531.14, 552.74] | 119.72 [118.43, 121.02] | 0.27 [0.20, 0.34] | 0.00 [0.00, 0.00] |
| R | flop | raise | fold | 9780 | 19.86 [19.35, 20.36] | 307.66 [303.59, 311.73] | 307.66 [303.59, 311.73] | 0.00 [0.00, 0.00] | 0.63 [0.47, 0.80] | 0.00 [0.00, 0.00] |
| R | flop | raise | raise | 17600 | 35.73 [35.15, 36.32] | -398.86 [-433.02, -364.70] | 568.03 [549.41, 586.64] | 559.64 [552.21, 567.07] | 0.56 [0.43, 0.70] | 0.00 [0.00, 0.00] |
| R | preflop | jam | all | 2393 | 100.00 [100.00, 100.00] | -477.31 [-560.41, -394.21] | 727.87 [685.99, 769.75] | 622.77 [615.53, 630.02] | 0.42 [0.16, 0.68] | 0.00 [0.00, 0.00] |
| R | preflop | jam | call | 2367 | 98.91 [98.50, 99.33] | -498.52 [-582.10, -414.94] | 719.90 [677.70, 762.10] | 629.62 [622.77, 636.46] | 0.21 [0.03, 0.40] | 0.00 [0.00, 0.00] |
| R | preflop | jam | fold | 26 | 1.09 [0.67, 1.50] | 1453.85 [1339.89, 1567.80] | 1453.85 [1339.89, 1567.80] | 0.00 [0.00, 0.00] | 19.23 [4.08, 34.38] | 0.00 [0.00, 0.00] |
| R | preflop | raise | all | 84949 | 100.00 [100.00, 100.00] | -232.68 [-243.48, -221.88] | 390.14 [383.55, 396.74] | 339.64 [335.07, 344.21] | 0.09 [0.06, 0.12] | 0.00 [0.00, 0.00] |
| R | preflop | raise | call | 38704 | 45.56 [45.09, 46.03] | -91.90 [-105.16, -78.64] | 484.71 [477.01, 492.42] | 136.62 [135.56, 137.67] | 0.04 [0.02, 0.07] | 0.00 [0.00, 0.00] |
| R | preflop | raise | fold | 15002 | 17.66 [17.30, 18.02] | 143.35 [140.99, 145.72] | 143.35 [140.99, 145.72] | 0.00 [0.00, 0.00] | 0.12 [0.06, 0.18] | 0.00 [0.00, 0.00] |
| R | preflop | raise | raise | 31243 | 36.78 [36.23, 37.33] | -587.64 [-609.75, -565.53] | 391.49 [378.26, 404.71] | 754.24 [749.48, 758.99] | 0.12 [0.07, 0.18] | 0.00 [0.00, 0.00] |
| R | river | jam | all | 797 | 100.00 [100.00, 100.00] | -575.91 [-706.55, -445.27] | 656.21 [590.96, 721.46] | 286.70 [272.40, 301.00] | 2.38 [1.33, 3.44] | 0.00 [0.00, 0.00] |
| R | river | jam | call | 717 | 89.96 [87.84, 92.09] | -842.40 [-973.79, -711.01] | 527.20 [461.21, 593.18] | 318.69 [304.87, 332.51] | 1.39 [0.54, 2.25] | 0.00 [0.00, 0.00] |
| R | river | jam | fold | 80 | 10.04 [7.91, 12.16] | 1812.50 [1792.22, 1832.78] | 1812.50 [1792.22, 1832.78] | 0.00 [0.00, 0.00] | 11.25 [4.31, 18.19] | 0.00 [0.00, 0.00] |
| R | river | raise | all | 12569 | 100.00 [100.00, 100.00] | -57.13 [-88.41, -25.86] | 530.94 [514.03, 547.85] | 457.49 [446.50, 468.48] | 3.38 [2.98, 3.79] | 0.00 [0.00, 0.00] |
| R | river | raise | call | 5320 | 42.33 [41.26, 43.40] | 50.62 [29.77, 71.47] | 347.05 [336.10, 357.99] | 100.00 [100.00, 100.00] | 2.07 [1.69, 2.45] | 0.00 [0.00, 0.00] |
| R | river | raise | fold | 1286 | 10.23 [9.62, 10.85] | 644.01 [626.32, 661.71] | 644.01 [626.32, 661.71] | 0.00 [0.00, 0.00] | 8.24 [6.67, 9.82] | 0.00 [0.00, 0.00] |
| R | river | raise | raise | 5963 | 47.44 [46.33, 48.55] | -304.48 [-364.60, -244.36] | 670.62 [638.63, 702.61] | 875.10 [858.97, 891.23] | 3.50 [2.91, 4.10] | 0.00 [0.00, 0.00] |
| R | turn | jam | all | 1333 | 100.00 [100.00, 100.00] | -496.40 [-598.59, -394.21] | 691.90 [641.17, 742.63] | 473.52 [457.86, 489.18] | 3.15 [2.21, 4.09] | 0.00 [0.00, 0.00] |
| R | turn | jam | call | 1227 | 92.05 [90.58, 93.51] | -671.56 [-776.35, -566.76] | 619.40 [566.56, 672.24] | 514.43 [499.44, 529.42] | 1.55 [0.86, 2.24] | 0.00 [0.00, 0.00] |
| R | turn | jam | fold | 106 | 7.95 [6.49, 9.42] | 1531.13 [1481.41, 1580.86] | 1531.13 [1481.41, 1580.86] | 0.00 [0.00, 0.00] | 21.70 [13.83, 29.57] | 0.00 [0.00, 0.00] |
| R | turn | raise | all | 26434 | 100.00 [100.00, 100.00] | -124.57 [-149.49, -99.64] | 565.07 [551.66, 578.49] | 340.71 [334.48, 346.93] | 2.30 [2.07, 2.53] | 0.00 [0.00, 0.00] |
| R | turn | raise | call | 11790 | 44.60 [43.86, 45.34] | -98.73 [-124.24, -73.21] | 509.71 [496.05, 523.37] | 102.76 [102.09, 103.42] | 1.28 [1.07, 1.49] | 0.00 [0.00, 0.00] |
| R | turn | raise | fold | 2760 | 10.44 [9.99, 10.89] | 523.95 [513.07, 534.83] | 523.95 [513.07, 534.83] | 0.00 [0.00, 0.00] | 5.54 [4.68, 6.41] | 0.00 [0.00, 0.00] |
| R | turn | raise | raise | 11884 | 44.96 [44.21, 45.71] | -300.82 [-344.05, -257.58] | 629.54 [606.37, 652.72] | 655.90 [645.54, 666.26] | 2.56 [2.22, 2.90] | 0.00 [0.00, 0.00] |

**Disjoint last-response partitions.** Every hand occurs once; return contributions sum to each arm's absolute native-pressure return. This cannot assign a causal action or street EV.

| Last response | R hands | O hands | R net chips lost / exposed hand [95%] | O net chips lost / exposed hand [95%] | O − R return contribution BB/100 [95%] |
| --- | --- | --- | --- | --- | --- |
| flop/jam/call | 1406 | 1316 | -651.49 [-751.51, -551.48] | -565.35 [-695.67, -435.03] | -2.33 [-5.37, 0.70] |
| flop/jam/fold | 165 | 116 | 1270.30 [1244.21, 1296.39] | 1342.24 [1283.81, 1400.68] | 0.73 [0.05, 1.42] |
| flop/raise/call | 1201 | 1043 | -390.59 [-495.04, -286.15] | -427.90 [-569.46, -286.34] | -0.31 [-2.80, 2.18] |
| flop/raise/fold | 9780 | 10042 | 307.66 [303.59, 311.73] | 331.08 [325.16, 337.00] | -4.28 [-5.78, -2.79] |
| flop/raise/raise | 4386 | 4916 | -533.45 [-582.08, -484.82] | -463.87 [-523.38, -404.36] | -0.80 [-5.42, 3.81] |
| no-facing-response | 8229 | 9869 | 50.00 [50.00, 50.00] | 50.00 [50.00, 50.00] | -1.11 [-1.24, -0.99] |
| preflop/jam/call | 2367 | 2197 | -498.52 [-582.10, -414.94] | -379.61 [-485.64, -273.57] | -4.69 [-8.46, -0.92] |
| preflop/jam/fold | 26 | 88 | 1453.85 [1339.89, 1567.80] | 1395.45 [1349.72, 1441.18] | -1.15 [-1.63, -0.68] |
| preflop/raise/call | 1094 | 1602 | -431.26 [-528.22, -334.31] | -389.26 [-501.36, -277.16] | 2.06 [-0.50, 4.62] |
| preflop/raise/fold | 15002 | 16975 | 143.35 [140.99, 145.72] | 166.66 [162.55, 170.76] | -9.20 [-10.55, -7.85] |
| preflop/raise/raise | 9400 | 11818 | -760.60 [-784.58, -736.61] | -679.57 [-705.81, -653.34] | 11.96 [7.42, 16.50] |
| river/jam/call | 717 | 415 | -842.40 [-973.79, -711.01] | -925.30 [-1144.77, -705.83] | -2.98 [-4.86, -1.11] |
| river/jam/fold | 80 | 125 | 1812.50 [1792.22, 1832.78] | 1576.80 [1536.75, 1616.85] | -0.71 [-1.48, 0.06] |
| river/raise/call | 5320 | 3222 | 50.62 [29.77, 71.47] | 52.20 [18.09, 86.32] | 1.37 [-0.41, 3.15] |
| river/raise/fold | 1286 | 725 | 644.01 [626.32, 661.71] | 683.17 [656.12, 710.23] | 4.52 [3.60, 5.43] |
| river/raise/raise | 3484 | 2017 | -409.67 [-471.38, -347.96] | -470.30 [-573.98, -366.62] | -6.49 [-10.21, -2.77] |
| turn/jam/call | 1227 | 1216 | -671.56 [-776.35, -566.76] | -707.24 [-841.35, -573.13] | 0.49 [-2.35, 3.33] |
| turn/jam/fold | 106 | 181 | 1531.13 [1481.41, 1580.86] | 1386.19 [1343.81, 1428.56] | -1.20 [-1.99, -0.41] |
| turn/raise/call | 903 | 618 | -415.50 [-539.37, -291.64] | -327.99 [-516.90, -139.09] | -2.34 [-4.46, -0.22] |
| turn/raise/fold | 2760 | 1847 | 523.95 [513.07, 534.83] | 551.38 [535.83, 566.93] | 5.80 [4.62, 6.99] |
| turn/raise/raise | 4789 | 3380 | -459.22 [-507.45, -410.98] | -386.12 [-460.34, -311.91] | -12.13 [-16.42, -7.84] |

## O-R: all panel, lineage and position contrasts

| Panel | Lineage | Position | Blocks | BB/100 [95%] |
| --- | --- | --- | --- | --- |
| uniform | all three | both | 256 | 40.43 [-6.39, 87.25] |
| uniform | all three | button | 256 | 85.42 [14.93, 155.90] |
| uniform | all three | big_blind | 256 | -4.56 [-64.90, 55.78] |
| uniform | 1 | both | 256 | 50.29 [-16.17, 116.75] |
| uniform | 1 | button | 256 | 70.51 [-30.32, 171.34] |
| uniform | 1 | big_blind | 256 | 30.08 [-54.93, 115.09] |
| uniform | 2 | both | 256 | 50.98 [-16.95, 118.90] |
| uniform | 2 | button | 256 | 99.22 [0.86, 197.57] |
| uniform | 2 | big_blind | 256 | 2.73 [-81.37, 86.84] |
| uniform | 3 | both | 256 | 20.02 [-46.06, 86.10] |
| uniform | 3 | button | 256 | 86.52 [-13.38, 186.43] |
| uniform | 3 | big_blind | 256 | -46.48 [-131.32, 38.35] |
| passive | all three | both | 256 | 49.87 [14.47, 85.27] |
| passive | all three | button | 256 | 47.14 [-14.72, 108.99] |
| passive | all three | big_blind | 256 | 52.60 [-10.26, 115.47] |
| passive | 1 | both | 256 | 51.66 [-0.05, 103.37] |
| passive | 1 | button | 256 | 24.02 [-59.03, 107.08] |
| passive | 1 | big_blind | 256 | 79.30 [1.87, 156.72] |
| passive | 2 | both | 256 | 24.61 [-27.83, 77.05] |
| passive | 2 | button | 256 | 41.41 [-36.42, 119.24] |
| passive | 2 | big_blind | 256 | 7.81 [-71.59, 87.21] |
| passive | 3 | both | 256 | 73.34 [22.37, 124.31] |
| passive | 3 | button | 256 | 75.98 [-12.46, 164.42] |
| passive | 3 | big_blind | 256 | 70.70 [-8.96, 150.37] |
| minraise-cap2 | all three | both | 256 | -8.20 [-65.83, 49.42] |
| minraise-cap2 | all three | button | 256 | -41.02 [-124.17, 42.14] |
| minraise-cap2 | all three | big_blind | 256 | 24.61 [-62.67, 111.89] |
| minraise-cap2 | 1 | both | 256 | -25.29 [-98.01, 47.42] |
| minraise-cap2 | 1 | button | 256 | -59.18 [-167.25, 48.89] |
| minraise-cap2 | 1 | big_blind | 256 | 8.59 [-108.01, 125.20] |
| minraise-cap2 | 2 | both | 256 | -15.62 [-96.80, 65.55] |
| minraise-cap2 | 2 | button | 256 | -50.00 [-172.33, 72.33] |
| minraise-cap2 | 2 | big_blind | 256 | 18.75 [-93.02, 130.52] |
| minraise-cap2 | 3 | both | 256 | 16.31 [-61.71, 94.33] |
| minraise-cap2 | 3 | button | 256 | -13.87 [-126.22, 98.49] |
| minraise-cap2 | 3 | big_blind | 256 | 46.48 [-61.96, 154.93] |
| pressure-cap2 | all three | both | 256 | 9.37 [-37.82, 56.57] |
| pressure-cap2 | all three | button | 256 | 12.37 [-48.10, 72.84] |
| pressure-cap2 | all three | big_blind | 256 | 6.38 [-62.37, 75.13] |
| pressure-cap2 | 1 | both | 256 | 38.77 [-26.85, 104.39] |
| pressure-cap2 | 1 | button | 256 | 60.35 [-17.61, 138.32] |
| pressure-cap2 | 1 | big_blind | 256 | 17.19 [-79.58, 113.96] |
| pressure-cap2 | 2 | both | 256 | -0.59 [-65.27, 64.10] |
| pressure-cap2 | 2 | button | 256 | 7.42 [-82.11, 96.95] |
| pressure-cap2 | 2 | big_blind | 256 | -8.59 [-98.50, 81.31] |
| pressure-cap2 | 3 | both | 256 | -10.06 [-71.37, 51.25] |
| pressure-cap2 | 3 | button | 256 | -30.66 [-121.39, 60.06] |
| pressure-cap2 | 3 | big_blind | 256 | 10.55 [-79.63, 100.72] |
| tight_passive | all three | both | 256 | -1.37 [-8.11, 5.37] |
| tight_passive | all three | button | 256 | -10.29 [-22.29, 1.71] |
| tight_passive | all three | big_blind | 256 | 7.55 [-0.20, 15.31] |
| tight_passive | 1 | both | 256 | -3.61 [-13.83, 6.60] |
| tight_passive | 1 | button | 256 | -13.87 [-32.38, 4.64] |
| tight_passive | 1 | big_blind | 256 | 6.64 [-1.73, 15.01] |
| tight_passive | 2 | both | 256 | -0.98 [-11.93, 9.98] |
| tight_passive | 2 | button | 256 | -15.23 [-30.26, -0.21] |
| tight_passive | 2 | big_blind | 256 | 13.28 [-2.47, 29.03] |
| tight_passive | 3 | both | 256 | 0.49 [-10.62, 11.59] |
| tight_passive | 3 | button | 256 | -1.76 [-22.74, 19.23] |
| tight_passive | 3 | big_blind | 256 | 2.73 [-4.53, 10.00] |
| loose_passive | all three | both | 256 | 19.27 [-5.25, 43.80] |
| loose_passive | all three | button | 256 | 19.66 [-19.85, 59.17] |
| loose_passive | all three | big_blind | 256 | 18.88 [-21.04, 58.80] |
| loose_passive | 1 | both | 256 | 18.85 [-14.96, 52.65] |
| loose_passive | 1 | button | 256 | 8.79 [-48.65, 66.22] |
| loose_passive | 1 | big_blind | 256 | 28.91 [-17.77, 75.58] |
| loose_passive | 2 | both | 256 | 3.91 [-31.23, 39.04] |
| loose_passive | 2 | button | 256 | 1.56 [-50.85, 53.97] |
| loose_passive | 2 | big_blind | 256 | 6.25 [-40.74, 53.24] |
| loose_passive | 3 | both | 256 | 35.06 [3.14, 66.98] |
| loose_passive | 3 | button | 256 | 48.63 [-0.60, 97.87] |
| loose_passive | 3 | big_blind | 256 | 21.48 [-29.15, 72.12] |
| tight_aggressive | all three | both | 256 | -8.46 [-28.41, 11.49] |
| tight_aggressive | all three | button | 256 | -17.84 [-48.66, 12.99] |
| tight_aggressive | all three | big_blind | 256 | 0.91 [-21.83, 23.66] |
| tight_aggressive | 1 | both | 256 | -10.25 [-37.98, 17.47] |
| tight_aggressive | 1 | button | 256 | -15.43 [-53.72, 22.86] |
| tight_aggressive | 1 | big_blind | 256 | -5.08 [-39.06, 28.91] |
| tight_aggressive | 2 | both | 256 | -20.90 [-42.49, 0.70] |
| tight_aggressive | 2 | button | 256 | -43.75 [-82.43, -5.07] |
| tight_aggressive | 2 | big_blind | 256 | 1.95 [-17.07, 20.97] |
| tight_aggressive | 3 | both | 256 | 5.76 [-21.36, 32.89] |
| tight_aggressive | 3 | button | 256 | 5.66 [-38.10, 49.42] |
| tight_aggressive | 3 | big_blind | 256 | 5.86 [-25.92, 37.64] |
| loose_aggressive | all three | both | 256 | 14.19 [-26.58, 54.96] |
| loose_aggressive | all three | button | 256 | 57.16 [-3.64, 117.97] |
| loose_aggressive | all three | big_blind | 256 | -28.78 [-80.72, 23.16] |
| loose_aggressive | 1 | both | 256 | 21.78 [-32.14, 75.69] |
| loose_aggressive | 1 | button | 256 | 55.27 [-20.32, 130.86] |
| loose_aggressive | 1 | big_blind | 256 | -11.72 [-80.57, 57.13] |
| loose_aggressive | 2 | both | 256 | -9.38 [-54.68, 35.93] |
| loose_aggressive | 2 | button | 256 | 17.58 [-56.34, 91.49] |
| loose_aggressive | 2 | big_blind | 256 | -36.33 [-85.86, 13.20] |
| loose_aggressive | 3 | both | 256 | 30.18 [-23.48, 83.83] |
| loose_aggressive | 3 | button | 256 | 98.63 [20.21, 177.06] |
| loose_aggressive | 3 | big_blind | 256 | -38.28 [-110.57, 34.01] |
| pot_pressure | all three | both | 256 | -0.65 [-25.93, 24.63] |
| pot_pressure | all three | button | 256 | -13.93 [-61.77, 33.90] |
| pot_pressure | all three | big_blind | 256 | 12.63 [-4.07, 29.33] |
| pot_pressure | 1 | both | 256 | 7.71 [-27.10, 42.53] |
| pot_pressure | 1 | button | 256 | 16.60 [-46.50, 79.70] |
| pot_pressure | 1 | big_blind | 256 | -1.17 [-30.37, 28.03] |
| pot_pressure | 2 | both | 256 | -6.45 [-39.80, 26.91] |
| pot_pressure | 2 | button | 256 | -23.83 [-89.56, 41.90] |
| pot_pressure | 2 | big_blind | 256 | 10.94 [0.29, 21.59] |
| pot_pressure | 3 | both | 256 | -3.22 [-36.14, 29.70] |
| pot_pressure | 3 | button | 256 | -34.57 [-92.80, 23.66] |
| pot_pressure | 3 | big_blind | 256 | 28.12 [-3.01, 59.26] |
| train_pressure | all three | both | 256 | -2.02 [-32.19, 28.15] |
| train_pressure | all three | button | 256 | 14.84 [-28.57, 58.26] |
| train_pressure | all three | big_blind | 256 | -18.88 [-61.98, 24.22] |
| train_pressure | 1 | both | 256 | 1.07 [-39.09, 41.23] |
| train_pressure | 1 | button | 256 | 14.65 [-48.40, 77.69] |
| train_pressure | 1 | big_blind | 256 | -12.50 [-65.92, 40.92] |
| train_pressure | 2 | both | 256 | -5.08 [-45.96, 35.80] |
| train_pressure | 2 | button | 256 | -2.73 [-63.10, 57.63] |
| train_pressure | 2 | big_blind | 256 | -7.42 [-67.95, 53.11] |
| train_pressure | 3 | both | 256 | -2.05 [-38.34, 34.24] |
| train_pressure | 3 | button | 256 | 32.62 [-19.04, 84.27] |
| train_pressure | 3 | big_blind | 256 | -36.72 [-87.21, 13.77] |
| native-pressure | all three | both | 12288 | -22.82 [-31.83, -13.80] |
| native-pressure | all three | button | 12288 | -12.85 [-25.90, 0.20] |
| native-pressure | all three | big_blind | 12288 | -32.78 [-45.98, -19.58] |
| native-pressure | 1 | both | 12288 | -11.36 [-24.04, 1.32] |
| native-pressure | 1 | button | 12288 | -4.17 [-22.17, 13.83] |
| native-pressure | 1 | big_blind | 12288 | -18.55 [-36.98, -0.11] |
| native-pressure | 2 | both | 12288 | -23.62 [-36.18, -11.05] |
| native-pressure | 2 | button | 12288 | -20.49 [-38.44, -2.54] |
| native-pressure | 2 | big_blind | 12288 | -26.74 [-45.14, -8.34] |
| native-pressure | 3 | both | 12288 | -33.48 [-46.02, -20.93] |
| native-pressure | 3 | button | 12288 | -13.90 [-32.08, 4.29] |
| native-pressure | 3 | big_blind | 12288 | -53.06 [-70.92, -35.20] |
| selective-stackoff | all three | both | 256 | -2.21 [-22.31, 17.88] |
| selective-stackoff | all three | button | 256 | -9.38 [-39.03, 20.28] |
| selective-stackoff | all three | big_blind | 256 | 4.95 [-13.99, 23.88] |
| selective-stackoff | 1 | both | 256 | -12.01 [-34.16, 10.14] |
| selective-stackoff | 1 | button | 256 | -23.63 [-57.86, 10.60] |
| selective-stackoff | 1 | big_blind | 256 | -0.39 [-26.24, 25.46] |
| selective-stackoff | 2 | both | 256 | 3.71 [-23.11, 30.53] |
| selective-stackoff | 2 | button | 256 | -5.08 [-46.99, 36.84] |
| selective-stackoff | 2 | big_blind | 256 | 12.50 [-19.26, 44.26] |
| selective-stackoff | 3 | both | 256 | 1.66 [-24.37, 27.69] |
| selective-stackoff | 3 | button | 256 | 0.59 [-37.12, 38.29] |
| selective-stackoff | 3 | big_blind | 256 | 2.73 [-19.46, 24.92] |
| lbr | all three | both | 2048 | 36.13 [17.55, 54.70] |
| lbr | all three | button | 2048 | 44.85 [16.54, 73.16] |
| lbr | all three | big_blind | 2048 | 27.41 [0.50, 54.32] |
| lbr | 1 | both | 2048 | 42.63 [16.67, 68.59] |
| lbr | 1 | button | 2048 | 57.47 [19.18, 95.76] |
| lbr | 1 | big_blind | 2048 | 27.78 [-8.76, 64.33] |
| lbr | 2 | both | 2048 | 40.34 [15.84, 64.85] |
| lbr | 2 | button | 2048 | 37.33 [0.62, 74.04] |
| lbr | 2 | big_blind | 2048 | 43.36 [8.25, 78.47] |
| lbr | 3 | both | 2048 | 25.42 [-0.26, 51.09] |
| lbr | 3 | button | 2048 | 39.75 [1.81, 77.68] |
| lbr | 3 | big_blind | 2048 | 11.08 [-25.53, 47.70] |

## C-R: all panel, lineage and position contrasts

| Panel | Lineage | Position | Blocks | BB/100 [95%] |
| --- | --- | --- | --- | --- |
| uniform | all three | both | 256 | 24.48 [-19.08, 68.04] |
| uniform | all three | button | 256 | 78.39 [13.59, 143.18] |
| uniform | all three | big_blind | 256 | -29.43 [-82.64, 23.79] |
| uniform | 1 | both | 256 | 29.59 [-36.24, 95.42] |
| uniform | 1 | button | 256 | 54.49 [-45.95, 154.93] |
| uniform | 1 | big_blind | 256 | 4.69 [-80.79, 90.17] |
| uniform | 2 | both | 256 | 5.86 [-66.41, 78.13] |
| uniform | 2 | button | 256 | 80.47 [-20.76, 181.69] |
| uniform | 2 | big_blind | 256 | -68.75 [-162.04, 24.54] |
| uniform | 3 | both | 256 | 37.99 [-32.91, 108.89] |
| uniform | 3 | button | 256 | 100.20 [-0.24, 200.63] |
| uniform | 3 | big_blind | 256 | -24.22 [-113.38, 64.94] |
| passive | all three | both | 256 | 42.84 [10.17, 75.51] |
| passive | all three | button | 256 | 38.80 [-18.58, 96.19] |
| passive | all three | big_blind | 256 | 46.88 [-5.65, 99.40] |
| passive | 1 | both | 256 | 49.71 [-2.60, 102.01] |
| passive | 1 | button | 256 | 28.71 [-55.20, 112.62] |
| passive | 1 | big_blind | 256 | 70.70 [-4.06, 145.47] |
| passive | 2 | both | 256 | -3.12 [-56.53, 50.28] |
| passive | 2 | button | 256 | 4.30 [-74.91, 83.50] |
| passive | 2 | big_blind | 256 | -10.55 [-86.83, 65.73] |
| passive | 3 | both | 256 | 81.93 [29.75, 134.12] |
| passive | 3 | button | 256 | 83.40 [-7.40, 174.20] |
| passive | 3 | big_blind | 256 | 80.47 [4.60, 156.34] |
| minraise-cap2 | all three | both | 256 | 14.13 [-37.39, 65.64] |
| minraise-cap2 | all three | button | 256 | 24.61 [-42.95, 92.17] |
| minraise-cap2 | all three | big_blind | 256 | 3.65 [-74.99, 82.29] |
| minraise-cap2 | 1 | both | 256 | -20.02 [-100.04, 60.00] |
| minraise-cap2 | 1 | button | 256 | 13.48 [-88.14, 115.09] |
| minraise-cap2 | 1 | big_blind | 256 | -53.52 [-181.77, 74.74] |
| minraise-cap2 | 2 | both | 256 | 30.86 [-45.49, 107.21] |
| minraise-cap2 | 2 | button | 256 | 33.20 [-75.55, 141.96] |
| minraise-cap2 | 2 | big_blind | 256 | 28.52 [-81.95, 138.98] |
| minraise-cap2 | 3 | both | 256 | 31.54 [-50.27, 113.36] |
| minraise-cap2 | 3 | button | 256 | 27.15 [-84.01, 138.31] |
| minraise-cap2 | 3 | big_blind | 256 | 35.94 [-73.42, 145.29] |
| pressure-cap2 | all three | both | 256 | 14.45 [-27.57, 56.48] |
| pressure-cap2 | all three | button | 256 | 8.46 [-45.17, 62.10] |
| pressure-cap2 | all three | big_blind | 256 | 20.44 [-39.69, 80.58] |
| pressure-cap2 | 1 | both | 256 | 15.14 [-48.87, 79.14] |
| pressure-cap2 | 1 | button | 256 | 52.54 [-26.57, 131.65] |
| pressure-cap2 | 1 | big_blind | 256 | -22.27 [-117.04, 72.51] |
| pressure-cap2 | 2 | both | 256 | 18.75 [-45.67, 83.17] |
| pressure-cap2 | 2 | button | 256 | -10.94 [-96.09, 74.21] |
| pressure-cap2 | 2 | big_blind | 256 | 48.44 [-38.66, 135.53] |
| pressure-cap2 | 3 | both | 256 | 9.47 [-52.79, 71.73] |
| pressure-cap2 | 3 | button | 256 | -16.21 [-103.70, 71.28] |
| pressure-cap2 | 3 | big_blind | 256 | 35.16 [-58.00, 128.32] |
| tight_passive | all three | both | 256 | 0.39 [-8.19, 8.97] |
| tight_passive | all three | button | 256 | -7.42 [-21.62, 6.78] |
| tight_passive | all three | big_blind | 256 | 8.20 [-0.83, 17.23] |
| tight_passive | 1 | both | 256 | -3.81 [-15.60, 7.98] |
| tight_passive | 1 | button | 256 | -15.04 [-33.66, 3.59] |
| tight_passive | 1 | big_blind | 256 | 7.42 [-8.22, 23.06] |
| tight_passive | 2 | both | 256 | 5.47 [-7.51, 18.44] |
| tight_passive | 2 | button | 256 | -2.73 [-21.98, 16.51] |
| tight_passive | 2 | big_blind | 256 | 13.67 [-1.86, 29.21] |
| tight_passive | 3 | both | 256 | -0.49 [-13.37, 12.40] |
| tight_passive | 3 | button | 256 | -4.49 [-29.18, 20.20] |
| tight_passive | 3 | big_blind | 256 | 3.52 [-3.82, 10.86] |
| loose_passive | all three | both | 256 | 13.87 [-9.82, 37.55] |
| loose_passive | all three | button | 256 | 9.24 [-26.85, 45.34] |
| loose_passive | all three | big_blind | 256 | 18.49 [-17.86, 54.84] |
| loose_passive | 1 | both | 256 | 5.96 [-31.77, 43.69] |
| loose_passive | 1 | button | 256 | -13.09 [-73.59, 47.42] |
| loose_passive | 1 | big_blind | 256 | 25.00 [-24.04, 74.04] |
| loose_passive | 2 | both | 256 | -4.10 [-37.47, 29.26] |
| loose_passive | 2 | button | 256 | -10.16 [-60.83, 40.52] |
| loose_passive | 2 | big_blind | 256 | 1.95 [-41.44, 45.34] |
| loose_passive | 3 | both | 256 | 39.75 [5.10, 74.40] |
| loose_passive | 3 | button | 256 | 50.98 [-1.98, 103.93] |
| loose_passive | 3 | big_blind | 256 | 28.52 [-21.32, 78.35] |
| tight_aggressive | all three | both | 256 | -10.81 [-24.22, 2.60] |
| tight_aggressive | all three | button | 256 | -24.09 [-47.45, -0.73] |
| tight_aggressive | all three | big_blind | 256 | 2.47 [-14.53, 19.48] |
| tight_aggressive | 1 | both | 256 | -17.68 [-36.46, 1.11] |
| tight_aggressive | 1 | button | 256 | -38.48 [-68.53, -8.43] |
| tight_aggressive | 1 | big_blind | 256 | 3.12 [-19.32, 25.57] |
| tight_aggressive | 2 | both | 256 | -25.78 [-45.04, -6.52] |
| tight_aggressive | 2 | button | 256 | -46.88 [-79.53, -14.22] |
| tight_aggressive | 2 | big_blind | 256 | -4.69 [-32.58, 23.20] |
| tight_aggressive | 3 | both | 256 | 11.04 [-14.90, 36.97] |
| tight_aggressive | 3 | button | 256 | 13.09 [-28.06, 54.24] |
| tight_aggressive | 3 | big_blind | 256 | 8.98 [-22.14, 40.11] |
| loose_aggressive | all three | both | 256 | -2.54 [-35.32, 30.25] |
| loose_aggressive | all three | button | 256 | -6.25 [-56.96, 44.46] |
| loose_aggressive | all three | big_blind | 256 | 1.17 [-44.49, 46.84] |
| loose_aggressive | 1 | both | 256 | -4.00 [-51.23, 43.22] |
| loose_aggressive | 1 | button | 256 | -19.34 [-90.93, 52.26] |
| loose_aggressive | 1 | big_blind | 256 | 11.33 [-52.51, 75.17] |
| loose_aggressive | 2 | both | 256 | -26.56 [-70.17, 17.05] |
| loose_aggressive | 2 | button | 256 | -44.92 [-115.94, 26.09] |
| loose_aggressive | 2 | big_blind | 256 | -8.20 [-67.38, 50.98] |
| loose_aggressive | 3 | both | 256 | 22.95 [-27.45, 73.35] |
| loose_aggressive | 3 | button | 256 | 45.51 [-29.02, 120.04] |
| loose_aggressive | 3 | big_blind | 256 | 0.39 [-66.63, 67.41] |
| pot_pressure | all three | both | 256 | 8.53 [-11.81, 28.87] |
| pot_pressure | all three | button | 256 | 0.65 [-36.75, 38.05] |
| pot_pressure | all three | big_blind | 256 | 16.41 [-0.29, 33.10] |
| pot_pressure | 1 | both | 256 | 16.89 [-20.43, 54.22] |
| pot_pressure | 1 | button | 256 | 31.05 [-36.68, 98.79] |
| pot_pressure | 1 | big_blind | 256 | 2.73 [-27.15, 32.61] |
| pot_pressure | 2 | both | 256 | 13.09 [-20.18, 46.35] |
| pot_pressure | 2 | button | 256 | 8.20 [-56.33, 72.74] |
| pot_pressure | 2 | big_blind | 256 | 17.97 [1.66, 34.28] |
| pot_pressure | 3 | both | 256 | -4.39 [-33.34, 24.55] |
| pot_pressure | 3 | button | 256 | -37.30 [-86.38, 11.77] |
| pot_pressure | 3 | big_blind | 256 | 28.52 [-2.76, 59.79] |
| train_pressure | all three | both | 256 | 2.41 [-26.34, 31.16] |
| train_pressure | all three | button | 256 | 17.06 [-23.01, 57.13] |
| train_pressure | all three | big_blind | 256 | -12.24 [-51.15, 26.67] |
| train_pressure | 1 | both | 256 | 8.69 [-33.78, 51.16] |
| train_pressure | 1 | button | 256 | 22.46 [-41.44, 86.36] |
| train_pressure | 1 | big_blind | 256 | -5.08 [-62.76, 52.60] |
| train_pressure | 2 | both | 256 | -9.38 [-52.42, 33.67] |
| train_pressure | 2 | button | 256 | -8.98 [-71.49, 53.52] |
| train_pressure | 2 | big_blind | 256 | -9.77 [-65.20, 45.67] |
| train_pressure | 3 | both | 256 | 7.91 [-25.98, 41.81] |
| train_pressure | 3 | button | 256 | 37.70 [-11.55, 86.94] |
| train_pressure | 3 | big_blind | 256 | -21.88 [-65.65, 21.90] |
| native-pressure | all three | both | 12288 | -19.70 [-27.69, -11.70] |
| native-pressure | all three | button | 12288 | -12.59 [-24.14, -1.04] |
| native-pressure | all three | big_blind | 12288 | -26.80 [-38.27, -15.33] |
| native-pressure | 1 | both | 12288 | -2.10 [-14.75, 10.56] |
| native-pressure | 1 | button | 12288 | 2.94 [-14.80, 20.68] |
| native-pressure | 1 | big_blind | 12288 | -7.13 [-25.44, 11.18] |
| native-pressure | 2 | both | 12288 | -24.49 [-37.12, -11.86] |
| native-pressure | 2 | button | 12288 | -16.64 [-34.68, 1.40] |
| native-pressure | 2 | big_blind | 12288 | -32.34 [-50.32, -14.37] |
| native-pressure | 3 | both | 12288 | -32.51 [-44.89, -20.12] |
| native-pressure | 3 | button | 12288 | -24.08 [-41.75, -6.41] |
| native-pressure | 3 | big_blind | 12288 | -40.93 [-58.64, -23.23] |
| selective-stackoff | all three | both | 256 | -4.17 [-22.13, 13.80] |
| selective-stackoff | all three | button | 256 | -7.55 [-33.54, 18.44] |
| selective-stackoff | all three | big_blind | 256 | -0.78 [-19.77, 18.21] |
| selective-stackoff | 1 | both | 256 | -14.55 [-35.66, 6.56] |
| selective-stackoff | 1 | button | 256 | -27.54 [-64.41, 9.33] |
| selective-stackoff | 1 | big_blind | 256 | -1.56 [-23.34, 20.21] |
| selective-stackoff | 2 | both | 256 | 8.98 [-15.15, 33.12] |
| selective-stackoff | 2 | button | 256 | 12.50 [-21.56, 46.56] |
| selective-stackoff | 2 | big_blind | 256 | 5.47 [-27.50, 38.43] |
| selective-stackoff | 3 | both | 256 | -6.93 [-32.51, 18.64] |
| selective-stackoff | 3 | button | 256 | -7.62 [-44.32, 29.08] |
| selective-stackoff | 3 | big_blind | 256 | -6.25 [-31.50, 19.00] |
| lbr | all three | both | 2048 | 25.68 [9.49, 41.88] |
| lbr | all three | button | 2048 | 34.47 [10.47, 58.48] |
| lbr | all three | big_blind | 2048 | 16.89 [-6.00, 39.79] |
| lbr | 1 | both | 2048 | 20.34 [-5.72, 46.40] |
| lbr | 1 | button | 2048 | 35.50 [-1.34, 72.33] |
| lbr | 1 | big_blind | 2048 | 5.18 [-30.76, 41.11] |
| lbr | 2 | both | 2048 | 41.76 [17.39, 66.13] |
| lbr | 2 | button | 2048 | 46.31 [10.03, 82.60] |
| lbr | 2 | big_blind | 2048 | 37.21 [2.84, 71.57] |
| lbr | 3 | both | 2048 | 14.95 [-10.04, 39.95] |
| lbr | 3 | button | 2048 | 21.61 [-14.64, 57.85] |
| lbr | 3 | big_blind | 2048 | 8.30 [-27.94, 44.54] |

## O-C: all panel, lineage and position contrasts

| Panel | Lineage | Position | Blocks | BB/100 [95%] |
| --- | --- | --- | --- | --- |
| uniform | all three | both | 256 | 15.95 [-17.39, 49.30] |
| uniform | all three | button | 256 | 7.03 [-42.68, 56.75] |
| uniform | all three | big_blind | 256 | 24.87 [-20.88, 70.62] |
| uniform | 1 | both | 256 | 20.70 [-32.52, 73.92] |
| uniform | 1 | button | 256 | 16.02 [-73.31, 105.34] |
| uniform | 1 | big_blind | 256 | 25.39 [-29.29, 80.07] |
| uniform | 2 | both | 256 | 45.12 [-2.56, 92.80] |
| uniform | 2 | button | 256 | 18.75 [-37.71, 75.21] |
| uniform | 2 | big_blind | 256 | 71.48 [-0.76, 143.72] |
| uniform | 3 | both | 256 | -17.97 [-69.77, 33.83] |
| uniform | 3 | button | 256 | -13.67 [-95.82, 68.47] |
| uniform | 3 | big_blind | 256 | -22.27 [-93.98, 49.45] |
| passive | all three | both | 256 | 7.03 [-15.39, 29.46] |
| passive | all three | button | 256 | 8.33 [-27.56, 44.23] |
| passive | all three | big_blind | 256 | 5.73 [-24.83, 36.29] |
| passive | 1 | both | 256 | 1.95 [-33.17, 37.08] |
| passive | 1 | button | 256 | -4.69 [-55.28, 45.90] |
| passive | 1 | big_blind | 256 | 8.59 [-40.28, 57.46] |
| passive | 2 | both | 256 | 27.73 [-3.42, 58.89] |
| passive | 2 | button | 256 | 37.11 [-19.30, 93.52] |
| passive | 2 | big_blind | 256 | 18.36 [-26.24, 62.96] |
| passive | 3 | both | 256 | -8.59 [-48.66, 31.47] |
| passive | 3 | button | 256 | -7.42 [-65.51, 50.67] |
| passive | 3 | big_blind | 256 | -9.77 [-59.33, 39.80] |
| minraise-cap2 | all three | both | 256 | -22.33 [-58.91, 14.25] |
| minraise-cap2 | all three | button | 256 | -65.62 [-120.39, -10.86] |
| minraise-cap2 | all three | big_blind | 256 | 20.96 [-30.26, 72.19] |
| minraise-cap2 | 1 | both | 256 | -5.27 [-60.77, 50.22] |
| minraise-cap2 | 1 | button | 256 | -72.66 [-154.71, 9.40] |
| minraise-cap2 | 1 | big_blind | 256 | 62.11 [-16.62, 140.84] |
| minraise-cap2 | 2 | both | 256 | -46.48 [-102.51, 9.54] |
| minraise-cap2 | 2 | button | 256 | -83.20 [-165.12, -1.29] |
| minraise-cap2 | 2 | big_blind | 256 | -9.77 [-89.55, 70.02] |
| minraise-cap2 | 3 | both | 256 | -15.23 [-75.15, 44.68] |
| minraise-cap2 | 3 | button | 256 | -41.02 [-130.00, 47.97] |
| minraise-cap2 | 3 | big_blind | 256 | 10.55 [-69.57, 90.66] |
| pressure-cap2 | all three | both | 256 | -5.08 [-33.45, 23.29] |
| pressure-cap2 | all three | button | 256 | 3.91 [-36.02, 43.83] |
| pressure-cap2 | all three | big_blind | 256 | -14.06 [-56.24, 28.11] |
| pressure-cap2 | 1 | both | 256 | 23.63 [-22.02, 69.28] |
| pressure-cap2 | 1 | button | 256 | 7.81 [-57.01, 72.63] |
| pressure-cap2 | 1 | big_blind | 256 | 39.45 [-21.00, 99.90] |
| pressure-cap2 | 2 | both | 256 | -19.34 [-62.66, 23.99] |
| pressure-cap2 | 2 | button | 256 | 18.36 [-38.57, 75.29] |
| pressure-cap2 | 2 | big_blind | 256 | -57.03 [-122.93, 8.86] |
| pressure-cap2 | 3 | both | 256 | -19.53 [-68.04, 28.98] |
| pressure-cap2 | 3 | button | 256 | -14.45 [-88.87, 59.96] |
| pressure-cap2 | 3 | big_blind | 256 | -24.61 [-89.53, 40.32] |
| tight_passive | all three | both | 256 | -1.76 [-8.00, 4.48] |
| tight_passive | all three | button | 256 | -2.86 [-12.41, 6.68] |
| tight_passive | all three | big_blind | 256 | -0.65 [-5.80, 4.49] |
| tight_passive | 1 | both | 256 | 0.20 [-9.57, 9.96] |
| tight_passive | 1 | button | 256 | 1.17 [-10.45, 12.80] |
| tight_passive | 1 | big_blind | 256 | -0.78 [-16.46, 14.90] |
| tight_passive | 2 | both | 256 | -6.45 [-15.64, 2.75] |
| tight_passive | 2 | button | 256 | -12.50 [-30.85, 5.85] |
| tight_passive | 2 | big_blind | 256 | -0.39 [-1.72, 0.94] |
| tight_passive | 3 | both | 256 | 0.98 [-4.08, 6.03] |
| tight_passive | 3 | button | 256 | 2.73 [-7.01, 12.48] |
| tight_passive | 3 | big_blind | 256 | -0.78 [-3.45, 1.89] |
| loose_passive | all three | both | 256 | 5.40 [-8.99, 19.80] |
| loose_passive | all three | button | 256 | 10.42 [-10.80, 31.63] |
| loose_passive | all three | big_blind | 256 | 0.39 [-19.81, 20.60] |
| loose_passive | 1 | both | 256 | 12.89 [-11.20, 36.98] |
| loose_passive | 1 | button | 256 | 21.88 [-10.07, 53.82] |
| loose_passive | 1 | big_blind | 256 | 3.91 [-29.53, 37.34] |
| loose_passive | 2 | both | 256 | 8.01 [-12.10, 28.12] |
| loose_passive | 2 | button | 256 | 11.72 [-21.22, 44.65] |
| loose_passive | 2 | big_blind | 256 | 4.30 [-20.28, 28.87] |
| loose_passive | 3 | both | 256 | -4.69 [-28.01, 18.64] |
| loose_passive | 3 | button | 256 | -2.34 [-38.86, 34.18] |
| loose_passive | 3 | big_blind | 256 | -7.03 [-34.55, 20.49] |
| tight_aggressive | all three | both | 256 | 2.34 [-10.59, 15.28] |
| tight_aggressive | all three | button | 256 | 6.25 [-13.82, 26.32] |
| tight_aggressive | all three | big_blind | 256 | -1.56 [-19.84, 16.72] |
| tight_aggressive | 1 | both | 256 | 7.42 [-12.46, 27.30] |
| tight_aggressive | 1 | button | 256 | 23.05 [-6.12, 52.21] |
| tight_aggressive | 1 | big_blind | 256 | -8.20 [-35.12, 18.71] |
| tight_aggressive | 2 | both | 256 | 4.88 [-14.52, 24.29] |
| tight_aggressive | 2 | button | 256 | 3.12 [-26.12, 32.37] |
| tight_aggressive | 2 | big_blind | 256 | 6.64 [-24.89, 38.18] |
| tight_aggressive | 3 | both | 256 | -5.27 [-21.13, 10.58] |
| tight_aggressive | 3 | button | 256 | -7.42 [-38.56, 23.71] |
| tight_aggressive | 3 | big_blind | 256 | -3.12 [-7.60, 1.35] |
| loose_aggressive | all three | both | 256 | 16.73 [-8.41, 41.88] |
| loose_aggressive | all three | button | 256 | 63.41 [27.16, 99.66] |
| loose_aggressive | all three | big_blind | 256 | -29.95 [-67.41, 7.52] |
| loose_aggressive | 1 | both | 256 | 25.78 [-8.01, 59.57] |
| loose_aggressive | 1 | button | 256 | 74.61 [27.47, 121.75] |
| loose_aggressive | 1 | big_blind | 256 | -23.05 [-74.51, 28.41] |
| loose_aggressive | 2 | both | 256 | 17.19 [-15.39, 49.76] |
| loose_aggressive | 2 | button | 256 | 62.50 [12.16, 112.84] |
| loose_aggressive | 2 | big_blind | 256 | -28.12 [-76.70, 20.45] |
| loose_aggressive | 3 | both | 256 | 7.23 [-26.70, 41.15] |
| loose_aggressive | 3 | button | 256 | 53.12 [1.06, 105.19] |
| loose_aggressive | 3 | big_blind | 256 | -38.67 [-88.44, 11.09] |
| pot_pressure | all three | both | 256 | -9.18 [-22.01, 3.65] |
| pot_pressure | all three | button | 256 | -14.58 [-39.44, 10.27] |
| pot_pressure | all three | big_blind | 256 | -3.78 [-9.94, 2.39] |
| pot_pressure | 1 | both | 256 | -9.18 [-22.74, 4.38] |
| pot_pressure | 1 | button | 256 | -14.45 [-40.73, 11.82] |
| pot_pressure | 1 | big_blind | 256 | -3.91 [-11.04, 3.23] |
| pot_pressure | 2 | both | 256 | -19.53 [-40.82, 1.76] |
| pot_pressure | 2 | button | 256 | -32.03 [-72.78, 8.72] |
| pot_pressure | 2 | big_blind | 256 | -7.03 [-20.44, 6.38] |
| pot_pressure | 3 | both | 256 | 1.17 [-20.78, 23.13] |
| pot_pressure | 3 | button | 256 | 2.73 [-39.92, 45.39] |
| pot_pressure | 3 | big_blind | 256 | -0.39 [-10.82, 10.04] |
| train_pressure | all three | both | 256 | -4.43 [-22.03, 13.18] |
| train_pressure | all three | button | 256 | -2.21 [-23.40, 18.97] |
| train_pressure | all three | big_blind | 256 | -6.64 [-34.89, 21.61] |
| train_pressure | 1 | both | 256 | -7.62 [-27.84, 12.60] |
| train_pressure | 1 | button | 256 | -7.81 [-37.84, 22.22] |
| train_pressure | 1 | big_blind | 256 | -7.42 [-34.78, 19.93] |
| train_pressure | 2 | both | 256 | 4.30 [-26.03, 34.62] |
| train_pressure | 2 | button | 256 | 6.25 [-30.73, 43.23] |
| train_pressure | 2 | big_blind | 256 | 2.34 [-46.98, 51.67] |
| train_pressure | 3 | both | 256 | -9.96 [-37.80, 17.88] |
| train_pressure | 3 | button | 256 | -5.08 [-37.18, 27.02] |
| train_pressure | 3 | big_blind | 256 | -14.84 [-60.45, 30.76] |
| native-pressure | all three | both | 12288 | -3.12 [-9.55, 3.31] |
| native-pressure | all three | button | 12288 | -0.26 [-9.38, 8.86] |
| native-pressure | all three | big_blind | 12288 | -5.98 [-15.17, 3.20] |
| native-pressure | 1 | both | 12288 | -9.26 [-19.36, 0.83] |
| native-pressure | 1 | button | 12288 | -7.11 [-21.17, 6.95] |
| native-pressure | 1 | big_blind | 12288 | -11.42 [-25.96, 3.13] |
| native-pressure | 2 | both | 12288 | 0.87 [-9.02, 10.77] |
| native-pressure | 2 | button | 12288 | -3.85 [-17.79, 10.08] |
| native-pressure | 2 | big_blind | 12288 | 5.60 [-8.31, 19.50] |
| native-pressure | 3 | both | 12288 | -0.97 [-10.85, 8.91] |
| native-pressure | 3 | button | 12288 | 10.18 [-4.11, 24.48] |
| native-pressure | 3 | big_blind | 12288 | -12.13 [-26.02, 1.77] |
| selective-stackoff | all three | both | 256 | 1.95 [-5.10, 9.01] |
| selective-stackoff | all three | button | 256 | -1.82 [-12.25, 8.61] |
| selective-stackoff | all three | big_blind | 256 | 5.73 [-4.18, 15.63] |
| selective-stackoff | 1 | both | 256 | 2.54 [-12.91, 17.99] |
| selective-stackoff | 1 | button | 256 | 3.91 [-12.66, 20.48] |
| selective-stackoff | 1 | big_blind | 256 | 1.17 [-24.95, 27.30] |
| selective-stackoff | 2 | both | 256 | -5.27 [-18.50, 7.95] |
| selective-stackoff | 2 | button | 256 | -17.58 [-41.76, 6.61] |
| selective-stackoff | 2 | big_blind | 256 | 7.03 [-3.83, 17.90] |
| selective-stackoff | 3 | both | 256 | 8.59 [-2.29, 19.48] |
| selective-stackoff | 3 | button | 256 | 8.20 [-3.39, 19.80] |
| selective-stackoff | 3 | big_blind | 256 | 8.98 [-9.53, 27.50] |
| lbr | all three | both | 2048 | 10.45 [-2.99, 23.88] |
| lbr | all three | button | 2048 | 10.38 [-8.75, 29.50] |
| lbr | all three | big_blind | 2048 | 10.51 [-8.61, 29.64] |
| lbr | 1 | both | 2048 | 22.29 [1.76, 42.82] |
| lbr | 1 | button | 2048 | 21.97 [-7.36, 51.31] |
| lbr | 1 | big_blind | 2048 | 22.61 [-6.50, 51.71] |
| lbr | 2 | both | 2048 | -1.42 [-20.80, 17.97] |
| lbr | 2 | button | 2048 | -8.98 [-36.69, 18.73] |
| lbr | 2 | big_blind | 2048 | 6.15 [-21.57, 33.87] |
| lbr | 3 | both | 2048 | 10.46 [-9.23, 30.16] |
| lbr | 3 | button | 2048 | 18.14 [-9.67, 45.94] |
| lbr | 3 | big_blind | 2048 | 2.78 [-25.78, 31.34] |

## O-T: all panel, lineage and position contrasts

| Panel | Lineage | Position | Blocks | BB/100 [95%] |
| --- | --- | --- | --- | --- |
| uniform | all three | both | 256 | 35.38 [-15.83, 86.60] |
| uniform | all three | button | 256 | 35.61 [-42.24, 113.46] |
| uniform | all three | big_blind | 256 | 35.16 [-28.31, 98.62] |
| uniform | 1 | both | 256 | 37.50 [-28.95, 103.95] |
| uniform | 1 | button | 256 | 42.97 [-57.76, 143.69] |
| uniform | 1 | big_blind | 256 | 32.03 [-41.98, 106.04] |
| uniform | 2 | both | 256 | 49.90 [-20.08, 119.89] |
| uniform | 2 | button | 256 | 40.82 [-66.45, 148.09] |
| uniform | 2 | big_blind | 256 | 58.98 [-26.66, 144.63] |
| uniform | 3 | both | 256 | 18.75 [-43.31, 80.81] |
| uniform | 3 | button | 256 | 23.05 [-73.59, 119.68] |
| uniform | 3 | big_blind | 256 | 14.45 [-64.46, 93.36] |
| passive | all three | both | 256 | -2.18 [-35.81, 31.45] |
| passive | all three | button | 256 | -15.43 [-69.23, 38.37] |
| passive | all three | big_blind | 256 | 11.07 [-32.85, 54.98] |
| passive | 1 | both | 256 | 12.11 [-29.09, 53.31] |
| passive | 1 | button | 256 | -3.52 [-62.72, 55.68] |
| passive | 1 | big_blind | 256 | 27.73 [-27.45, 82.91] |
| passive | 2 | both | 256 | -10.84 [-51.65, 29.97] |
| passive | 2 | button | 256 | -16.60 [-87.55, 54.35] |
| passive | 2 | big_blind | 256 | -5.08 [-58.88, 48.72] |
| passive | 3 | both | 256 | -7.81 [-52.73, 37.11] |
| passive | 3 | button | 256 | -26.17 [-96.61, 44.27] |
| passive | 3 | big_blind | 256 | 10.55 [-46.56, 67.66] |
| minraise-cap2 | all three | both | 256 | -57.32 [-112.51, -2.14] |
| minraise-cap2 | all three | button | 256 | -101.24 [-179.30, -23.18] |
| minraise-cap2 | all three | big_blind | 256 | -13.41 [-97.73, 70.90] |
| minraise-cap2 | 1 | both | 256 | -56.25 [-123.97, 11.47] |
| minraise-cap2 | 1 | button | 256 | -128.52 [-220.48, -36.55] |
| minraise-cap2 | 1 | big_blind | 256 | 16.02 [-90.55, 122.58] |
| minraise-cap2 | 2 | both | 256 | -91.89 [-167.40, -16.39] |
| minraise-cap2 | 2 | button | 256 | -123.24 [-232.33, -14.16] |
| minraise-cap2 | 2 | big_blind | 256 | -60.55 [-168.61, 47.51] |
| minraise-cap2 | 3 | both | 256 | -23.83 [-91.03, 43.37] |
| minraise-cap2 | 3 | button | 256 | -51.95 [-155.54, 51.64] |
| minraise-cap2 | 3 | big_blind | 256 | 4.30 [-89.96, 98.55] |
| pressure-cap2 | all three | both | 256 | -30.83 [-77.02, 15.37] |
| pressure-cap2 | all three | button | 256 | -33.40 [-95.91, 29.11] |
| pressure-cap2 | all three | big_blind | 256 | -28.26 [-94.10, 37.59] |
| pressure-cap2 | 1 | both | 256 | 3.71 [-51.72, 59.14] |
| pressure-cap2 | 1 | button | 256 | -14.06 [-90.51, 62.38] |
| pressure-cap2 | 1 | big_blind | 256 | 21.48 [-59.42, 102.39] |
| pressure-cap2 | 2 | both | 256 | -55.76 [-115.31, 3.78] |
| pressure-cap2 | 2 | button | 256 | -45.12 [-121.25, 31.02] |
| pressure-cap2 | 2 | big_blind | 256 | -66.41 [-155.10, 22.29] |
| pressure-cap2 | 3 | both | 256 | -40.43 [-96.38, 15.52] |
| pressure-cap2 | 3 | button | 256 | -41.02 [-126.11, 44.07] |
| pressure-cap2 | 3 | big_blind | 256 | -39.84 [-113.06, 33.37] |
| tight_passive | all three | both | 256 | -3.35 [-14.81, 8.11] |
| tight_passive | all three | button | 256 | -10.35 [-29.15, 8.44] |
| tight_passive | all three | big_blind | 256 | 3.65 [-5.54, 12.83] |
| tight_passive | 1 | both | 256 | -6.84 [-25.66, 11.99] |
| tight_passive | 1 | button | 256 | -18.75 [-41.75, 4.25] |
| tight_passive | 1 | big_blind | 256 | 5.08 [-17.32, 27.47] |
| tight_passive | 2 | both | 256 | 1.66 [-10.45, 13.77] |
| tight_passive | 2 | button | 256 | 1.76 [-21.96, 25.48] |
| tight_passive | 2 | big_blind | 256 | 1.56 [-3.31, 6.43] |
| tight_passive | 3 | both | 256 | -4.88 [-17.37, 7.61] |
| tight_passive | 3 | button | 256 | -14.06 [-37.87, 9.75] |
| tight_passive | 3 | big_blind | 256 | 4.30 [-2.45, 11.04] |
| loose_passive | all three | both | 256 | 1.79 [-21.91, 25.49] |
| loose_passive | all three | button | 256 | -1.24 [-43.16, 40.69] |
| loose_passive | all three | big_blind | 256 | 4.82 [-18.10, 27.74] |
| loose_passive | 1 | both | 256 | 14.65 [-17.12, 46.42] |
| loose_passive | 1 | button | 256 | 16.02 [-32.40, 64.43] |
| loose_passive | 1 | big_blind | 256 | 13.28 [-22.77, 49.33] |
| loose_passive | 2 | both | 256 | -4.59 [-33.91, 24.73] |
| loose_passive | 2 | button | 256 | -11.52 [-63.30, 40.26] |
| loose_passive | 2 | big_blind | 256 | 2.34 [-24.93, 29.61] |
| loose_passive | 3 | both | 256 | -4.69 [-31.13, 21.75] |
| loose_passive | 3 | button | 256 | -8.20 [-54.58, 38.18] |
| loose_passive | 3 | big_blind | 256 | -1.17 [-32.03, 29.69] |
| tight_aggressive | all three | both | 256 | -6.35 [-24.57, 11.88] |
| tight_aggressive | all three | button | 256 | -21.03 [-52.87, 10.82] |
| tight_aggressive | all three | big_blind | 256 | 8.33 [-9.17, 25.83] |
| tight_aggressive | 1 | both | 256 | 2.73 [-19.14, 24.61] |
| tight_aggressive | 1 | button | 256 | -8.98 [-48.44, 30.47] |
| tight_aggressive | 1 | big_blind | 256 | 14.45 [-4.36, 33.26] |
| tight_aggressive | 2 | both | 256 | -8.11 [-28.54, 12.33] |
| tight_aggressive | 2 | button | 256 | -26.37 [-60.42, 7.68] |
| tight_aggressive | 2 | big_blind | 256 | 10.16 [-11.85, 32.17] |
| tight_aggressive | 3 | both | 256 | -13.67 [-37.98, 10.64] |
| tight_aggressive | 3 | button | 256 | -27.73 [-70.64, 15.17] |
| tight_aggressive | 3 | big_blind | 256 | 0.39 [-22.46, 23.24] |
| loose_aggressive | all three | both | 256 | -5.37 [-43.75, 33.01] |
| loose_aggressive | all three | button | 256 | 25.98 [-33.99, 85.94] |
| loose_aggressive | all three | big_blind | 256 | -36.72 [-80.00, 6.57] |
| loose_aggressive | 1 | both | 256 | 10.94 [-35.76, 57.64] |
| loose_aggressive | 1 | button | 256 | 54.30 [-21.30, 129.90] |
| loose_aggressive | 1 | big_blind | 256 | -32.42 [-82.37, 17.53] |
| loose_aggressive | 2 | both | 256 | -6.35 [-50.07, 37.37] |
| loose_aggressive | 2 | button | 256 | 13.48 [-57.29, 84.24] |
| loose_aggressive | 2 | big_blind | 256 | -26.17 [-75.85, 23.50] |
| loose_aggressive | 3 | both | 256 | -20.70 [-65.91, 24.51] |
| loose_aggressive | 3 | button | 256 | 10.16 [-63.33, 83.64] |
| loose_aggressive | 3 | big_blind | 256 | -51.56 [-106.09, 2.96] |
| pot_pressure | all three | both | 256 | -13.12 [-38.25, 12.01] |
| pot_pressure | all three | button | 256 | -25.33 [-75.12, 24.47] |
| pot_pressure | all three | big_blind | 256 | -0.91 [-9.93, 8.10] |
| pot_pressure | 1 | both | 256 | -2.34 [-28.56, 23.87] |
| pot_pressure | 1 | button | 256 | -9.77 [-63.97, 44.44] |
| pot_pressure | 1 | big_blind | 256 | 5.08 [-8.67, 18.83] |
| pot_pressure | 2 | both | 256 | -16.31 [-50.67, 18.05] |
| pot_pressure | 2 | button | 256 | -31.45 [-97.89, 35.00] |
| pot_pressure | 2 | big_blind | 256 | -1.17 [-18.63, 16.29] |
| pot_pressure | 3 | both | 256 | -20.70 [-51.42, 10.01] |
| pot_pressure | 3 | button | 256 | -34.77 [-95.33, 25.80] |
| pot_pressure | 3 | big_blind | 256 | -6.64 [-17.26, 3.98] |
| train_pressure | all three | both | 256 | -20.74 [-47.57, 6.10] |
| train_pressure | all three | button | 256 | -26.50 [-69.81, 16.81] |
| train_pressure | all three | big_blind | 256 | -14.97 [-47.34, 17.39] |
| train_pressure | 1 | both | 256 | -25.39 [-60.25, 9.47] |
| train_pressure | 1 | button | 256 | -48.44 [-99.29, 2.41] |
| train_pressure | 1 | big_blind | 256 | -2.34 [-50.24, 45.55] |
| train_pressure | 2 | both | 256 | -13.77 [-51.31, 23.77] |
| train_pressure | 2 | button | 256 | -13.48 [-67.92, 40.96] |
| train_pressure | 2 | big_blind | 256 | -14.06 [-63.10, 34.97] |
| train_pressure | 3 | both | 256 | -23.05 [-60.95, 14.86] |
| train_pressure | 3 | button | 256 | -17.58 [-73.15, 38.00] |
| train_pressure | 3 | big_blind | 256 | -28.52 [-75.98, 18.95] |
| native-pressure | all three | both | 12288 | -29.79 [-38.53, -21.04] |
| native-pressure | all three | button | 12288 | -24.78 [-37.76, -11.79] |
| native-pressure | all three | big_blind | 12288 | -34.80 [-47.13, -22.46] |
| native-pressure | 1 | both | 12288 | -31.18 [-42.88, -19.48] |
| native-pressure | 1 | button | 12288 | -23.36 [-40.27, -6.46] |
| native-pressure | 1 | big_blind | 12288 | -39.00 [-55.68, -22.32] |
| native-pressure | 2 | both | 12288 | -34.30 [-45.97, -22.63] |
| native-pressure | 2 | button | 12288 | -37.65 [-54.66, -20.65] |
| native-pressure | 2 | big_blind | 12288 | -30.94 [-47.49, -14.39] |
| native-pressure | 3 | both | 12288 | -23.88 [-35.42, -12.34] |
| native-pressure | 3 | button | 12288 | -13.31 [-30.13, 3.51] |
| native-pressure | 3 | big_blind | 12288 | -34.45 [-50.60, -18.30] |
| selective-stackoff | all three | both | 256 | -8.69 [-23.97, 6.59] |
| selective-stackoff | all three | button | 256 | -17.64 [-42.60, 7.31] |
| selective-stackoff | all three | big_blind | 256 | 0.26 [-13.25, 13.77] |
| selective-stackoff | 1 | both | 256 | -9.18 [-23.85, 5.49] |
| selective-stackoff | 1 | button | 256 | -16.02 [-38.59, 6.56] |
| selective-stackoff | 1 | big_blind | 256 | -2.34 [-16.34, 11.65] |
| selective-stackoff | 2 | both | 256 | -10.25 [-31.94, 11.43] |
| selective-stackoff | 2 | button | 256 | -21.29 [-55.89, 13.32] |
| selective-stackoff | 2 | big_blind | 256 | 0.78 [-19.55, 21.12] |
| selective-stackoff | 3 | both | 256 | -6.64 [-23.62, 10.34] |
| selective-stackoff | 3 | button | 256 | -15.62 [-48.32, 17.07] |
| selective-stackoff | 3 | big_blind | 256 | 2.34 [-17.21, 21.90] |
| lbr | all three | both | 2048 | 12.78 [-6.42, 31.98] |
| lbr | all three | button | 2048 | 26.86 [-1.94, 55.67] |
| lbr | all three | big_blind | 2048 | -1.30 [-27.77, 25.16] |
| lbr | 1 | both | 2048 | 16.55 [-7.84, 40.94] |
| lbr | 1 | button | 2048 | 32.23 [-3.68, 68.13] |
| lbr | 1 | big_blind | 2048 | 0.88 [-33.57, 35.33] |
| lbr | 2 | both | 2048 | 22.53 [-2.60, 47.66] |
| lbr | 2 | button | 2048 | 27.54 [-9.57, 64.65] |
| lbr | 2 | big_blind | 2048 | 17.53 [-15.81, 50.87] |
| lbr | 3 | both | 2048 | -0.74 [-25.05, 23.57] |
| lbr | 3 | button | 2048 | 20.83 [-16.07, 57.72] |
| lbr | 3 | big_blind | 2048 | -22.31 [-55.01, 10.38] |

## R: street coverage and LBR accounting

| Panel | Lineage | Street | decisions | missing | zero_mass | positive_mass | current | lbr_decisions | lbr_incomplete | lbr_over_soft_budget | Terminal hands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lbr | 1 | flop | 2764 | 6 | 0 | 0 | 2758 | 2684 | 0 | 0 | 1336 |
| lbr | 1 | preflop | 5330 | 0 | 0 | 0 | 5330 | 4480 | 0 | 0 | 1929 |
| lbr | 1 | river | 376 | 12 | 0 | 0 | 364 | 422 | 0 | 0 | 321 |
| lbr | 1 | turn | 993 | 10 | 0 | 0 | 983 | 1070 | 0 | 0 | 510 |
| loose_aggressive | 1 | flop | 208 | 0 | 0 | 0 | 208 | 0 | 0 | 0 | 76 |
| loose_aggressive | 1 | preflop | 431 | 0 | 0 | 0 | 431 | 0 | 0 | 0 | 328 |
| loose_aggressive | 1 | river | 78 | 0 | 0 | 0 | 78 | 0 | 0 | 0 | 65 |
| loose_aggressive | 1 | turn | 129 | 1 | 0 | 0 | 128 | 0 | 0 | 0 | 43 |
| loose_passive | 1 | flop | 279 | 2 | 0 | 0 | 277 | 0 | 0 | 0 | 70 |
| loose_passive | 1 | preflop | 410 | 1 | 0 | 0 | 409 | 0 | 0 | 0 | 235 |
| loose_passive | 1 | river | 180 | 2 | 0 | 0 | 178 | 0 | 0 | 0 | 178 |
| loose_passive | 1 | turn | 207 | 1 | 0 | 0 | 206 | 0 | 0 | 0 | 29 |
| minraise-cap2 | 1 | flop | 544 | 1 | 0 | 0 | 543 | 0 | 0 | 0 | 102 |
| minraise-cap2 | 1 | preflop | 704 | 0 | 0 | 0 | 704 | 0 | 0 | 0 | 160 |
| minraise-cap2 | 1 | river | 249 | 3 | 0 | 0 | 246 | 0 | 0 | 0 | 177 |
| minraise-cap2 | 1 | turn | 372 | 2 | 0 | 0 | 370 | 0 | 0 | 0 | 73 |
| native-pressure | 1 | flop | 25631 | 73 | 0 | 0 | 25558 | 0 | 0 | 0 | 6597 |
| native-pressure | 1 | preflop | 41681 | 24 | 0 | 0 | 41657 | 0 | 0 | 0 | 10871 |
| native-pressure | 1 | river | 5993 | 143 | 0 | 0 | 5850 | 0 | 0 | 0 | 3759 |
| native-pressure | 1 | turn | 12436 | 207 | 0 | 0 | 12229 | 0 | 0 | 0 | 3349 |
| passive | 1 | flop | 448 | 0 | 0 | 0 | 448 | 0 | 0 | 0 | 0 |
| passive | 1 | preflop | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 64 |
| passive | 1 | river | 437 | 2 | 0 | 0 | 435 | 0 | 0 | 0 | 437 |
| passive | 1 | turn | 448 | 1 | 0 | 0 | 447 | 0 | 0 | 0 | 11 |
| pot_pressure | 1 | flop | 108 | 39 | 0 | 0 | 69 | 0 | 0 | 0 | 62 |
| pot_pressure | 1 | preflop | 348 | 65 | 0 | 0 | 283 | 0 | 0 | 0 | 414 |
| pot_pressure | 1 | river | 21 | 12 | 0 | 0 | 9 | 0 | 0 | 0 | 15 |
| pot_pressure | 1 | turn | 38 | 17 | 0 | 0 | 21 | 0 | 0 | 0 | 21 |
| pressure-cap2 | 1 | flop | 409 | 0 | 0 | 0 | 409 | 0 | 0 | 0 | 114 |
| pressure-cap2 | 1 | preflop | 704 | 0 | 0 | 0 | 704 | 0 | 0 | 0 | 241 |
| pressure-cap2 | 1 | river | 140 | 1 | 0 | 0 | 139 | 0 | 0 | 0 | 98 |
| pressure-cap2 | 1 | turn | 234 | 0 | 0 | 0 | 234 | 0 | 0 | 0 | 59 |
| selective-stackoff | 1 | flop | 162 | 0 | 0 | 0 | 162 | 0 | 0 | 0 | 62 |
| selective-stackoff | 1 | preflop | 330 | 0 | 0 | 0 | 330 | 0 | 0 | 0 | 353 |
| selective-stackoff | 1 | river | 67 | 0 | 0 | 0 | 67 | 0 | 0 | 0 | 62 |
| selective-stackoff | 1 | turn | 105 | 0 | 0 | 0 | 105 | 0 | 0 | 0 | 35 |
| tight_aggressive | 1 | flop | 105 | 0 | 0 | 0 | 105 | 0 | 0 | 0 | 56 |
| tight_aggressive | 1 | preflop | 293 | 0 | 0 | 0 | 293 | 0 | 0 | 0 | 412 |
| tight_aggressive | 1 | river | 17 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 15 |
| tight_aggressive | 1 | turn | 48 | 0 | 0 | 0 | 48 | 0 | 0 | 0 | 29 |
| tight_passive | 1 | flop | 106 | 2 | 0 | 0 | 104 | 0 | 0 | 0 | 60 |
| tight_passive | 1 | preflop | 276 | 1 | 0 | 0 | 275 | 0 | 0 | 0 | 407 |
| tight_passive | 1 | river | 27 | 2 | 0 | 0 | 25 | 0 | 0 | 0 | 26 |
| tight_passive | 1 | turn | 45 | 1 | 0 | 0 | 44 | 0 | 0 | 0 | 19 |
| train_pressure | 1 | flop | 170 | 8 | 0 | 0 | 162 | 0 | 0 | 0 | 61 |
| train_pressure | 1 | preflop | 355 | 7 | 0 | 0 | 348 | 0 | 0 | 0 | 354 |
| train_pressure | 1 | river | 69 | 13 | 0 | 0 | 56 | 0 | 0 | 0 | 57 |
| train_pressure | 1 | turn | 107 | 11 | 0 | 0 | 96 | 0 | 0 | 0 | 40 |
| uniform | 1 | flop | 313 | 0 | 0 | 0 | 313 | 0 | 0 | 0 | 130 |
| uniform | 1 | preflop | 592 | 0 | 0 | 0 | 592 | 0 | 0 | 0 | 281 |
| uniform | 1 | river | 76 | 0 | 0 | 0 | 76 | 0 | 0 | 0 | 56 |
| uniform | 1 | turn | 139 | 0 | 0 | 0 | 139 | 0 | 0 | 0 | 45 |
| lbr | 2 | flop | 2762 | 2 | 0 | 0 | 2760 | 2571 | 0 | 0 | 1256 |
| lbr | 2 | preflop | 5424 | 0 | 0 | 0 | 5424 | 4581 | 0 | 0 | 2006 |
| lbr | 2 | river | 352 | 2 | 0 | 0 | 350 | 417 | 0 | 0 | 317 |
| lbr | 2 | turn | 977 | 15 | 0 | 0 | 962 | 1054 | 0 | 0 | 517 |
| loose_aggressive | 2 | flop | 234 | 0 | 0 | 0 | 234 | 0 | 0 | 0 | 78 |
| loose_aggressive | 2 | preflop | 431 | 0 | 0 | 0 | 431 | 0 | 0 | 0 | 304 |
| loose_aggressive | 2 | river | 108 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 88 |
| loose_aggressive | 2 | turn | 148 | 0 | 0 | 0 | 148 | 0 | 0 | 0 | 42 |
| loose_passive | 2 | flop | 299 | 1 | 0 | 0 | 298 | 0 | 0 | 0 | 63 |
| loose_passive | 2 | preflop | 411 | 0 | 0 | 0 | 411 | 0 | 0 | 0 | 215 |
| loose_passive | 2 | river | 207 | 0 | 0 | 0 | 207 | 0 | 0 | 0 | 205 |
| loose_passive | 2 | turn | 235 | 2 | 0 | 0 | 233 | 0 | 0 | 0 | 29 |
| minraise-cap2 | 2 | flop | 494 | 0 | 0 | 0 | 494 | 0 | 0 | 0 | 100 |
| minraise-cap2 | 2 | preflop | 715 | 0 | 0 | 0 | 715 | 0 | 0 | 0 | 182 |
| minraise-cap2 | 2 | river | 200 | 3 | 0 | 0 | 197 | 0 | 0 | 0 | 156 |
| minraise-cap2 | 2 | turn | 319 | 5 | 0 | 0 | 314 | 0 | 0 | 0 | 74 |
| native-pressure | 2 | flop | 21689 | 102 | 0 | 0 | 21587 | 0 | 0 | 0 | 4783 |
| native-pressure | 2 | preflop | 41480 | 47 | 0 | 0 | 41433 | 0 | 0 | 0 | 12458 |
| native-pressure | 2 | river | 5890 | 156 | 0 | 0 | 5734 | 0 | 0 | 0 | 3902 |
| native-pressure | 2 | turn | 12291 | 222 | 0 | 0 | 12069 | 0 | 0 | 0 | 3433 |
| passive | 2 | flop | 459 | 0 | 0 | 0 | 459 | 0 | 0 | 0 | 0 |
| passive | 2 | preflop | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 53 |
| passive | 2 | river | 453 | 1 | 0 | 0 | 452 | 0 | 0 | 0 | 453 |
| passive | 2 | turn | 459 | 1 | 0 | 0 | 458 | 0 | 0 | 0 | 6 |
| pot_pressure | 2 | flop | 121 | 39 | 0 | 0 | 82 | 0 | 0 | 0 | 63 |
| pot_pressure | 2 | preflop | 349 | 67 | 0 | 0 | 282 | 0 | 0 | 0 | 399 |
| pot_pressure | 2 | river | 29 | 14 | 0 | 0 | 15 | 0 | 0 | 0 | 21 |
| pot_pressure | 2 | turn | 55 | 19 | 0 | 0 | 36 | 0 | 0 | 0 | 29 |
| pressure-cap2 | 2 | flop | 362 | 0 | 0 | 0 | 362 | 0 | 0 | 0 | 96 |
| pressure-cap2 | 2 | preflop | 715 | 0 | 0 | 0 | 715 | 0 | 0 | 0 | 267 |
| pressure-cap2 | 2 | river | 126 | 2 | 0 | 0 | 124 | 0 | 0 | 0 | 102 |
| pressure-cap2 | 2 | turn | 203 | 3 | 0 | 0 | 200 | 0 | 0 | 0 | 47 |
| selective-stackoff | 2 | flop | 187 | 0 | 0 | 0 | 187 | 0 | 0 | 0 | 54 |
| selective-stackoff | 2 | preflop | 331 | 0 | 0 | 0 | 331 | 0 | 0 | 0 | 326 |
| selective-stackoff | 2 | river | 104 | 2 | 0 | 0 | 102 | 0 | 0 | 0 | 91 |
| selective-stackoff | 2 | turn | 140 | 0 | 0 | 0 | 140 | 0 | 0 | 0 | 41 |
| tight_aggressive | 2 | flop | 127 | 0 | 0 | 0 | 127 | 0 | 0 | 0 | 57 |
| tight_aggressive | 2 | preflop | 297 | 0 | 0 | 0 | 297 | 0 | 0 | 0 | 395 |
| tight_aggressive | 2 | river | 22 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 20 |
| tight_aggressive | 2 | turn | 64 | 0 | 0 | 0 | 64 | 0 | 0 | 0 | 40 |
| tight_passive | 2 | flop | 119 | 0 | 0 | 0 | 119 | 0 | 0 | 0 | 51 |
| tight_passive | 2 | preflop | 277 | 0 | 0 | 0 | 277 | 0 | 0 | 0 | 393 |
| tight_passive | 2 | river | 41 | 0 | 0 | 0 | 41 | 0 | 0 | 0 | 41 |
| tight_passive | 2 | turn | 69 | 0 | 0 | 0 | 69 | 0 | 0 | 0 | 27 |
| train_pressure | 2 | flop | 192 | 9 | 0 | 0 | 183 | 0 | 0 | 0 | 58 |
| train_pressure | 2 | preflop | 356 | 8 | 0 | 0 | 348 | 0 | 0 | 0 | 337 |
| train_pressure | 2 | river | 91 | 15 | 0 | 0 | 76 | 0 | 0 | 0 | 76 |
| train_pressure | 2 | turn | 133 | 8 | 0 | 0 | 125 | 0 | 0 | 0 | 41 |
| uniform | 2 | flop | 309 | 1 | 0 | 0 | 308 | 0 | 0 | 0 | 104 |
| uniform | 2 | preflop | 607 | 0 | 0 | 0 | 607 | 0 | 0 | 0 | 284 |
| uniform | 2 | river | 62 | 1 | 0 | 0 | 61 | 0 | 0 | 0 | 49 |
| uniform | 2 | turn | 171 | 2 | 0 | 0 | 169 | 0 | 0 | 0 | 75 |
| lbr | 3 | flop | 2801 | 0 | 0 | 0 | 2801 | 2523 | 0 | 0 | 1324 |
| lbr | 3 | preflop | 5317 | 0 | 0 | 0 | 5317 | 4469 | 0 | 0 | 2001 |
| lbr | 3 | river | 342 | 9 | 0 | 0 | 333 | 385 | 0 | 0 | 299 |
| lbr | 3 | turn | 919 | 7 | 0 | 0 | 912 | 970 | 0 | 0 | 472 |
| loose_aggressive | 3 | flop | 206 | 0 | 0 | 0 | 206 | 0 | 0 | 0 | 65 |
| loose_aggressive | 3 | preflop | 431 | 0 | 0 | 0 | 431 | 0 | 0 | 0 | 326 |
| loose_aggressive | 3 | river | 98 | 1 | 0 | 0 | 97 | 0 | 0 | 0 | 76 |
| loose_aggressive | 3 | turn | 141 | 0 | 0 | 0 | 141 | 0 | 0 | 0 | 45 |
| loose_passive | 3 | flop | 279 | 1 | 0 | 0 | 278 | 0 | 0 | 0 | 68 |
| loose_passive | 3 | preflop | 410 | 0 | 0 | 0 | 410 | 0 | 0 | 0 | 235 |
| loose_passive | 3 | river | 188 | 1 | 0 | 0 | 187 | 0 | 0 | 0 | 187 |
| loose_passive | 3 | turn | 210 | 2 | 0 | 0 | 208 | 0 | 0 | 0 | 22 |
| minraise-cap2 | 3 | flop | 490 | 3 | 0 | 0 | 487 | 0 | 0 | 0 | 92 |
| minraise-cap2 | 3 | preflop | 703 | 0 | 0 | 0 | 703 | 0 | 0 | 0 | 185 |
| minraise-cap2 | 3 | river | 224 | 1 | 0 | 0 | 223 | 0 | 0 | 0 | 166 |
| minraise-cap2 | 3 | turn | 333 | 1 | 0 | 0 | 332 | 0 | 0 | 0 | 69 |
| native-pressure | 3 | flop | 23745 | 67 | 0 | 0 | 23678 | 0 | 0 | 0 | 5451 |
| native-pressure | 3 | preflop | 41045 | 13 | 0 | 0 | 41032 | 0 | 0 | 0 | 11695 |
| native-pressure | 3 | river | 6282 | 184 | 0 | 0 | 6098 | 0 | 0 | 0 | 4129 |
| native-pressure | 3 | turn | 12612 | 276 | 0 | 0 | 12336 | 0 | 0 | 0 | 3301 |
| passive | 3 | flop | 447 | 0 | 0 | 0 | 447 | 0 | 0 | 0 | 0 |
| passive | 3 | preflop | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 65 |
| passive | 3 | river | 436 | 0 | 0 | 0 | 436 | 0 | 0 | 0 | 436 |
| passive | 3 | turn | 447 | 1 | 0 | 0 | 446 | 0 | 0 | 0 | 11 |
| pot_pressure | 3 | flop | 96 | 35 | 0 | 0 | 61 | 0 | 0 | 0 | 52 |
| pot_pressure | 3 | preflop | 349 | 69 | 0 | 0 | 280 | 0 | 0 | 0 | 420 |
| pot_pressure | 3 | river | 21 | 8 | 0 | 0 | 13 | 0 | 0 | 0 | 15 |
| pot_pressure | 3 | turn | 43 | 20 | 0 | 0 | 23 | 0 | 0 | 0 | 25 |
| pressure-cap2 | 3 | flop | 369 | 3 | 0 | 0 | 366 | 0 | 0 | 0 | 91 |
| pressure-cap2 | 3 | preflop | 703 | 0 | 0 | 0 | 703 | 0 | 0 | 0 | 266 |
| pressure-cap2 | 3 | river | 152 | 1 | 0 | 0 | 151 | 0 | 0 | 0 | 110 |
| pressure-cap2 | 3 | turn | 217 | 1 | 0 | 0 | 216 | 0 | 0 | 0 | 45 |
| selective-stackoff | 3 | flop | 163 | 0 | 0 | 0 | 163 | 0 | 0 | 0 | 61 |
| selective-stackoff | 3 | preflop | 331 | 0 | 0 | 0 | 331 | 0 | 0 | 0 | 353 |
| selective-stackoff | 3 | river | 73 | 0 | 0 | 0 | 73 | 0 | 0 | 0 | 67 |
| selective-stackoff | 3 | turn | 102 | 1 | 0 | 0 | 101 | 0 | 0 | 0 | 31 |
| tight_aggressive | 3 | flop | 100 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 40 |
| tight_aggressive | 3 | preflop | 299 | 0 | 0 | 0 | 299 | 0 | 0 | 0 | 419 |
| tight_aggressive | 3 | river | 25 | 0 | 0 | 0 | 25 | 0 | 0 | 0 | 23 |
| tight_aggressive | 3 | turn | 57 | 0 | 0 | 0 | 57 | 0 | 0 | 0 | 30 |
| tight_passive | 3 | flop | 100 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 43 |
| tight_passive | 3 | preflop | 276 | 0 | 0 | 0 | 276 | 0 | 0 | 0 | 412 |
| tight_passive | 3 | river | 37 | 0 | 0 | 0 | 37 | 0 | 0 | 0 | 37 |
| tight_passive | 3 | turn | 58 | 0 | 0 | 0 | 58 | 0 | 0 | 0 | 20 |
| train_pressure | 3 | flop | 166 | 6 | 0 | 0 | 160 | 0 | 0 | 0 | 53 |
| train_pressure | 3 | preflop | 358 | 6 | 0 | 0 | 352 | 0 | 0 | 0 | 355 |
| train_pressure | 3 | river | 79 | 11 | 0 | 0 | 68 | 0 | 0 | 0 | 66 |
| train_pressure | 3 | turn | 115 | 12 | 0 | 0 | 103 | 0 | 0 | 0 | 38 |
| uniform | 3 | flop | 321 | 1 | 0 | 0 | 320 | 0 | 0 | 0 | 118 |
| uniform | 3 | preflop | 597 | 0 | 0 | 0 | 597 | 0 | 0 | 0 | 281 |
| uniform | 3 | river | 77 | 0 | 0 | 0 | 77 | 0 | 0 | 0 | 56 |
| uniform | 3 | turn | 157 | 1 | 0 | 0 | 156 | 0 | 0 | 0 | 57 |

## O: street coverage and LBR accounting

| Panel | Lineage | Street | decisions | missing | zero_mass | positive_mass | current | lbr_decisions | lbr_incomplete | lbr_over_soft_budget | Terminal hands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lbr | 1 | flop | 3059 | 1 | 0 | 3058 | 0 | 2954 | 0 | 0 | 1354 |
| lbr | 1 | preflop | 5248 | 0 | 0 | 5248 | 0 | 4485 | 0 | 0 | 1890 |
| lbr | 1 | river | 383 | 2 | 1 | 380 | 0 | 444 | 0 | 0 | 323 |
| lbr | 1 | turn | 1090 | 0 | 1 | 1089 | 0 | 1180 | 0 | 0 | 529 |
| loose_aggressive | 1 | flop | 207 | 0 | 0 | 207 | 0 | 0 | 0 | 0 | 76 |
| loose_aggressive | 1 | preflop | 431 | 0 | 0 | 431 | 0 | 0 | 0 | 0 | 330 |
| loose_aggressive | 1 | river | 79 | 0 | 0 | 79 | 0 | 0 | 0 | 0 | 59 |
| loose_aggressive | 1 | turn | 122 | 0 | 0 | 122 | 0 | 0 | 0 | 0 | 47 |
| loose_passive | 1 | flop | 272 | 1 | 0 | 271 | 0 | 0 | 0 | 0 | 68 |
| loose_passive | 1 | preflop | 409 | 0 | 0 | 409 | 0 | 0 | 0 | 0 | 242 |
| loose_passive | 1 | river | 173 | 3 | 1 | 169 | 0 | 0 | 0 | 0 | 171 |
| loose_passive | 1 | turn | 202 | 1 | 0 | 201 | 0 | 0 | 0 | 0 | 31 |
| minraise-cap2 | 1 | flop | 470 | 0 | 0 | 470 | 0 | 0 | 0 | 0 | 133 |
| minraise-cap2 | 1 | preflop | 695 | 0 | 0 | 695 | 0 | 0 | 0 | 0 | 191 |
| minraise-cap2 | 1 | river | 145 | 0 | 0 | 145 | 0 | 0 | 0 | 0 | 110 |
| minraise-cap2 | 1 | turn | 262 | 0 | 0 | 262 | 0 | 0 | 0 | 0 | 78 |
| native-pressure | 1 | flop | 20984 | 11 | 12 | 20961 | 0 | 0 | 0 | 0 | 6054 |
| native-pressure | 1 | preflop | 41091 | 2 | 1 | 41088 | 0 | 0 | 0 | 0 | 13609 |
| native-pressure | 1 | river | 3659 | 40 | 21 | 3598 | 0 | 0 | 0 | 0 | 2356 |
| native-pressure | 1 | turn | 8669 | 21 | 24 | 8624 | 0 | 0 | 0 | 0 | 2557 |
| passive | 1 | flop | 439 | 0 | 0 | 439 | 0 | 0 | 0 | 0 | 0 |
| passive | 1 | preflop | 512 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 73 |
| passive | 1 | river | 424 | 0 | 1 | 423 | 0 | 0 | 0 | 0 | 424 |
| passive | 1 | turn | 439 | 0 | 0 | 439 | 0 | 0 | 0 | 0 | 15 |
| pot_pressure | 1 | flop | 98 | 29 | 0 | 69 | 0 | 0 | 0 | 0 | 62 |
| pot_pressure | 1 | preflop | 345 | 64 | 0 | 281 | 0 | 0 | 0 | 0 | 421 |
| pot_pressure | 1 | river | 15 | 9 | 0 | 6 | 0 | 0 | 0 | 0 | 10 |
| pot_pressure | 1 | turn | 30 | 14 | 0 | 16 | 0 | 0 | 0 | 0 | 19 |
| pressure-cap2 | 1 | flop | 332 | 0 | 0 | 332 | 0 | 0 | 0 | 0 | 107 |
| pressure-cap2 | 1 | preflop | 695 | 0 | 0 | 695 | 0 | 0 | 0 | 0 | 284 |
| pressure-cap2 | 1 | river | 95 | 0 | 0 | 95 | 0 | 0 | 0 | 0 | 72 |
| pressure-cap2 | 1 | turn | 168 | 0 | 0 | 168 | 0 | 0 | 0 | 0 | 49 |
| selective-stackoff | 1 | flop | 154 | 0 | 0 | 154 | 0 | 0 | 0 | 0 | 59 |
| selective-stackoff | 1 | preflop | 331 | 0 | 0 | 331 | 0 | 0 | 0 | 0 | 363 |
| selective-stackoff | 1 | river | 64 | 0 | 0 | 64 | 0 | 0 | 0 | 0 | 59 |
| selective-stackoff | 1 | turn | 98 | 0 | 0 | 98 | 0 | 0 | 0 | 0 | 31 |
| tight_aggressive | 1 | flop | 98 | 0 | 0 | 98 | 0 | 0 | 0 | 0 | 52 |
| tight_aggressive | 1 | preflop | 300 | 0 | 0 | 300 | 0 | 0 | 0 | 0 | 422 |
| tight_aggressive | 1 | river | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 11 |
| tight_aggressive | 1 | turn | 41 | 0 | 0 | 41 | 0 | 0 | 0 | 0 | 27 |
| tight_passive | 1 | flop | 97 | 1 | 0 | 96 | 0 | 0 | 0 | 0 | 57 |
| tight_passive | 1 | preflop | 276 | 0 | 0 | 276 | 0 | 0 | 0 | 0 | 416 |
| tight_passive | 1 | river | 28 | 2 | 0 | 26 | 0 | 0 | 0 | 0 | 27 |
| tight_passive | 1 | turn | 39 | 2 | 0 | 37 | 0 | 0 | 0 | 0 | 12 |
| train_pressure | 1 | flop | 173 | 4 | 0 | 169 | 0 | 0 | 0 | 0 | 68 |
| train_pressure | 1 | preflop | 354 | 2 | 0 | 352 | 0 | 0 | 0 | 0 | 353 |
| train_pressure | 1 | river | 81 | 16 | 0 | 65 | 0 | 0 | 0 | 0 | 63 |
| train_pressure | 1 | turn | 103 | 10 | 0 | 93 | 0 | 0 | 0 | 0 | 28 |
| uniform | 1 | flop | 303 | 0 | 0 | 303 | 0 | 0 | 0 | 0 | 123 |
| uniform | 1 | preflop | 597 | 0 | 0 | 597 | 0 | 0 | 0 | 0 | 294 |
| uniform | 1 | river | 66 | 0 | 0 | 66 | 0 | 0 | 0 | 0 | 46 |
| uniform | 1 | turn | 127 | 0 | 0 | 127 | 0 | 0 | 0 | 0 | 49 |
| lbr | 2 | flop | 3019 | 0 | 0 | 3019 | 0 | 2909 | 0 | 0 | 1377 |
| lbr | 2 | preflop | 5226 | 0 | 0 | 5226 | 0 | 4486 | 0 | 0 | 1908 |
| lbr | 2 | river | 393 | 2 | 1 | 390 | 0 | 461 | 0 | 0 | 333 |
| lbr | 2 | turn | 1038 | 0 | 0 | 1038 | 0 | 1144 | 0 | 0 | 478 |
| loose_aggressive | 2 | flop | 190 | 0 | 0 | 190 | 0 | 0 | 0 | 0 | 66 |
| loose_aggressive | 2 | preflop | 430 | 0 | 0 | 430 | 0 | 0 | 0 | 0 | 342 |
| loose_aggressive | 2 | river | 73 | 0 | 0 | 73 | 0 | 0 | 0 | 0 | 56 |
| loose_aggressive | 2 | turn | 122 | 0 | 0 | 122 | 0 | 0 | 0 | 0 | 48 |
| loose_passive | 2 | flop | 269 | 0 | 0 | 269 | 0 | 0 | 0 | 0 | 68 |
| loose_passive | 2 | preflop | 410 | 0 | 0 | 410 | 0 | 0 | 0 | 0 | 244 |
| loose_passive | 2 | river | 170 | 1 | 0 | 169 | 0 | 0 | 0 | 0 | 169 |
| loose_passive | 2 | turn | 201 | 0 | 0 | 201 | 0 | 0 | 0 | 0 | 31 |
| minraise-cap2 | 2 | flop | 476 | 0 | 0 | 476 | 0 | 0 | 0 | 0 | 137 |
| minraise-cap2 | 2 | preflop | 693 | 0 | 0 | 693 | 0 | 0 | 0 | 0 | 192 |
| minraise-cap2 | 2 | river | 149 | 0 | 0 | 149 | 0 | 0 | 0 | 0 | 112 |
| minraise-cap2 | 2 | turn | 254 | 0 | 0 | 254 | 0 | 0 | 0 | 0 | 71 |
| native-pressure | 2 | flop | 20775 | 6 | 6 | 20763 | 0 | 0 | 0 | 0 | 6062 |
| native-pressure | 2 | preflop | 41190 | 6 | 2 | 41182 | 0 | 0 | 0 | 0 | 13617 |
| native-pressure | 2 | river | 3622 | 28 | 19 | 3575 | 0 | 0 | 0 | 0 | 2319 |
| native-pressure | 2 | turn | 8630 | 23 | 20 | 8587 | 0 | 0 | 0 | 0 | 2578 |
| passive | 2 | flop | 437 | 0 | 0 | 437 | 0 | 0 | 0 | 0 | 0 |
| passive | 2 | preflop | 512 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 75 |
| passive | 2 | river | 422 | 0 | 0 | 422 | 0 | 0 | 0 | 0 | 422 |
| passive | 2 | turn | 437 | 0 | 0 | 437 | 0 | 0 | 0 | 0 | 15 |
| pot_pressure | 2 | flop | 93 | 30 | 0 | 63 | 0 | 0 | 0 | 0 | 57 |
| pot_pressure | 2 | preflop | 342 | 61 | 0 | 281 | 0 | 0 | 0 | 0 | 425 |
| pot_pressure | 2 | river | 15 | 9 | 0 | 6 | 0 | 0 | 0 | 0 | 10 |
| pot_pressure | 2 | turn | 32 | 17 | 0 | 15 | 0 | 0 | 0 | 0 | 20 |
| pressure-cap2 | 2 | flop | 329 | 0 | 0 | 329 | 0 | 0 | 0 | 0 | 112 |
| pressure-cap2 | 2 | preflop | 693 | 0 | 0 | 693 | 0 | 0 | 0 | 0 | 291 |
| pressure-cap2 | 2 | river | 91 | 0 | 0 | 91 | 0 | 0 | 0 | 0 | 65 |
| pressure-cap2 | 2 | turn | 155 | 0 | 0 | 155 | 0 | 0 | 0 | 0 | 44 |
| selective-stackoff | 2 | flop | 144 | 0 | 0 | 144 | 0 | 0 | 0 | 0 | 58 |
| selective-stackoff | 2 | preflop | 331 | 0 | 0 | 331 | 0 | 0 | 0 | 0 | 372 |
| selective-stackoff | 2 | river | 64 | 0 | 0 | 64 | 0 | 0 | 0 | 0 | 60 |
| selective-stackoff | 2 | turn | 90 | 0 | 0 | 90 | 0 | 0 | 0 | 0 | 22 |
| tight_aggressive | 2 | flop | 82 | 0 | 0 | 82 | 0 | 0 | 0 | 0 | 45 |
| tight_aggressive | 2 | preflop | 302 | 0 | 0 | 302 | 0 | 0 | 0 | 0 | 435 |
| tight_aggressive | 2 | river | 11 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 10 |
| tight_aggressive | 2 | turn | 35 | 0 | 0 | 35 | 0 | 0 | 0 | 0 | 22 |
| tight_passive | 2 | flop | 89 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 49 |
| tight_passive | 2 | preflop | 276 | 0 | 0 | 276 | 0 | 0 | 0 | 0 | 423 |
| tight_passive | 2 | river | 28 | 0 | 0 | 28 | 0 | 0 | 0 | 0 | 28 |
| tight_passive | 2 | turn | 41 | 1 | 0 | 40 | 0 | 0 | 0 | 0 | 12 |
| train_pressure | 2 | flop | 161 | 5 | 0 | 156 | 0 | 0 | 0 | 0 | 60 |
| train_pressure | 2 | preflop | 355 | 7 | 0 | 348 | 0 | 0 | 0 | 0 | 363 |
| train_pressure | 2 | river | 68 | 14 | 0 | 54 | 0 | 0 | 0 | 0 | 53 |
| train_pressure | 2 | turn | 101 | 12 | 0 | 89 | 0 | 0 | 0 | 0 | 36 |
| uniform | 2 | flop | 290 | 0 | 0 | 290 | 0 | 0 | 0 | 0 | 113 |
| uniform | 2 | preflop | 597 | 0 | 0 | 597 | 0 | 0 | 0 | 0 | 306 |
| uniform | 2 | river | 66 | 0 | 0 | 66 | 0 | 0 | 0 | 0 | 47 |
| uniform | 2 | turn | 129 | 0 | 0 | 129 | 0 | 0 | 0 | 0 | 46 |
| lbr | 3 | flop | 2999 | 0 | 0 | 2999 | 0 | 2905 | 0 | 0 | 1358 |
| lbr | 3 | preflop | 5249 | 0 | 0 | 5249 | 0 | 4486 | 0 | 0 | 1927 |
| lbr | 3 | river | 373 | 2 | 1 | 370 | 0 | 436 | 0 | 0 | 315 |
| lbr | 3 | turn | 1024 | 0 | 1 | 1023 | 0 | 1114 | 0 | 0 | 496 |
| loose_aggressive | 3 | flop | 198 | 0 | 0 | 198 | 0 | 0 | 0 | 0 | 65 |
| loose_aggressive | 3 | preflop | 429 | 0 | 0 | 429 | 0 | 0 | 0 | 0 | 335 |
| loose_aggressive | 3 | river | 79 | 0 | 0 | 79 | 0 | 0 | 0 | 0 | 61 |
| loose_aggressive | 3 | turn | 130 | 0 | 0 | 130 | 0 | 0 | 0 | 0 | 51 |
| loose_passive | 3 | flop | 272 | 1 | 0 | 271 | 0 | 0 | 0 | 0 | 65 |
| loose_passive | 3 | preflop | 410 | 0 | 0 | 410 | 0 | 0 | 0 | 0 | 241 |
| loose_passive | 3 | river | 170 | 0 | 0 | 170 | 0 | 0 | 0 | 0 | 169 |
| loose_passive | 3 | turn | 207 | 0 | 0 | 207 | 0 | 0 | 0 | 0 | 37 |
| minraise-cap2 | 3 | flop | 473 | 1 | 0 | 472 | 0 | 0 | 0 | 0 | 124 |
| minraise-cap2 | 3 | preflop | 696 | 0 | 0 | 696 | 0 | 0 | 0 | 0 | 193 |
| minraise-cap2 | 3 | river | 156 | 1 | 1 | 154 | 0 | 0 | 0 | 0 | 117 |
| minraise-cap2 | 3 | turn | 273 | 0 | 1 | 272 | 0 | 0 | 0 | 0 | 78 |
| native-pressure | 3 | flop | 20722 | 3 | 3 | 20716 | 0 | 0 | 0 | 0 | 5876 |
| native-pressure | 3 | preflop | 41133 | 6 | 1 | 41126 | 0 | 0 | 0 | 0 | 13721 |
| native-pressure | 3 | river | 3825 | 15 | 25 | 3785 | 0 | 0 | 0 | 0 | 2447 |
| native-pressure | 3 | turn | 8795 | 20 | 27 | 8748 | 0 | 0 | 0 | 0 | 2532 |
| passive | 3 | flop | 440 | 0 | 0 | 440 | 0 | 0 | 0 | 0 | 0 |
| passive | 3 | preflop | 512 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 72 |
| passive | 3 | river | 426 | 0 | 0 | 426 | 0 | 0 | 0 | 0 | 426 |
| passive | 3 | turn | 440 | 0 | 1 | 439 | 0 | 0 | 0 | 0 | 14 |
| pot_pressure | 3 | flop | 100 | 31 | 0 | 69 | 0 | 0 | 0 | 0 | 59 |
| pot_pressure | 3 | preflop | 344 | 63 | 0 | 281 | 0 | 0 | 0 | 0 | 419 |
| pot_pressure | 3 | river | 17 | 10 | 0 | 7 | 0 | 0 | 0 | 0 | 12 |
| pot_pressure | 3 | turn | 38 | 17 | 0 | 21 | 0 | 0 | 0 | 0 | 22 |
| pressure-cap2 | 3 | flop | 334 | 1 | 0 | 333 | 0 | 0 | 0 | 0 | 101 |
| pressure-cap2 | 3 | preflop | 696 | 0 | 0 | 696 | 0 | 0 | 0 | 0 | 288 |
| pressure-cap2 | 3 | river | 107 | 0 | 0 | 107 | 0 | 0 | 0 | 0 | 79 |
| pressure-cap2 | 3 | turn | 172 | 0 | 0 | 172 | 0 | 0 | 0 | 0 | 44 |
| selective-stackoff | 3 | flop | 150 | 0 | 0 | 150 | 0 | 0 | 0 | 0 | 59 |
| selective-stackoff | 3 | preflop | 331 | 0 | 0 | 331 | 0 | 0 | 0 | 0 | 366 |
| selective-stackoff | 3 | river | 66 | 0 | 0 | 66 | 0 | 0 | 0 | 0 | 60 |
| selective-stackoff | 3 | turn | 92 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 27 |
| tight_aggressive | 3 | flop | 93 | 0 | 0 | 93 | 0 | 0 | 0 | 0 | 47 |
| tight_aggressive | 3 | preflop | 301 | 0 | 0 | 301 | 0 | 0 | 0 | 0 | 425 |
| tight_aggressive | 3 | river | 11 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 10 |
| tight_aggressive | 3 | turn | 43 | 0 | 0 | 43 | 0 | 0 | 0 | 0 | 30 |
| tight_passive | 3 | flop | 96 | 0 | 0 | 96 | 0 | 0 | 0 | 0 | 52 |
| tight_passive | 3 | preflop | 276 | 0 | 0 | 276 | 0 | 0 | 0 | 0 | 416 |
| tight_passive | 3 | river | 28 | 0 | 0 | 28 | 0 | 0 | 0 | 0 | 28 |
| tight_passive | 3 | turn | 45 | 1 | 0 | 44 | 0 | 0 | 0 | 0 | 16 |
| train_pressure | 3 | flop | 164 | 4 | 0 | 160 | 0 | 0 | 0 | 0 | 60 |
| train_pressure | 3 | preflop | 355 | 5 | 0 | 350 | 0 | 0 | 0 | 0 | 360 |
| train_pressure | 3 | river | 73 | 13 | 0 | 60 | 0 | 0 | 0 | 0 | 55 |
| train_pressure | 3 | turn | 106 | 11 | 0 | 95 | 0 | 0 | 0 | 0 | 37 |
| uniform | 3 | flop | 289 | 0 | 0 | 289 | 0 | 0 | 0 | 0 | 113 |
| uniform | 3 | preflop | 601 | 0 | 0 | 601 | 0 | 0 | 0 | 0 | 301 |
| uniform | 3 | river | 65 | 1 | 0 | 64 | 0 | 0 | 0 | 0 | 49 |
| uniform | 3 | turn | 132 | 0 | 0 | 132 | 0 | 0 | 0 | 0 | 49 |

## C: street coverage and LBR accounting

| Panel | Lineage | Street | decisions | missing | zero_mass | positive_mass | current | lbr_decisions | lbr_incomplete | lbr_over_soft_budget | Terminal hands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lbr | 1 | flop | 3063 | 1 | 0 | 0 | 3062 | 2977 | 0 | 0 | 1388 |
| lbr | 1 | preflop | 5160 | 0 | 0 | 0 | 5160 | 4430 | 0 | 0 | 1880 |
| lbr | 1 | river | 336 | 0 | 0 | 0 | 336 | 403 | 0 | 0 | 297 |
| lbr | 1 | turn | 1031 | 0 | 0 | 0 | 1031 | 1121 | 0 | 0 | 531 |
| loose_aggressive | 1 | flop | 203 | 0 | 0 | 0 | 203 | 0 | 0 | 0 | 72 |
| loose_aggressive | 1 | preflop | 429 | 0 | 0 | 0 | 429 | 0 | 0 | 0 | 331 |
| loose_aggressive | 1 | river | 73 | 0 | 0 | 0 | 73 | 0 | 0 | 0 | 58 |
| loose_aggressive | 1 | turn | 131 | 0 | 0 | 0 | 131 | 0 | 0 | 0 | 51 |
| loose_passive | 1 | flop | 272 | 0 | 0 | 0 | 272 | 0 | 0 | 0 | 60 |
| loose_passive | 1 | preflop | 410 | 0 | 0 | 0 | 410 | 0 | 0 | 0 | 241 |
| loose_passive | 1 | river | 183 | 0 | 0 | 0 | 183 | 0 | 0 | 0 | 182 |
| loose_passive | 1 | turn | 212 | 2 | 0 | 0 | 210 | 0 | 0 | 0 | 29 |
| minraise-cap2 | 1 | flop | 472 | 0 | 0 | 0 | 472 | 0 | 0 | 0 | 139 |
| minraise-cap2 | 1 | preflop | 697 | 0 | 0 | 0 | 697 | 0 | 0 | 0 | 191 |
| minraise-cap2 | 1 | river | 151 | 0 | 0 | 0 | 151 | 0 | 0 | 0 | 111 |
| minraise-cap2 | 1 | turn | 254 | 1 | 0 | 0 | 253 | 0 | 0 | 0 | 71 |
| native-pressure | 1 | flop | 21877 | 2 | 0 | 0 | 21875 | 0 | 0 | 0 | 5979 |
| native-pressure | 1 | preflop | 40907 | 3 | 0 | 0 | 40904 | 0 | 0 | 0 | 13111 |
| native-pressure | 1 | river | 4389 | 30 | 0 | 0 | 4359 | 0 | 0 | 0 | 2927 |
| native-pressure | 1 | turn | 9453 | 31 | 0 | 0 | 9422 | 0 | 0 | 0 | 2559 |
| passive | 1 | flop | 441 | 0 | 0 | 0 | 441 | 0 | 0 | 0 | 0 |
| passive | 1 | preflop | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 71 |
| passive | 1 | river | 430 | 0 | 0 | 0 | 430 | 0 | 0 | 0 | 430 |
| passive | 1 | turn | 441 | 0 | 0 | 0 | 441 | 0 | 0 | 0 | 11 |
| pot_pressure | 1 | flop | 95 | 31 | 0 | 0 | 64 | 0 | 0 | 0 | 59 |
| pot_pressure | 1 | preflop | 347 | 65 | 0 | 0 | 282 | 0 | 0 | 0 | 423 |
| pot_pressure | 1 | river | 15 | 8 | 0 | 0 | 7 | 0 | 0 | 0 | 10 |
| pot_pressure | 1 | turn | 32 | 16 | 0 | 0 | 16 | 0 | 0 | 0 | 20 |
| pressure-cap2 | 1 | flop | 336 | 0 | 0 | 0 | 336 | 0 | 0 | 0 | 116 |
| pressure-cap2 | 1 | preflop | 697 | 0 | 0 | 0 | 697 | 0 | 0 | 0 | 282 |
| pressure-cap2 | 1 | river | 96 | 0 | 0 | 0 | 96 | 0 | 0 | 0 | 70 |
| pressure-cap2 | 1 | turn | 159 | 1 | 0 | 0 | 158 | 0 | 0 | 0 | 44 |
| selective-stackoff | 1 | flop | 152 | 0 | 0 | 0 | 152 | 0 | 0 | 0 | 55 |
| selective-stackoff | 1 | preflop | 332 | 0 | 0 | 0 | 332 | 0 | 0 | 0 | 362 |
| selective-stackoff | 1 | river | 71 | 0 | 0 | 0 | 71 | 0 | 0 | 0 | 66 |
| selective-stackoff | 1 | turn | 104 | 0 | 0 | 0 | 104 | 0 | 0 | 0 | 29 |
| tight_aggressive | 1 | flop | 95 | 0 | 0 | 0 | 95 | 0 | 0 | 0 | 50 |
| tight_aggressive | 1 | preflop | 295 | 0 | 0 | 0 | 295 | 0 | 0 | 0 | 422 |
| tight_aggressive | 1 | river | 14 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 13 |
| tight_aggressive | 1 | turn | 43 | 0 | 0 | 0 | 43 | 0 | 0 | 0 | 27 |
| tight_passive | 1 | flop | 96 | 0 | 0 | 0 | 96 | 0 | 0 | 0 | 52 |
| tight_passive | 1 | preflop | 276 | 0 | 0 | 0 | 276 | 0 | 0 | 0 | 416 |
| tight_passive | 1 | river | 28 | 0 | 0 | 0 | 28 | 0 | 0 | 0 | 28 |
| tight_passive | 1 | turn | 45 | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 16 |
| train_pressure | 1 | flop | 169 | 7 | 0 | 0 | 162 | 0 | 0 | 0 | 58 |
| train_pressure | 1 | preflop | 356 | 4 | 0 | 0 | 352 | 0 | 0 | 0 | 356 |
| train_pressure | 1 | river | 84 | 13 | 0 | 0 | 71 | 0 | 0 | 0 | 66 |
| train_pressure | 1 | turn | 108 | 7 | 0 | 0 | 101 | 0 | 0 | 0 | 32 |
| uniform | 1 | flop | 320 | 0 | 0 | 0 | 320 | 0 | 0 | 0 | 122 |
| uniform | 1 | preflop | 593 | 0 | 0 | 0 | 593 | 0 | 0 | 0 | 282 |
| uniform | 1 | river | 71 | 0 | 0 | 0 | 71 | 0 | 0 | 0 | 52 |
| uniform | 1 | turn | 152 | 0 | 0 | 0 | 152 | 0 | 0 | 0 | 56 |
| lbr | 2 | flop | 2901 | 0 | 0 | 0 | 2901 | 2797 | 0 | 0 | 1354 |
| lbr | 2 | preflop | 5240 | 0 | 0 | 0 | 5240 | 4429 | 0 | 0 | 1950 |
| lbr | 2 | river | 336 | 0 | 0 | 0 | 336 | 409 | 0 | 0 | 304 |
| lbr | 2 | turn | 933 | 0 | 0 | 0 | 933 | 1075 | 0 | 0 | 488 |
| loose_aggressive | 2 | flop | 205 | 0 | 0 | 0 | 205 | 0 | 0 | 0 | 65 |
| loose_aggressive | 2 | preflop | 422 | 0 | 0 | 0 | 422 | 0 | 0 | 0 | 329 |
| loose_aggressive | 2 | river | 91 | 0 | 0 | 0 | 91 | 0 | 0 | 0 | 70 |
| loose_aggressive | 2 | turn | 138 | 0 | 0 | 0 | 138 | 0 | 0 | 0 | 48 |
| loose_passive | 2 | flop | 275 | 0 | 0 | 0 | 275 | 0 | 0 | 0 | 61 |
| loose_passive | 2 | preflop | 410 | 0 | 0 | 0 | 410 | 0 | 0 | 0 | 238 |
| loose_passive | 2 | river | 188 | 0 | 0 | 0 | 188 | 0 | 0 | 0 | 186 |
| loose_passive | 2 | turn | 214 | 1 | 0 | 0 | 213 | 0 | 0 | 0 | 27 |
| minraise-cap2 | 2 | flop | 475 | 0 | 0 | 0 | 475 | 0 | 0 | 0 | 130 |
| minraise-cap2 | 2 | preflop | 697 | 0 | 0 | 0 | 697 | 0 | 0 | 0 | 187 |
| minraise-cap2 | 2 | river | 159 | 0 | 0 | 0 | 159 | 0 | 0 | 0 | 121 |
| minraise-cap2 | 2 | turn | 267 | 0 | 0 | 0 | 267 | 0 | 0 | 0 | 74 |
| native-pressure | 2 | flop | 21714 | 2 | 0 | 0 | 21712 | 0 | 0 | 0 | 6420 |
| native-pressure | 2 | preflop | 40794 | 4 | 0 | 0 | 40790 | 0 | 0 | 0 | 13128 |
| native-pressure | 2 | river | 4003 | 19 | 0 | 0 | 3984 | 0 | 0 | 0 | 2623 |
| native-pressure | 2 | turn | 8640 | 24 | 0 | 0 | 8616 | 0 | 0 | 0 | 2405 |
| passive | 2 | flop | 441 | 0 | 0 | 0 | 441 | 0 | 0 | 0 | 0 |
| passive | 2 | preflop | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 71 |
| passive | 2 | river | 429 | 0 | 0 | 0 | 429 | 0 | 0 | 0 | 429 |
| passive | 2 | turn | 441 | 0 | 0 | 0 | 441 | 0 | 0 | 0 | 12 |
| pot_pressure | 2 | flop | 104 | 33 | 0 | 0 | 71 | 0 | 0 | 0 | 57 |
| pot_pressure | 2 | preflop | 344 | 63 | 0 | 0 | 281 | 0 | 0 | 0 | 417 |
| pot_pressure | 2 | river | 24 | 11 | 0 | 0 | 13 | 0 | 0 | 0 | 18 |
| pot_pressure | 2 | turn | 41 | 21 | 0 | 0 | 20 | 0 | 0 | 0 | 20 |
| pressure-cap2 | 2 | flop | 348 | 0 | 0 | 0 | 348 | 0 | 0 | 0 | 112 |
| pressure-cap2 | 2 | preflop | 697 | 0 | 0 | 0 | 697 | 0 | 0 | 0 | 275 |
| pressure-cap2 | 2 | river | 107 | 0 | 0 | 0 | 107 | 0 | 0 | 0 | 80 |
| pressure-cap2 | 2 | turn | 171 | 0 | 0 | 0 | 171 | 0 | 0 | 0 | 45 |
| selective-stackoff | 2 | flop | 151 | 0 | 0 | 0 | 151 | 0 | 0 | 0 | 48 |
| selective-stackoff | 2 | preflop | 331 | 0 | 0 | 0 | 331 | 0 | 0 | 0 | 364 |
| selective-stackoff | 2 | river | 84 | 1 | 0 | 0 | 83 | 0 | 0 | 0 | 76 |
| selective-stackoff | 2 | turn | 112 | 1 | 0 | 0 | 111 | 0 | 0 | 0 | 24 |
| tight_aggressive | 2 | flop | 94 | 0 | 0 | 0 | 94 | 0 | 0 | 0 | 48 |
| tight_aggressive | 2 | preflop | 298 | 0 | 0 | 0 | 298 | 0 | 0 | 0 | 423 |
| tight_aggressive | 2 | river | 23 | 0 | 0 | 0 | 23 | 0 | 0 | 0 | 20 |
| tight_aggressive | 2 | turn | 45 | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 21 |
| tight_passive | 2 | flop | 100 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 52 |
| tight_passive | 2 | preflop | 276 | 0 | 0 | 0 | 276 | 0 | 0 | 0 | 412 |
| tight_passive | 2 | river | 38 | 0 | 0 | 0 | 38 | 0 | 0 | 0 | 38 |
| tight_passive | 2 | turn | 49 | 0 | 0 | 0 | 49 | 0 | 0 | 0 | 10 |
| train_pressure | 2 | flop | 172 | 5 | 0 | 0 | 167 | 0 | 0 | 0 | 58 |
| train_pressure | 2 | preflop | 353 | 4 | 0 | 0 | 349 | 0 | 0 | 0 | 355 |
| train_pressure | 2 | river | 88 | 17 | 0 | 0 | 71 | 0 | 0 | 0 | 70 |
| train_pressure | 2 | turn | 112 | 9 | 0 | 0 | 103 | 0 | 0 | 0 | 29 |
| uniform | 2 | flop | 300 | 0 | 0 | 0 | 300 | 0 | 0 | 0 | 121 |
| uniform | 2 | preflop | 593 | 0 | 0 | 0 | 593 | 0 | 0 | 0 | 300 |
| uniform | 2 | river | 66 | 0 | 0 | 0 | 66 | 0 | 0 | 0 | 48 |
| uniform | 2 | turn | 122 | 0 | 0 | 0 | 122 | 0 | 0 | 0 | 43 |
| lbr | 3 | flop | 3029 | 0 | 0 | 0 | 3029 | 2878 | 0 | 0 | 1347 |
| lbr | 3 | preflop | 5265 | 0 | 0 | 0 | 5265 | 4437 | 0 | 0 | 1927 |
| lbr | 3 | river | 379 | 1 | 0 | 0 | 378 | 411 | 0 | 0 | 324 |
| lbr | 3 | turn | 1002 | 0 | 0 | 0 | 1002 | 1102 | 0 | 0 | 498 |
| loose_aggressive | 3 | flop | 211 | 0 | 0 | 0 | 211 | 0 | 0 | 0 | 74 |
| loose_aggressive | 3 | preflop | 424 | 0 | 0 | 0 | 424 | 0 | 0 | 0 | 324 |
| loose_aggressive | 3 | river | 82 | 0 | 0 | 0 | 82 | 0 | 0 | 0 | 64 |
| loose_aggressive | 3 | turn | 132 | 0 | 0 | 0 | 132 | 0 | 0 | 0 | 50 |
| loose_passive | 3 | flop | 279 | 1 | 0 | 0 | 278 | 0 | 0 | 0 | 72 |
| loose_passive | 3 | preflop | 410 | 0 | 0 | 0 | 410 | 0 | 0 | 0 | 234 |
| loose_passive | 3 | river | 179 | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 178 |
| loose_passive | 3 | turn | 207 | 0 | 0 | 0 | 207 | 0 | 0 | 0 | 28 |
| minraise-cap2 | 3 | flop | 485 | 1 | 0 | 0 | 484 | 0 | 0 | 0 | 113 |
| minraise-cap2 | 3 | preflop | 702 | 0 | 0 | 0 | 702 | 0 | 0 | 0 | 187 |
| minraise-cap2 | 3 | river | 174 | 0 | 0 | 0 | 174 | 0 | 0 | 0 | 133 |
| minraise-cap2 | 3 | turn | 292 | 0 | 0 | 0 | 292 | 0 | 0 | 0 | 79 |
| native-pressure | 3 | flop | 22084 | 5 | 0 | 0 | 22079 | 0 | 0 | 0 | 5877 |
| native-pressure | 3 | preflop | 40558 | 8 | 0 | 0 | 40550 | 0 | 0 | 0 | 12908 |
| native-pressure | 3 | river | 4517 | 31 | 0 | 0 | 4486 | 0 | 0 | 0 | 2998 |
| native-pressure | 3 | turn | 10056 | 18 | 0 | 0 | 10038 | 0 | 0 | 0 | 2793 |
| passive | 3 | flop | 446 | 0 | 0 | 0 | 446 | 0 | 0 | 0 | 0 |
| passive | 3 | preflop | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 66 |
| passive | 3 | river | 435 | 0 | 0 | 0 | 435 | 0 | 0 | 0 | 435 |
| passive | 3 | turn | 446 | 0 | 0 | 0 | 446 | 0 | 0 | 0 | 11 |
| pot_pressure | 3 | flop | 109 | 39 | 0 | 0 | 70 | 0 | 0 | 0 | 67 |
| pot_pressure | 3 | preflop | 346 | 64 | 0 | 0 | 282 | 0 | 0 | 0 | 411 |
| pot_pressure | 3 | river | 20 | 9 | 0 | 0 | 11 | 0 | 0 | 0 | 14 |
| pot_pressure | 3 | turn | 36 | 17 | 0 | 0 | 19 | 0 | 0 | 0 | 20 |
| pressure-cap2 | 3 | flop | 356 | 1 | 0 | 0 | 355 | 0 | 0 | 0 | 98 |
| pressure-cap2 | 3 | preflop | 702 | 0 | 0 | 0 | 702 | 0 | 0 | 0 | 275 |
| pressure-cap2 | 3 | river | 112 | 0 | 0 | 0 | 112 | 0 | 0 | 0 | 83 |
| pressure-cap2 | 3 | turn | 195 | 0 | 0 | 0 | 195 | 0 | 0 | 0 | 56 |
| selective-stackoff | 3 | flop | 162 | 0 | 0 | 0 | 162 | 0 | 0 | 0 | 62 |
| selective-stackoff | 3 | preflop | 331 | 0 | 0 | 0 | 331 | 0 | 0 | 0 | 354 |
| selective-stackoff | 3 | river | 73 | 0 | 0 | 0 | 73 | 0 | 0 | 0 | 66 |
| selective-stackoff | 3 | turn | 106 | 0 | 0 | 0 | 106 | 0 | 0 | 0 | 30 |
| tight_aggressive | 3 | flop | 104 | 0 | 0 | 0 | 104 | 0 | 0 | 0 | 57 |
| tight_aggressive | 3 | preflop | 299 | 0 | 0 | 0 | 299 | 0 | 0 | 0 | 414 |
| tight_aggressive | 3 | river | 16 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 15 |
| tight_aggressive | 3 | turn | 43 | 0 | 0 | 0 | 43 | 0 | 0 | 0 | 26 |
| tight_passive | 3 | flop | 103 | 0 | 0 | 0 | 103 | 0 | 0 | 0 | 57 |
| tight_passive | 3 | preflop | 276 | 0 | 0 | 0 | 276 | 0 | 0 | 0 | 409 |
| tight_passive | 3 | river | 32 | 0 | 0 | 0 | 32 | 0 | 0 | 0 | 32 |
| tight_passive | 3 | turn | 47 | 1 | 0 | 0 | 46 | 0 | 0 | 0 | 14 |
| train_pressure | 3 | flop | 175 | 6 | 0 | 0 | 169 | 0 | 0 | 0 | 65 |
| train_pressure | 3 | preflop | 357 | 9 | 0 | 0 | 348 | 0 | 0 | 0 | 350 |
| train_pressure | 3 | river | 77 | 13 | 0 | 0 | 64 | 0 | 0 | 0 | 60 |
| train_pressure | 3 | turn | 109 | 10 | 0 | 0 | 99 | 0 | 0 | 0 | 37 |
| uniform | 3 | flop | 299 | 0 | 0 | 0 | 299 | 0 | 0 | 0 | 119 |
| uniform | 3 | preflop | 598 | 0 | 0 | 0 | 598 | 0 | 0 | 0 | 295 |
| uniform | 3 | river | 56 | 1 | 0 | 0 | 55 | 0 | 0 | 0 | 41 |
| uniform | 3 | turn | 137 | 0 | 0 | 0 | 137 | 0 | 0 | 0 | 57 |

## T: street coverage and LBR accounting

| Panel | Lineage | Street | decisions | missing | zero_mass | positive_mass | current | lbr_decisions | lbr_incomplete | lbr_over_soft_budget | Terminal hands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lbr | 1 | flop | 3029 | 0 | 0 | 3029 | 0 | 2757 | 0 | 0 | 1183 |
| lbr | 1 | preflop | 5541 | 0 | 0 | 5541 | 0 | 4423 | 0 | 0 | 1898 |
| lbr | 1 | river | 573 | 1 | 0 | 572 | 0 | 638 | 0 | 0 | 501 |
| lbr | 1 | turn | 1236 | 0 | 0 | 1236 | 0 | 1290 | 0 | 0 | 514 |
| loose_aggressive | 1 | flop | 230 | 0 | 0 | 230 | 0 | 0 | 0 | 0 | 78 |
| loose_aggressive | 1 | preflop | 427 | 0 | 0 | 427 | 0 | 0 | 0 | 0 | 308 |
| loose_aggressive | 1 | river | 93 | 0 | 0 | 93 | 0 | 0 | 0 | 0 | 73 |
| loose_aggressive | 1 | turn | 145 | 0 | 0 | 145 | 0 | 0 | 0 | 0 | 53 |
| loose_passive | 1 | flop | 293 | 1 | 0 | 292 | 0 | 0 | 0 | 0 | 75 |
| loose_passive | 1 | preflop | 409 | 1 | 0 | 408 | 0 | 0 | 0 | 0 | 220 |
| loose_passive | 1 | river | 194 | 1 | 0 | 193 | 0 | 0 | 0 | 0 | 194 |
| loose_passive | 1 | turn | 218 | 1 | 0 | 217 | 0 | 0 | 0 | 0 | 23 |
| minraise-cap2 | 1 | flop | 536 | 0 | 0 | 536 | 0 | 0 | 0 | 0 | 102 |
| minraise-cap2 | 1 | preflop | 721 | 0 | 0 | 721 | 0 | 0 | 0 | 0 | 152 |
| minraise-cap2 | 1 | river | 263 | 0 | 0 | 263 | 0 | 0 | 0 | 0 | 196 |
| minraise-cap2 | 1 | turn | 364 | 0 | 0 | 364 | 0 | 0 | 0 | 0 | 62 |
| native-pressure | 1 | flop | 27165 | 3 | 2 | 27160 | 0 | 0 | 0 | 0 | 5119 |
| native-pressure | 1 | preflop | 40904 | 0 | 0 | 40904 | 0 | 0 | 0 | 0 | 10240 |
| native-pressure | 1 | river | 8834 | 16 | 10 | 8808 | 0 | 0 | 0 | 0 | 5875 |
| native-pressure | 1 | turn | 16574 | 11 | 21 | 16542 | 0 | 0 | 0 | 0 | 3342 |
| passive | 1 | flop | 465 | 0 | 0 | 465 | 0 | 0 | 0 | 0 | 0 |
| passive | 1 | preflop | 512 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 47 |
| passive | 1 | river | 456 | 0 | 0 | 456 | 0 | 0 | 0 | 0 | 456 |
| passive | 1 | turn | 465 | 0 | 0 | 465 | 0 | 0 | 0 | 0 | 9 |
| pot_pressure | 1 | flop | 123 | 45 | 0 | 78 | 0 | 0 | 0 | 0 | 73 |
| pot_pressure | 1 | preflop | 350 | 67 | 0 | 283 | 0 | 0 | 0 | 0 | 399 |
| pot_pressure | 1 | river | 25 | 12 | 0 | 13 | 0 | 0 | 0 | 0 | 19 |
| pot_pressure | 1 | turn | 43 | 22 | 0 | 21 | 0 | 0 | 0 | 0 | 21 |
| pressure-cap2 | 1 | flop | 421 | 0 | 0 | 421 | 0 | 0 | 0 | 0 | 116 |
| pressure-cap2 | 1 | preflop | 721 | 0 | 0 | 721 | 0 | 0 | 0 | 0 | 226 |
| pressure-cap2 | 1 | river | 164 | 0 | 0 | 164 | 0 | 0 | 0 | 0 | 125 |
| pressure-cap2 | 1 | turn | 233 | 0 | 0 | 233 | 0 | 0 | 0 | 0 | 45 |
| selective-stackoff | 1 | flop | 176 | 0 | 0 | 176 | 0 | 0 | 0 | 0 | 72 |
| selective-stackoff | 1 | preflop | 333 | 0 | 0 | 333 | 0 | 0 | 0 | 0 | 339 |
| selective-stackoff | 1 | river | 87 | 0 | 0 | 87 | 0 | 0 | 0 | 0 | 79 |
| selective-stackoff | 1 | turn | 106 | 0 | 0 | 106 | 0 | 0 | 0 | 0 | 22 |
| tight_aggressive | 1 | flop | 118 | 0 | 0 | 118 | 0 | 0 | 0 | 0 | 67 |
| tight_aggressive | 1 | preflop | 295 | 0 | 0 | 295 | 0 | 0 | 0 | 0 | 400 |
| tight_aggressive | 1 | river | 19 | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 18 |
| tight_aggressive | 1 | turn | 51 | 0 | 0 | 51 | 0 | 0 | 0 | 0 | 27 |
| tight_passive | 1 | flop | 119 | 1 | 0 | 118 | 0 | 0 | 0 | 0 | 72 |
| tight_passive | 1 | preflop | 276 | 1 | 0 | 275 | 0 | 0 | 0 | 0 | 393 |
| tight_passive | 1 | river | 34 | 0 | 0 | 34 | 0 | 0 | 0 | 0 | 34 |
| tight_passive | 1 | turn | 48 | 0 | 0 | 48 | 0 | 0 | 0 | 0 | 13 |
| train_pressure | 1 | flop | 197 | 7 | 0 | 190 | 0 | 0 | 0 | 0 | 71 |
| train_pressure | 1 | preflop | 357 | 8 | 0 | 349 | 0 | 0 | 0 | 0 | 333 |
| train_pressure | 1 | river | 85 | 16 | 0 | 69 | 0 | 0 | 0 | 0 | 72 |
| train_pressure | 1 | turn | 123 | 11 | 0 | 112 | 0 | 0 | 0 | 0 | 36 |
| uniform | 1 | flop | 346 | 0 | 0 | 346 | 0 | 0 | 0 | 0 | 116 |
| uniform | 1 | preflop | 604 | 0 | 0 | 604 | 0 | 0 | 0 | 0 | 266 |
| uniform | 1 | river | 92 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 68 |
| uniform | 1 | turn | 173 | 0 | 0 | 173 | 0 | 0 | 0 | 0 | 62 |
| lbr | 2 | flop | 3010 | 0 | 0 | 3010 | 0 | 2770 | 0 | 0 | 1225 |
| lbr | 2 | preflop | 5441 | 0 | 0 | 5441 | 0 | 4457 | 0 | 0 | 1859 |
| lbr | 2 | river | 621 | 0 | 0 | 621 | 0 | 693 | 0 | 0 | 530 |
| lbr | 2 | turn | 1201 | 0 | 0 | 1201 | 0 | 1292 | 0 | 0 | 482 |
| loose_aggressive | 2 | flop | 220 | 0 | 0 | 220 | 0 | 0 | 0 | 0 | 73 |
| loose_aggressive | 2 | preflop | 432 | 0 | 0 | 432 | 0 | 0 | 0 | 0 | 318 |
| loose_aggressive | 2 | river | 82 | 0 | 0 | 82 | 0 | 0 | 0 | 0 | 66 |
| loose_aggressive | 2 | turn | 136 | 0 | 0 | 136 | 0 | 0 | 0 | 0 | 55 |
| loose_passive | 2 | flop | 292 | 0 | 0 | 292 | 0 | 0 | 0 | 0 | 70 |
| loose_passive | 2 | preflop | 409 | 0 | 0 | 409 | 0 | 0 | 0 | 0 | 222 |
| loose_passive | 2 | river | 196 | 3 | 0 | 193 | 0 | 0 | 0 | 0 | 195 |
| loose_passive | 2 | turn | 222 | 4 | 0 | 218 | 0 | 0 | 0 | 0 | 25 |
| minraise-cap2 | 2 | flop | 536 | 0 | 0 | 536 | 0 | 0 | 0 | 0 | 116 |
| minraise-cap2 | 2 | preflop | 720 | 0 | 0 | 720 | 0 | 0 | 0 | 0 | 149 |
| minraise-cap2 | 2 | river | 265 | 0 | 0 | 265 | 0 | 0 | 0 | 0 | 195 |
| minraise-cap2 | 2 | turn | 353 | 0 | 0 | 353 | 0 | 0 | 0 | 0 | 52 |
| native-pressure | 2 | flop | 28136 | 6 | 2 | 28128 | 0 | 0 | 0 | 0 | 5351 |
| native-pressure | 2 | preflop | 40854 | 0 | 0 | 40854 | 0 | 0 | 0 | 0 | 9862 |
| native-pressure | 2 | river | 9046 | 18 | 9 | 9019 | 0 | 0 | 0 | 0 | 5943 |
| native-pressure | 2 | turn | 16856 | 8 | 2 | 16846 | 0 | 0 | 0 | 0 | 3420 |
| passive | 2 | flop | 464 | 0 | 0 | 464 | 0 | 0 | 0 | 0 | 0 |
| passive | 2 | preflop | 512 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 48 |
| passive | 2 | river | 453 | 0 | 0 | 453 | 0 | 0 | 0 | 0 | 453 |
| passive | 2 | turn | 464 | 0 | 0 | 464 | 0 | 0 | 0 | 0 | 11 |
| pot_pressure | 2 | flop | 119 | 43 | 0 | 76 | 0 | 0 | 0 | 0 | 68 |
| pot_pressure | 2 | preflop | 352 | 70 | 0 | 282 | 0 | 0 | 0 | 0 | 404 |
| pot_pressure | 2 | river | 28 | 16 | 0 | 12 | 0 | 0 | 0 | 0 | 20 |
| pot_pressure | 2 | turn | 42 | 21 | 0 | 21 | 0 | 0 | 0 | 0 | 20 |
| pressure-cap2 | 2 | flop | 416 | 0 | 0 | 416 | 0 | 0 | 0 | 0 | 117 |
| pressure-cap2 | 2 | preflop | 720 | 0 | 0 | 720 | 0 | 0 | 0 | 0 | 230 |
| pressure-cap2 | 2 | river | 166 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 125 |
| pressure-cap2 | 2 | turn | 231 | 0 | 0 | 231 | 0 | 0 | 0 | 0 | 40 |
| selective-stackoff | 2 | flop | 172 | 0 | 0 | 172 | 0 | 0 | 0 | 0 | 65 |
| selective-stackoff | 2 | preflop | 331 | 0 | 0 | 331 | 0 | 0 | 0 | 0 | 342 |
| selective-stackoff | 2 | river | 90 | 0 | 0 | 90 | 0 | 0 | 0 | 0 | 80 |
| selective-stackoff | 2 | turn | 112 | 0 | 0 | 112 | 0 | 0 | 0 | 0 | 25 |
| tight_aggressive | 2 | flop | 115 | 0 | 0 | 115 | 0 | 0 | 0 | 0 | 63 |
| tight_aggressive | 2 | preflop | 297 | 0 | 0 | 297 | 0 | 0 | 0 | 0 | 405 |
| tight_aggressive | 2 | river | 23 | 0 | 0 | 23 | 0 | 0 | 0 | 0 | 20 |
| tight_aggressive | 2 | turn | 47 | 0 | 0 | 47 | 0 | 0 | 0 | 0 | 24 |
| tight_passive | 2 | flop | 113 | 0 | 0 | 113 | 0 | 0 | 0 | 0 | 61 |
| tight_passive | 2 | preflop | 276 | 0 | 0 | 276 | 0 | 0 | 0 | 0 | 399 |
| tight_passive | 2 | river | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 40 |
| tight_passive | 2 | turn | 53 | 0 | 0 | 53 | 0 | 0 | 0 | 0 | 12 |
| train_pressure | 2 | flop | 188 | 8 | 0 | 180 | 0 | 0 | 0 | 0 | 66 |
| train_pressure | 2 | preflop | 356 | 12 | 0 | 344 | 0 | 0 | 0 | 0 | 343 |
| train_pressure | 2 | river | 88 | 13 | 0 | 75 | 0 | 0 | 0 | 0 | 70 |
| train_pressure | 2 | turn | 116 | 11 | 0 | 105 | 0 | 0 | 0 | 0 | 33 |
| uniform | 2 | flop | 333 | 0 | 0 | 333 | 0 | 0 | 0 | 0 | 122 |
| uniform | 2 | preflop | 610 | 0 | 0 | 610 | 0 | 0 | 0 | 0 | 271 |
| uniform | 2 | river | 95 | 0 | 0 | 95 | 0 | 0 | 0 | 0 | 69 |
| uniform | 2 | turn | 162 | 0 | 0 | 162 | 0 | 0 | 0 | 0 | 50 |
| lbr | 3 | flop | 2895 | 0 | 0 | 2895 | 0 | 2658 | 0 | 0 | 1172 |
| lbr | 3 | preflop | 5562 | 0 | 0 | 5562 | 0 | 4452 | 0 | 0 | 1970 |
| lbr | 3 | river | 582 | 0 | 0 | 582 | 0 | 647 | 0 | 0 | 500 |
| lbr | 3 | turn | 1128 | 0 | 0 | 1128 | 0 | 1194 | 0 | 0 | 454 |
| loose_aggressive | 3 | flop | 227 | 0 | 0 | 227 | 0 | 0 | 0 | 0 | 68 |
| loose_aggressive | 3 | preflop | 433 | 0 | 0 | 433 | 0 | 0 | 0 | 0 | 312 |
| loose_aggressive | 3 | river | 90 | 0 | 0 | 90 | 0 | 0 | 0 | 0 | 73 |
| loose_aggressive | 3 | turn | 148 | 0 | 0 | 148 | 0 | 0 | 0 | 0 | 59 |
| loose_passive | 3 | flop | 297 | 0 | 0 | 297 | 0 | 0 | 0 | 0 | 73 |
| loose_passive | 3 | preflop | 409 | 0 | 0 | 409 | 0 | 0 | 0 | 0 | 216 |
| loose_passive | 3 | river | 201 | 1 | 0 | 200 | 0 | 0 | 0 | 0 | 200 |
| loose_passive | 3 | turn | 224 | 2 | 0 | 222 | 0 | 0 | 0 | 0 | 23 |
| minraise-cap2 | 3 | flop | 514 | 0 | 0 | 514 | 0 | 0 | 0 | 0 | 111 |
| minraise-cap2 | 3 | preflop | 718 | 0 | 0 | 718 | 0 | 0 | 0 | 0 | 163 |
| minraise-cap2 | 3 | river | 246 | 0 | 0 | 246 | 0 | 0 | 0 | 0 | 180 |
| minraise-cap2 | 3 | turn | 340 | 0 | 0 | 340 | 0 | 0 | 0 | 0 | 58 |
| native-pressure | 3 | flop | 26628 | 2 | 1 | 26625 | 0 | 0 | 0 | 0 | 5387 |
| native-pressure | 3 | preflop | 40945 | 1 | 1 | 40943 | 0 | 0 | 0 | 0 | 10425 |
| native-pressure | 3 | river | 8443 | 15 | 13 | 8415 | 0 | 0 | 0 | 0 | 5589 |
| native-pressure | 3 | turn | 15703 | 15 | 6 | 15682 | 0 | 0 | 0 | 0 | 3175 |
| passive | 3 | flop | 462 | 0 | 0 | 462 | 0 | 0 | 0 | 0 | 0 |
| passive | 3 | preflop | 512 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 50 |
| passive | 3 | river | 451 | 0 | 0 | 451 | 0 | 0 | 0 | 0 | 451 |
| passive | 3 | turn | 462 | 0 | 0 | 462 | 0 | 0 | 0 | 0 | 11 |
| pot_pressure | 3 | flop | 122 | 44 | 0 | 78 | 0 | 0 | 0 | 0 | 68 |
| pot_pressure | 3 | preflop | 352 | 70 | 0 | 282 | 0 | 0 | 0 | 0 | 400 |
| pot_pressure | 3 | river | 25 | 14 | 0 | 11 | 0 | 0 | 0 | 0 | 19 |
| pot_pressure | 3 | turn | 46 | 22 | 0 | 24 | 0 | 0 | 0 | 0 | 25 |
| pressure-cap2 | 3 | flop | 393 | 0 | 0 | 393 | 0 | 0 | 0 | 0 | 112 |
| pressure-cap2 | 3 | preflop | 718 | 0 | 0 | 718 | 0 | 0 | 0 | 0 | 241 |
| pressure-cap2 | 3 | river | 154 | 0 | 0 | 154 | 0 | 0 | 0 | 0 | 114 |
| pressure-cap2 | 3 | turn | 220 | 0 | 0 | 220 | 0 | 0 | 0 | 0 | 45 |
| selective-stackoff | 3 | flop | 172 | 0 | 0 | 172 | 0 | 0 | 0 | 0 | 62 |
| selective-stackoff | 3 | preflop | 331 | 0 | 0 | 331 | 0 | 0 | 0 | 0 | 343 |
| selective-stackoff | 3 | river | 85 | 0 | 0 | 85 | 0 | 0 | 0 | 0 | 80 |
| selective-stackoff | 3 | turn | 114 | 0 | 0 | 114 | 0 | 0 | 0 | 0 | 27 |
| tight_aggressive | 3 | flop | 122 | 0 | 0 | 122 | 0 | 0 | 0 | 0 | 60 |
| tight_aggressive | 3 | preflop | 295 | 0 | 0 | 295 | 0 | 0 | 0 | 0 | 400 |
| tight_aggressive | 3 | river | 27 | 0 | 0 | 27 | 0 | 0 | 0 | 0 | 25 |
| tight_aggressive | 3 | turn | 55 | 0 | 0 | 55 | 0 | 0 | 0 | 0 | 27 |
| tight_passive | 3 | flop | 116 | 0 | 0 | 116 | 0 | 0 | 0 | 0 | 58 |
| tight_passive | 3 | preflop | 276 | 0 | 0 | 276 | 0 | 0 | 0 | 0 | 396 |
| tight_passive | 3 | river | 44 | 0 | 0 | 44 | 0 | 0 | 0 | 0 | 44 |
| tight_passive | 3 | turn | 59 | 0 | 0 | 59 | 0 | 0 | 0 | 0 | 14 |
| train_pressure | 3 | flop | 200 | 8 | 0 | 192 | 0 | 0 | 0 | 0 | 62 |
| train_pressure | 3 | preflop | 352 | 4 | 0 | 348 | 0 | 0 | 0 | 0 | 333 |
| train_pressure | 3 | river | 88 | 18 | 0 | 70 | 0 | 0 | 0 | 0 | 73 |
| train_pressure | 3 | turn | 130 | 21 | 0 | 109 | 0 | 0 | 0 | 0 | 44 |
| uniform | 3 | flop | 336 | 0 | 0 | 336 | 0 | 0 | 0 | 0 | 117 |
| uniform | 3 | preflop | 593 | 0 | 0 | 593 | 0 | 0 | 0 | 0 | 275 |
| uniform | 3 | river | 81 | 0 | 0 | 81 | 0 | 0 | 0 | 0 | 64 |
| uniform | 3 | turn | 164 | 0 | 0 | 164 | 0 | 0 | 0 | 0 | 56 |

## Retained earlier exploratory direct test

Fresh root 202610060901; three matched pairs, 12,288 blocks. Outside the frozen release rule and separate from the later nine-pair confirmation. Source summary and independent audit are retained.

| Earlier matched-pair result | BB/100 [95%] |
| --- | --- |
| Overall | -14.19 [-18.97, -9.40] |
| Pair 1 | -13.73 [-20.48, -6.97] |
| Pair 2 | -15.31 [-22.04, -8.59] |
| Pair 3 | -13.52 [-20.32, -6.71] |

## Audit artifacts

[Arena](hu20-cfr-plus-artifacts/arena-audit.json), [fresh direct](hu20-cfr-plus-artifacts/direct-nine-audit.json), [earlier direct](hu20-cfr-plus-artifacts/direct-audit.json), [full-export turn/river](hu20-cfr-plus-artifacts/full-export-turn-loss-audit.json), [pressure reconciliation](hu20-cfr-plus-artifacts/pressure-description.json).
