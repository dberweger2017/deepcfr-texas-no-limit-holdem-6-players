# CFR+ progress against v0.4.0

PR [#171](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/171) is merged. Review found no unresolved issue; 27 native parity tests and four Rust tests pass, and all GitHub CI checks are green. Three 1B-node lineages are training from main `60f516d` with `--regret-floor 0`, seeds 2026100601/02/03 and production traverser-reach averaging.

The [predeclared protocol](../hu20-cfr-plus-confirmation.md) reuses #165's frozen panels and release rule. Native pressure gets 12,288 blocks, versus 2,048 previously: the previous per-block variance predicts a paired 95% half-width of 7.72 BB/100. LBR gets 2,048 blocks and each other panel 256. A separate direct candidate–v0.4.0 match gets 12,288 blocks. All compute is local M1/M4; no rental or paid compute.

**Results and the release decision are pending.** The completed page will show direct-match win rate, absolute panel performance, paired panel changes, absolute LBR-opponent win rates, exact turn/river bench E/Q with blueprint and witness references, a chart and artifact hashes. Panel changes measure relative performance against shared opponents; they are distinct from direct head-to-head play. The exact bench trains only on fixed turn/river spots and does not establish full-game strength.

The candidate ships only if its primary average−v0.4.0 contrast has LBR lower bound >0, native-pressure lower bound >−10, and no other panel upper bound <−20 BB/100, with paired 95% intervals. Passing still requires the owner's confirmation before releasing v0.4.1.
