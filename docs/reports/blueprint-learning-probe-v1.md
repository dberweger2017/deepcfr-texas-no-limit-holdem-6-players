# Early blueprint learning probe

The retained M4 checkpoint at iteration 8,733 and 5,834,622 entries was queried directly at one fixed six-player, 100 BB, first-to-act preflop decision. This avoided the unmeasured large-policy export on the 16 GiB M4. The source checkpoint SHA-256 is `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`. The [existing diagnostic](../../src/blueprint/diagnostics.py) holds the public decision fixed and varies only the acting player's two cards across all 169 canonical classes.

| Policy derived from checkpoint | Trained classes | Distinct distributions | AA fold / call / min / pot | 72o fold / call / min / pot |
| --- | ---: | ---: | --- | --- |
| Current regret-matched | 169/169 | 105 | 0% / 99.45% / 0.55% / 0% | 100% / 0% / 0% / 0% |
| Iteration-weighted average | 169/169 | 169 | 0.02% / 44.17% / 31.77% / 24.04% | 85.94% / 0.91% / 12.66% / 0.49% |

This establishes that the checkpoint is no longer a uniform table at this decision and that card class affects the action distribution. It does **not** show profitable or well-balanced play. In particular, the current policy almost always limps AA here, while the average policy mixes calls and raises. That difference could reflect noisy current regrets or a real strategic issue; the one-decision probe cannot distinguish them. The [bounded RunPod check](../blueprint-runpod-check.md) therefore exports and evaluates **both** policies against an empty-table control on paired fresh deals. A weak or inconclusive result will be retained, not tuned away with repeated short tests.
