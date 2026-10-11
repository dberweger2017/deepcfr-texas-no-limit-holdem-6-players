# v0.5.0: heads-up 100 BB

The first heads-up 100 BB release: a 1B-node average with public-history translation, playable locally against a human or in a self-play spectator view.

- **Model:** #207's linear-CFR opponent-sampled average, seed 2026100601, 1,000,002,065 nodes. Bet sizes outside its menu are translated to the nearest supported public history (512 states, 128 events).
- **Results:** it beats four of five scripted opponents at 100 BB:

  | Opponent | BB/100 |
  |---|---|
  | check/call | +101 |
  | random | +77 |
  | loose-aggressive | +57 |
  | tight-aggressive | +36 |

  Against pot-size pressure it's about break-even: −9 [−39, 20] BB/100. Two independent seeds show the same pattern.
- **Limits:** no external benchmark has been played, and #215's formal recipe qualification failed on one inconclusive comparison. See the model card for every number and limit.
- **Next:** a 2B checkpoint from the same seed already beats this model by +29.5 BB/100 head to head. Larger HU100 releases will follow, and v0.5.5 adds 200 BB for a Slumbot benchmark.

**Separate runtime.** v0.5.0 runs with `--v050`. The HU20 releases are unchanged and stay the default for the HU20 table: v0.4.2, with v0.4.1 and v0.4.0 still selectable.

**Assets:** the model, model card, these notes, install instructions, `release-manifest.json`, `SHA256SUMS`, and a standard-library verifier, `verify_v050_bundle.py`. The verifier checks every byte and the publication binding to this release's source commit.
