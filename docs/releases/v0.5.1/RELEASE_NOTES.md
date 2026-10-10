# v0.5.1: heads-up 100 BB, twice the training

The same recipe as v0.5.0, trained to 2B nodes instead of 1B. **It beats v0.5.0 head to head by +29.51 [26.82, 32.20] BB/100.** {{SEEDS_SHORT}}

- **Model:** #223's linear-CFR opponent-sampled average, seed 2026100601, 2,000,000,460 nodes, 54.6M information sets. Translation for off-menu bet sizes stays on (512 states, 128 events).
- **Scripted opponents** (translation on), BB/100:

  | Opponent | BB/100 |
  |---|---|
  | check/call | +132 |
  | random | +107 |
  | tight-aggressive | +55 |
  | loose-aggressive | +42 |
  | pot-size pressure | −37 |

  None of the differences from v0.5.0 on the same deals is significant. Pot-size pressure still beats it.
- **Limits:** no external benchmark has been played. See the model card for every number and limit.
- **Next:** a 4B checkpoint already beats this model head to head, and v0.5.5 adds 200 BB for a Slumbot benchmark.

**Running it.** v0.5.1 runs with `--hu100-release`, the same flag that now also loads v0.5.0. It peaks at about 3.8 GB of RAM, like v0.5.0, and takes about five minutes to load. The HU20 releases are unchanged: v0.4.2 stays the default on the HU20 table.

**Assets:** the model, model card, these notes, install instructions, `release-manifest.json`, `SHA256SUMS`, and a standard-library verifier, `verify_hu100_bundle.py`. The verifier checks every byte and the publication binding to this release's source commit.
