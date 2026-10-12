# v0.5.2: heads-up 100 BB, 4B nodes

The same recipe and training run as v0.5.1, continued from 2B to 4B nodes. **It beats v0.5.1 head to head by +17.90 [14.98, 20.83] BB/100.** Two independent seeds repeat the gain over their own 2B models: +18.67 and +19.42 BB/100 (#236).

- **Model:** #226's linear-CFR opponent-sampled average, seed 2026100601, 4,000,000,174 nodes, 70.5M information sets. Translation for off-menu bet sizes stays on (512 states, 128 events).
- **Scripted opponents** (translation on), BB/100:

  | Opponent | BB/100 |
  |---|---|
  | check/call | +140 |
  | random | +138 |
  | loose-aggressive | +82 |
  | tight-aggressive | +49 |
  | pot-size pressure | −49 |

  No opponent shows a severe regression from v0.5.1 on the same deals. Loose-aggressive improves by +35; pot-size pressure still beats both models.
- **Limits:** no external benchmark has been played. See the model card for every number and limit.
- **Next:** v0.5.5 adds 200 BB for a Slumbot benchmark.

**Running it.** v0.5.2 runs with `--hu100-release`, which also loads v0.5.0 and v0.5.1 bundles. It peaks at about 4 GB of RAM, like v0.5.1, and takes about six and a half minutes to load. The HU20 releases are unchanged: v0.4.2 stays the default on the HU20 table.

**Assets:** the model, model card, these notes, install instructions, `release-manifest.json`, `SHA256SUMS`, and a standard-library verifier, `verify_hu100_bundle.py`. The verifier checks every byte and the publication binding to this release's source commit.
