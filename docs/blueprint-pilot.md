# First tabular blueprint test

The goal is to find a faster route to playing strength, not to reproduce every Pluribus implementation choice. The first test is an end-to-end plumbing and resource pilot. It does **not** qualify a model for v0.5 or establish that tabular CFR has learned good poker.

## Frozen first test

[`configs/blueprint/pilot-v1.json`](../configs/blueprint/pilot-v1.json) declares six-player, 100 BB, no-rake cash hands, two training iterations, one fresh dealt root per traverser per iteration, a 30,000-node phase cap, a 100,000-entry cap, and a five-minute cap per iteration. A two-block validation arena run checks that the frozen export can play legal hands against existing styles. The evaluation is a smoke test; its sample is too small for a win-rate claim.

Each iteration holds its policy fixed, samples chance and opponent actions, and expands every traverser action. It accumulates iteration-weighted regrets and own-reach-weighted strategy counts in a tabular store. Updates publish only after the complete six-role iteration finishes inside all bounds. If a bound fails, the previous checkpoint remains valid and the failed iteration is not published. Deal and action seeds are derived from the plan seed, iteration, role and sample, so an iteration-boundary resume does not depend on an opaque random stream.

The pilot action menu includes fold/check/call when legal and at most minimum and pot-sized raises, plus a short-stack all-in. It caps raises per street to keep the first tree measurable. Preflop cards use 169 canonical rank/suit classes. Later streets use a small made-hand/draw/board-texture bucket. Public actions are retained in order with coarse raise-size labels. This is an intentionally small information abstraction. Its collisions and limited raise menu may make the policy weak; the pilot tests the pipeline and measures its cost before any larger abstraction or search design is chosen.

The key and policy code receive `Observation`, not privileged `Hand` or deck state. The simulator may use hidden worlds to calculate terminal payoffs during training. The exported player receives only observations. Unknown abstract information sets fall back to a uniform distribution over the pilot menu and are counted as a coverage weakness in later analysis, not treated as evidence of learned play.

## Machine and artifact plan

The always-on M4 host `m4` reports 10 CPU cores and 16 GiB RAM. Use it for this pilot; the 16 GiB local machines are reserved for lightweight development checks. The saved checkpoint and compressed inference export are distinct. The export is hash-pinned in the arena manifest and cannot resume training. The arena copies its exact bytes into the run artifact for reproduction.

Run from the repository root on the M4:

```bash
.venv/bin/python -m scripts.train_blueprint \
  --plan configs/blueprint/pilot-v1.json \
  --out results/blueprint-pilot-v1
```

To resume a completed iteration into a new artifact directory, pass `--resume results/blueprint-pilot-v1/checkpoint.json.gz` and a new `--out`. Never overwrite a failed or completed run. Report wall time, nodes, entries, peak resident memory, model hash, evaluation validity, and all failures. A rental decision follows the measured entry growth and throughput. CPU RAM is the likely constraint for a larger tabular blueprint; a GPU is justified only by a later measured neural workload.

## Acceptance and next decision

The first test passes as plumbing if a full iteration finishes without invalid actions or non-finite values, fresh-process resume matches uninterrupted training, and the frozen export plays and reproduces in the arena. The test does not require positive BB/100. If the six-player tree exceeds a cap, retain the failure and reduce branching or change the sampling method in a separate measured comparison. Do not raise the cap blindly on a 16 GiB host.

The pilot writes `card_probe.json`: all 169 canonical hands at the same first-to-act preflop public decision, with trained/unseen counts and policy distributions for AA and 72o. This measures whether the artifact even distinguishes cards at that decision. It is not a poker-quality score. Use this probe and broader unseen-infoset coverage before a longer campaign. A strength claim still requires the predeclared fresh, multi-seed scripted-opponent test in [`ROADMAP.md`](../ROADMAP.md).
