# Multi-street representation campaign

This campaign extends the bounded multi-street pilot to 48 fresh canonical
flop families: 24 train, 8 tuning, 8 validation, and 8 sealed-test families.
Each family owns four holdings and flop/turn/river open/facing contexts, for
1,152 contexts total and 576 training contexts. The materialized family list
and its generator seed are part of the plan fingerprint. Every prior observed
flop family is excluded before context construction.

The two existing fixed continuation profiles remain in force: `uniform` and
`increasing`, combined with the existing 1:2 schedule. The three representation
arms are `scaled_baseline`, `learned_separate_card_branch`, and
`explicit_visible_features`. Fits inspect train and tuning targets only; one
duration from 1,024, 2,048, or 4,096 is selected per arm by the existing
selection module, averaged across three seeds. Validation opens selected
recipes only, and sealed test opens the baseline and every qualifier once.
The diagnostic does not promote a model on playing-strength evidence.

## Calibration

Calibration uses a fixed training-only subset covering all six
`street:situation` strata. The same predetermined world stream is nested at
8, 16, 32, 64, and 128 worlds. For each context, every legal action is
evaluated in every world. Sufficiency uses within-world paired differences
`Q(world, action_a) - Q(world, action_b)` for every unordered action pair.
It never uses marginal Q standard errors or a difference between continuation
profiles.

For each context, the largest one-SE among its action pairs is retained. The
stratum summary is the 90th percentile of those context errors. The frozen
precision rule is one SE <= 0.10 BB, with the first N >= 16 whose current and
all larger prefixes pass. N=8 is retained in the trace. If no allowed prefix
passes, N=128 is frozen with `unresolved_at_maximum`; the run continues and
reports the unresolved stratum explicitly. Calibration is a precision
planning diagnostic, not a confidence guarantee.

Production references use a separate hash-derived stream namespace and seed.
Prefix indices are stable, so retrying or parallelizing contexts does not
change a completed world stream. Cache entries are atomically published and
bind the source, expanded plan, stream key, selected N, world count/seeds,
visible observation, hidden assignment/runout fingerprint, and action order.
Invalid or stale entries are rejected before fitting.

## Cost estimate and command

The reference budget is estimated from measured pilot seconds by street and
situation, multiplied by the declared context count and calibrated world
counts, then padded for retry and cache/setup overhead. The paid ceiling is
$10 including rental, storage, setup, and retrieval. The current live quote
for the planned host is 32 vCPU / 64 GB at $1.12 per hour; the orchestration
uses one Torch thread per job and can run up to nine jobs in parallel when the
host memory check permits it. No GPU or new algorithm is part of this
campaign.

Run locally for a tiny calibration/verification smoke first, then use the
same manifest and measured estimate to drive the authorized paid run:

```bash
python -m scripts.check_multistreet_campaign \
  --plan configs/holdem/multistreet-campaign.json \
  --out results/multistreet-campaign \
  --cache results/multistreet-campaign-cache
```

Completed per-context cache batches survive interruption. `--verify` checks
the campaign manifest, every cache hash and provenance field, every fit
checkpoint, selection record, qualification record, and sealed-test reload.
