# Multi-street reliable-target representation pilot

This diagnostic extends the reliable-target representation check to legal
six-player flop, turn, and river prefixes. It is a finite regression pilot and
does not claim full-game strength or promote a policy.

Each sampled world is built from the hero holding, the public board visible at
the decision, a joint opponent range conditioned on those visible cards, and
future board cards drawn without replacement after all hole cards. The engine
may retain future cards internally, but every model input is the identical
hero observation for all worlds in one context. Fixed public continuation
profiles produce the action values; root values for every legal candidate are
paired across independent worlds and report Monte Carlo standard errors.

The three arms are `scaled_baseline`, `learned_separate_card_branch`, and
`explicit_visible_features`. The latter uses made-hand ranks only when enough
cards are visible, suit-symmetric public rank and suit patterns, and explicit
straight/flush completion potential. It receives no hidden cards, equity,
future cards, or target values.

Canonical flop ancestors own every descendant street, situation, and runout.
The frozen plan excludes the prior representation and nested card-diversity
flop families. Duration selection uses a separate tuning family and chooses
one of 1,024, 2,048, or 4,096 steps per architecture by equal street/group
weighted decision cost, averaged across three seeds with earliest ties. Only
the selected duration is evaluated on validation. Qualification uses paired
reference-noise uncertainty, decision cost, and relative error; it has no
training-RMSE gate. Sealed test is opened once for the baseline and every
qualifier.

The bounded local smoke uses two independent worlds and short durations to
exercise reference construction, phase ordering, fresh reload, artifact
hashes, and corruption rejection. It is infrastructure evidence only. The
full configuration is intentionally not run until the reference cost and
uncertainty review is complete.

```bash
python -m scripts.check_multistreet_representation \
  --plan configs/holdem/multistreet-representation.json \
  --out results/multistreet-representation
```

The existing river protocols and artifacts remain unchanged.
