# Experimental native-reopening HU20 model card

## Status and measured quality

Draft [PR #115](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/115).
Three fresh 20M-node uncapped B policies and three paired capped A controls
completed, using frozen final-current C extraction. No policy is promoted.
The [audited M4 comparison](reports/hu20-native-reopening-m4.md) measures:

| Fixed attack | Capped A target BB/100 | Uncapped B target BB/100 | Paired B−A, 97.5% interval |
| --- | ---: | ---: | --- |
| Native pressure | −254.75 | +102.20 | +356.94 [+337.01, +376.87] |
| Original-cap2 LBR | −130.27 | −95.88 | +34.39 [−5.55, +74.33] |

The native-pressure gain repeats across all three seeds. The aggregate LBR
safeguard passes its prespecified −10-BB/100 margin because the lower bound
is −5.55. LBR improvement is statistically inconclusive, and B still loses
substantially to that responder. Individual-seed and role intervals remain
wide. These are realized conditional benchmark profits, not exact profile
exploitability or strength against humans.

B also regresses against the original-cap2 always-minraise/check-call control
(−32.12 BB/100 relative to A) and passive check-call (−12.90), with exploratory
95% intervals below zero. The target remains profitable against those rules
in absolute terms. Restricted sizing, coarse cards, sparse river histories
and uniform missing-key fallback remain measured or architectural limits.
No three-player, six-player, tournament or professional-strength claim follows.

## Game, extraction and information

Two players, standard 52-card Hold'em, 20BB stacks reset every hand, 0.5/1
blinds, no ante or rake. Native betting legality, minimum raises, reopening,
all-ins, refunds and settlement are authoritative. The abstract sizes remain
min/pot/conditional-jam with the existing deduplication and jam condition.
`raise_cap=None` removes the count restriction; it does **not** model every
integer size. Folding is absent whenever checking is legal.

- Game: `hu20-native-reopening-20bb-52card-no-ante-rake-v1`.
- Schema: `hu20-native-reopening-ordered-history-card-v1`.
- Inference format: `holdem-hu20-native-reopening-blueprint-v1`.
- Menu: `hu20-min-pot-conditional-jam-native-reopening-v1`.
- Training: from-zero K1 external sampling, the existing linear-iteration
  regret convention, one root per seat, unchanged cards and ordered history.
- Playing extraction: final current C, fixed before outcomes. No averaging,
  card-abstraction change, pruning or checkpoint migration was introduced.

Only the acting player's own cards and public immutable observation enter
lookup. Unknown entries use uniform probabilities on this same expanded menu,
with explicit counts. Native-legal opponent sizes outside the restricted menu
can still lead to unsupported histories; removing the raise cap does not solve
all off-tree handling. The common LBR uses original-cap2 responder actions,
full compatible ranges and target-specific likelihoods, with four future
samples and five soft seconds. Its one-step/checkdown approximation does not
measure the exact best response.

## Fixed-order playable artifact

Use **B seed 2026093001**, the first seed in the frozen order, regardless of
its profit. Final checkpoint SHA-256:
`cb4ca3348398f358c611d92467ae9f891de41f64e374cd847c0103ccb2d62822`.
Final current-policy SHA-256:
`198dae3eb36b6a830ab106ba13032ecab10fc1f2fc12bc4bb9bc06dcce9a01a6`.
All six runs and 24 milestone checkpoint/export pairs are recorded in the
[artifact manifest](reports/hu20-native-reopening-m4-artifacts/manifest.json)
and [model specifications](reports/hu20-native-reopening-m4-artifacts/models.json).

From a checkout of draft #115 with its installed Python environment:

```sh
mkdir -p results/hu20-native-demo
scp m4:/Users/dberweger/Local/hu20-native-reopening-ab/results/hu20-native-reopening-m4-20260928/training/B-2026093001/current-3.json.gz results/hu20-native-demo/current-3.json.gz
python -m scripts.play_hu20_native \
  --policy results/hu20-native-demo/current-3.json.gz \
  --sha256 198dae3eb36b6a830ab106ba13032ecab10fc1f2fc12bc4bb9bc06dcce9a01a6 \
  --history results/hu20-native-human-first.jsonl
python -m scripts.play_hu20_native --replay results/hu20-native-human-first.jsonl
```

The adapter rejects a capped artifact or hash/game mismatch. It prints the
version and model hash, presents B's trained menu, alternates the button,
resets stacks and hides bot cards until legitimate disclosure. Choose a new
history path for another session. The verified M4 smoke completed and replayed
20 hands, with 72 trained and one fallback bot decision; its transcript and
histories are retained. It is a usability check, not a poker-strength result.

## Resources and next decision

All 14,057 confirmation LBR decisions completed, maximum 0.602 seconds and no
soft overruns. Training used under 1 GiB per run; the entire measured campaign,
including native audit, peaked at 2.16 GiB, with unchanged swap and 3.97 hours
elapsed from preflight start. Large artifacts remain on M4 with [retrieval
instructions](reports/hu20-native-reopening-m4.md#audit-lineage-and-retention).
The earlier [HU20](hu20-model-card.md) and [TP20](tp20-model-card.md) demos and
published artifact hashes are preserved.

The proposed next experiment is one separately authorized HU20 same-recipe
work-scaling confirmation, retaining the saved 20M A/B baselines, fresh
confirmation deals, the primary attacks and regressing controls. No campaign,
rental, merge or promotion is scheduled by this model card.

## Subsequent scaling limitation (#116)

The [dual-Mac continuation](reports/hu20-scaling-both-macs.md) stopped on a missing deployed observation fixture, retaining a first-seed 40M export and a second-seed 34.291M partial checkpoint. No new confirmation hand was played, and no 100M candidate was produced. The partial artifacts have no measured strength gain and do not replace this verified 20M model or its human-play command. The historical #115 measurements above remain the applicable evidence.

## M4-only scaling recovery (#116)

The separately authorized [M4-only recovery](reports/hu20-scaling-m4-recovery.md) produced all three 100M checkpoints. Against the original-cap2 bounded LBR, their role-balanced target profit improved by 25.94 BB/100 versus each lineage's 20M policy (two-sided 97.5% block interval [7.43, 44.45]), but remained **−73.14 BB/100 in absolute terms**. The native-pressure contrast was +19.39 [2.86, 35.93] BB/100 on average; one lineage regressed by 11.37 BB/100. The LBR is a limited attacker, so this does not certify robustness or full-game exploitability. Only 73,728 of 608,256 planned confirmation hands ran; the control and intermediate-checkpoint panels remain pending. The first-seed 100M human-play command was smoke-tested and replayed with bot cards hidden, while the verified 20M model and older human interfaces remain available. No default model is promoted.
