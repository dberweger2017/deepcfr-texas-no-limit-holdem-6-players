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

## Completed scaling diagnostics (#116)

The separately authorized [M4 diagnostic completion](reports/hu20-scaling-diagnostics.md) evaluated all 99 panels deferred by the original recovery window, without retraining. Across the three lineages, B100M versus its own B20M improved against original-cap2 minraise by **+47.36 [34.47, 60.24] BB/100** and passive check-call by **+13.63 [4.43, 22.82]**, using exploratory two-sided 95% paired-block intervals. The 40M/80M playing estimates vary by rule and do not form an intermediate LBR curve.

The six smaller secondary panels are mixed. Against pot pressure, B100M earned only **+3.29 BB/100** in aggregate, and 454 of 1,757 target decisions used fallback after native sizes sometimes left the target menu. Against the bounded LBR, B100M still lost **−73.14 BB/100** even though its fallback rate in those games was low. These are distinct measured limitations; neither establishes a single cause or full-game exploitability. The native audit and independent arithmetic now cover all **608,256** planned hands, with no model promotion. The fixed-first-seed B100M [human-play command](reports/hu20-scaling-m4-recovery.md#retained-artifacts-and-reproduction) and the earlier 20M commands remain available.

## Saved-policy diagnosis (#117)

The [M4 decision diagnosis](reports/hu20-b100-diagnosis-m4.md) retains all three B20/40/80/100M policies and makes no change to this playable model. On a fresh 512-block, two-position original-cap2 bounded-LBR schedule, B100M earned **−83.72 BB/100** in absolute target profit and improved **+40.38 [exploratory 95% 7.34, 73.42] BB/100** over its own B20M predecessor. Adjacent 40→80M and 80→100M intervals include zero, so this is not evidence of continued gains at a known rate. One retained B20M LBR decision was soft-limited during an owner-requested process pause; the report includes its bounded sensitivity rather than dropping it.

The 60-decision information-safe conditional audit found positive held-out local action gaps at some trained and heavily visited keys, but its uniform compatible opponent range does not reconstruct the actual LBR posterior. Four same-key concrete-card probes include a suit-equivalent preflop negative control with an apparent difference, so they do not justify declaring the card abstraction the cause. A deterministic nearest-size lookup translated only 53 of 382 attempted pot-pressure decisions to trained keys and changed profit by **+1.24 [−5.20, 7.68] BB/100**; the one-third-pot control moved −6.16 [−13.98, 1.65]. It has no demonstrated playing benefit and is not part of the human-facing policy. Native chip state remained exact in the test. The bounded LBR is not exact exploitability, and none of these tests certifies broad HU20 strength.

The next recommendation is a small posterior-conditioned conditional-value measurement on new outcome-blind decisions before selecting a training recipe. It belongs on the M4. The fixed-first-seed B100M human-play command above and the older #112/#113/#115 interfaces remain available; no model is promoted.

## Posterior diagnostic limitation (#128)

The [stability-gated M4 audit](reports/hu20-posterior-audit-v2-m4.md)
completed its timer checks and all 409,892 prospective likelihood calls.
Four of five fixed cases passed, but one seed-3 button flop failed: independent
four-sample estimated ranges had TV 0.374–0.383, and the 16-sample comparison
assigned 19.3–22.3% mass to holdings omitted by the smaller estimates.
The supervisor therefore prohibited all primary conditional values and later
controls. No posterior-conditioned gap or new playing-strength measurement
was produced; these missing values are not zero gaps. Sixteen samples are
not an exact posterior, and five cases do not certify all 24 selected decisions.
This result identifies insufficient reliability of this diagnostic budget,
without identifying sparse training, card/history abstraction or sizing as
the cause of the model's remaining LBR loss. The one next recommendation is a
prospectively frozen larger-likelihood stability measurement on the same five
cases, after an outcome-free M4 cost preflight. No training change, paid run,
model promotion or alteration to the verified human-play command follows.

## Exploratory 500M continuation (#136, analysis ongoing)

The [three-lineage campaign report](reports/hu20-500m-campaign.md) retains the
original B100M policies and human-play commands. Its complete exploratory
B100M→B500M broad comparison does not demonstrate bounded-LBR improvement:
the paired change is **−7.51 [95% −22.72, +7.70] BB/100**, with B500M absolute
profit **−72.48 [−88.56, −56.40]**. Native-pressure change is +5.49
[−6.82, +17.80]. Selective-stackoff and pot-pressure aggregate changes are
positive on their exploratory intervals, but seed/position variation and
regressions remain; selective full-stack losses total 134→119/24,576 hands,
with seed 1 increasing 36→50. These unadjusted exploratory results do not
certify that rare catastrophic mistakes are fixed or locate a learning plateau.
The complete 150M profile also has inconclusive LBR (+3.06 [−10.72, +16.85]),
pressure and selective-stackoff changes versus 100M. Selective full-stack
losses remain 134/24,576 hands at 150M; seed/position regressions persist.
The 200/300/400M broad groups and separate fresh 97.5% final confirmation
remain pending. No B500M model is promoted and no training intervention or
release declaration follows from these preliminary endpoints.
