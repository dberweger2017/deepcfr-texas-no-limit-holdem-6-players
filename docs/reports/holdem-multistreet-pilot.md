# Multi-street representation pilot

PR #83 delivers the bounded implementation and smoke evidence for the next
reliable-target representation task. No broad fit campaign or model promotion
was run.

The committed end-to-end smoke used one training seed, two independent hidden
world draws per context, and 1/2-step checkpoints for the three architecture
arms. It materialized one facing river context for each train, tuning,
validation, and test split. Duration selection chose the earliest checkpoint
with the lowest tuning cost for each arm, validation opened only after those
choices, no candidate qualified, and the sealed test contained the baseline
only. Runtime was 1.14 seconds locally. A fresh verifier reproduced the six
fit records and ten artifact hashes; a modified checkpoint and a duplicated
sealed-test roster were both rejected.

The six-case reference smoke covered open and facing flop, turn, and river
roots with two worlds and the full fixed uniform continuation. Open-root costs
were:

| street | nodes | seconds | action-value SE status |
| --- | ---: | ---: | --- |
| flop | 52,972 | 7.7 | estimated from two worlds |
| turn | 12,812 | 2.0 | estimated from two worlds |
| river | 1,752 | 0.3 | estimated from two worlds |

Facing roots were 126 nodes and about 0.02 seconds each. These are paired
Monte Carlo diagnostic estimates with two draws, not population confidence
intervals. A one-world reference is explicitly marked insufficient for
uncertainty. The full uniform profile is retained as a correctness smoke; a
cheaper fixed fold/call continuation is a separate design choice for the next
cost-controlled task.

The frozen plan has three seeds, 1,024/2,048/4,096 checkpoints, separate
tuning and validation families, equal street/group selection weights, fresh
canonical flop ancestors excluded from the prior representation and nested
card-diversity plans, and paired reference-noise screening. Its four flop
families and four-world defaults are pilot configuration, not evidence of
generalization. The next task should broaden families only after the reference
cost and continuation policy are reviewed.
