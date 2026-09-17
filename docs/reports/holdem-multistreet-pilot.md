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

## Retained all-street integration pilot

After the bounded smoke, a committed-source integration pilot ran all 24
contexts in the frozen plan with two worlds per context, seed 991, and short
1/2-step checkpoints. It completed in 100.446 seconds and passed a fresh
`--verify` reload. The retained artifacts are [the integration report](../../results/multistreet-integration/report.json),
[the integration plan](../../results/multistreet-integration-plan.json), and
[the source archive](../../results/multistreet-source-310258a.tar.gz).

The report records revision `310258a3b428cf643f082dcaf5c8afaff13bc25d`, source
fingerprint `380146b2ccef45b6f194470574e9e4c776b6aeb26207b0d4d1f6fbf6dcd37394`,
plan hash `c5022e068a346c7286922cf00c887693ea536a7b81454e6196790febb83881ef`,
and context hash `f5feaa4ad35cbe61f5801233dda7a368214dda28611b42437d4c54f30f250806`.
The source archive SHA-256 is `b8c36e3fbf60637d3ed40d31794f76d58398eacb70277c3272b2d7fd6af21b82`.

Duration selection chose baseline 1, learned separate card branch 2, and
explicit visible features 1. The learned card branch qualified on this one
seed (validation gain 0.1275 BB, paired reference SE 0.0334 BB, lower bound
0.0606 BB); explicit features did not (gain 0). The sealed test therefore ran
the baseline and learned-card arm once. Test paired SE for the learned arm was
0.0208 BB. This is bounded one-seed, two-world integration evidence, not a
multi-seed claim or production promotion; it requires independent confirmation
before any architecture decision.
