# Fresh heads-up 20BB blueprint protocol

The fixed M4 campaign is complete. See the [confirmation report](reports/hu20-m4.md),
the [development extraction decision](reports/hu20-development-m4.md), and
the [playable model card](hu20-model-card.md). This document records the
protocol as frozen before confirmation.

This experiment starts three independent K1 external-sampling trainers from
zero regrets. The game is two-seat, standard 52-card Hold'em with 20 BB
starting stacks, 0.5/1 blinds, no ante or rake, and a fresh 20 BB stack for
each evaluation hand. The native engine handles blind order, ties, refunds,
all-ins and settlement. The model does not reuse any six-seat checkpoint.

`hu20-20bb-52card-no-ante-rake-v2` and
`hu20-ordered-history-card-baseline-v2` identify the game and abstraction.
The betting menu keeps legacy min/pot/conditional-jam sizing and a two-raise
cap. It removes fold when check is legal during both training and inference.
The key is button-relative by construction and retains the ordered abstract
betting history, player status and action labels. Postflop cards use the
earlier blueprint descriptor as a named baseline component. This is a
restricted betting game with a lossy card/history abstraction, not an exact
full-game solve or a Pluribus-quality equity abstraction.

The development/resource preflight is one million traversal nodes from a new
seed. It measures throughput, entry and revisit growth, RSS, system pressure,
swap, and one full-size snapshot/collector/export. Based on that measurement,
record one final common work budget in `configs/blueprint/hu20-m4.json` before
the three main runs. The initial proposal is 20 million traversal nodes per
seed. The four checkpoint milestones are 10%, 25%, 50%, and 100% of that
budget. The late extraction window is eight equal-weight completed profiles
from 50% through 100%. Each profile uses the same predeclared preflop root
samples per seat. A missing preflop average falls back explicitly to current;
each missing postflop snapshot contributes uniform policy. The legacy
`Node.average` is never used as the extracted policy.

Development evaluation uses fresh paired two-position blocks across six
balanced opponent types, including a uniform player on the new menu.
Current and windowed policy results and saved early checkpoints form the
learning curve. Before confirmation outcomes are opened, a committed decision
file identifies one final playing extraction. Confirmation uses an entirely
different frozen schedule, initially proposed as 4,096 blocks per opponent;
the block is the independent deal/rotation cluster. The primary contrast is
the within-block average across three seeds of trained final policy minus
the untrained new-menu uniform policy, equally weighted across opponents.
Use a two-sided 95% block-clustered interval. Individual opponents,
checkpoints and direct seed cross-play are diagnostic.

The campaign uses one training process at a time, 10.5 GiB process RSS and
separate swap/pressure monitoring, with a hard ten-hour wall cap including
training, extraction and evaluation. The final report will retain every
attempt and failure and will distinguish learned chip profit from conditional
reference-game quality. A positive result does not imply six-seat or
tournament readiness.
