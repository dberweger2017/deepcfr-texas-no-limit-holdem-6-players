# Experimental three-player 20BB learner

## Status

Draft [PR #113](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/113).
The three fresh 20M-node seeds and all fixed comparisons have completed. The
[verified report](reports/tp20-m4.md) establishes aggregate trained minus same-game
uniform **+58.84 BB/100 [95% block CI +53.92, +63.75]** on the restricted six-lineup
three-player panel. Tight-passive regresses; absolute five-scripted-lineup profit
remains unestablished. No default player is promoted, and no six-player, tournament
or professional-strength qualification is claimed.

## Game and policy

Three actual players, 20BB stacks reset each hand, standard 52-card Hold'em,
0.5/1 blinds, no rake/ante. A fold preserves the three-seat game and its earlier
public history. Training and play use the restricted min/pot/conditional-jam
menu, with at most two raises per street, and exclude folding when a check is
free. Legal off-menu opponent actions remain part of observed history.

Game: `tp20-20bb-52card-no-ante-rake-v1`.
Schema: `tp20-ordered-history-card-baseline-v1`.
Portable inference format: `holdem-tp20-blueprint-v1`.
Postflop cards use `legacy-postflop-descriptor-v1`.

All three seeds begin with zero regrets and use K1 external sampling. Playing
extraction is final-current C, fixed before outcomes. An absent information key
uses uniform probabilities on the same restricted menu and is counted as a
fallback. It does not substitute a scripted opponent. The agent receives only
its own immutable observation; the two bot seats do not share private cards.

## Artifacts and play

The M4 checkout is `/Users/dberweger/Local/tp20-pr113`; artifacts are in
`results/tp20-m4-20260928/training/SEED`. Fixed seed order is
2026092901, 2026092902, 2026092903. Demo bots use the first two, irrespective
of their benchmark scores. All final checkpoint and inference hashes and early
checkpoint lineage are in the [report](reports/tp20-m4.md) and
[sealed inventory](reports/tp20-m4-manifest.json). Both demo inference files were
retrieved and hash-verified locally; 20 smoke hands completed and replayed.

Retrieve and play from a checkout of this draft PR with its installed Python environment:

```sh
mkdir -p results/tp20-demo
scp m4:/Users/dberweger/Local/tp20-pr113/results/tp20-m4-20260928/training/2026092901/current.json.gz results/tp20-demo/2026092901.json.gz
scp m4:/Users/dberweger/Local/tp20-pr113/results/tp20-m4-20260928/training/2026092902/current.json.gz results/tp20-demo/2026092902.json.gz
python -m scripts.play_tp20 \
  --policies results/tp20-demo/2026092901.json.gz results/tp20-demo/2026092902.json.gz \
  --hashes c12ac92e9512413e6d83ed806dd0b7a60d88ef8418099294049b14af5700edd6 77d704f9dbdfae569917f9293f8580fa9c9eea4777eb5c44d7ae3d5f5d47962b \
  --history results/tp20-human.jsonl
python -m scripts.play_tp20 --replay results/tp20-human.jsonl
```

Each hand resets all three stacks and rotates the button. The human chooses
from the saved training menu. Bot private cards are hidden until legitimately
exposed by the engine. Histories retain native actions, deal seeds, model
hashes, final chip ledgers and public event hashes for replay. HU20 remains a
separate game with its intact [existing model card](hu20-model-card.md).

## Evidence and limitations

The [frozen protocol](tp20-pilot.md) completed all checkpoint curves, primary
and mixed lineups, and early/final crossplay against two other independent
final checkpoints. All three seed aggregates and early-to-final crossplay
estimates are positive. The full audit verified 1,072 files, 132 campaign
attempts and 382,464 legal paired evaluation hands. Peak RSS was 2.39 GiB,
with unchanged swap and 2.27-hour total campaign time. CI passed 834 tests.

On the independent set, median final turn updates remain five and median river
updates one or two; the river set contains only 53 decisions. Own-trajectory
coverage is higher and is reported separately. Missing-key fallbacks remain
explicit. Scripted-panel returns and update density are not exploitability
certificates. Review the retained negative and inconclusive opponent-specific
results before another separately authorized experiment; no automatic
six-player run follows.

## Audited robustness limitations

The [PR #114 diagnostic](reports/robustness-m4.md) evaluates all three saved
final checkpoints. Target pressure/pressure profit is −25.80 BB/100
[exploratory 95% block interval −40.00, −11.61] inside the training menu and
−279.86 [−291.61, −268.11] under native pressure. Minraise/minraise also
exposes menu losses (−79.93 [−102.91, −56.95]). Trained-minus-uniform
improvements remain positive, without removing these absolute weaknesses.
Native pressure produces off-menu preflop fallback; restricted-menu later
streets also have sparse trained coverage. Exposure counts do not assign a
whole losing hand to one decision. HU local-response results do not measure
TP exploitability: three-player worst-case quality remains unmeasured.
Existing human-plus-two-bots play and saved artifacts remain unchanged.
