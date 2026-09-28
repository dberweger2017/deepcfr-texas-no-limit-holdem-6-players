# Experimental three-player 20BB learner

## Status

Draft [PR #113](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/113).
The fixed M4 campaign is running; final trained hashes and playing results are
not yet available. No default player is promoted. This is not a six-player,
tournament or professional-strength model.

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
2026092901, 2026092902, 2026092903. Demo bots will use the first two, irrespective
of their benchmark scores. The completed report will list every training and
inference SHA-256 and exact retrieval commands. Do not use guessed hashes.

After retrieving those two `current.json.gz` files and their verified hashes:

```sh
python -m scripts.play_tp20 \
  --policies results/tp20-demo/2026092901.json.gz results/tp20-demo/2026092902.json.gz \
  --hashes FIRST_VERIFIED_SHA256 SECOND_VERIFIED_SHA256 \
  --history results/tp20-human.jsonl
python -m scripts.play_tp20 --replay results/tp20-human.jsonl
```

Each hand resets all three stacks and rotates the button. The human chooses
from the saved training menu. Bot private cards are hidden until legitimately
exposed by the engine. Histories retain native actions, deal seeds, model
hashes, final chip ledgers and public event hashes for replay. HU20 remains a
separate game with its intact [existing model card](hu20-model-card.md).

## Evidence to obtain

The [frozen protocol](tp20-pilot.md) requires checkpoint learning curves,
independent and own-trajectory street density, paired same-game uniform
confirmation, six equally weighted primary lineups, mixed-lineup diagnostics,
independent-seed early/final crossplay and complete resource/artifact audits.
Scripted-panel returns or update density are not an exploitability certificate.
The report retains negative, inconclusive and incomplete outcomes.
