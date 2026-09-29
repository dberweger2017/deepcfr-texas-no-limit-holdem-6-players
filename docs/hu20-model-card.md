# Heads-up 20BB blueprint model card

**Status:** experimental, trained and playable in draft PR #112. Three
independent from-zero seeds completed; none is promoted as a repository
default. The [M4 report](reports/hu20-m4.md) confirms a +38.31 BB/100
mean improvement [95% block interval +33.98, +42.64] over the untrained
same-game policy on the declared six-opponent fixed-stack panel.

**Supported play:** heads-up fixed-stack Hold'em, 20 BB per hand, 0.5/1
blinds, standard deck, no ante or rake, and the saved restricted action menu.
The human-play command presents exactly that menu. Each hand resets stacks
and alternates the button. The inference artifact must pass SHA-256 and
versioned game/schema checks before play.

**Learning method:** ordinary K1 external-sampling tabular regret updates
from zero regrets, with three independent seeds and 20M completed traversal
nodes per seed. Final current `C` was frozen from development evidence before
fresh confirmation. Eight-snapshot `A` remains available as a diagnostic,
with explicit uniform missing-profile and current preflop fallbacks. It is
not an exact reach-weighted CFR average.

**Information:** only the acting player's private cards and public
observations enter policy lookup. The postflop descriptor and ordered public
history are deliberately retained from the earlier baseline. Different
underlying histories can share an abstract key. No exact full-game
exploitability or textbook convergence guarantee is claimed for this
imperfect-recall abstraction.

**Known limits:** a fixed 20 BB stack, heads-up only, restricted raise sizes,
two-raise cap, no ante/rake or tournament payouts, and coarse postflop card
features. A manual session is usability feedback, not a strength estimate.
The model must not be used as a six-player checkpoint. The trained policy
regressed relative to uniform against pot-pressure and tight-passive on the
held-out panel and still lost in absolute profit to loose-aggressive and
pot-pressure. The panel improvement does not demonstrate strength against
human opponents or exact full-game exploitability.

**Artifacts:** the fixed-order demo is seed `2026092801`, with final training
checkpoint SHA-256
`bc88e6798bc1cee2ebec0e3ea395f95cff5c2703017e0afbb8917ddffea1074a`
and policy-index SHA-256
`c1b5ab0b1064f93a059247fbce43624c08d943cff351c20ae605b78b05d9af0f`.
The other two independent seeds, all snapshots and every evaluation attempt
are inventoried in the [report artifacts](reports/hu20-m4-artifacts/inventory.json).
Large files stay at `/Users/dberweger/Local/hu20-pr112/results/hu20-m4-20260927`
on `ssh m4`. With the repository virtual environment active, copy the seed-1
artifact into your local checkout and play:

```sh
mkdir -p results/hu20-demo-2026092801
scp m4:/Users/dberweger/Local/hu20-pr112/results/hu20-m4-20260927/training/2026092801/policy-index.sqlite results/hu20-demo-2026092801/
scp m4:/Users/dberweger/Local/hu20-pr112/results/hu20-m4-20260927/training/2026092801/policy-manifest.json results/hu20-demo-2026092801/
python -m scripts.play_hu20 --index results/hu20-demo-2026092801/policy-index.sqlite --manifest results/hu20-demo-2026092801/policy-manifest.json --arm C --history results/hu20-human-first.jsonl
python -m scripts.play_hu20 --replay results/hu20-human-first.jsonl
```

The loader verifies the index hash and game identity before play. The terminal
shows the board, human cards, pot, stacks, position and menu; bot cards stay
hidden until legitimate disclosure. Each hand resets to 20 BB and alternates
the button. The session prints cumulative chip profit, counts lookup
fallbacks and writes a replayable history. Use a new `--history` path for
another session; an existing file is never overwritten. A 20-hand smoke
completed and replayed byte-identical public-event hashes, including river
hands. Measured M4 trained decision latency was 0.043–0.044 ms mean and
0.271 ms maximum across the three seeds' confirmation hands.

## Audited robustness limitations

The [PR #114 diagnostic](reports/robustness-m4.md) uses all three saved seeds.
Final menu pressure profit is +35.13 BB/100 [exploratory 95% block interval
24.52, 45.75], but native raises beyond the training cap expose −264.01
[−277.08, −250.95]. An observation-only, menu-restricted local response
produces target profit −111.96 [−143.48, −80.45]; all response decisions
complete without soft overruns. The 20M target improves over 2M, but
10M-to-20M is inconclusive. These are realized profits against fixed attacks,
not exact full-game exploitability. Native off-menu-history pressure lookups
missed in the measured panels; this is a deployment limitation, not proof
that every off-menu event misses. Card-abstraction causation is unestablished.
No checkpoint, extraction or playable interface changed; no model is promoted.

## Separate native-reopening experiment

[Draft #115's separately versioned candidate](hu20-native-reopening-model-card.md)
trains without the artificial raise-count cap and compares fresh A/B policies.
It passes the prespecified native-pressure contrast and aggregate LBR
non-inferiority margin, but still loses to LBR and regresses against two
controls. This preserves every #112 model, artifact and play command above;
it does not promote or replace the capped baseline. See the
[audited comparison](reports/hu20-native-reopening-m4.md) for full limitations.
