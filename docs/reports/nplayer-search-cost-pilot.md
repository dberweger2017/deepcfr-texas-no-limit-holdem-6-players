# Three-player cost calibration, October 17, 2026

The first fixed task completed: 1,000 three-player 20 BB blueprint self-play
hands on bb128. It establishes the cost of baseline play and supplies fixed
legal observations for search qualification. **Zero search solves ran.** No
playing-strength or search-speed comparison is available yet.

## Measured baseline

| Measurement | Result |
|---|---:|
| Total measurement wall time | 34.122 s |
| Observed native process wall time | 34.066 s |
| Sampled peak child RSS | 4,184,338,432 bytes (3.90 GiB) |
| Trace size | 2,438,056 bytes |
| Hands | 1,000 |
| Decisions | 5,868 |
| Preflop / flop / turn / river decisions | 3,186 / 1,389 / 808 / 485 |
| Policy misses, by street | 0 / 0 / 0 / 2 |

Both times include startup, model/table loading, native play, replay using a
clone of the same native engine, and trace writing. Monitoring polls every
second and adds sensor/read overhead. This is one observation on a shared host;
it is not an isolated inference benchmark or an independently implemented
engine replay. Peak RSS is sampled, not a guaranteed maximum.

The process used CPU 0, nice 19, idle I/O priority, one thread, and a 16 GiB
address-space ceiling. It completed without reaching the [declared guards](../nplayer-search-cost-pilot.md).
Existing bb128/M4 jobs, affinity, locks and protocols were unchanged.

## Fixed search inputs

The extractor keeps up to eight distinct observations per street/unfolded-player
stratum, using the lowest canonical observation hashes. It never uses payoffs.
There are **43 selected observations**:

| Street | Two unfolded players: opportunities / selected | Three unfolded players: opportunities / selected |
|---|---:|---:|
| Flop | 1,320 / 8 | 69 / 8 |
| Turn | 786 / 8 | 22 / 8 |
| River | 482 / 8 | 3 / 3 |

Unfolded includes all-in players; these are not counts of players able to act.
Rare three-player river situations need explicit coverage in a later comparison.
The selected set is for engineering and cost, not a representative strength
sample. Search must receive only each row's `observation`, excluding source-hand
metadata. Inputs contain the acting seat's cards, observed board, public player
state, legal bounds and past public actions. Hidden cards, deck, seeds, future
actions and rewards are excluded; regression tests exercise this boundary.

## What remains before a search comparison

The current Phase 2 K50 model works in native N-player inference. Older Python
search rejects K50, and native N-player play has no search arm. The next concrete
task is the current-model adapter: key/menu/probability parity, legal observations,
public-range sampling and continuation correctness. Changed regret/sampling math
requires independent review before dependent larger research.

After qualification, time 12 fixed decisions at minimum work, then the matched
observation set across a declared grid of street horizons, solver iterations and
sampling counts. Separate loading/range construction/solving and report tail
latency as well as median. Only then quote approximately 1,000 full hands per
eligible configuration. Learned continuation values require trained, held-out
validation before inclusion. These axes can be combined; they are not three
mutually exclusive algorithms.

For scale only, this baseline contains 2,682 postflop decisions across all seats.
An added mean one second per decision would add about 45 minutes; ten seconds
would add about 7.5 hours. These are arithmetic scenarios, **not measured search
estimates**: search changes trajectories, one-seat deployment has fewer calls,
and independent solver repetitions add work. An eight-hour estimate should
follow the measured task, not define work to fill.

## Provenance and reproduction

Measurement script source: `2fe300e2`. Binary source:
`71aa93612ac9c5de1a51862d94ce430db5c41e1b`; native tree:
`3d9b4382741fbda72451b7c9b8c39cf9cf9dd483`.
Input model and binary hashes are pinned in the
[protocol](../nplayer-search-cost-pilot.md) and
[PR292 member index](nplayer-blueprint-phase2-artifacts/archive-model-members.json).
PR292 was confirmed merged before reading its retained inputs in place.
The native loader verified the model SHA256 before play. Seed `202610173401`
is cost-only, excluded from future confirmation.

Raw output is retained at
`bb128:/home/dberweger/nplayer-search-cost-20261017/baseline/`, with admission,
resource samples, stdout, stderr, result and complete trace. Local ignored copy:
`results/nplayer-search-cost-20261017/`. Trace SHA256:
`d39d160c1d0d193b29056872c74c5a1e2a3d724d4b77dbdc0b281fef2e75399f`.
Selected observations SHA256:
`7fbf487d58661b9f5e7d7da3e6c3a5e9565fa664ca15edcf2df356b32c966a59`.
Local trace readback matches the native reported hash. No files were removed.
Drive archival is pending; these paths are active retained evidence, not a
completed archive receipt.

Retrieve into a fresh ignored destination and verify the trace hash above:

```sh
scp -r bb128:/home/dberweger/nplayer-search-cost-20261017/baseline FRESH_DESTINATION/
python -m scripts.extract_nplayer_cost_observations \
  --trace FRESH_DESTINATION/baseline/hands.jsonl \
  --out FRESH_DESTINATION/observations.json --per-stratum 8
```

The qualified binary, tables and model remain under
`bb128:/home/dberweger/nplayer-blueprint-phase2-20261015/` with archive retrieval
provenance in PR292's index. No model, raw trace or selected observation payload
is committed to Git.
