# Native multiplayer traversal: second Phase 1 gate

The owner confirmed the independent rules/information review of
[PR288](https://github.com/dberweger2017/CFR-NLHE-6max/pull/288) at `231b3eb6` on
October 15, 2026. No blocking findings remain in that review. This work starts
on `feature/nplayer-traversal-known-games`, initially stacked on that parent.
The owner subsequently merged PR288; PR289 now targets main. The owner still
decides every merge.

## Algorithm and scope

One generic external-sampling traversal serves finite-game adapters and the
reviewed N-player cash game. Each complete iteration visits every traverser in
seat order, with one independent sampled deal per seat and a frozen strategy
snapshot for that iteration. Regret increments have linear iteration weight.
The existing HU implementation and serialization stay unchanged.

Preserve opponent-sampled averaging: accumulate the current policy at sampled
non-traverser decisions. Scope can stop at a chosen street, without allocating
average arrays beyond that street. This practical multiplayer average is not
an unbiased own-reach CFR average: other sampled players affect visitation.
OpenSpiel explicitly documents this distinction in its external-sampling
implementation. Exact per-player best-response gains of the actual stored
average, rather than an equilibrium guarantee, are the acceptance evidence.

Checkpoint identity includes N, game/rules, seed, menu/card abstraction,
averaging scope and complete-iteration state. Resume must reproduce fresh
bytes. Frozen mode completes the boundary iteration and then discards new-key
updates, while continuing updates to existing keys and reporting misses.

## Known games and references

- Kuhn: three players, four distinct ranked cards, one-chip antes, one possible
  one-chip bet, no raises. Match Szafron, Gibson and Sturtevant (2013), including
  Tables 1–3 and the equilibrium utility family, not an arbitrary HU extension.
  [Original paper](https://poker.cs.ualberta.ca/publications/AAMAS13-3pkuhn.pdf).
- Leduc: OpenSpiel's multiplayer convention, `players=3`, two copies of each of
  four ranks, one private card each, one public card between two rounds,
  one-chip antes, raise increments 2/4 and at most two bets/raises per round.
  Suits are strategically irrelevant; ties split evenly. This is distinct
  from six-card heads-up Leduc. Reference source:
  [OpenSpiel 48401890](https://github.com/google-deepmind/open_spiel/tree/48401890ee9857e611678302371378175a8e4c6b/open_spiel/games/leduc_poker).
- Sampling/averaging reference:
  [OpenSpiel external sampling](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/external_sampling_mccfr.py).

Exact evaluation enumerates chance and legal actions. A best response chooses
one action per information set after aggregating hidden histories with
counterfactual reach; it must never maximize separately for each hidden deal.
Validate the evaluator against analytical Kuhn profiles, deliberately exploitable
profiles, and an independent reference. Rules/information tests cover all seats,
hidden-card invariance, replay and restoration. Regret tests compare expected
sampled increments with independently enumerated full-tree increments.

## Compute and acceptance

All substantial work runs on idle bb128 under its on-host 85°C watchdog, at
most eight workload threads, 8 GiB family RSS and 32 GiB free disk. M1 edits and
light checks are allowed; M4 is reserved. Pilot budgets measure cost only.
Before any sampled-training pilot, the main acceptance is fixed as follows:

- Three seeds, **2026101501, 2026101502, 2026101503**, for each game.
- Kuhn: **2,000,000 complete iterations** per seed; every player's exact
  average-policy BR gain must be **<=0.005 chips/hand**. Its value vector must
  be within 0.005 chips/player of the published utility family for some
  beta in [0,1/4]. Analytical profile tests must independently match to 1e-12.
- Leduc: **500,000 complete iterations** per seed; every player's exact
  average-policy BR gain must be **<=0.025 chips/hand**.
- Report fixed logarithmic intermediate checkpoints and the final endpoint;
  judge the final stored average only, without selecting a better earlier
  iterate or seed. Each main job is bounded to **30 minutes**. First a
  **1,000-iteration** pilot per game, <=120 seconds each, must support that
  runtime quote. Abort admission if the measured cost does not fit.

Do not extend a failed or inconclusive
experiment to obtain a pass. Preserve failures and resource receipts.

Admission pilot: seed 2026101590, 1,000 iterations per game. Kuhn used 0.0064 s
and Leduc 0.244 s including all intermediate exact evaluations. Even scaling
those inclusive times linearly quotes under 13 s / 123 s per main seed; the
30-minute cap has ample margin. Analytical/native uniform-policy evaluations
also agree with independently installed OpenSpiel **2.0.2** to 1e-10. The main
runner retains all six endpoints and also compares complete fresh/resumed
checkpoint and average-export bytes at the first seed of each game.

Before dependent size sampling: both games must meet the predeclared per-player
BR thresholds, Kuhn must match the published result, recovery/frozen/averaging
tests and existing HU byte gates must pass, and the owner must arrange an
independent review of the traversal, information keys and regret math. No
release, search or long NLHE training is part of this gate.

## First fixed endpoint (October 15)

Source `1578d66fe7e47bed1d72ee34651893d76d625824`: all three Kuhn seeds pass;
maximum per-player BR gains are 0.000861, 0.002171 and 0.003183 chips/hand.
Published-family utility errors are below 0.000591 chips/player. All three
Leduc seeds fail the 0.025 limit: maximum gains are 0.042425, 0.044560 and
0.031817. Independent OpenSpiel values/BRs match, and both full fresh/resumed
checkpoint/average byte comparisons pass. All six planned endpoints are retained;
the combined run lasts 112.58 seconds including build and watchdog closure.

No threshold is relaxed and no dependent size/smoke work starts. The owner
approved a new fixed confirmation on October 15: Leduc only, three fresh seeds
2026101511–2026101513, 2,000,000 complete iterations each, unchanged 0.025
per-player limit, approximately three minutes quoted from the measured
10.5–10.7 seconds per 500,000 iterations, capped at ten minutes total. The prior
failed endpoint stays recorded. The runner also checks exact split/resume for
the first confirmation seed inside that same cap.

The approved confirmation passes at source `7900a8019ed6b868891c0b733adfa0da768f1a1e`:
maximum per-player BR gains are **0.021380, 0.021521 and 0.019345 chips/hand**.
All nine gains are below the unchanged 0.025 limit, and independent OpenSpiel
values and best responses agree to 1e-10. Fresh and split/resumed checkpoint
and average-export bytes match. The complete guarded job, including build,
reference checks and recovery, takes **182.97 seconds**, below its 600-second
cap. This resolves the known-game threshold; independent traversal review and
the remaining HU checks still gate dependent work. Full results belong in
the [traversal report](reports/nplayer-traversal.md).

Cash adapter correctness tests use at most 20,000-node targets and complete
iteration boundaries; they are not the later size experiment or smoke run.
They check all N=2–6 at short stacks, three-player 20-BB exact resume, frozen
membership and scoped averages with the pinned K50 inputs. The existing HU
native suites and identity gates (HU100, #250 fresh/resume, HU20 v0.4.1) remain
required; each bounded identity job has a fifteen-minute on-host guard.
