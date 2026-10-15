# N-player native rules: Phase 1 parity gate

[PR288](https://github.com/dberweger2017/CFR-NLHE-6max/pull/288), October 15, 2026.
Scientific source `35483892a3aba6e51cc1ba3f1c0963c7e3fb9c34`, native tree
`a863c89425bb5f36f231dad52018c24e7b744f18`, based on main `ac674b41`.
[Predeclared protocol](../nplayer-phase1-parity.md).

**One million three-player hands match the pinned engine with zero mismatches.**
This is the first, rules-only gate of the owner's v0.6 Phase 1 request. The new
2–6-player game is isolated from the existing HU trainer. Generic traversal,
Kuhn/Leduc validation, information keys, N-player checkpoints, size measurement
and the bounded training/export smoke remain dependent work. Independent rules
and information review is required before that work; the owner decides merging.

## Game and information boundary

`native/hu20-trainer/src/nplayer.rs` supports parameterized blinds and positive
unequal integer stacks up to the engine's 10^12-chip input limit. It implements
clockwise action/dealing order, the big-blind option, short blinds, full and
short all-ins, per-player cumulative reopening, automatic board runouts,
side-pot eligibility, uncalled refunds and separate-pot odd-chip distribution.
Folded contribution levels merge when eligibility is unchanged, avoiding extra
rounding events. At settlement, contributions/bets/pot clear, final stacks
include winnings/refunds, and rewards sum to zero.

The core is privileged simulator state. `observe(seat)` contains only that
seat's private cards, the currently visible board, public stacks/status, table
configuration and complete actor-labelled public betting/reveal history. Its
borrowed fields cannot reach the simulator or another seat's cards. Inactive
observers receive no actionable choices. Live showdown/muck disclosure remains
the existing Python hand interface's responsibility; this PR adds no disclosure
profile or playing-policy loader.

Tests verify hidden-card/future-card invariance at all four streets, for every
observer at N=2–6. Invalid actions leave the parent unchanged. Every checked
transition saves/restores the complete state and history and repeats the action
to reproduce the same child. N=2-specific blind placement is the actual cash-game
rule; betting, reopening, accounting and settlement otherwise iterate over N.

## Differential gate

The referee is the installed `pokers` commit
[`5db20e3d5d6862b32a7402035c1340b622d3b005`](https://github.com/dberweger2017/pokers/blob/5db20e3d5d6862b32a7402035c1340b622d3b005/RULES.md).
The runner checks the installation's VCS metadata before generation. It supplies
the same complete shuffled deck to both implementations; engine transitions
and the native evaluator are independent. The checker compares the initial and
every post-action state: actor, decision street, visible board, all hole-card
dealing, stacks, street bets, contributions, folded status, pot, legal choices,
call payments and raise bounds, then final stacks/rewards. Terminal actor and
legal actions normalize to absent, as in the public hand contract. Complete
public history and Python `pots_for` contribution-pot/refund accounting also
match. All hole cards are used only by this privileged test, never policy inputs.

Main run, seeds **202610160000–202611159999**:

- **1,000,000 hands / 5,981,964 decisions**, zero mismatches.
- **738,454 unequal-stack hands**, **125,000 forced all-in hands**,
  **277,737 side-pot hands**, **287,741 refund hands**, and
  **392,534 multiway showdowns**.
- Decision streets: **2,966,320 preflop**, **1,465,833 flop**,
  **928,761 turn**, **621,050 river**.
- **259.42 seconds**, one streaming worker. Including watchdog closure:
  261.01 seconds, **26.67 MiB** sampled family RSS, **61°C** peak CPU,
  zero swap growth or throttle-counter change.

The eight fixed seed classes cover 20/100-BB equal stacks, arbitrary unequal
stacks, small stacks at blind/raise boundaries, passive late-street play and
large integer stacks. Blinds vary among 1/2, 3/7 and 50/100; buttons rotate.
Non-forced modes include arbitrary off-menu raises and free folds. The gate
does not measure training speed, information-table growth or playing strength.

[Complete compact result](nplayer-game-parity-artifacts/million--result.json)
includes the native checker SHA256 and the SHA256 of every successful complete
oracle trace in order. Traces stream through memory instead of becoming a large
raw fixture; source, seed interval and engine pin reproduce them exactly. A
first mismatch is retained as a complete hand and causes a nonzero exit. Tests
inject corrupt states, settlement and history to verify the checker fails.

Supplemental **1,000 hands each at N=2,4,5,6** pass. Fixed cases verify reopening
after multiple short all-ins; a previous full all-in not reopening a later
caller; short-BB bring-ins; checks before a short opening all-in; separate side
pots/refunds; tied odd chips; two spare chips awarded to different winners; and
folded dead money not creating extra pot boundaries.

## Heads-up preservation and tests

The existing HU game, key, CFR, RNG, table, checkpoint, export and telemetry files
are unchanged. The only existing Rust source edit exports the new module from
`lib.rs`. The normal remapped build is source-clean and records the toolchain
and native tree in its [receipt](nplayer-game-parity-artifacts/candidate-build.json).
Adding a module changes executable bytes; output compatibility is checked on
the complete compressed checkpoint/export bytes, without normalization.

Six HU20/HU100/HU200 × v1/v2 comparisons at 100,000 nodes match the qualified
main binary exactly for all **18 checkpoint/current/average outputs**. The main
binary is #283's pinned `d7049a7a…ad16436`; its native source tree equals the PR's
main base before adding this module. HU100's fixed #204/#206 capacity endpoint
also reproduces checkpoint SHA256 `792a675ce6d45d4de8d1b7f3fc6d976548610810f11da8f336c3926f37f8d416`
fresh and resumed from 20M: **39,438,279 nodes / 30,080 iterations**.
[Exact commands and hashes](nplayer-game-parity-artifacts/hu-identities--receipt.json).

#250's HU200 K50 seed 2026101302 reproduces its fresh **20,002,695-node /
8,728-iteration** checkpoint and the same checkpoint from a 10M-node prefix.
Both complete 270,143,830-byte gzip files have SHA256
`4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
The 10M prefix also matches `fc102fc2…de9e5db`. All three complete K50 input
files were rehashed against #163's pins before loading.
[Checkpoint receipts](nplayer-game-parity-artifacts/gate-results--checkpoints.json).

The pinned v0.4.1 HU20 regression reaches **1,000,000,110 nodes / 2,126,271
iterations / 4,319,080 entries**. Its complete 142,677,367-byte average export
matches SHA256 `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`.
Training, export and hashing finish in **793.50 seconds**, inside the 15-minute
guard, at **423.60 MiB** sampled family RSS and **65°C** peak CPU.
[HU20 receipts](nplayer-game-parity-artifacts/hu20-release-identity--result.json)
and [exact commands](nplayer-game-parity-artifacts/hu20-release-identity--commands.json).

- Rust: **26 library + 5 CLI tests pass**.
- New focused Python checker tests: **23 pass**.
- Existing native suites plus K50 evaluation, compact policy, streaming audit
  and checkpoint diagnosis, including those new checks: **342 pass**, followed
  by the initially skipped optional real-K50-input check: **1 passes** with the
  reverified tables. **343 distinct Python checks pass in total**.
- Staged repository artifact check passes. No checkpoint, export, raw trace or
  table payload is added to Git.
- Both CI matrix jobs pass on scientific source `35483892`; the final evidence
  commit changes documentation/receipts only.

## Failures, resources and review

The first on-host build's 26 Rust tests and example compilation passed; the
initial Python pilot failed at import because the engine's `Stage` enum is
unhashable. Replacing the dictionary lookup with equality comparisons fixes the
harness. No hand had been generated or compared in that failed attempt. Its
complete log and failed watchdog receipt are retained. The corrected 1,000-hand
pilot passed before the distinct million-hand seed interval. No game mismatch
was observed or suppressed.

All load ran on idle cygnus-x1, with an on-host 85°C watchdog (stricter than the
requested 90°C ceiling), at most eight assigned CPUs, four build jobs, 8-GiB
family RSS cap and 32-GiB disk floor. Final Rust tests were serial and native
regression jobs used four Rayon workers; Python numeric tests used the qualified
strict-MKL setting. M1 performed source/edit/static checks; M4 was not accessed.
Guard summaries and raw-log hashes are in
[resource receipts](nplayer-game-parity-artifacts/guards.json).
Across the closed jobs, the highest sampled CPU temperature is **68°C**, the
largest sampled family RSS is **1.96 GiB**, and no swap growth or throttle-counter
increase is observed. All compute has exited; only the initial import attempt
has a failed guard result. The scientific implementation's CI matrix also passes.

Author diff review covers rules, information access and parity failure paths;
independent review is still pending. This is evidence of parity with the selected
engine and fixed rules regressions, not a mathematical proof over every hand.
No N-player CFR, equilibrium, exploitability, strength or release claim follows.
The protected evidence root and read-only dependencies are indexed in
[RESULTS_INDEX](../../RESULTS_INDEX.md); originals remain intact.
