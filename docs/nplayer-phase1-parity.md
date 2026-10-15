# N-player rules gate

Phase 1 of v0.6, first PR. Base: `ac674b41`. Owner request October 15, 2026.
No dependent traversal, size experiment, training campaign, release or merge is
authorized by this gate. The owner decides merges and arranges independent
review of rules/information access before dependent work starts.

## Contract and implementation boundary

The referee is `pokers` at `5db20e3d5d6862b32a7402035c1340b622d3b005`, the
requirements pin and its [RULES.md](https://github.com/dberweger2017/pokers/blob/5db20e3d5d6862b32a7402035c1340b622d3b005/RULES.md).
`native/hu20-trainer/src/nplayer.rs` implements integer-chip 2–6-player hands.
Stacks and blinds are parameters, including short blinds and unequal stacks,
within the engine's 10^12-chip per-input limit. No rake, ante or disclosure rule
is added. `observe` supplies the owner's cards, visible board and public action
history; the core retains simulated hidden cards solely for transitions/payoffs.

The existing HU game, keys, traversal, RNG, checkpoint and export implementations
remain unchanged. The new game is not yet wired to a trainer. This isolates the
rules review from the subsequent generic-traversal and checkpoint-format review.
The N-player core uses u64 chips because six supported engine stacks can exceed
u32; this does not change legacy HU arithmetic or bytes.

## Predeclared checks

1. Focused Rust tests: order/blinds for all 2–6 seats and all buttons; individual
   and cumulative reopening; checking before a short opening all-in; short BB
   bring-ins; side pots, refunds, folded-money pot coalescing and odd chips;
   invalid-action atomicity; save/restore and public-observation invariance.
2. Streaming differential test against the installed, verified engine pin.
   Compare initial and every post-action state, including actor, public cards,
   each player's stack/bets/contributions/fold status, pot, legal choices and
   exact raise/call bounds. Compare complete final stacks/rewards and independently
   reconstruct contribution pots/refunds with `src/game/types.py:pots_for`.
   The native checker also verifies save/restore and exact action replay at every
   transition. Successful complete traces are SHA256-hashed; retain each first
   failure as a complete replayable hand and stop that gate.
3. Pilot: 1,000 three-player hands, seed 202610150000, for correctness/cost only.
   Then at least **1,000,000 three-player hands**, distinct consecutive seeds
   starting 202610160000. Four disjoint 250,000-hand shards may run concurrently.
   Seed modulo eight selects equal 20/100 BB, random unequal stacks, tiny stacks
   around blind/raise boundaries, passive late-street play, or large integer
   stacks. Every eighth hand forces raises to all-in and calls, without folds.
   All modes rotate buttons and use 1/2, 3/7 or 50/100 blinds. Non-forced modes
   include off-menu raises, minimum/pot/all-in sizes and free folds.
   Supplemental 1,000 hands each at N=2,4,5,6, plus fixed adversarial fixtures.
4. Existing native Rust/Python tests, HU100 recovery gates, and #250's fresh
   20M / 10M→20M HU200 K50 identity. Expected final SHA256:
   `4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
   Compare tiny HU20/HU100/HU200 checkpoints and exports to the unchanged main
   binary as well. Reproduce #204/#206's HU100 capacity endpoint fresh and from
   20M, including checkpoint SHA256 `792a675c…f8d416`. The final bounded legacy
   identity check regenerates the v0.4.1 HU20 average at its existing 1B endpoint,
   expecting its pinned `571e1982…6b74d` SHA256 and 142,677,367 bytes. This is
   regression-only; no new policy or playing-strength experiment is selected.
   Any discrepancy blocks dependent work.

## Resources and evidence

Write code on M1; light checks there are owner-authorized subject to pressure.
Use idle cygnus-x1 (`bb128`) for builds/tests. Recheck active processes before
load; consult the owner if another campaign is present. At most eight workload
threads, four Cargo jobs. The existing on-host qualification watchdog stops at
85°C, stricter than the owner's 90°C ceiling, samples every <=2 seconds, and
kills its owned process group on resource/read failure, interruption or deadline.
Use separate bounded jobs: build/tests <=30 minutes, million-hand parity <=60
minutes, HU identity <=15 minutes. Set 8 GiB family RSS and 32 GiB disk floor.
No M4 access. No training beyond bounded regression identities.

Primary working root: `/home/dberweger/nplayer-phase1-20261015/` on bb128.
Read-only Python environment and K50 inputs may use the qualified #283 paths;
reverify hashes before any K50 read. No owning PR roots are copied or cleaned.
Retain all logs, source pins, commands, trace hashes, failures, resource telemetry
and compact results. Index generated evidence and archive at closeout according
to `docs/artifact-storage.md`; keep large files outside Git. Scientific failures
must be explained and fixed by a replayable regression, never erased or bypassed.

## Later gates

After independent parity review: generic external-sampling linear CFR rotating
all N traversers; opponent-sampled averages with a street limit; frozen membership,
telemetry, resume and N-tagged compatible checkpoints. Three-player Kuhn and Leduc
must exercise that same traversal and exact per-player best responses. Check
Szafron, Gibson and Sturtevant (2013) against the implemented Kuhn convention.
Then measure 20/100-BB table growth and scope the bounded 3-handed smoke/export
for cygnus's 125 GiB. No multiplayer equilibrium guarantee is claimed.
