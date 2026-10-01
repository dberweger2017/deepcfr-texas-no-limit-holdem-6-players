# HU20 fixed-sweep river re-solving protocol

I test a separate opt-in HU20 adapter over the existing full-range river CFR.
B500M current plays earlier streets; river play samples the solver's average.
This is a conditional restricted river game, not full no-limit exploitability,
safe Libratus solving, a true opponent posterior or model promotion.

## Frozen engineering definitions

- Two seats, 2,000 chips each at hand start, blinds 50/100; no rake/ante.
- Enumerate all 1,081 holdings per seat compatible with the five-card board.
  Multiply each seat's own observed pre-river action likelihoods under the
  immutable blueprint. Join the two factors with exact card disjointness.
  On-menu likelihoods remain exact, including zeros. Existing corrected
  off-menu raise likelihood uses its documented distance kernel/0.01 floor;
  count these approximations. No actual holding conditions the public solve.
  A wholly zero or incompatible range fails explicitly, without silent repair.
- River tree retains #108's min/pot/conditional-jam menu and two ordinary
  raises per street, removes free folds and inserts observed exact river raises
  even beyond that cap. Full range does not mean every possible future sizing.
- Fixed complete Linear CFR sweeps and average extraction. A wall/RSS watchdog
  aborts an incomplete experiment; it does not choose a shorter live strategy.
- Retain the profile on-tree. After off-tree actions rebuild from the river
  round root, inserting all observed raises. Freeze the entire action matrix
  at every hero node where an action was already taken. New solves cannot alter
  those holding-specific likelihoods. This consistency is not safe solving.
- Cache a whole profile by canonical public root, model/range identity, tree
  version, sweep count, inserted public actions and frozen matrices. Never use
  actual cards or random seeds in that identity. Bounded LRU; no partial solves.
- Log exact wagers, range support/trained/missing/off-menu likelihood counts,
  completed sweeps, solves/re-solves/cache hits, runtime, RSS and failures.
  Earlier-street blueprint use is distinct from river delegation (none).

## Correctness and cost gate

Generated policies/tiny ranges verify native terminal settlement, independent
scalar best responses, average weighting, deterministic profiles, hidden-card
invariance, exact holdings, cache reuse and prior-action constraints across
later off-tree re-solving. Re-run existing #108 tests unchanged.

Before outcomes, measure 250/500/1,000/2,000 sweeps on three declared HU20 roots
(limped, 4BB flop raise, 12BB flop raise, then check through). Use fixed seeds
202610010301–303, first B500M lineage by fixed order, full public ranges.
Capture restricted-game profile quality, setup/solve time and RSS. Record any
unmet milestone rather than silently reducing sweeps. Curve watchdog 20 minutes,
6 GiB process peak. No B500M playing outcome selects the setting.

After that timing evidence, freeze a practical paired hand budget and sweep
setting in a committed plan before opening playing outcomes. Use all three
verified B500M lineages, fresh identical deals and both positions; keep each
cheap/style/selective-stackoff panel separate. No bounded LBR in this task.
Retain every failure, native replay and exact action record; abort on invalid
play/incomplete solves. Report seed/position paired BB/100 uncertainty, tails,
range coverage, solve/cache counts and incremental cost. Whole-hand returns
are not individual-bet EV. No training, M4 access, paid compute or promotion.
