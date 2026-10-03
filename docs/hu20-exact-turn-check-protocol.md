# HU20 exact turn check: frozen protocol

Frozen before any main-root loss values. This amends draft #145 with an exact
turn/river diagnostic after the original flop roots failed memory admission.
The original flop protocol and resource failures remain retained. Draft #146's
history-alias audit was completed first. **Turn results do not answer the flop
question.** No training, promotion, rental or automatic merge is authorized.

## Population, selection and policies

Set A has 67 unique start-of-turn roots behind 74 v1 LBR-facing turn decisions in
the hash-pinned #143 hands. There are 134 live turn roots among 768 LBR-panel
hands. Select 16 roots, two in each preflop pot-type × physical-button stratum,
with seed 202610020205. The selected roots contain 17 observed decisions. Keep
all 74 decision references and empirical preflop ranges in the corpus evidence;
only selected roots receive exact values. All 273 flop decisions remain the
separate #146 exposure population, not turn equilibrium evidence.

Set B is 32 unselected start-of-turn roots from 3,000 fresh first-lineage B500M
current-policy self-play deals. The population has 1,088 live roots and 1,912
earlier terminals. Deal/action seeds are 202610020201/202610020202; button
alternates. Sampling stops before any turn action. Select four roots in each
limped/min-raised/pot-raised/3-bet × button stratum with seed 202610020203.
Selection uses public roots, strata, multiplicity and measured cost, never
turn loss values or outcomes. Three declared cost-pilot roots are excluded.

All six B500M exports are used: current and stored CFR average for each training
seed 2026093001, 2026093002 and 2026093003. Exact paths/hashes are in
`configs/diagnostics/hu20-exact-turn-check.json`; input provenance is #141/#143.
The same selected roots are used for every export. Each policy supplies its own
`public_ranges`, conditioning on its preflop **and flop** likelihoods. Literal
zero likelihood is retained, and zero-support roots are disclosed as exclusions.

The 48 roots × six exports give 288 spot-policy jobs and 576 target-seat results.
Each root weight is multiplicity / stratum inclusion probability. Set B drives
the decision; Set A addresses selected turn folds. Order seed 202610020206
interleaves sets, pot types, buttons and rotating export identities. All corpus
identities are reproduced by native replay. Frozen corpus and admission hashes
are recorded in the machine-readable config.

## Values and projections

Use the exact native-reopening HU20 menu compiled by replaying `choices()`.
Integer BB = 100 chips. Solve the exact-card turn→river subgame to equilibrium
exploitability ≤0.2% of root pot. Record the achieved value for every spot. A
residual above 0.5% is reported and excluded; values between 0.2% and 0.5% remain
eligible with an explicit unmet-target flag.

For each target seat, lock only the target and call external `compute_mes_ev`.
Report respondent BR minus its primary equilibrium value, in BB and root-pot %:

- `e_bp`: the real export, per hand and native menu action.
- `e_v1proj`: own-range × own-action-reach weighted equilibrium projection onto
  the full real v1 key, pooling public lines and physical river runouts in this
  root. Blocked holdings have no weight.
- `e_v1proj_line`: the same descriptor projection independently within each
  public line, pooling river runouts there.
- `e_eq50`, `e_eq200`: per-turn-root equity buckets at fixed K=50/200 per street.
  Turn histograms have 20 bins of exact equity against a uniform opponent over
  every legal river. Assignment uses cumulative L1 / EMD and mean centroids,
  seeds 202610020254/202610020404. River buckets are pooled equity quantiles.
  This convention does not claim that Euclidean k-means optimizes EMD.
- `alias_cost = (e_v1proj - e_v1proj_line) / e_bp`. Keep signed differences;
  nonpositive denominators produce an undefined ratio, never clipping.

Record both-blueprint solver EV for every main job. At all target turn bet nodes
retain range-wide blueprint/equilibrium folds under common equilibrium joint
reach, and individual excess-fold hands with exact equity decile and full v1
key. Report observed Set A nodes separately, with inverse root-inclusion weights
per observed decision; multiplicity is not applied twice.

Secondary values replace the respondent's range with #143's empirical LBR
preflop range on that line/position, filtered by the turn board. The target range
stays primary. This is literally preflop empirical weighting, without a flop
likelihood update. API traversal remains `compute_mes_ev`; the reference is the
**primary equilibrium strategy reweighted to that range**, not an independently
solved secondary equilibrium. Record total/retained/unsupported empirical mass
and report paired secondary summaries only on common complete-support roots.
All partial-support per-job results remain evidence. Secondary values do not
drive the decision.

## Qualification and admission

Gate K: 100,000 samples across native turn tree types and physical runouts,
full `information_key` equals the factored key with zero mismatches. V1: every
line/action/chip amount matches native replay; zero mismatches in every main
tree. V2: 300 independent native terminal samples per real preflight root,
including fold/showdown; integer payouts agree. V3: current-binary river-only
toy BR values agree with `profile_quality`. V4: each of the three real-export
preflight roots locks both real B500M policies and falls within the independent
20,000-deal native Monte Carlo 95% interval. Current-tool EV is identical to the
qualified lock EV. V4 qualifies the mapping; it is not repeated as hundreds of
independent 95% gate tests. V5 is recorded per main root as above. Also qualify
both v1 projections against an independent Python oracle across 412 aliased
public nodes, and secondary weights against a native river oracle and ten
primary-identity comparisons per pilot.

The complete-cost pilot was declared before its values in
`docs/hu20-exact-turn-check-preflight.md`. Its three fixed seed-202610010901
roots are excluded from the main corpus. Native compressed full-pipeline costs
are 360.58 / 128.78 / 89.52 seconds (limped/min-raised/3-bet), including all ten
primary and ten secondary BRs. Peak owned RSS is 2.60 / 1.28 / 0.94 GiB; swap
does not grow. Both API memory modes are retained before allocation.

Admission estimate: twofold contingency × six exports × sum over selected roots
of [10 fixed export seconds + pilot seconds × native node count / pilot nodes].
Pot-raised roots use the min-raised reference. This gives **21.0559 M4 hours**.
All 67 A roots plus even 16 B roots would give 25.3003 hours, so both populations
are sampled and coverage is explicit. The cost model is an estimate, not a
promise; a hard cumulative 24-hour main wall clock survives resume. Do not change
selection after seeing values. An incomplete campaign receives no hypothesis
decision.

M4 admission budget is **4 GiB owned-process RSS**, from measured reclaimable
memory with at most 80% admitted. Record processes, RSS, pressure and swap at
admission and per job. One external solver, `nice -n 10`, `RAYON_NUM_THREADS=2`.
Run swap baseline is 781,587,578 bytes; growth >1 GiB stops owned work. Stop on
any nonfinite value, failed qualification gate, resource guard or numerical
error. Equilibrium loop is bounded by 10,000 iterations/900 seconds; complete
solver job by 1,200 seconds. Preparation has its own guarded process.

Call API `memory_usage()` before allocation. All preflight roots fit native
compressed menus, with zero-weight holdings omitted. Declared fallback ladder
is native → cap 3 → cap 2 (jams retained; cap 2 matches LBR). A capped export
requires removed target blueprint **and equilibrium** reach audits; >2% removed
target mass excludes the spot. No cap is silently admitted without this audit.
If a native root exceeds the present implementation's admission, stop and retain
its estimate before a separately audited capped continuation. Stop if a 3-bet
root cannot fit. No paid host is launched; an oversize continuation requires an
owner-approved quote. The existing M4 jobs and TensorBoard remain untouched.

The AGPL library/custom harness stay in `~/Local/hu20-exact-flop-tool`, outside
this MIT repository, upstream pinned to
`9d1509fe5077d019825f833eed04b16d342dfda1`. Commit only Python, documentation
and results/hashes. The frozen source inventory includes all external sources,
binary, native engine, six exports and three hand archives.

## Frozen report and decision

Use the common eligible-root intersection across all six exports for primary
pooled, per-lineage, extraction and seat summaries. Report excluded roots and
source support. The pooled estimand first averages export/seat values at a root,
then uses reach weights. Bootstrap 2,000 draws of independent roots within the
eight sampling strata, paired across every export/seat; seed 202610020207.
Ratios are ratios of pooled means on each draw. Never discard undefined ratio
draws to manufacture an interval. At least 16 eligible B roots and at least two
per stratum are required; otherwise no decision. Report BB and root-pot % with
95% intervals, R, alias cost, and `mean e_eq200 / mean e_v1proj`.

Turn-scope descriptive share is (134 / 768) × mean `e_bp` / 0.65 BB per LBR
hand. Retain the #145 point thresholds, with their narrower turn interpretation:

- Share <10%: H3-turn; little measured turn contribution, audit earlier streets
  including flop and preflop ranges.
- R ≥0.7 and share ≥25%: H1-turn heuristic; high feasible-v1 projection loss.
  Recommend better buckets plus exact-card search as a proposed intervention.
- R ≤0.3: H2-turn feasible witness; substantially better turn/river play can be
  represented within these isolated roots. Propose a native fast trainer audit.
- Otherwise: mixed turn/river evidence.
- Alias cost ≥0.25: material public-line pooling; ≤0.10: small signed difference;
  otherwise intermediate. This annotation does not override the R/share rule.
- H0-turn if Set A's common positive-equilibrium-reach selected-node fold rate
  is within three percentage points of blueprint, with ≥90% of the entire
  frozen selected-decision weight covered. Excluded roots remain in that
  denominator. Bootstrap fold differences over roots, seed 202610020208.

These classifications concern point thresholds; report paired ratio intervals
and their uncertainty without selecting a different threshold. The projections
are feasible strategies, not the abstraction's own equilibria. High projection
loss is an upper bound on the abstraction's minimum loss and therefore a
heuristic, not a proof of H1. An isolated-root feasible strategy need not be one
globally consistent blueprint across roots. **Within-root alias cost cannot
measure preflop/flop aliases inherited across different turn roots.** Per-turn
buckets are more favourable than global blueprint buckets. Earlier-street
errors are excluded by conditioning on given ranges. Descriptive LBR accounting
is not a causal decomposition. Report convergence residuals, all failures,
resources, exclusions, hashes and the original flop limitation plainly.

## Monitoring and delivery

Atomic per-job results permit resume. Solver JSONL carries iteration, residual,
elapsed and RSS; orchestrator counters include jobs and roots completed by set.
`scripts/monitor_turn_check.py` bridges the owned nested streams to the existing
server under `hu20-exact-turn-check-20261002/main-01`, with monitoring-only mean
and bootstrap panels, fold panels, validation and resource panels. `status.md`
heartbeat records stage, jobs done/total, ETA and last error. Do not restart the
existing server on 127.0.0.1:6006. Owner connection:

```sh
ssh -N -L 6006:127.0.0.1:6006 m4
```

Open http://localhost:6006 on M1. Post draft #145 comments at preflight done,
protocol frozen, 72/144/288 completed jobs (25/50/100%) and report done. Commit
and push completed subtasks. Final report is
`docs/reports/hu20-exact-turn-check.md`; update ROADMAP Current position. Both
PRs remain drafts, with no automatic merge.
