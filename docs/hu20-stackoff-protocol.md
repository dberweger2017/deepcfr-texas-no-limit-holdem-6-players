# Post-Luna HU20 stackoff regression — frozen v1

I am testing whether the large-wager pattern motivating the analysis in
[#127](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/127)
and [#130](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/130)
reappears against one reproducible opponent. This is a **post-Luna stress test**,
not independent confirmation, exact exploitability, a human-strength test or
permission to change a model. The thresholds below were chosen before this
evaluation, after seeing the historical Luna report. I will not tune them to
produce a loss or select favorable seeds. The opponent version is frozen for
future scaling comparisons.

## Opponent: `selective-stackoff-hu20-v1`

The opponent receives only its immutable acting-player observation, including
its own cards and public betting/board information. It never receives the
simulator, hidden rival cards, future deck, target probabilities or a private
human journal. All actions come from
`choices(view, raise_cap=None, free_fold=False)` and pass native validation.

- **Preflop strong:** TT–AA or AK, suited or unsuited.
- **Preflop medium:** other pocket pairs, any other ace, or both ranks at
  least ten. Other hands are weak.
- **Postflop strong:** best-five category two pair or better, except playing
  the exact five-card board on the river.
- **Postflop medium:** one pair whose paired rank is present in the hole
  cards. Remaining hands, including board-only river hands, are weak for calls.
- **Large wager:** the opponent's exact additional native call amount is at
  least **800 chips / 8BB**, rather than raise-to or total hand investment.
  Strong hands call; others fold. This threshold deliberately follows the
  descriptive Luna question and is not evidence selected independently of it.
- **Small wager:** call amount is at most **200 chips / 2BB** and at most
  one third of the current pot. A zero call also qualifies.
- Strong hands min-raise with probability **0.35** against small wagers or
  when checked to, if `min` is available; otherwise they check/call.
  Medium hands call only small wagers. Weak hands fold to wagers. With no
  wager all hands check unless the strong-hand min-bet branch is taken.

These are coarse card-selection rules, not equity estimates, nut detection or
an optimal response. Board texture, draws and blockers can make them wrong.
An abstract minimum raise can still be substantial after a previous raise;
the engine's exact bound is authoritative. There is no off-tree translation.

Exactly one `random.Random.random()` draw is consumed at every opponent
decision, including forced choices. Each hand/rotation creates its own stream;
there is no learning or memory across hands. Existing evaluation stream seeds
separate deal, target actions and opponent actions. The same logical opponent
stream is paired across checkpoints, lineages and rotations, with divergence
after different decisions retained rather than resampled to force agreement.

## Fixed artifacts and schedule

`configs/diagnostics/hu20-stackoff-v1.json` pins all twelve inference/checkpoint
hash pairs from #116's sealed model index. B20M/B40M/B80M/B100M refer to work
milestones, not hand counts; actual node overshoot remains in the original
training records. Each is current-policy extraction in the unchanged uncapped
HU20 game: two seats, 2,000 chips each, 50/100 blinds, no rake/ante, reset every
hand. No training, new extraction, seed selection or policy promotion occurs.

For each saved policy, use fresh independent deal blocks and both target-seat
rotations. The existing harness sets button to block modulo two. Thus each
block supplies one button/SB and one BB hand. Deals and logical action streams
are shared across all twelve policies. Panels have separate fixed roots.

| Panel | Paired blocks per policy |
| --- | ---: |
| Selective-stackoff v1 | 1,024 |
| Existing native pressure | 256 |
| Original-cap2 pressure, original-cap2 minraise, native minraise, passive | 128 each |
| Six existing secondary opponents, including uniform | 64 each |
| Existing original-cap2 bounded LBR: four chance samples, five soft seconds | 128 |

Total: **55,296 hands**, with the broader scripted/LBR panel retained. This
sample does not replace #116/#117's larger historical schedules or mix their
estimates. Historical results will be linked separately. LBR remains the
existing implementation; all timer-limited batches are retained and reported.
No higher LBR budget or favorable rerun is allowed after returns are inspected.

The new schedule roots are `202610020001`, `202610020101`–`202610020105`,
`202610020201`–`202610020206`, and `202610020301`. They are disjoint from the
published campaign roots. Record the plan digest and scientific source SHA
before the first model load. Use one sequential M1 worker, at most 6GiB RSS,
0.5GiB swap growth, at least 1GiB free disk, and a two-hour absolute safety cap.
An incomplete run retains its attempted hands and reports missing cells;
it is not silently restarted or represented as the complete schedule.
M4 is only a coordinated read-only artifact-transfer source for this task.

## Tail metrics and uncertainty

Report from the **target policy's perspective**, per lineage/checkpoint/panel
and actual position, with counts and denominators:

- Overall and positional BB/100, plus paired changes from that lineage's
  B20M and adjacent checkpoints.
- Target decisions whose menu offers a raise leaving the rival at least
  8BB to call; actual such raises; exact all-in raises separately.
- The immediate rival fold/call/raise response to each large target raise,
  with terminal/no-response cases explicit. Whole-hand returns after the
  first large raise partition into folded/continued/no response/no large
  raise. Repeated raises do not duplicate whole-hand profit.
- Exact +20BB full-stack wins and −20BB full-stack losses. Near-full wins or
  losses do not count as full stacks. Conditional averages include all losses.
- Trained/fallback target decisions by street and large-wager exposure,
  plus LBR requested/completed batches and timer limitations.

Use the existing arena Student-t interval over independent paired-deal block
means, with its minimum-block/no-variation rules. First average the two
positions within each block. For aggregate results average the three fixed
lineages within a shared block; do not count their common deals as three
independent samples. Position estimates likewise cluster shared lineages by
deal block. Checkpoint differences are paired before calculating intervals.
Intervals are exploratory, unadjusted 95%, conditional on these three saved
lineages; show individual lineages rather than inferring a population of seeds.
Small subgroup counts receive descriptive counts/returns, not significance
claims. **Whole-hand subgroup returns are not individual-bet EV.**

## Policy inspection

Select at most two contexts for each B100M lineage × actual position × small
river bet/raise kind, by the lowest canonical public-context hash, from the new
selective-stackoff panel. Selection uses public state and a trained actual
lookup with at least **300 visits**, never profit or aggression probability.
Record empty strata. Exact board, public action history, stacks/commitments,
position and concrete ordered menu define a matched public situation.

For each selected context and each saved checkpoint, query every two-card
holding compatible with the public board/disclosures. Do not exclude the real
unrevealed opponent cards. Group results by the existing card descriptor and
by concrete best-five category; include hole cards, visits, trained/fallback,
menu names/amounts, large-raise probability and all-in-raise probability in
raw diagnostic rows. Report all-query coverage and the ≥300-visit subset.
Do not count multiple concrete hands sharing a key as independent trained
entries or call comparable weak/strong aggression proof of an error.

All hands undergo existing native replay, public-event digest and chip-payoff
verification. Fixture tests cover information isolation, restricted/native
legality, RNG determinism, exact boundaries and dashboard arithmetic. Generated
fixtures and the twelve real saved artifacts are distinguished in the report.

## Running and inspecting the regression

I use a separate source checkout and an ignored input directory containing the
exact relative filenames in the plan. The sealed #116 model index and final
manifest identify their compressed hashes; I copy existing exports/checkpoints,
without training or recompressing them. The evaluator verifies every export and
streams every checkpoint node to check its current-policy extraction and visit
count. It retains no mutable trainer state.

```sh
python -m scripts.evaluate_hu20_stackoff \
  --plan configs/diagnostics/hu20-stackoff-v1.json \
  --inputs results/hu20-stackoff-inputs --out results/hu20-stackoff-run
python -m scripts.report_hu20_stackoff --run results/hu20-stackoff-run
python -m scripts.inspect_hu20_stackoff \
  --run results/hu20-stackoff-run --inputs results/hu20-stackoff-inputs
python -m pytest tests/diagnostics/test_stackoff.py -q
```

The output directory must be new. I do not silently resume an incomplete campaign
or exclude failed hands. `result.json` and `attempts.json` retain completion,
resource guards, timings and attempted panels; `manifest.json` pins source,
plan, inputs and initial output hashes. Each completed generated hand is checked
against native replay before entering metrics. `summary.json` includes aggregate,
position, individual-lineage, paired checkpoint and paired between-lineage
estimates. `contexts.json` records selection and empty strata. The inspection
outputs contain exact own-card queries, visits, menus and probabilities, with
unique information keys distinguished from compatible concrete holdings.

Inspection is a separate bounded read-only process with the same memory, swap,
disk and maximum-duration guards. It never changes the frozen opponent or the
completed evaluation. Timing and hashes added by reporting/inspection must be
recorded in the final evidence manifest as well as the evaluator manifest.

All these hands are newly generated simulator evidence. They are not private
human session journals. There is no browser/service change or action translation.

I render the measured tables and package compact evidence with:

```sh
python -m scripts.export_hu20_stackoff \
  --run results/hu20-stackoff-run \
  --out docs/reports/hu20-stackoff-artifacts
```

`dashboard.md` contains paired checkpoint/position estimates, per-lineage tails,
whole-hand return partitions, late-street exposure and bounded-LBR execution.
`decisions.csv.gz` retains exact cards, native amounts, concrete menus, keys,
visits and probabilities for generated decisions. It does not duplicate hand
profit onto each action. `generated-hands.jsonl.gz` retains exact action prefixes,
deal seeds, event digests, payoff and telemetry for native replay. These are
fresh simulator fixtures, never private human records. Every raw matched-holding
inspection is retained separately. `evidence-manifest.json` hashes the compact
outputs and links them to the scientific run manifest; large model binaries
remain outside ordinary Git history.

I verify packaged hashes, native replays, raw-CSV tail arithmetic and the complete
paired summary without reloading model binaries:

```sh
python -m scripts.check_hu20_stackoff \
  --evidence docs/reports/hu20-stackoff-artifacts
```
