# HU100 inference action translation protocol

This campaign uses only the M4 (10 cores, 16 GB), in an isolated checkout on
`feature/hu100-action-translation`. No training, native trainer edits, paid
compute, released defaults, web changes, releases or cleanup are authorized.

## Design declared before implementation and final play

The optional policy translates missing exact keys only. Existing stored keys,
including zero-mass keys, keep their current behavior. Default is disabled.
The real observation and its current legal min/pot/conditional-jam menu always
supply the cards, player flags and action names in the lookup key. Only past
raise labels may change; sampling still uses one unchanged policy RNG draw.

Replay the public betting topology with a small heads-up betting reducer, using
starting stacks, blinds, actors and public action kinds. It never constructs a
simulator, reads a deck, or receives hidden cards or seeds. For an observed raise
that was in the original real menu, retain that menu action's name in the
translated replay. At an off-menu raise branch over the translated legal raise
menu. Calls/checks, actors and street transitions remain fixed. A witness must
reach the same acting seat, player folded/all-in flags and current menu names.
This restricts translation to off-menu histories and makes entirely on-menu
opponents an exact behavioral control, including their missing-key fallback.

Distance per raise is `abs(x/(1+x) - y/(1+y))`, where x is observed paid/pre-action
pot and y is witness paid/witness pre-action pot (pot floor one big blind).
This bounded pseudo-harmonic coordinate measures proportional size changes and
compresses huge jams; exact rational arithmetic makes ordering deterministic.
Sum distances across raises. Prefer fewer changes of raise all-in status before
size distance, preserving each all-in suffix whenever a viable witness permits.
Break ties lexicographically by witness raise-to targets, then labels. Best-first
search returns the first positive-mass stored key with matching real menu names.
Limit to 512 popped public states and 128 history events, returning uniform if
none is found within the bound. No wall-clock deadline changes choices.
Record exact/translated/uniform, selected key, summed distance, suffix changes,
search work and measured lookup latency for every target decision.

## Pinned input and evaluation

Restore #204's terminal average (39,438,279 nodes) from Stage 1 Drive ID
`12LEcZWKJCwtTCCYcMxW578Gm3rDy9emK`, native Research-Cloud archive indexed in
RESULTS_INDEX.md. Whole SHA256
`9f01443ef9008aa112112ca25bee71048f74bd07e5ebb9e364ce243efac22f4b`;
manifest SHA256 `681ca26c9f6b7fe4c25525786595ebc6423b809f4f84b87ac3fde2f9705d27ef`.
Required member `research/terminal/average.gz`, 193,277,097 bytes, SHA256
`ba62d13536120a9d549f2f3ff84bcb2a96fbd8143ac2fc8477addab368dee0c4`.
Verify whole ZIP, manifest and member before use; record exact commands/provenance.

Compare the identical average with translation disabled/enabled against random,
check_call, tight_aggressive, loose_aggressive and pot_pressure. Reuse #203/#204
arena scheduling, scripted policies, replay and report arithmetic. Stacks reset
100 BB each hand, blinds 50/100, no rake. Paired fresh deals, both seat rotations,
identical target and rival action streams across translation options. No model
selection. Pilot and final roots will be pinned and checked against #197, #200,
#203 and #204 schedules (plus all other HU100 schedule roots in source/evidence).

Primary: translated minus untranslated pot_pressure, unadjusted paired-block
Student-t 95% interval; improvement requires lower bound >0. Secondary random
safeguard: lower bound >-20 BB/100, severe regression if upper bound <-20.
The other three are descriptive controls: require identical hand and action
results; any difference blocks acceptance for investigation. One lineage and
scripted opponents do not establish general strength or release eligibility.

Timing-only pilot: 16 blocks/opponent, both options, full replay and reproduction.
Do not read pilot outcomes. Freeze 2,048 blocks/opponent if projected cost fits
resource admission; otherwise stop and report rather than reduce after looking
at outcomes. Final execution budget is 3 times measured load plus per-block
play/replay/reproduction projection, plus 120 seconds for reporting. Post the
measured budget in the PR description before any final play. No extensions,
pooling, rerunning failed final hands or outcome-dependent sample changes.

## Guards, validation and closeout

Whole process-family RSS <3 GiB, system used memory <10 GiB, swap growth <256 MiB,
disk free >20 GiB, AC required, normal pressure at admission; sample every 200 ms.
Stop on information leak, invalid action, accounting error or any guard breach,
and post the issue with its single immediate next step (SOMA) on the PR.
One independent source review before final play and one independent evidence
review at closeout. Fixtures cover exact/translated/not-found/hidden-deal
invariance, suffix handling, bound and RNG invariance, and public-reducer parity.

Report BB/100/intervals, street rates, distance histogram and lookup latency.
Every final hand must replay and reproduce, including decision telemetry
(excluding elapsed time). Archive all inputs, source, raw hands/decisions,
partials, reviews and resources in a member-hashed ZIP under
`~/Local/Research-Cloud/PR-<n>-hu100-action-translation/`; verify local members and
separately confirm native upload plus cloud ID/name/size/parent. Index restore
commands/hashes. Merge only after green checks and no open findings, then add one
short ROADMAP Current position entry. No deletion or eviction of any synced file.

## Source review correction before final play

The first complete timing-only pilot at source `2cc06e8`, root
`2026100820511`, is retained as `results/action-translation/original-pilot`.
The reviewer found host guards refreshed at five seconds despite the declared
200 ms cadence. No breach or invalid play was observed, and no pilot outcomes
were read. Corrected source refreshes all guards each polling iteration, with
200 ms target cadence and actual timestamps retained; a mocked transient AC
breach verifies the stop. The corrected timing-only pilot uses fresh root
`2026100820521`; final root remains `2026100820512`. Both are checked against all
#197/#200/#203/#204 roots and the first pilot before final admission. The first
quote is superseded; the corrected pilot alone determines the sample and budget.
Default disabled inference keeps the existing reader/sampler path; telemetry
is explicitly enabled only by this research evaluator.
