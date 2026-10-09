# HU200 public-history translation experiment

Prospective protocol, October 9, 2026. Base current main `a3358fc` (merged #217).
This intervention tests a hypothesis motivated by #217's 934 proven unsupported
pot-pressure decisions and 34.5% positive-mass river coverage. Fewer fallbacks
are not evidence of a playing gain. No training, live Slumbot match, release,
default change, automatic adoption, model selection or cleanup is authorized.

## Frozen policy and intervention

Freeze #216's 100,000,034-node opponent-sampled average, seed 2026100905,
20,575,288 entries, iteration 52136 (the model index/header is authoritative).
SHA256 `1e9613547ad6721f2559e419caf296a781a914a4e92ecd4ef35f4b483c66060b`,
510,277,592 bytes. Verify hash, source checkpoint, node count, iteration, seed,
game/schema/menu, average rule and uniform fallback from the index before use.
Use its existing immutable M1 path and #216 archive restoration; copy neither
model nor native runtime. Keep HU200 200BB equal stacks, blinds 50/100, 0.01 chip
unit, no ante/rake, v1 cards and min/pot/conditional-jam menu unchanged.

Translation off versus on differs only in public-history lookup. Explicit HU200
schema/game/table identity must match. Keep #205's HU100 behavior, bounds,
distance, all-in preference, tie ordering, exact/zero-mass behavior and single
unchanged policy RNG draw. Only missing exact keys can translate; current real
legal actions/amounts, cards, board and flags remain authoritative. An originally
on-menu raise preserves its menu name; off-menu raises branch over legal menu
witnesses. Prefer fewer all-in-label changes, then summed exact rational
`abs(paid/(max(pot,BB)+paid) - witness_paid/(max(witness_pot,BB)+witness_paid))`,
then lexicographic raise targets and labels. Limit 512 popped states and 128
history events. Bound failure uses the same uniform fallback, never a timed
search or extra random draw. An unsupported open jam remains a known limitation.
Use legal observations only; hidden world and private streams stay in the
separate evaluator, never policy inputs. Independent correctness review clears
translation and evaluator before calibration/final science.

## Pairing, primary and regressions

Five opponents: random, check_call, tight_aggressive, loose_aggressive,
pot_pressure. Identical fresh deals, rotated candidate seats and identical
private candidate/rival action streams across off/on. Button alternates by block;
stacks reset every hand. SHA256 label-separated schedule follows #217's codec.
Final root **2026100908**, excluded timing root **2026100909**. Check deal seeds
against #216 smoke, #217 timing/final, prior HU100 schedules including #205/#207,
and between current timing/final before calibration. No training seed reuse is
represented as a fresh training lineage.

Primary: **on minus off against pot_pressure**, two-sided paired-block Student-t
95% interval. Demonstrated improvement requires lower bound >0. Report practical
size in BB/100 and absolute off/on descriptive 95% intervals. Each block averages
its two seat rotations; blocks, not hands, are independent units. Constant
series has a flagged zero-width arithmetic interval. No aggregate opponent score
or training-seed/general-strength claim.

Random safeguard: paired 95% lower >−20 BB/100; severe regression if upper <−20,
otherwise inconclusive when safeguard fails. Check_call/tight/loose are on-menu
controls: require identical complete actions, events, settlements and decision
menus/probabilities. Any difference stops acceptance and requires investigation,
even if its outcome is favorable. Report all five gains and absolute results.
A positive primary plus passing safeguards supports a separate fresh confirmation;
it does not automatically adopt translation. An inconclusive primary retains the
option as experimental; a demonstrated loss or a correctness/control failure
calls for another approach. Remaining coverage gaps and sparse covered learning
inform suggestions without attributing causal losses to them.

## Timing-only admission and stopping

Target **2,048 paired blocks/opponent /40,960 hands** within one **3,600s** clock,
including input verification, source snapshot, calibration, evaluation, all replay,
policy/telemetry reproduction, native parity, primary report/readback and local
ZIP/readback/closeout. Development, small checks, independent reviews and PR/report
administration precede or follow science. Never infer runtime from #217's six
minutes. One compact model load per timing/final worker; measure fixed load and
source costs separately from scalable per-opponent off/on play, trace, witness
verification and reproduction. Native process startup/supervision overhead is
reported separately; admission conservatively scales its complete measured cost.

Calibration: **32 blocks/opponent/arm**, complete verification; produce costs and
raw traces only, no payoff summaries. Do not inspect calibration payoffs. Freeze
largest common sample among **2048,1024,512,256,128,64,32** satisfying
`2*(fixed reload + (calibration play/replay + supervised native parity)*n/32)+600s`
in remaining time. Reserve `2*calibration evidence bytes*n/32 +1GiB` above the
admitted disk floor. All adjustment is timing/storage-only and frozen before
outcomes. If none fits, stop with timing/partial evidence. No outcome-driven
retry, extension, optional stopping, pooling or repeated final hands.

Single M1 worker and exclusive `/tmp/deepcfr-m1-research.lock`; check #217 current
status, threads/process ownership and #216 status before input use. M4 and other
PR roots untouched. Initial >=4GiB available RAM, normal pressure, AC, >=15%
system free, 3/4GiB family soft/hard ceilings, original-baseline +512MiB swap and
3,000,000,000-byte total swap. Guards sample about every 0.5s and retain actual
gaps and kernel command peaks. Soft/time/resource/information/action/accounting/
reproduction errors latch off science and preserve failures/partials. Archive
only if remaining resources/time permit; no silent readmission or scientific
restart. Default **15.5GiB disk floor**; a proposed evaluation-only **12GiB**
exception requires the owner's pending answer and must be recorded before launch.
No deletion or offloading is part of resolving admission.

## Verification and outputs

Every action: immutable public observation replay, legal actions/bounds and real
current menu. Every hand: chip conservation, complete engine events/settlement
replay and candidate/rival reproduction. Compare all probabilities, lookup modes,
keys, deterministic search telemetry and choices; exclude measured elapsed time
only. Independently replay every translated witness through the real engine:
legal menu choices, preserved on-menu names, actor/street/cards/flags, current
menu names, positive stored key, distance and all-in changes must agree. Replay
all real hands with #216's hash-pinned native runtime for actor/legal/menu/key/
settlement parity. Reporter verifies trace hashes, full frozen coordinates,
paired deals, raw payoffs and controls before arithmetic. Independent final
review recounts gains/uncertainty, absolute results and telemetry from raw records.

Report exact/translated/uniform rates and missing/zero reasons by street/opponent/
arm; bound hits, deterministic state counts, all-in changes, distance distribution,
lookup latency (mean/p50/p95/p99/max) including translated subset, observed total
play/witness/reproduction costs and fixed loading costs. Rates follow each
policy's trajectory and do not causally assign profit to individual fallbacks.

## Storage and unmerged handoff

Create and locally verify one ZIP with source, frozen protocol/plan, input hashes/
restoration, review receipts, calibration/final traces and fixtures, raw resources,
logs, failures/partials, primary report and checks. Manifest every member's bytes,
SHA256, original path and mtime; verify every local member and whole ZIP SHA256.
Check the owning PR's current status first; the user explicitly authorizes this
task's own open evidence archive. Place directly in existing Research-Cloud under
`PR-<number>-HU200-action-translation/`, discover actual Drive folder/archive IDs
when available, index restoration and upload-pending owner/later-agent handoff.
No upload wait, remote-byte or upload-acceptance claim, duplicate model archive,
original deletion, synced deletion or forced offloading. Mutable archive-monitor
and final administrative receipts remain in compact Git records. Updated report,
roadmap/index, passing relevant checks and independent final review accompany an
unmerged PR ready for owner review.
