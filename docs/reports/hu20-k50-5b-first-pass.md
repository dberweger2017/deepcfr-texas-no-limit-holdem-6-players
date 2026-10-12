# HU20 K50 5B first pass

**Gates verified; sequential K50 training is underway.** All three v1 5B
controls are hash-verified and fully audited. No direct pilot or final play has
started; there is no strength result or adoption/release decision. [Owner-authorized protocol](../hu20-k50-5b-first-pass.md).

## M4 release and preparation

PR236 merged October 12 at 04:29:05 UTC. The 06:30 Madrid admission check found
zero matching research processes on M4 and no denied process metadata. This
thread alone disabled its admission schedule, then created an isolated worktree
at main `045c793a4abd2e371d4d77de663fd3dd2f00316b`. M1 did no research compute.
The owner requested waiting until 06:00 Madrid, then 30-minute checks; that
strategy and its superseded earlier ETA are recorded on [PR238](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/238).

The prospective protocol and strict resource supervisor are committed at
`b4c1f21c`; nine guard/process-family tests and staged artifact/whitespace checks
pass. Native training source is unchanged. Its supported option is `--nodes`,
so the intended 5B command uses `--nodes 5000000000` with complete iterations.

## Initial preventive storage admission

At initial admission, M4 had 22.01 GiB free, normal memory pressure, 89% free
memory, AC, and 1,347,818,618 bytes of system swap. After Python setup and small
source files, it has **21.87 GiB free**. The mandatory floor is **16 GiB**.

The conservative full-campaign reserve needs **35.72 GiB free**, rounded to a
**38 GiB entry target**. It includes 10.44 GiB for retained inputs, gates, six
checkpoint/export sets, traces, environment/build and source, plus the larger
of sequential audit scratch (9.28 GiB) and final archive space. The scratch
advisory uses the existing capacity planner's 1,024 bytes per projected key;
K50 uses 1.4 times #182's measured 6,315,718 keys, with 10% forecast headroom.
This is a prospective estimate, not a measured K50 footprint or a new guard.
[Complete formulas and provenance](hu20-k50-5b-first-pass-artifacts/storage-admission.json).
Actual hydration, outputs, scratch and pilot costs still need stage admission.

This initial refusal was preventive, **not a guard breach or scientific
failure**. The owner later authorized individually admitted stages while
provisioning storage. M4 had 40.50 GiB free at guarded build admission. The
six-hour clock began at **10:20:26 UTC**, with fixed deadline **16:20:26 UTC /
18:20:26 Madrid**; it is never reset. Table and control hashes are now verified. No other PR root or synced file was deleted,
offloaded or changed. No paid compute, guard relaxation or sample change.

The 38 GiB entry target ceased to be a blanket hold after the owner replied
**“on it, you can start though.”** Storage was then provisioned above the full
forecast. Every later stage still reserves its outputs and scratch above the
unchanged 16 GiB floor. A measured time/precision
quote above the owner's limits stops dependent work for a revised decision.

## Verified gates and controls

Current-main native source is unchanged. Binary SHA256
`aa7d3a5ef197f113c82a3e861abdc3081af9f5262f96f991e093d6d5a7a0add6`;
14 native tests pass. Seed602 1B exactly reproduces archived checkpoint SHA256
`a0ef695d6807aa4554ba01b32d27f64c39ccdc72d03be5a21c99cf67e52ad716`,
including max_entries=1B. Fresh K50 4M and its own 2M-midpoint resume reach
4,000,386 actual nodes /11,456 iterations and have **785,675 exact rows**,
including IEEE signed zero and all three table hashes. Only documented
native_state recovery metadata differs. No hash mismatch or guard breach.

All three v1 5B checkpoint ZIP/manifest/member pins verify. Current-main
uniform averages and paired current exports pass full audits. Seed601's
average reproduces archived `ab8ce0087f6bfefde447ebe48ad754c094db13318803df890488eb1283453193`.
[Complete preliminary receipts and exact density](hu20-k50-5b-first-pass-artifacts/preliminary-gates-and-controls.json).

The measured v1 1B train/save took 469.35 s. A three-seed K50 reserve scales
this by 5 nodes, 13% K50 overhead and 25% headroom: **9,944.37 s**, plus a
provisional **7,200 s** downstream reserve, within **19,642.40 s** remaining at
launch. This admits sequential terminal training, not final play; the measured
completed-model direct pilot must still pass the fixed time/precision rules.

The owner accepted **“Use reached-key quantiles with coverage disclosed.”**
Whole-table density is exact. Street quantiles cover distinct stored keys
reached during frozen direct matches, with the classified fraction and remaining
unclassified keys disclosed. Missing reached keys remain separate from stored
zero-visit keys. No full-table street allocation is inferred from hashes.

## Required remaining evidence

Three sequential terminal K50 5B runs with full exports/audits, fresh root
inventory, cost/precision pilot admission, canonical final plan posted before
play, both fixed direct contrasts, complete independent replay, seed/seat splits,
reached-key street distributions, primary decision, member-hashed ZIP and native /
independent cloud acceptance, RESULTS_INDEX/ROADMAP updates, final review and
green checks remain pending. The PR remains draft. No incomplete science may
merge as a finished campaign.

## Prospective direct-play preparation

The original pilot/primary/secondary roots are fresh and reserved before any
direct outcome. [Inventory](hu20-k50-5b-first-pass-artifacts/root-reservation.json)
covers readable local configs, reports, manifests, plans, the index and public
GitHub issue/PR bodies; only this task's proposed reservations matched.
No other PR work root was read.

The one-use wrapper separates fixed model loading from per-hand play, checks
the unchanged time/precision limits, requires the exact canonical plan on PR238
before final play, and independently replays both contrasts before readout.
Eight source tests pass for cost projection, paired seed/seat weighting,
independent arithmetic and corrupt replay evidence. Pilot and final play remain
pending until all three terminal K50 exports pass their full audits.
