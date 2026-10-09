# HU100 qualification preparation: disk admission refused

[PR211](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/211) · draft, unmerged.

Latest [complete storage planning estimate](hu100-seed-qualification-artifacts/storage-planning-20261009.json):
**110 GiB free for 4,096 blocks**, or **130 GiB for 8,192**. M4 currently
has **29.448 GiB free**: another **80.552/100.552 GiB** is needed. This
supersedes the earlier 32.865 GiB model-only figure as a planning target.

For 4,096 blocks, forecast originals total 45.102 GiB: models/restored inputs
13.101, fixed pilot/final/reproduction snapshots 22.544, raw play/reproduction
8.457 and fixed pilot/build/resources/metadata reserve 1.000. One archive copy
adds 45.102; the 15.5 GiB floor and 3 GiB transient save/recovery reserve yield
108.704 GiB, rounded to 110. For 8,192, raw storage doubles and the same
fixed snapshot bytes yield 125.617 GiB, rounded to 130. Snapshot bytes are
never block-scaled. No archive-compression savings are assumed.

The inventory read only #207's ZIP directory and its 324,414-byte embedded
manifest; the pinned manifest SHA256 matches. No model/raw member contents,
whole ZIP rehash, extraction, copy or evidence mutation. Forecast terminal
sizes use the unchanged entry ceiling, early saves get 10% size allowance,
and variable raw traces get 2x size allowance. These are conservative planning
estimates from measured #207 bytes, not actual new-seed output or final pilot
admission. No campaign started; full timing and stable/exclusive admission
remain required once storage fits.

The requested three-lineage qualification has **not started**. The October 9
05:46 UTC [readiness refresh](hu100-seed-qualification-artifacts/readiness-20261009.json)
supersedes the initial blocker: #207's archive-failure and end evidence closeout
are resolved; upload acceptance is explicitly handed to the owner, and only
final exact-head CI was pending at that snapshot. All checks subsequently
passed and #207 merged at `6e68bc9`; current main is integrated into this PR,
with all upstream science/source/evidence retained. The dependency prerequisite
is now clear; the owner upload-verification handoff remains unchanged. The M4's fresh disk
planning reference now refuses dependent work.

Current M4: **23.841 GiB free**, normal pressure, 87% system-free, AC,
1,447.94 MiB total swap and no visible research process. These are reconnaissance
snapshots, not exclusive stable-host admission or a new fixed swap baseline.
At #207's measured early+terminal checkpoint/current/average sizes, the two
new seeds need **8.683 GiB originals**; with an archive copy and the fixed
15.5 GiB floor, the model-only reference requires **32.865 GiB free**.
That is **9.024 GiB more than available**, excluding restored input copies,
arena snapshots, raw hands/full reproduction, pilot/partials and seed-size
headroom. Fresh seeds may differ in size; this reference is not a guaranteed
lower bound and cannot replace the full measured time/disk quote.

The latest dependency receipts verify #207's 20,517,119,304-byte retry ZIP,
all 948 member size/SHA256 readbacks, retained original failure/partial and
passing independent science/end/integration reviews. Its packing/readback
took 120.42 seconds. Cloud ID/metadata and remote-byte acceptance remain
owner-delegated and are not claimed. We performed no model/ZIP read, copy,
retrieval, build, pilot, training, final play or cleanup in this refresh.

The following initial preflight remains a historical record. Live GitHub
checks at October 8 22:01 UTC (October 9 locally) found #204–#206 merged,
but #207 open/draft with its archive guard failure unresolved. Its scientific
report records successful fixed 1B training and full final replay/reproduction;
there was no accepted archive or final independent evidence closeout then.
That initial owner's-prerequisite refusal is superseded by the refresh above.

[Prospective protocol](../hu100-seed-qualification.md) freezes existing seed
2026100601 and fresh seeds 2026100901/2026100902, sequential unchanged-recipe
training, early and terminal saves, separate multiplicity-adjusted growth and
translation families, practical interpretation, timing-only sample selection
and one six-hour absolute computation deadline. It is a design preparation;
campaign runner integration, measured admission quote and execution review
remain required before launch.

The owner subsequently sets this campaign's swap-growth guard to **3 GB
(3,000,000,000 bytes)** above its fixed fresh baseline. Other resource ceilings
remain inherited from #207. Its historical 512 MiB archive breach remains
recorded; changing this campaign's guard supplies neither archive acceptance
nor the missing evidence closeout.

[Read-only preflight receipt](hu100-seed-qualification-artifacts/preflight.json)
retains current PR states/comments/checks and three M4 host observations at
22:01:29/35/41 UTC. No trainer/Python/cargo/arena/packer process was visible
apart from the inventory shell itself. M4 has 10 cores/16 GiB, AC and pressure
level 1; free disk was about **45.37 GiB**. Swap was **1,344.94 MiB** in all
three observations, **781.38 MiB above #207's 563.56 MiB original baseline**,
still above its +512 MiB stop threshold. This does not establish the cause of
the previous breach or waive it. These brief snapshots are reconnaissance,
not the required fresh stable admission or exclusive research ownership.

The incomplete synced ZIP still measures **3,354,661,317 bytes**, modification
time `1791496020`, matching the prior stop receipt. The failure latch remains
115 bytes with that same modification time. Neither was copied, changed or
rehashed; all #207 originals remain untouched. No model retrieval, build,
timing pilot, new training, final play, archive retry, cleanup or baseline reset
occurred. No new campaign computation deadline was started.

There is **no admitted six-hour time/disk quote**: #207's successful science
timings inform preparation, but its failed archive supplies no measured
successful full-packing/readback cost. Fresh measured pilots, storage of all
originals/snapshots/reproduction/archive copies, and a recovery/closeout reserve
must pass before launch. The blocked preparation produced only compact Git
documents/status receipts; there are no new research payloads or accepted new
Research-Cloud archive to index.

Independent [preparation review](hu100-seed-qualification-artifacts/preparation-review.json)
found no blocker for documentation-only preparation and confirmed #207 remains
unresolved. Its qualification-safeguard clarification was applied: acceptance
also needs all three fixed endpoints, integrity/replay/reproduction/storage/
evidence gates, matching on-menu controls and no random severe-regression flag.
This review certifies neither a future runner nor #207's raw science/archive.
Staged artifact and whitespace checks pass on Python 3.11. An initial macOS
Python 3.9 artifact invocation failed on unsupported `zip(strict=True)`;
the supported interpreter passed. A local CI prose-only invocation lacked PR
event context and refused; this data-receipt diff correctly uses full host CI.
Final-head CI status and the amendment review remain on PR211.

Recommendation now: **retain the prospective v0.5 recipe as unqualified**.
#207's single-seed loose-aggressive and pot-translation gains remain promising;
its tight gain was inconclusive. Whether either effect repeats across seeds is
unanswered. Preserve #207's completed closeout and pending owner upload handoff,
admit this campaign only after enough free space,
fresh exclusive/stable-host evidence and the full six-hour time/disk quote.
No other PR cleanup or synced eviction funds this campaign. This PR grants
no automatic launch, release, recipe change, deeper training or merge.
