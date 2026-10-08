# Native recovery and HU100 table growth — preparation status

**Tools coded; campaign not run.** Latest owner instruction permits lightweight
local coding now. M4 remains reserved to #188; setup, tests and training wait
for its merge and worker-side closeout. [Frozen campaign protocol](../native-recovery-hu100.md).
[Evidence PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197)
will receive qualification and measured evidence after admission.

At **October 7, 22:55 Madrid**, #188 was OPEN and M4 was actively executing its
guarded final evaluation: supervisor PID 15411, controller PID 15412 and three
workers. This is a point-in-time observation, not future admission. Its science
completion and worker-side closeout remain protected. No M4 files/environments
were changed and no campaign process was launched.

The initial thread-bound wake is **October 8, 04:00 Madrid /02:00 UTC**. It will
disable its own timer first and check the merge/idle gates. While waiting, one
15-minute admission timer is permitted; during execution, one lightweight
30-minute status timer replaces it. Every timer ends at completion/deadline.

**ETA:** earliest conditional start 04:00 Madrid; hard finish/report target
**12:00 Madrid on October 8**. No measured HU100 finish estimate exists yet.
Remaining qualification, HU20 reference/recovery and measured pilot costs
consume the same eight-hour maximum window. A late #188 merge/closeout shortens
the available budget. HU100 ends at 10B total nodes, capacity or time; reaching
10B is not promised. Upload acceptance may remain pending at the deadline and
will be labeled as such.

| Work | Status |
| --- | --- |
| Isolated branch and scheduled wake | Prepared; no duplicate launch |
| Tooling extensions | Coded locally; execution deferred |
| Independent correctness review | Round 3: no blocking source findings at `63b7d3e`; runtime qualification pending |
| Build/test qualification | Not run locally; pending #188 merge and M4 admission |
| HU20 pinned 1B reference current/average audit | Not run |
| Retained 500M →1B complete-state/probability equivalence | Not run |
| HU100 100k/1M/5M/10M pilot and audit | Not run; blocked on HU20 correctness |
| HU100 each 1B through 10B/capacity/time | Not run |
| Research-Cloud archive/member/upload verification | No campaign artifacts yet |

All future work retains 5.5-GiB aggregate RSS, ≤0.5-GiB campaign swap growth,
≥15.5-GiB free disk, AC power, serialization headroom and an external hard guard.
No merge, release, publication, arenas, paid compute or increased limits.

The first independent source review of `55958f9` found six blockers: an extra
save after a stop during serialization, a missing pilot recovery gate, incomplete
early-audit admission, baseline continuity, executed source/binary binding and
unvalidated reserves. The revised code addresses those findings and adds
deterministic fixtures for all-row/signed-zero mismatches, controlled versus hard
stops, admission/reserve failures, duplicate launches and stop-during-save.
The following rounds reviewed those responses; source review is not a passed
test or scientific result.

The second source review of `6cdb680` confirmed closure of the stop-during-save,
recovery-gated pilot, four early audits/both guards and baseline objections. It
found two remaining blockers: growth launch could miss changed pilot inputs,
and export/audit memory was absent from capacity forecasting. The latest source
binds/rechecks the complete pilot prerequisite manifest before a launch claim,
requires all eight measured pilot export/audit RSS peaks, and reserves twice
their forecast at the proposed table ceiling below 5.5 GiB. New fixtures cover
missing/changed prerequisites before claim and incomplete/underreserved memory
measurements. None of these fixtures have been executed locally.

The third independent review inspected `63b7d3e` against merged #196
(`97d0cb9deb25e7937cf232f458dcea6fc71ff800`) and found **no blocking source
findings**. It confirmed complete prerequisite membership/rehashing before the
durable claim, all eight measured export/audit RSS requirements, the 2× memory
forecast below 5.5 GiB, and the earlier comparator/stop/claim fixes. All prior
source objections are closed. Review task:
`node:delegated-task:command%3Amcp%3Acb89182f-41c8-40cb-998f-3f002cda0873%3Adelegate-task%3Apr197-tooling-review-round3`.

This is **source closure only**. The reviewer ran no tests/builds, changed no
files, used no M4 access and posted nothing to GitHub. Runtime qualification,
the repository-artifact check and scientific equivalence remain pending until
#188 is merged and worker-side closeout completes. The follow-up commit records
this review in documentation only; it changes no reviewed tooling behavior.
