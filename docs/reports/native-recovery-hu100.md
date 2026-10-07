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
**10:00 Madrid on October 8**. No measured HU100 finish estimate exists yet.
Remaining qualification, HU20 reference/recovery and measured pilot costs
consume the same six-hour maximum window. A late #188 merge/closeout shortens
the available budget. HU100 ends at 10B total nodes, capacity or time; reaching
10B is not promised. Upload acceptance may remain pending at the deadline and
will be labeled as such.

| Work | Status |
| --- | --- |
| Isolated branch and scheduled wake | Prepared; no duplicate launch |
| Tooling extensions | Coded locally; execution deferred |
| Independent correctness review | Two source reviews received; additional gate/memory fixes prepared for round 3 |
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
Follow-up independent review and actual runtime qualification remain required;
source review is not a passed test or scientific result.

The second source review of `6cdb680` confirmed closure of the stop-during-save,
recovery-gated pilot, four early audits/both guards and baseline objections. It
found two remaining blockers: growth launch could miss changed pilot inputs,
and export/audit memory was absent from capacity forecasting. The latest source
binds/rechecks the complete pilot prerequisite manifest before a launch claim,
requires all eight measured pilot export/audit RSS peaks, and reserves twice
their forecast at the proposed table ceiling below 5.5 GiB. New fixtures cover
missing/changed prerequisites before claim and incomplete/underreserved memory
measurements. Round 3 independent review and M4 runtime qualification are still
required; none of these fixtures have been executed locally.
