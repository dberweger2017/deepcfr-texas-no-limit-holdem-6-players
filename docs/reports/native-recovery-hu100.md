# Native recovery and HU100 table growth — preparation status

**Prepared, not run.** The owner requires #188 to be merged before implementation
or worker setup/training. [Frozen campaign protocol](../native-recovery-hu100.md).
The preparation PR will receive implementation, tests, independent review and
measured evidence after that gate and genuine M4 availability.

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
Implementation/setup, review, HU20 reference/recovery and measured pilot costs
consume the same six-hour maximum window. A late #188 merge/closeout shortens
the available budget. HU100 ends at 10B total nodes, capacity or time; reaching
10B is not promised. Upload acceptance may remain pending at the deadline and
will be labeled as such.

| Work | Status |
| --- | --- |
| Isolated branch and scheduled wake | Prepared; no duplicate launch |
| Tooling extensions, qualification and independent correctness review | Pending #188 merge and M4 admission |
| HU20 pinned 1B reference current/average audit | Not run |
| Retained 500M →1B complete-state/probability equivalence | Not run |
| HU100 100k/1M/5M/10M pilot and audit | Not run; blocked on HU20 correctness |
| HU100 each 1B through 10B/capacity/time | Not run |
| Research-Cloud archive/member/upload verification | No campaign artifacts yet |

All future work retains 5.5-GiB aggregate RSS, ≤0.5-GiB campaign swap growth,
≥15.5-GiB free disk, AC power, serialization headroom and an external hard guard.
No merge, release, publication, arenas, paid compute or increased limits.
