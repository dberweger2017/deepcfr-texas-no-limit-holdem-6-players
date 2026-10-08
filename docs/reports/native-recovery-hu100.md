# Native recovery and HU100 table growth — preparation status

**HU20 reference running on M4; qualification passed.** On October 8 the owner
explicitly authorized M4 use before #188 merge, then extended the hard deadline
to **12:00 Madrid /10:00 UTC**. Actual inspection found no competing heavy
research worker, AC power and about 100 GiB free disk. Isolated M4 checkout and
environment preserve other agents' source/dependencies. [Frozen campaign protocol](../native-recovery-hu100.md).
[Evidence PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197)
records qualified tooling; scientific audits are still pending.

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
HU20 reference/recovery and measured pilot costs
consume the same eight-hour maximum window. A late #188 merge/closeout shortens
the available budget. HU100 ends at 10B total nodes, capacity or time; reaching
10B is not promised. Upload acceptance may remain pending at the deadline and
will be labeled as such.

| Work | Status |
| --- | --- |
| Isolated branch and scheduled wake | Prepared; no duplicate launch |
| Tooling extensions | Qualified on M4; owner-release/deadline amendment independently reviewed |
| Independent correctness review | Round 4: no blocking source findings at `6431839`; runtime qualified |
| Build/test qualification | M4 release build, 7 Rust tests, 81 Python fixtures and artifact check passed at `6431839` |
| HU20 pinned 1B reference current/average audit | Training started 07:57:12 Madrid; exports/audit pending |
| Retained 500M →1B complete-state/probability equivalence | Not run |
| HU100 100k/1M/5M/10M pilot and audit | Not run; blocked on HU20 correctness |
| HU100 each 1B through 10B/capacity/time | Not run |
| Research-Cloud archive/member/upload verification | Historical input verified; campaign archive/upload pending |

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

The fourth independent source review found no blocking issues in `6431839`
against `e10001d`, confirming the exact owner-release receipt and noon deadline
while retaining idle/resource/identity/claim guards. M4 qualification finished
with 81 Python fixtures and 7 Rust library tests passed, release build and
repository-artifact check passed. Sampled aggregate peak **356,220,928 bytes**,
zero swap growth; all guarded attempts complete. Binary SHA256
`d6ecd69ce54b1afaf50e6df64edf104f1d14a77681de4d2c227b759015b2cf29`.
The M1 system-Python artifact-check attempt failed because Python 3.9 lacks
`zip(strict=True)`; the supported environment passed. No check was waived.
Historical parent retrieval is running under the same hard limits; science
awaits its verified whole-archive/member hashes. Admission timer disabled,
30-minute compact execution timer active, deadline timer now 12:00 Madrid.

At **07:57:12 Madrid /05:57:12 UTC**, fresh reference attempt `reference-02`
claimed execution durably and started native training. Earlier operator failures
(runpy invocation, then missing `gh` in detached PATH) occurred before any launch
claim or scientific work; logs and prepared `reference-01` remain retained. The
installed M4 GitHub CLI returned HTTP401, so an ignored operator adapter accepts
only the two fixed read-only PR188 status invocations, checks the public GitHub
REST response identity, and records live response provenance. It reported #188
MERGED at 05:34:01 UTC; no credential was transferred and the qualified
`6431839` source/binary was unchanged. The wrapper advances reference → recovery
→ pilot only through the reviewed gates. Growth awaits measured pilot admission.

Historical #182 retrieval completed under the external guard in **427.96 s**,
with **51,593,216-byte** sampled aggregate peak and zero swap growth. It verified
the complete indexed ZIP, embedded manifest, and **171,794,336-byte** retained
500M member before atomically accepting `inputs/historical-500M.json.gz`.
No equivalence or HU100 result is claimed by this status update.
