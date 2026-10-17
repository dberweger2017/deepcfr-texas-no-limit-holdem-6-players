# HU200 river search: proposed strength counts and resource admission

October 17, 2026. **Historical proxy proposal, superseded for final sizing by
the owner-approved [fixed calibration](hu200-river-calibration-protocol.md).**
The [measured final-count proposal](hu200-river-search-final-proposal.md) now recommends
131,072 direct blocks plus 2,048 per script, pending owner choice and storage
admission. The old options and storage failures below remain evidence of the original
full-body design. They are not current admission claims. The evidence-retention
gate now passes, two/three-worker throughput is measured, and the fixed
128 direct plus seven by 32 calibration completed within its original two-hour
cap. Calibration roots are reserved and excluded from finals. Final counts and
venue remain the owner's decision before any final hands.

## Measured inputs and uncertainty

The fixed 24-spot M4 cost study has mean complete service time **5.290501 s**,
p95 **9.648689 s**, 2.91 GB search-family peak RSS and at least 5.32 GB host
available memory. All searches complete, without fallback or missing actual EV.
Full output parity passes against the original implementation. Search work stays
50 iterations, six threads, the same full ranges/menu and complete prior locks.

For an initial power calculation, use the largest measured independent direct
block SD from #306's three blend-versus-seed matches: **879.355850 BB/100**.
That is a historical planning proxy, **not measured search-versus-blend variance**.
At two-sided alpha .05, the normal approximation
`n = ceil(((z(.975) + z(.8)) * SD / 10)^2)` requires **60,693 blocks**.
Rounding up to 65,536 gives approximately 82.9% power under that variance model.
Actual power can differ when search changes the distribution.

The direct cost projection uses #306's maximum observed blend river-decision
rate, **0.389908 decisions per hand**, across those matches. For the panel,
assume one searched decision per search-arm hand: two per paired block. This
panel rate is an exposed assumption, not a measured search frequency. The
24-spot sample is stratified and cannot estimate how often river search occurs.
Include 25% time headroom for verification and overhead; these estimates are
planning values rather than a hard runtime guarantee.

[Arithmetic and source hashes, computed on the guarded M4](reports/hu200-river-search-artifacts/strength-count-proposal.json).

## Count options before play

| Proposal | Direct duplicate blocks | Paired blocks per each of seven scripts | Historical-proxy power for +10 direct | Single-worker time with 25% headroom | Full native evidence projection |
|---|---:|---:|---:|---:|---:|
| Powered design |65,536|1,024|82.9%|120.2 h|13.28 TB|
| Smaller time option |8,192|256|17.7%|18.3 h|2.02 TB|

The powered design has 131,072 direct primary hands and 28,672 panel primary
hands. The smaller option has 16,384 and 7,168, respectively. A panel block
contains two seat rotations for each arm on common deals; a direct block
contains two rotations of the search-versus-blueprint duel. Repeated records
from replay/reproduction are not independent samples.

The smaller option has an approximately **27.22 BB/100** direct effect threshold
for 80% power under the same proxy. It does not satisfy the requested +10 target.
No result-dependent extension from that option to the powered design is allowed.
The owner must choose the design before hands; neither option is authorized here.

Any approved final declaration must pin fresh roots, both arms, exact counts,
search work and timeout policy before play. The direct nominal 95% block interval
is primary. The seven paired scripted contrasts receive both nominal 95% and
seven-comparison adjusted intervals, with pot_pressure prominent. It must also
predeclare adverse-result flags, verification barriers, action/EV/fallback
telemetry and the opponent-range diagnosis. No new default or Slumbot play is
implied by a positive research finding.

## Storage and concurrency are not admitted yet

This adapter retains **4,869,006,899 bytes for 24 decisions**, approximately
202.88 MB per decision, including complete native matrices. Under the exposed
frequency assumptions, raw evidence alone projects to 13.28 TB for the powered
design and 2.02 TB for the smaller design. These are extrapolations from the
fixed sample; new trees may differ. Neither fits the current storage arrangement:
bb128 had approximately 1.89 TB free before this incremental preservation, and
the M4 has only about 3.81 GB above its unchanged 16 GiB free-disk floor.

Another full fixed-24 concurrent pilot needs approximately 4.87 GB of evidence,
so it is not admitted. No two- or three-search throughput has been measured.
The single-worker memory result must not be presented as a concurrent result,
and runtime must not be divided by three without measurement. Do not delete
open-PR originals, change the disk floor, or run scientific work on bb128.

**Recommendation:** resolve evidence-volume plumbing before strength, then use
a declared variance pilot to replace the historical proxies. A concrete next
scope would preserve complete played-policy/support/fallback ledgers and
telemetry for every decision; preserve every native request and checksum;
retain full native bodies for an outcome-independent, predeclared diagnostic
sample and all failures. That would change evidence retention only, with the
same ranges, menus, iterations, locks and solver. It needs an explicit owner
scope decision and measured byte admission, exact-output parity, and another
cost check before use. Moving/removing any closed M4 scratch after verified
bb128 preservation also requires specific authorization for this open PR.
This proposal does not perform those changes or remove any file.

After storage is admitted, measure up to three concurrent unchanged searches
under the existing memory/disk guards and update the time estimate. The owner
then approves final counts and runtime before the committed final strength plan.

For that calibration, propose **128 direct duplicate blocks plus 32 paired
blocks per each of the seven scripts**, subject to owner approval before any
pilot hands, with a proposed cap of **two M4 hours including verification**.
Commit disjoint fresh roots and a protocol first. Use it only to
measure cost, river-decision frequency and independent-block variance; keep
mean effects and winning/losing hands out of engineering decisions. Its hands
never enter the final test. Freeze the solver before this pilot and make no
strategy/work changes after its outcomes. Use a predeclared variance-only
sizing rule, report variance uncertainty, and propose final counts to the owner
before the main study. No calibration hands are authorized by this document.

The 120-hour/13.28-TB calculation explains why a historical-proxy main study
cannot simply start now. It is not a recommendation to commit five days before
measuring search-specific cost/frequency/variance. The smaller main study also
does not establish the desired sensitivity.

The PR stays draft until the authorized research and storage closeout finish,
unless the owner explicitly changes its scope. A final-review date cannot yet
be fixed from this cost result alone.
