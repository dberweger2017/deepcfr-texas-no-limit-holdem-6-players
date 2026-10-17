# HU200 river search: final counts and venue proposal

October 17, 2026, PR317. **Approved on the M4 with up to three searches.**
The [final declaration](hu200-river-final-protocol.md) fixes counts, roots, analysis
and the owner-authorized verified block transfer/local retirement.
The remaining text preserves the decision proposal and its original assumptions.
The fixed, disjoint calibration is complete; no final hands have been played.
Recommend **131,072 direct duplicate blocks and 2,048 paired blocks per each of
seven scripts**, on the M4 at three searches, subject to storage admission.
This exceeds 24 hours, so it is a proposal, not an authorized launch.

## Calibration and sizing

All **352 blocks / 1,152 primary hands** independently replay and reproduce with
fresh players and zero native reproduction solves. The original clock includes
launch plumbing repair/revalidation: **604.32 s** to worker closure, below its
7,200 s cap. All roots remain excluded from finals. Search is unchanged.

There are **202 searched decisions, 67 changed actions, zero fallbacks/timeouts
and zero missing successful-decision EVs**. Three-worker peak family RSS is
**10.326 GB**, minimum host available **6.033 GB**; both memory checks pass.
M4 compute was released **October 17 at 08:49:03 Madrid** after cable preservation.
Realized mean effects remain withheld from this variance-only planning readout.
No solver/range/opponent tuning follows calibration outcomes.

Direct duplicate-block SD is **1,086.84 BB/100**, normal-model 95% SD interval
**[968.04, 1,239.15]**, one-sided 95% upper SD **1,212.92**. The declared sizing
rule uses that upper SD, two-sided alpha .05 and +10 BB/100, rounded up to a power
of two. It gives **131,072 blocks**, approximately **84.7% power** under that
model; the plug-in estimate rounds to the same count. This is approximate power
conditional on the variance model, not a heavy-tail guarantee.

| Calibration panel | Blocks | Nonzero block scores | Block SD, BB/100 | Searched decisions |
|---|---:|---:|---:|---:|
|direct|128|90|1,086.84|98|
|random|32|2|114.96|6|
|check_call|32|13|250.08|60|
|tight_aggressive|32|0|0 observed|1|
|loose_aggressive|32|8|1,191.48|22|
|**pot_pressure**|**32**|**2**|**1,555.10**|**5**|
|half_pot_pressure|32|0|0 observed|6|
|matched_pot_pressure|32|2|1,542.47|4|

Panel counts follow the predeclared minimum of direct/64. They do **not** target
80% power for each opponent. In particular, the small pot_pressure calibration
implies approximately **±85.5 nominal / ±117.3 seven-way adjusted BB/100** final
halfwidths at 2,048 blocks, using its upper-SD model. These are unstable precision
proxies from only two nonzero pilot blocks. Zero observed variance elsewhere
does not establish zero uncertainty or exclude rare losses. All final nominal
and seven-way adjusted intervals, including inconclusive ones, must be reported.

## Two compute quotes

| Count option | Direct blocks | Paired blocks per script | Approximate +10 direct power | M4, three searches | Cygnus-x1 conditional trainer proxy |
|---|---:|---:|---:|---:|---:|
|**Recommended powered design**|**131,072**|**2,048**|**84.7%**|**73.4 h**|**448.6 h / 18.7 days**|
|One-day M4 option; misses requested power|32,768|512|32.0%|18.3 h|112.2 h / 4.7 days|

The recommended design has **319,488 primary hands**, the smaller option 79,872.
The M4 quote uses measured calibration block costs at the admitted three-worker
configuration, including per-hand replay/reproduction/checksums and load/setup,
with **25% headroom**. A one-sided 95% normal block-cost upper projection with the
same headroom is **87.4 h** (smaller option **21.8 h**). These are compute planning
quotes, not guaranteed finish times; final storage transport and archive/upload
closeout are additional and unmeasured. The smaller design targets roughly
+18.8 BB/100 for 80% power and cannot be extended after seeing results.

**Cygnus is not search-benchmarked and is unavailable while #301 computes.**
The quoted 448.6 h uses [PR283's pre-XMP trainer compute ratio](reports/cygnus-x1-qualification.md#exact-recovery-and-matched-hu200-k50-trainer-cost),
0.41145 of one M4 worker, after undoing the measured M4 fixed-spot contention
ratio. It is a hardware proxy from a different workload, not measured search
speed. One six-thread native search plus runtime headroom respects the eight
busy-thread limit; two six-thread searches are not proposed. Assuming anywhere
from one-M4-worker speed down to that trainer ratio gives **184.6–448.6 h**
(smaller option **46.1–112.2 h**). This is a scenario range, not a confidence
interval or a guaranteed bound. Current XMP/thermal changes and different solver
instruction paths may change it substantially. After #301 releases compute,
require fixed-spot build/output parity, cost and thermal qualification, then
owner acceptance of the resulting quote before any final hand there.

## Storage admission and next decision

The calibration retains **278.3 MB** of complete hand/decision evidence with the
predeclared native-body sampling. The powered projection is **90.6 GB**, smaller
option **22.6 GB**; rare larger bodies and every failure can increase it. The
M4 has only about **2.38 GB above its unchanged 16 GiB disk floor**, so neither
final design is admitted as a wholly local run. The bb128 research cache has
capacity, but remains storage-only while #301 computes.

For the recommended M4 design, propose a bounded transfer of closed, verified
final blocks to bb128, with per-file SHA256 readback before retiring those exact
owned local block files. This needs the owner's specific authorization because
open-PR originals are protected. Existing research roots, inputs and calibration
originals stay retained. The transfer/resume plumbing and its measured overhead
must qualify before a final protocol and launch. Alternatively provide enough
M4 scratch space, or select cygnus after its qualification. No file retirement
or final strength hand occurs under this proposal.

The owner chooses counts/venue and storage treatment. Then commit a final plan
with fresh roots, fixed counts/work, stopping/adverse-result rules, complete
verification barrier, all intervals, changed-action/EV/fallback telemetry and
the HU20-style range-model diagnosis before playing. No Slumbot run is implied.
The PR stays draft until its planned final research and preservation are done;
final-review ETA depends on that choice and actual closeout.

[Calibration summary and uncertainty](reports/hu200-river-search-artifacts/calibration-summary.json) ·
[Arithmetic and both venue quotes](reports/hu200-river-search-artifacts/calibration-final-count-proposal.json) ·
[All-hands barrier](reports/hu200-river-search-artifacts/calibration-all-hands-verified.json) ·
[Prospective calibration protocol](hu200-river-calibration-protocol.md).
