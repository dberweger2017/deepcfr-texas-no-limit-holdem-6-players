# Reduce HU100 export and audit memory

Owner-requested engineering work, October 8, 2026. No training, recipe change,
policy selection or merge. Branch `feature/hu100-export-audit-memory` starts at
merged #197 (`7dca996`). #200's completed experiment was still OPEN at admission;
its latest protocol/report at `acef171` were read through Git without touching
its protected roots. Use #197's accepted archives, not #200's live dependencies.

## Frozen measurement plan

Free Apple M4, 16 GiB, sequential children. Initial inventory: no competing heavy
worker, AC, normal pressure, 82% system-free, 440.81 MiB swap used, 84 GiB disk.
Recheck immediately before execution; refuse competing work or unsafe resources.
A single persisted start/deadline gives **1,800 seconds total** for input retrieval,
hash verification, M4 focused qualification, timing pilot, before/after benchmark
and complete equivalence verification. No clock reset or guard increase.

Whole owned family RSS **10 GiB**, system pressure level 1 /free percentage ≥15,
full ceiling plus 2 GiB headroom before each heavy phase, original captured swap
baseline/+0.5 GiB, disk floor **15.5 GiB**, AC. Existing external supervisor guards
all phases; fresh admissions precede heavy children. Stop on guard failure,
correctness mismatch or insufficient time. Never kill another worker.

Hash the original accepted #197 whole ZIPs and manifests; restore all five HU100
checkpoint/current/average sets and the pinned uninterrupted HU20 1B set into a
fresh ignored input root. Verify every selected member against its embedded
manifest and HU100 model index. Record archive URL/member/hash/source provenance.

Baseline source `7dca996`, candidate committed source and both native binary
hashes are retained. Reuse the same pinned Python environment without changing
it. Fresh processes run identical inputs in fixed before/after order. Separate
small-table timing pilot; 2× timing allowance and a 60-second reserve gate each
subsequent input. Prioritize largest HU100 (3,255,387 entries) and HU20 reference
(4,319,080 entries), then remaining HU100 sizes if they fit the same deadline.

Measure combined native current+average export, full Python audit and Python
extraction. Verify native gzip files against canonical byte hashes, Python
extraction gzip against baseline, and full audit receipts against baseline.
Retain every row, all metadata/order/numeric bits and all integrity checks.
Average inference behavior is covered by existing frozen-fixture tests.

Target 200-ms whole-family RSS sampling, including coordinator/supervisor and
all descendants, plus macOS `/usr/bin/time -l` command high-water RSS. Report
actual sample counts and separate the kernel command peak from sampled family
peak: neither establishes the exact simultaneous family peak. Sampling can miss
short-lived children/peaks and RSS double-counts shared pages; timing includes
measurement overhead and warm filesystem caches, with only one run per case.

Capacity recommendations retain 2× plus 10% headroom, account for training/save,
disk scratch and retained outputs, and limit extrapolation to 2× the largest
measured HU100 table. No estimate authorizes training or promises 1B/10B nodes.
Archive large inputs/outputs/source/logs/failures to a new Research-Cloud folder
with a member manifest and retrieval provenance. Preserve all originals.
