# O learning curve protocol

Question: does additional training still strengthen the seed **2026100601** linear-CFR opponent-sampled average under the v1 card abstraction? This is **one training lineage only**; findings do not generalize to other seeds or certify convergence. The v0.4.2 release decision belongs to the owner.

All exports use the default **uniform zero-mass rule**. The three #165 pilot checkpoints (100M, 500M and 1B training nodes) must match its campaign manifest. Build the native trainer from main; independently audit each average against its checkpoint accumulators and current export. The 1B export must equal SHA256 `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`; otherwise stop. Shipped R1 is SHA256 `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`.

## Labels and contrasts, declared before pilot

Positive BB/100 favors the first policy. **Better** means the nominal paired Student-t 95% interval lower bound >0; **worse** means its upper bound <0; otherwise **no detectable difference**. No multiplicity-adjusted claim.

| Role | Direct contrast |
| --- | --- |
| Primary A | O@1B vs O@100M |
| Primary B | O@1B vs O@500M |
| Secondary C | O@100M vs shipped R1 |
| Secondary D | O@500M vs shipped R1 |

Use #175's unchanged direct-match runner, reporter and independent auditor, with its frozen execution source `8bbfc457`. Duplicate deals swap physical seats; average the two seat returns inside independent deal blocks. Both players receive only their own observations and private action RNGs. One BB =100 chips, so net chips per hand numerically equals BB/100. Every action and settlement undergoes native replay during play and subsequent independent replay; raw-chip estimates, intervals and labels must agree.

Primary B determines the predeclared interpretation:

- **better:** still improving; test scaling beyond 1B;
- **no detectable difference:** plateau near 1B under v1 at this experiment's resolution;
- **worse:** investigate before scaling.

A nondetection does not establish equivalence or asymptotic convergence. Direct poker comparisons are not transitive.

## Outcome-blind pilot and frozen final

Reserve pilot root **202610065101** and different final root **202610065201**. Search tracked configs/reports and retained plans on both Macs before use; retain search receipts and any unreadable paths. Pilot **4,096 duplicate deal blocks per contrast**. Inspect only block SDs, timing, memory, completeness and failures; never means, intervals or labels. Pilot deals are excluded from final play.

Size each primary using Student-t(0.975,n−1) × pilot block SD / sqrt(n) ≤**2.7 BB/100**, rounding upward to 4,096-block increments, minimum 32,768. This gives headroom for the requested achieved half-width ≤**3 BB/100**. Both secondary contrasts receive the maximum primary count. Freeze all counts, the different final root, startup-inclusive time/replay estimate and canonical bundled plan SHA256 in a PR comment before final play. No outcome-driven extension; a width miss is reported as a miss.

**Free M4 only**, no RunPod or heavy M1 jobs. At most **three workers**, **6-GiB per-worker RSS**, **≥15 GiB free disk**, and a **two-hour final play/report/replay cap**. Preserve every partial/failure. Stop before final play if the outcome-blind forecast with 50% headroom exceeds one hour. Guards also cover preparation, reporting and audit processes.

The results chart plots 100M/500M direct performance against R1, with R1 at zero. Its 1B point uses #176's already-audited first-seed direct R1 comparison and is explicitly marked as historical on a different root; no additional match is introduced and no strength value is inferred transitively from A/B.

## Conditional scaled training and archive

Only if independently audited Primary B is **better**, start one M4 lineage immediately: `hu20-trainer train --seed 2026100601 --average-rule opponent-sampled --nodes 10000000000 --milestones 2000000000,5000000000 --out <path>/O-2026100601-{nodes}.json.gz`. Record timing and peak RSS, guard RAM/disk and stop before breach. Retain exact source, command, checkpoints and milestone receipts; post each milestone on the PR. Run no new matches until the owner decides. Keep the PR open while scaled training runs.

Archive all research inputs, exports, source/binary, environment, plans, raw traces, logs, audits, partials and failures as member-hashed ZIPs under `~/Local/Research-Cloud/PR-<n>-HU20-learning-curve/`. Verify every member by readback. Retain originals and never delete or evict synced files or other open PR roots; #166's M1 `~/Local/hu20-turn-search-arena-20261005/` is protected. Update the report and RESULTS_INDEX, review the diff and unresolved PR findings, and merge only with green required checks and nothing unresolved. No release or tag is authorized.
