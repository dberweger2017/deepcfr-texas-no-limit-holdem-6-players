# CFR+ floor control match protocol

Compare the exact #165 1B-node exports with 0.4.0-shield and shipped v0.4.0 R1, using free M4 compute only. No training or release decision. Source: merged #173; its direct runner, policy loader, duplicate-deal play, reporter and independent replay/audit remain unchanged.

Before the pilot, labels are **better** if the 95% interval lower bound > 0, **worse** if its upper bound < 0, and **no detectable difference** otherwise. Positive values favor the first policy. All intervals are nominal Student-t 95%, conditional on the three retained training lineages; no multiplicity-adjusted claim.

- **Primary A:** #165 T versus shipped R1, each seed 2026100601/02/03 and equal-weight three-lineage mean within each independent deal block.
- **Primary B:** shield versus #165 T directly, matching training seeds 2026100601/02/03, each lineage and equal-weight block aggregate. This controls the node budget and extraction, isolating the floor recipe at fixed node work; floor-dependent trajectories and iteration counts may differ.
- **Secondary:** #165 O versus shipped R1, each lineage and equal-weight block aggregate.

Every deal runs twice, with policies swapping physical seats; seats and lineages are averaged inside blocks, never counted as independent replicates. One BB = 100 chips, so mean net chips per hand numerically equals BB/100. Policies receive only their own observation and separate action streams. Every action and settlement undergoes native replay during play and a later independent audit, including raw-chip interval recomputation.

Reserve pilot root **202610061501** and different final root **202610061601**. Search tracked configurations/reports and retained campaign plans for earlier use before launch. Pilot: **2,048 duplicate blocks per pairing**, all nine pairs. Inspect only block SDs, costs, memory, coverage and failures; never pilot means, intervals or labels. Exclude pilot deals from the final run.

Outcome-blind sizing: for each primary family, use the maximum SD among its three individual block series and its three-lineage block aggregate. Solve Student-t(0.975, n−1) × SD / sqrt(n) ≤ **3.5 BB/100**, round n upward to a multiple of 4,096, with a minimum of 32,768. This adds headroom for the requested achieved half-width ≤4; actual width may differ. Give secondary O the same count as A. Freeze counts, a different final root, a startup-inclusive measured time estimate and canonical plan SHA256 in a PR comment before final play. No outcome-driven sample extension.

M4: at most three workers; unchanged 6-GiB per-worker RSS limit, extra 15-GiB free-disk guard, two-hour aggregate deadline for play/report/replay, and preserve all partials and failures. If the pilot projects more than an hour with 50% time headroom, stop before final launch and report the resource finding. All compute is local and free. Independent audits cover pilot and final hands; the pilot arithmetic audit runs only after final counts are frozen.

Plain interpretation uses both controls: if T is at least as good as R1 and shield is worse than T, the floor recipe explains the loss; if T is also worse than R1, budget/averaging contributes. A nondetection is not proof of equivalence, and these controls do not separate budget from averaging or certify general poker strength.

Archive all research files, exact inputs, source, environment, logs, partials, plans, raw traces and audits in a member-hashed ZIP under `~/Local/Research-Cloud/PR-<n>-HU20-floor-control/`. Verify every ZIP member by readback. Keep originals and never delete or evict synced files.
