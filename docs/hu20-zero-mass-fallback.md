# Zero-mass fallback play measurement

Use M1/free local compute only; no RunPod, training or release decisions. Review and merge #178 after every required check is green, then build merged main's native trainer in release mode. Pin #165's six T/O 1B checkpoints by its manifests, seeds 2026100601/02/03; re-export each with `--zero-mass current` as T′/O′ and run `cfr_average.audit` on every export. Pin original T/O and shipped v0.4.0 R1 by exact manifest hashes. Retain the existing trial export separately.

Before any pilot, labels are **better** if the nominal paired Student-t 95% interval lower bound >0, **worse** if upper bound <0, otherwise **no detectable difference**. Positive values favor the first policy. Intervals condition on the three retained training lineages, with no multiplicity-adjusted claim.

- Primary A: T′ versus matched-seed T, each lineage and equal-weight three-lineage deal-block aggregate.
- Primary B: O′ versus matched-seed O, each lineage and equal-weight three-lineage deal-block aggregate.
- Secondary: T′ versus shipped R1; O′ versus shipped R1; T′ versus matched-seed O. Report every lineage and aggregate.

Use #175's direct runner, loader, duplicate deals, seats swapped, paired blocks, matched seeds, reporter and independent action/settlement replay unchanged. Average seats and lineages inside blocks; never count them as independent replicates. One BB =100 chips, so mean net chips per hand numerically equals BB/100.

Reserve direct pilot root **202610063401**, final **202610063501**, and conditional pressure root **202610063601**. Search tracked files and readable retained plans for previous use before launch. Direct pilot: **2,048 duplicate blocks per pairing**, all fifteen pairs. Inspect only block SDs, runtime, memory, completeness and failures; never pilot means, intervals or labels. Exclude pilot deals from final estimates.

For each primary family, use the maximum SD among its three lineage block series and its aggregate. Solve Student-t(0.975,n−1) × SD / sqrt(n) ≤**3.5 BB/100**, round upward to multiples of **4,096**, minimum **32,768**. Give all secondary families the larger primary count. Freeze a different fresh root, counts, startup-inclusive estimate, individual plans and canonical bundle SHA256 in a PR comment before final play. Requested achieved primary half-width ≤4; report actual widths without outcome-driven extension. Pilot arithmetic audit follows final freezing.

At most **three M1 workers**, unchanged **6 GiB per-worker RSS guard**, **8 GiB free-disk floor**, and #175's **two-hour aggregate final play/report/replay cap**. Stop before final launch if the pilot's measured play-plus-audit projection with 50% headroom exceeds one hour. Monitor free disk throughout preparation, play, replay and archival; stop and report immediately below 8 GiB. Preserve every partial and failure. Keep research outputs gzip-compressed; a storage adapter compresses metadata/logs around unchanged runner/reporter/auditor functions without altering poker or statistics. Never delete or evict anything to make room.

After independently verified direct results, post all contrasts with intervals, mechanical labels and a plain reading. If either primary aggregate is better, run the predeclared small native-pressure check using #165's unchanged arena play/loader/panel: **12,288 paired deal blocks**, all four arms T′/T/O′/O, both seats and three matched lineages. Report T′−T and O′−O nominal paired 95% intervals, all lineage/position results and street coverage. Replay every action/settlement and independently recompute raw-chip statistics. No sample extension or release gate. Use the same worker/RSS/disk guards and a separate two-hour conditional-check safety cap.

Archive all research files, exact inputs, source, environment, plans, raw traces, logs, failures, audits and restoration instructions in a member-hashed ZIP under `~/Local/Research-Cloud/PR-<n>-HU20-zero-mass-fallback/`. Verify every member by readback. Retain originals and never delete or evict synced files. Owner-approved PR comments remain binding.
