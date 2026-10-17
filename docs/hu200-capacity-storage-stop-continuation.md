# PR301 continuation after the October 17 storage stop

The owner authorized continuing training and adjusting timing for this stop.
This is one continuation of PR301, not permission for future experiments or later
unattended retries. Original seeds, cap, nodes, trainer/export binaries, card
tables, audits, fixed evaluation counts and global replay/reproduction barrier
remain unchanged. No trainer, exporter or auditor implementation changes.

Adopt all three audited 10B recoveries and all earlier identity/memory gates.
Seed1302 alone continues from its own 10B checkpoint to20B in a fresh output
root. Seeds1421/1422 keep their completed20B checkpoints and existing exports:
restart only their interrupted every-row validation, using the unchanged sorted
auditor. Rehash all inputs and check every row, strict order, lineage and gzip
EOF/CRC. Existing exports are never rewritten. Successful audit receipts are
new files; old stop files, traces, failure receipt and partial attempts remain.
Seed1302's new20B save gets the original full independent save/export/audit path.

Reuse the qualified global scheduler/guards. Add only a reservation-policy hook:
existing jobs retain their exact20GB/23GB reservations, while read-only revalidation
and storage receipt jobs reserve64MiB for metadata. One two-thread native run may
overlap the two single-thread revalidations. At most three helpers and eight
reserved CPUs, original RSS/swap/temperature/disk floors and absolute deadline
remain. Binary/table/parent hash and fresh resource admission precede launch.

After each verified20B audit, move its now-inactive10B recovery and20B current
export to the owner-authorized HDD location, verifying source SHA, independent
destination readback and source identity before removing the NVMe copy. Keep20B
recoveries/averages on NVMe;5B averages/twins remain on HDD until needed. Never
move files still used by a job. This uses the owner's continuing storage authority
and preserves evidence. No other PR or shared source/dependency is moved.

Preserve the original clock and October16 one-hour ledger byte-for-byte. Seal one
new immutable ledger at restart admission: add only the elapsed interval from
`2026-10-17T09:21:22.269998Z` (recorded terminal storage failure) to that admission.
This interval closes once; reading the ledger again never adds credit. Admission
and first worker-start timestamps are recorded separately. No subsequent idle
period or failure receives an automatic extension. Evaluation's24h clock still
starts only at the first cache build after all training/audits.

Preserve root `training-failed.json` as historical evidence. The new attempt uses
`continuation-20261017/`, fresh guards/source/output/stop files, an explicit owner
approval and a superseding completion/failure receipt. Never restart the old
controller. Only all three successful20B audits and storage receipts publish the
new terminal completion. The same existing bounded event task receives the new
attempt's terminal event, with its own delivery ledger; no new schedule.

Before launch, declare the source and remaining work in PR301. Record coordinator
resumes/handbacks separately from job starts/stops, with UTC/Madrid timestamps,
reasons, background job state, deadline changes and ETA. Historical handbacks with
only completion evidence are marked as such. On any new failure, stop/report;
no later automatic scientific retry. Owner decides merge and subsequent use.
