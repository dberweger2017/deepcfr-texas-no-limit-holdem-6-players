# Native export: no sort workspace for sorted checkpoints

[#223](hu100-3b-ladder.md) stopped HU100 at 2B nodes before 3B. The current-policy export of its 54.6M-entry checkpoint used 13.58 GiB of temporary disk, and scaling that to 3B left too little free space for the 16 GiB floor. The workspace came from sorting every current-policy row through uncompressed text runs on disk. But the trainers write checkpoints in key order already.

`native/hu20-trainer` export now streams the current policy straight into its compressed staging file while keys strictly increase. Strictly increasing keys also rule out duplicates. If a checkpoint has any key out of order, the streamed attempt is discarded with all its staged files, and the export repeats with the existing on-disk sort. The output bytes are the same either way.

| #204's 39.4M-node HU100 checkpoint (7,643,261 entries) | Before | After |
|---|---:|---:|
| Peak size of the export directory, outputs included (sampled every 0.5 s) | 1.67 GB | 0.39 GB |
| Final current + average files | 0.38 GB | 0.38 GB |
| Wall time | 108 s | 112 s |

Average SHA256 `ba62d135…` (#205's pinned 39.4M average) and current SHA256 `606e8c2c…` are identical before and after. The workspace beyond the outputs drops from about 3.4× their size to none. At #223's 2B checkpoint, that saves about 13.6 GiB per export.

New Rust tests check three things. Streamed and sorted exports are byte-identical. A reversed checkpoint falls back, leaves no scratch behind, and produces the same bytes. Duplicate keys are rejected on either path. All existing native export, audit and capacity suites pass.
