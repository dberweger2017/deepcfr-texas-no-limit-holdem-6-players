# M1 research storage and Drive organization — October 5, 2026

The M1 had **13,264,736,256 bytes / 13.3 GB free**, about 98% used, at the initial check. It already uses the same `~/Local/Research-Cloud` shortcut as M4, pointing to `~/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results`.

## Storage findings and protected scopes

The primary poker checkout measured about **23.2 GB**, including **19.7 GB of Git administration**. This Git store is shared with the active #166 worktree and other checkouts; it was not archived, cleaned or rewritten. The primary `planning/` directory is another 1.5 GB with unclassified ownership and remains untouched. Other local projects and personal files are outside this archival scope. Google Drive's managed CloudStorage tree measured about 71.3 GB; no synced data was deleted and no cache offloading was forced.

Fresh PR reads established **#149, #162, #164 and #165 merged**, while **#146 and #166 are open**. The open alias-audit, turn-search arena, pilot, quote and solver folders, their inputs and active worktrees were excluded from archival. Source roots were checked for open file handles and stable identities. No worker was stopped and no source file was changed. A separate shallow clone was used for documentation instead of switching the primary checkout.

## Merged-run archives

| PR | M1 scopes | New archive bytes | Source records | Existing canonical references | New payload records |
| --- | --- | ---: | ---: | ---: | ---: |
| #149 | Five older M4 retrieval copies, compact sixth retrieval and M1 engineering/preflight | 310,667,830 | 14,506 / 39,092,760,723 logical bytes | 14,181 | 325 |
| #162 | `hu20-trainer-bench-monitor-20261005` | 304,963 | 47 / 1,747,814,837 logical bytes | 34 | 13 |
| #164 | `hu20-native-gate-20261005` | 1,336,346,201 | 61 / 1,336,339,851 logical bytes | 0 | 61 |

All **14,614 source records / 42,176,915,411 logical bytes** passed SHA256 and byte-count checks. Repeated snapshots and original hard links mean these logical totals are not additional physical disk usage or a promise of reclaimable space. The three new archives total **1,647,318,994 compressed bytes**.

The #149 supplement preserves all distinct M1 historical snapshots, clocks, source/manifests and engineering/preflight files. Content already preserved in the complete [M4 campaign](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view) is referenced by exact member path, SHA256 and byte count; it is not uploaded again. The canonical manifest SHA256 is `d5608ba75a785dfc5633f96dc27246d1b0c7781e478e67bc31972163565fca65`, verified against the owning-task receipt. The canonical tar prefix is `campaign/`. Historical changes and the M1-only real-pool preflight are preserved in the supplement.

The #162 supplement preserves local monitoring, reporting and retrieval provenance, alongside exact references to the already accepted [M4 trainer-bench archive](https://drive.google.com/file/d/1_6dRapLReZ9-sAMPaePNX-nehXowGwki/view). Its canonical manifest SHA256 is `62aefd4d8b9dc3e4380612f772d513384c2049e6fc708d3bba1334230cf0a049`. The separate local compressed copy is recorded as a whole-file reference, not copied into another archive.

The #164 whole archive preserves all native gate files, including the **pre-fix `native-exact/` seeding-bug lineages**, corrected `python-rng/` outputs and the hash-verified `python-reference/` inputs described in [PR164's history](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/164). Earlier failed evidence stays distinct from the corrected parity result.

| Archive | SHA256 | Manifest SHA256 |
| --- | --- | --- |
| `PR149-M1-retrieval-history-20261005.tar.gz` | `939bb2ee7d88832cd480251f2c922794f33ec631f7086c1c9979390b1371f619` | `155730ea8fa20ea29dc18058d6225c076e4cd5669e8bd4d439899a82372fb793` |
| `PR162-M1-monitor-20261005.tar.gz` | `a925013e40bac632f7aa9c3db8b8226aa1fac209fb4a737aec400488e5d8df74` | `833ab623061f3a335cbe55016eeef7ad60f9326b10b90b21c01fad473afeebd7` |
| `PR164-M1-native-gate-20261005.tar.gz` | `dfabd1bfedfa0c276b803d4144005ebb831e3bd5f801f4c307e2879fdd27ff57` | `35cedb6744eee9e84f3bcacbddb78ecac866b99434f9631b5d059a23335a0974` |

## Drive locations and upload checks

- #149: [M1-board-pooling / M1-retrieval-history-20261005](https://drive.google.com/drive/folders/13RWLf779O6DpKZQ2rBKuEg-PpHOH7xX8), [new supplement](https://drive.google.com/file/d/17woiEOQSDwHvyq6eNzLhm5U2cl0nCQan/view). Confirmed uploaded.
- #162: [PR-162 / M1-monitor-20261005](https://drive.google.com/drive/folders/1Kq-JMOj5u4Rlnvevvb8QXfWRb_nCSKzT), [new supplement](https://drive.google.com/file/d/1bC5YnXxAfvsmMJwM-5eD-bukXM6BTSZb/view). Confirmed uploaded.
- #164: [PR-164 / M1-native-gate](https://drive.google.com/drive/folders/1MCnYU8wXQ0y2aTkWS9NVVaJ9aBDksRp3), [whole native-gate archive](https://drive.google.com/file/d/13aotv71LxwK4QQ0x3XIpP4sz4m42K9B_/view). Confirmed uploaded.
- #165: the existing [campaign](https://drive.google.com/file/d/1xenvcfa5QfhGutNe5xNtDhFLYRqKcyxI/view) and [audit/source snapshot](https://drive.google.com/file/d/1olH4RanIwqGhmPae7Pw8GtvIwRr0e14n/view) remain canonical. Both separately retained M1 archive SHA256s were freshly recomputed and matched; native uploaded/not-uploading/no-conflict/exact-size and current cloud IDs/names/sizes/parents agree. No duplicate was uploaded.

The newly staged archives are APFS clone copies with freshly checked staged SHA256s. All three new archives have verified native FileProvider upload acceptance and current cloud name/size/parent readback. [Upload receipt](https://drive.google.com/file/d/1JPuWeSbPIbMsHhXpyghuo0o6gaVyqKnP/view). Every newly included tar payload was streamed and hash-verified; each canonical reference has a freshly hashed source and a matching immutable canonical manifest record. This uses the established desktop acceptance criterion, with no full cloud-byte re-download.

## Restoration, index and retention

Each archive includes `ARCHIVE-MANIFEST.json`, `RESTORE-README.txt`, `restore_snapshot.py` and the packaging helper. A manifest entry maps an original M1 path either to a new payload or to a canonical archive member/whole file. For #149/#162, obtain both the canonical archive and supplement; verify their archive hashes, extract them into fresh directories on a host with adequate storage, then use the restore helper. #162's whole-file reference also needs `--canonical-archive`. The helper checks input sizes/hashes and preserves original hard-link groups. A bounded restoration check passed on a real canonical member, a supplement payload and a preserved hard link.

Git administration, virtual environments, node modules and Python/test caches are excluded. No source symlinks were found in these selected scopes. The separate local source archives and originals remain. No deletion, synced payload removal or forced cache eviction occurred; uploading and indexing therefore does not reclaim the original files' disk space.

The shared root `RESULTS_INDEX.md` is updated in place with the same Drive file ID, retaining the M4 and older records. The previous Drive copy and complete archival/upload receipts are preserved in [Archive-receipts / M1-organization-20261005](https://drive.google.com/drive/folders/1MiIacZEfk8tZhoxUaTLHwmxtck5bxCdr). This is archival evidence only; no training, gameplay, algorithm change or paid compute occurred.
