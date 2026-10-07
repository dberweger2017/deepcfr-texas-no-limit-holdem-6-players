# Git, Google Drive and local research storage

This is the common storage contract for new work. [AGENTS.md](../AGENTS.md)
controls artifact preservation; [RESULTS_INDEX.md](../RESULTS_INDEX.md) remains
the canonical archive/retrieval index. A historical report's “retained” or
“upload pending” statement is a dated snapshot, not proof of current storage.

## What belongs where

**Git:** source, dependency pins, small configurations and deterministic test
fixtures, protocols, readable reports, modest plots, compact result summaries,
and archive/retrieval receipts. Keep enough context to understand the result
without downloading the full experiment. A filename such as `manifest.json`
does not make megabytes of raw data a compact receipt.

**Research-Cloud:** training checkpoints, inference exports, replay buffers,
bucket tables, raw hand/decision traces, bulky solver inputs/outputs, complete
logs, interrupted attempts, and generated research archives. Use the existing
`~/Local/Research-Cloud/PR-<number>-<description>/` folder on the owning machine;
record its actual Drive folder ID/URL. Both Macs refer to the same cloud folder.
Do not create a second folder merely because the spelling differs from an example.

**Ignored local working directories:** retrieve into `models/`, `results/` or
`planning/` outside synced folders. Keep active inputs where their campaign
expects them. Verify SHA256 before use. Released inference assets continue to
use their pinned GitHub release downloads; being named a release does not grant
permission to commit a model to Git.

## Required closeout record

Each campaign's entry in RESULTS_INDEX must name the owning PR and include:

- Source revision, run identity, and whether the record is final or partial.
- Actual Drive archive ID/URL and folder ID/URL, archive name, byte count and SHA256.
- Embedded member manifest path/hash and member size/SHA256 readback result.
- For each required model/input: archive member path, byte count, SHA256 and a
  usable retrieval command, or an exact link to its already indexed record.
- Separate upload evidence: current native uploaded/no-pending/no-conflicts
  status and independent cloud ID/name/size/parent confirmation. State explicitly
  whether remote archive bytes were downloaded and verified.
- Failures, partial attempts and restore dependencies. Never present a successful
  final bundle as if earlier failures did not occur.

A cloud listing proves metadata, not member contents. A local ZIP readback proves
local archive integrity, not upload completion. Both claims must stay distinct.
For manual retrieval, download the indexed archive to a fresh ignored path,
verify its whole SHA256, extract the named members and verify their SHA256s.
Never treat a link alone as an accepted archive or overwrite an active input.

## Repository guard

`python -m scripts.check_repository_artifacts` checks the staged index;
`--revision HEAD` checks a commit. It rejects new model/archive/database formats,
raw JSONL traces, tracked files under local research roots, and any blob larger
than **1 MiB**, including large plain JSON. Case variants and interrupted suffixes
are checked. Small JSON summaries, CSV tables, report images and source remain
eligible. This is a maintenance rule, not a content classifier: authors still
must follow the storage contract for data that passes the mechanical check.

[The exact-path retention list](../configs/repository-artifacts.json) pins existing
legacy payloads by Git blob ID and byte count. It is **not** a Drive manifest or a
list of files approved for deletion. Existing payloads remain to avoid breaking
replay, tests or active research. Changed bytes or copies at new paths fail.
Do not regenerate the retention list to admit new experiment output. A necessary
fixture exception needs a specific reviewable reason and exact bytes; a release
model exception additionally requires explicit owner approval and an exact-path
ignore exception, as AGENTS.md requires.

`.gitignore` prevents common accidental staging; CI also checks objects already
staged or force-added. Git blob IDs identify retained repository bytes and must
never be substituted for the archive/member SHA256s required for restoration.

## Removal and retirement

Before copying, archiving, moving or removing research evidence, read the owning
PR's **current** status. Protect every open PR root and its dependencies unless
the owner authorizes that specific action. Historical “merged” notes do not
replace the live check. Never delete inside synced folders or force offloading.

For merged evidence, removal needs current Google Drive upload acceptance plus a
dependency review of active runs, source references, tests and other worktrees.
The owner clarified on October 7 that confirmed Drive uploads are trusted for
cleanup: use existing manifests and provenance without downloading archives or
repeating archive/member hash audits. Check selected originals against recorded
paths, sizes and modification history; retain changed or uncertain files. This
cleanup preference does not change new-archive creation receipts or SHA256 checks
when retrieving models for research use. Write a receipt naming each removed
path, exact restoration archive/member/hash, and update RESULTS_INDEX. Owner
standing authorization applies only after these checks. Unknown provenance means
retain and record the gap, not delete.

Retiring code is separate from deleting evidence. First establish its callers,
fixtures, owning campaign and reproducible source revision. Keep mathematical,
rules, information-boundary and recovery regressions when replacing an approach.
No automatic archive upload, worktree cleanup, history rewrite or background
maintenance is scheduled by these instructions.
