# PR292 archive and cleaning-agent handoff

Snapshot: October 16, 2026. [PR292](https://github.com/dberweger2017/CFR-NLHE-6max/pull/292).
The owner authorized merge after this handoff and green CI. All scientific
work is complete; the learning rule passes on all three seeds. Archive
completion remains blocked by the OpenAI upload bridge. No new training,
evaluation, scientific retry, release, M4 access or unattended cleanup.

## Current state and deletion boundary

**Retain all scientific originals, archives, source and inputs.** There are
237 expected cloud pieces, of which **133 are uploaded and accepted**. The
other 104 are missing from the accepted archive. A partial archive cannot
restore the complete ZIP. Merge authorization does not waive upload acceptance
or dependency review before removing originals.

The [partial cloud index](../reports/nplayer-blueprint-phase2-artifacts/archive-main-cloud-partial.json)
records every accepted piece's Drive ID/URL, name, bytes, SHA256, order and
native/independent metadata observations. Do not upload those pieces again.
No remote archive bytes have been downloaded or verified. The
[block receipt](../reports/nplayer-blueprint-phase2-artifacts/archive-transport-block.json)
is the earlier failure-time snapshot; its 126 accepted count is superseded by
the subsequent 133 accepted observations in the partial cloud index.

The worker and recurring schedule are paused. No deletion is performed by this
handoff. Recheck owning PR status and current files/dependencies before any
later action, following [artifact storage](../artifact-storage.md) and AGENTS.md.
Never delete inside a synced folder or force offloading.

## Exact locations

| Location | Contents and retention |
| --- | --- |
| `bb128:/home/dberweger/nplayer-blueprint-phase2-20261015/` | Complete protected scientific root: training, checkpoints/exports, evaluations/traces, replay/reproduction, guards/resources/VRM/fan1, runtime/source, archives. |
| Same host root, `archives/main/evidence.zip` | Sealed full scientific ZIP, 15,887,630,020 bytes. |
| Same host root, `archives/main/evidence.zip.part0000` through `.part0118` | Original 119 host transport parts. Preserve all; they differ from the 237 cloud piece names. |
| Same host root, `training/<seed>/<milestone>m.gz` and `<milestone>m-average.gz` | Seeds 2026101541–1543; milestones 200, 500, 1000. All 18 checkpoints/exports protected. |
| `/Users/dberweger/.t3/worktrees/deepcfr-texas-no-limit-holdem-6-players/feature-nplayer-game-parity/planning/nplayer-phase2/` | Ignored M1 working root: current `HANDOFF.md`, transport helpers, upload/native receipts, partials, last guard and schedule identity. Keep the whole root. |
| M1 working root, `transport/spool/` | `*.ready.json`, `*.upload.json`, `*.accepted.json`, native observations and remaining `.bin` pieces. Persisted upload IDs are authoritative; reconcile before retry. |
| M1 working root, `transport/main-part.interrupted-bios-20261015.bin` | Preserved 94,264,320-byte interrupted SCP buffer; keep with its failure/log provenance. |
| M1 working root, `last-guard/` | Already retrieved small raw final archive-guard command/resource/result receipts for the remaining provenance ZIP. |
| [Research-Cloud folder](https://drive.google.com/drive/folders/1O7yc5RzlIRGD0iWn8nSXoyElcBWdriI9) | `PR-292-nplayer-blueprint-phase2`, ID `1O7yc5RzlIRGD0iWn8nSXoyElcBWdriI9`, parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. |
| `~/Local/Research-Cloud/PR-292-nplayer-blueprint-phase2/` on M1 | Symlink into the native Drive folder; resolve before FileProvider checks. Do not delete or force offload synced files. |

Native canonical folder on M1:
`/Users/dberweger/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results/PR-292-nplayer-blueprint-phase2/`.

Main ZIP SHA256:
`ea57f0e4910c458c5597c194da25342aaf912f2f098452e8e0982b1121dddb3f`.
Embedded `MANIFEST.json` SHA256:
`ae9903fa0a8fea8b59bddb55d0441528de7b87f66136071d4adc26fc4c44700c`.
All local member sizes/SHA256/CRC and original part concatenation passed before
transport. [Seal and 119 host-part pins](../reports/nplayer-blueprint-phase2-artifacts/archive-main.json).
[All 18 model member paths, bytes, hashes and source mtimes, plus runtime/input pins](../reports/nplayer-blueprint-phase2-artifacts/archive-model-members.json).

The separate `phase2-archive-guard-closeout.zip` is already accepted:
[Drive ID `1utIneB1zdFo0BcLR3_no3tJN48jd_oOl`](https://drive.google.com/file/d/1utIneB1zdFo0BcLR3_no3tJN48jd_oOl/view),
53,823 bytes, SHA256
`77691215a001e50f9edf1adcdaf341a229c3b0167d3b48c1a5fa35f3ecf99357`.
Do not upload it again. [Manifest and acceptance](../reports/nplayer-blueprint-phase2-artifacts/archive-closeout-cloud.json).

## Protected shared dependencies

Training source: `8ad2729c1fc9edad4afe8e406b28dd680b50a989`.
Evaluation/manifest source: `71aa93612ac9c5de1a51862d94ce430db5c41e1b`.
These scientific sources remain frozen; later integration/report commits do
not authorize rebuilding or rerunning their evidence.

The host `repo` shares Git objects with PR291 → PR289 → PR288 → #283.
Do not remove those checkouts or Git objects. The environment and K50 tables
also depend on #283. [K50 archive/input restoration](../reports/cygnus-x1-artifacts/input-restoration.json).
PR291's protected smoke input is
`bb128:/home/dberweger/nplayer-size-smoke-20261015/smoke/scoped-average.gz`,
225,158,368 bytes, SHA256
`fcbe8bd5733572fe9f9046840d6e277a0b11ce0ac96936546df0815be218485e`.
It is also included as `inputs/phase1-scoped-average.gz` in the main ZIP.
Do not clean other campaigns, shared environments, tables or active inputs.

## Resume archive work, without duplicating it

Read the ignored current `HANDOFF.md` first. Original worker PID41487 exited
during the owner's BIOS shutdown; `transport/worker-failure.json` and
`transfer-0063.log` are historical and retained. The explicitly resumed worker
is **PID36169**, script `transport_resume_owner20261016.py`, currently stopped
with SIGSTOP. Verify PID identity and state before considering SIGCONT; never
start a second worker. It has verified/split 79/119 source parts, yielding
158 ready pieces in a bounded 2-GiB spool. Forty host source parts still need
retrieval. bb128 is compute-free but will be needed for that file transfer.

The sole schedule is disabled, ID
`scheduled-task:command:mcp:4f5fb925-b540-427b-a123-7d38cba085f5:schedule-task:pr292-nplayer-phase2-progress-30m-20261015-v1`.
Do not re-enable it or retry a terminal stop merely because a previously queued
scheduled prompt arrives. No other campaign's schedule may be changed.

The bridge failed before Drive upload for `main-evidence.zip.part0066-1`:
first Azure HTTP403 AuthenticationFailed, then a request-send error to OpenAI's
files endpoint. Exact-name/parent reconciliation found no file. Preserve
`transport/bridge-staging-403-20261016.json`,
`bridge-staging-403-reconciliation.json`, `bridge-terminal-block-20261016.json`
and `post-stop-native-reconciliation.json`. No automatic ambiguous retry.
After owner-authorized continuation or recovery, reconcile again before any
upload, resume the existing worker, and use the existing `upload_batch.js`
invocation documented in the ignored handoff, with limit32 and at most64MiB
per piece. Persist every new ID before further steps. Acceptance requires
independent cloud ID/name/size/parent and native uploaded1/uploading0/conflicts0.

Finish all 237 pieces, ordered by `(source_index, sub_index)`; their byte sum
must equal the sealed ZIP size. Keep the original 119-part receipt separate.
Then produce the complete cloud manifest and a small M1 provenance ZIP with
last-guard receipts, raw transport/native evidence, helpers and every failure;
perform local member readback and native plus independent upload acceptance.
Index all IDs, pins and restoration commands. Do not claim remote-byte
verification without downloading bytes, or start further science.

## Retrieval and eventual cleanup

Until cloud completion, a required model can be retrieved directly from the
host into a fresh ignored directory, then checked against the member pins:

```sh
mkdir -p results/pr292-retrieval
rsync -a bb128:/home/dberweger/nplayer-blueprint-phase2-20261015/training/2026101541/1000m-average.gz results/pr292-retrieval/
shasum -a 256 results/pr292-retrieval/1000m-average.gz
```

Expected SHA256:
`990246065bfc1f8a13c5ca3456090689b5ab52105c860d4bca1531b3f5e3db70`;
705,933,433 bytes. Other models use their exact indexed member/hash.

For complete host-part restoration on a machine with sufficient space, copy
all 119 `evidence.zip.partNNNN` files into a fresh ignored directory, then run:

```sh
python3.11 -m scripts.restore_nplayer_phase2_archive --receipt docs/reports/nplayer-blueprint-phase2-artifacts/archive-main.json --parts results/pr292-host-parts --out results/pr292-restored --member training/2026101541/1000m-average.gz
```

The helper verifies each part, the concatenated ZIP, embedded manifest and
selected members. Cloud restoration requires the **future complete** cloud
manifest with all 237 downloaded pieces; do not pass the current partial
index. Each final part row must retain `member` as well as `cloud_name` for the
restorer. Avoid staging the full archive on the owner's M1.

After full acceptance, check current owning PR status, recorded paths/sizes/
modification history and all active dependencies before retiring any merged
original. Retain changed or uncertain files. Existing accepted receipts and
provenance suffice for cleanup; do not repeat expensive archive/hash audits
or download remote archives just for cleanup. Record every removed path,
restoration archive/member/hash and dependency decision in a cleanup receipt
and RESULTS_INDEX. Existing disposable-buffer removals are recorded in
`transport/disposable-removals.jsonl`; this handoff deletes nothing.
