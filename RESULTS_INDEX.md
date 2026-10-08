## PR202 HU100 export and audit memory (October 8)

[PR #202](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/202) is **open, ready for review; do not merge or clean its roots**. [Report](docs/reports/hu100-export-audit-memory.md) · [protocol](docs/hu100-export-audit-memory.md). Final production source `506743412fdbf7ec2cb3f4a1069b68d0effcb538`; later edits are harness admission logging, tests and documentation. Largest HU100 (11,042,440 actual nodes /3,255,387 entries): sampled whole-family export **3.172 →0.101 GiB**, audit **4.668 →0.141 GiB**, with approximately 2× time. All six inputs preserve compressed native current/average bytes, Python extraction bytes and complete audit receipts. [Equivalence](docs/reports/hu100-export-audit-memory-artifacts/final-equivalence.json) · [measurements](docs/reports/hu100-export-audit-memory-artifacts/final-measurements.json) · [resource limits/failures](docs/reports/hu100-export-audit-memory-artifacts/resources.json). Science/verification finished at 1,550.88 seconds, archive local verification/copy at 1,601.74 seconds of the original 1,800-second budget; subsequent asynchronous upload acceptance involved no new science. Initial setup failure and fresh-headroom refusal/never-started continuation remain disclosed. No training or guard increase.

**Accepted primary archive:** [hu100-export-audit-memory-M4-20261008.zip](https://drive.google.com/file/d/1-XHLJ-dM_DFMFTtXZYJZW7OmBeW2VLem/view), **4,901,349,021 bytes /309 verified members**, SHA256 `b5ae66eed7ed5e07c4fb8aa25cf3f72fa6de36a203798e568859783da380c554`. Native path `~/Local/Research-Cloud/PR-202-HU100-export-audit-memory/hu100-export-audit-memory-M4-20261008.zip`; [folder](https://drive.google.com/drive/folders/10d6DPp8i6uQrW5QaEaJEW364Dd7Vt7nv) under Research-Cloud `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Embedded `ARCHIVE-MANIFEST.json.gz` compressed SHA256 `f965f1e770d8738d68c5c253a6af3501b85ee872185080da0f19e79dd2ef08d3`, decompressed SHA256 `2c2193ce313c2c5a9e953b236051e5836f9685b7a8714bd267f1a33b743b0357`. [Local/member receipt](docs/reports/hu100-export-audit-memory-artifacts/archive-local-receipt.json) · [native/cloud acceptance](docs/reports/hu100-export-audit-memory-artifacts/archive-upload-acceptance.json) · [archive guard](docs/reports/hu100-export-audit-memory-artifacts/archive-guard.json). Native uploaded/not-uploading/no-conflicts and connector ID/name/size/parent match; no remote-byte download claimed.

**Accepted later metadata seal:** [hu100-export-audit-memory-closeout-20261008.zip](https://drive.google.com/file/d/1OizXJXcge0u2LtmpjsOvyIsXizPTEzDo/view), **140,594 bytes /31 verified members**, SHA256 `b36699a8803a5d3a13a7307723a1b194f05d2c815e2b2de9630c773c7c4cb30d`; embedded `ARCHIVE-MANIFEST.json` SHA256 `0e40f810fbf2dcd28659043ab82ffc6bfca19b34d49533618b278bb9ac621866`. [Local/member receipt](docs/reports/hu100-export-audit-memory-artifacts/closeout-archive-local-receipt.json) · [connector acceptance](docs/reports/hu100-export-audit-memory-artifacts/closeout-archive-upload-acceptance.json) · [independent evidence review](docs/reports/hu100-export-audit-memory-artifacts/evidence-review.json). Later reports, compact receipts, changed source modules/tests, admission fix, native upload acceptance and dated pre-final CI snapshot are sealed separately; manifest pins `6933530` and files, without changing frozen production source or primary ZIP. This M1 metadata-only work followed the scientific cap and restarted no M4 computation. Restore using the same whole-ZIP/manifest/member procedure. Its RESULTS_INDEX snapshot predates this self-indexing entry; current checked-in receipts are authoritative. No remote-byte download or final-head CI snapshot is claimed by this ZIP.

Fresh inputs were restored from the accepted merged #197 [followup ZIP](https://drive.google.com/file/d/1kMXJIUUB6YkYphhHSKsRp3Xacz_Oxno2/view) and [resource-stop ZIP](https://drive.google.com/file/d/1hNbCAU71BcYFxVBx2OSmqYtp0sXfhmcS/view), verifying whole ZIPs, embedded manifests and every selected member. [Retrieval receipt](docs/reports/hu100-export-audit-memory-artifacts/retrieval.json) pins all 18 original Drive IDs/member paths/sizes/SHA256 and local retrieval paths; it does not claim remote-byte retrieval. In the new primary ZIP these exact inputs are `results/inputs/<basename from local_path>`, including largest HU100 `hu100-11042440-checkpoint.gz` (114,633,184 bytes /SHA256 `234628a0390502f6b17f4bad3486c47c2a5aa7a297fc611e00287533ea1f4567`) and HU20 `hu20-reference-1b-checkpoint.gz` (214,220,678 bytes /SHA256 `4ee91d3977d98c0b6b462ed310396c6a7cbba48ef51716bb0fdbf6609dfbc825`). Source tar/binary identities, all before/initial/final outputs, raw 200-ms family samples, 5-second guard samples, pilots, failures and one-use continuation claims are preserved under `results/` and `bin/`.

Restore into a new ignored nonsynced directory: download the primary ZIP by its pinned Drive ID, then `shasum -a 256 DOWNLOADED_ZIP` and require the whole hash above. Run `python3 -m zipfile -e DOWNLOADED_ZIP results/retrieved/pr202/NEW_ROOT`; verify both compressed/decompressed manifest hashes and the sizes/SHA256 of every needed member against that manifest before use. Inputs additionally match the retrieval receipt. Extract each retained source tar separately; reuse frozen execution source and commands, not an automatic benchmark/training relaunch. Do not restore into another agent's root.

[Capacity estimate](docs/reports/hu100-export-audit-memory-artifacts/capacity.json) recommends a separately approved 30-minute free-M4 growth experiment toward 20M total nodes, at most 6,510,774 entries, one terminal checkpoint/audit set and unchanged guards. Training/save reserve 5.63 GiB, tools 0.62 GiB, additional disk 9.63 GiB plus retained inputs/15.5-GiB floor; fresh pilot admission required. This is advisory, limited to 2× measured HU100 entries, and gives no 1B/10B promise. Native table/save growth is the next memory bottleneck; average inference remains compact. M4 own root `~/Local/hu100-export-audit-memory-20261008` and M1 feature/baseline isolated checkouts remain intact. No cleanup, model Git addition, training or unattended work.

## PR190 global equity-bucket validation (October 8)

**Validation complete:** global K50 passes at **0.3917 BB [0.3614, 0.4234]**, versus fitted50 0.3874 and v1 0.6567. Global K200 is descriptively worse at 0.4233 [0.3795, 0.4740]. All 120 collections/120 held-out locks/18 replays qualify; every coverage cell passes and the independent raw/bootstrap audit agrees within 4.44e-16. [Report](docs/reports/hu20-global-bucket-validation.md), [compact evidence](docs/artifacts/pr190-validation-readout.json). The original swap stop, failed storage admission and interrupted output remain preserved. Owner-authorized continuation retained the original baseline/deadline and completed on October 8 at 10:39 CEST. No training or production key change.

**Execution provenance:** final frozen study attempt `continuation-01`, source revision `b3de451a01485c46d91f333b1128040e9257cb90`; `continuation-execution-source.tar.gz` in the final evidence ZIP has SHA256 `aad8556e13c9c7e9cf3c7c2abefe5f5316294967ffa72d9095d0b7081a99ea67`. Its adjacent `.json` pins launcher, admission, authorization and original budget hashes. The original failed attempt remains separate. The prepared global schedule is `prepared/manifest.json`, 134,997 bytes, SHA256 `476d0c08e043056e2073b1b858d36c1291541c16c5a81e1b908daa8f3cdcfaf6`, in the verified input ZIP. Current main was merged only after scientific completion; this does not change the source used for the frozen run.

Exact required members in `verified-inputs-prepared-pilots.zip` (verify that ZIP's pinned SHA256, then stream/extract the member and verify the row below):

| Input/member | Bytes | SHA256 |
|---|---:|---|
| `inputs/pr149/pooling-engineering-05-mac` | 1,340,544 | `fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f` |
| `inputs/pr149/prepared-03/manifest.json` | 73,912 | `037e74e7a8cd7fb713d451996c1959c35b0d8ac02410d9313bc73745d0f601cf` |
| `inputs/pr149/repo/docs/reports/hu20-board-pooling-artifacts/corpus.json` | 187,428 | `482af346e395ed27fa5ae87245ba1af4c051d8b4f0a41b0e749dc51677e4bc85` |
| `inputs/pr149/repo/docs/reports/hu20-board-pooling-artifacts/crossfit.json` | 3,822 | `ae6bd8bd9f2d049bbd8fcecaa3e41d273150479b936418881738b3c8c61daf67` |
| `inputs/tables/river-k200.bin` | 1,231,562,564 | `b0591710df149248b2feb89f0a9c808f9745be1f7e5c4eb63aa8a2d93b78864e` |
| `inputs/tables/river-k50.bin` | 1,231,562,564 | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |
| `inputs/tables/turn-k200.bin` | 139,600,524 | `ccec40009426fd64cac53816e6a286adec159eb2a22c479bb273029faa1b7f6c` |
| `inputs/tables/turn-k50.bin` | 139,600,524 | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |

**Canonical complete study:** [final evidence ZIP](https://drive.google.com/file/d/1_y9XnksKlahYcsuBqhmPht4M8l5HJszu/view), `~/Local/Research-Cloud/PR-190-HU20-bucket-validation/research-evidence-20261008.zip`: **881,002,949 bytes /1,551 verified members**, SHA256 `637533571613aaad53e67b846ba4d256fcf3a52be3ee9483213b164554217707`; `ARCHIVE-MANIFEST.json` SHA256 `3cbf9807f5253f2ef081124c2dbc911f837f1161c2b1db754f357351ce39c1cd`. Native uploaded/not-uploading/no-conflict and cloud ID/name/size/parent acceptance match; parent `1C0z2LwrqiUZgVwU_flkBwvxM0MV5LeDh`. [Complete receipt, prior archives and readout hashes](docs/artifacts/pr190-final-archive-receipt.json). No remote-byte redownload is claimed. Fitted-policy members are `run/crossfit-<evaluation-half>-<lineage>.json.gz` with their raw/gzip hashes in adjacent `.receipt.json` members. Full readout is `readout-20261008/summary.json` and independent arithmetic is `readout-20261008/independent-audit.json`, with exact member hashes in the receipt/manifest. All unique source versions, setup/archival failures, partials, input retrieval/admission and monitoring provenance are retained; unchanged previously archived members have exact locators in this ZIP's manifest.

All **258 native payload ZIPs** (120 collection, 120 lock, 18 replay; 5,181,180,575 total ZIP bytes) freshly pass archive/raw-member SHA256 checks, native upload acceptance and a matching Drive folder metadata listing. Each archive's exact Drive ID/URL, name, size, raw hash, ZIP hash and manifest hash is in `native-archive-inventory-20261008.json`, SHA256 `ab3d92b856eb03376b08aea0df2f159aa1d83d3f34a3e1249b80ea52462b6cc4`, inside the final evidence ZIP. Payload names are `native-<collect|relock|replay>-<spot-hash>-<lineage-index>.zip`; each has raw member `response.jsonl` and member-hashed `ARCHIVE-MANIFEST.json`. The inventory pins every job instead of relying on a filename pattern alone.

Restore into a fresh ignored local directory: verify the chosen ZIP SHA256 against this receipt/inventory, read `ARCHIVE-MANIFEST.json` with Python `zipfile.ZipFile(archive)`, and stream `archive.open(member)` to the destination; verify SHA256 and byte count against the member manifest before use. For gzip members, verify the encoded member first, then decompress and verify the raw hash in its adjacent receipt. Retrieve raw native ZIPs by their pinned Drive URLs, verify ZIP and `response.jsonl` hashes, and rebind absolute paths only inside the restored copy. Never overwrite/delete synced evidence. The final evidence manifest's `previous_archive_references` resolves unchanged inputs by exact archive/member/hash.

| Prior archive | Drive ID | ZIP SHA256 | Exact payload members |
|---|---|---|---|
| verified-inputs-prepared-pilots.zip | [1LM6DGFHa56yNQHVyW2_xJVKlnWdqXNx6](https://drive.google.com/file/d/1LM6DGFHa56yNQHVyW2_xJVKlnWdqXNx6/view?usp=drivesdk) | `56db3a114aaa1e5273c4445c178a6e516bdcf7eacf0e9e4477dbc5786e360e49` | `ARCHIVE-MANIFEST.json`; 1140 members |
| completed-derived-input-buffers.zip | [15qli8PyGl2HDK_G9sD0dS64tCQGDi4X3](https://drive.google.com/file/d/15qli8PyGl2HDK_G9sD0dS64tCQGDi4X3/view?usp=drivesdk) | `505ada0bdeb81dcc52089294bca9d7f080ecd109edb560d0324809e3b9e7f89f` | `ARCHIVE-MANIFEST.json`; 5 members |
| resource-stop-20261007.zip | [1ST9kX2SkhD_7KXVdFv0UTV_O2wokZSmA](https://drive.google.com/file/d/1ST9kX2SkhD_7KXVdFv0UTV_O2wokZSmA/view?usp=drivesdk) | `77ebc688432145e4e7cff06b69be715385ebabaa01e6e06afc76cf4d8373336e` | `ARCHIVE-MANIFEST.json`; 252 members |

The input/preparation/pilot ZIP retains the exact four full-deck table members `inputs/tables/turn-k50.bin`, `turn-k200.bin`, `river-k50.bin`, `river-k200.bin`; their raw hashes and retrieval provenance are in the embedded manifest and preparation receipts. Its qualified binary is `inputs/pr149/pooling-engineering-05-mac`, SHA256 `fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f`. Restored #149 frozen corpus/split, prepared requests, original reference views and baseline atomic results retain verified #149 archive provenance; absolute paths were rebound only in this working copy. The derived-buffer ZIP retains lossless gzip encodings with raw-member locators. [Late publication ZIP](https://drive.google.com/file/d/14l4tDYB6MBQimw1KoxQIXvr9ZLa3zre7/view), `publication-evidence-20261008.zip`, retains 29 additional check/acceptance/source records: **1,014,044 bytes**, SHA256 `9809c2391ed3c040c84db7d5c1cefc680175410e7561bda0623fc4db0b0cab11`; manifest SHA256 `1cfc29e364da2a2a137a9bc4999ffad51ccd3521b3c045d784055df2604e7fef`. Every member verifies and native/cloud upload acceptance matches; [terminal receipt](docs/artifacts/pr190-publication-archive-receipt.json). Use the same manifest-driven restoration procedure. Terminal receipts are also checked into this PR; local originals remain.


[Closeout ZIP](https://drive.google.com/file/d/1VIcz4Bk60-QaZPrKGZDyS0S5FvaTQvG7/view), `closeout-evidence-20261008.zip`, retains **39 members /608,353 bytes** of late publication, upstream merge/conflict, check/review and final PR-source records at `7f79f793040a24a13786b7645ec39edf40469651`. SHA256 `b0f38cb292eb7bd7d9529d783ae41a63e5d37651cb3d3d5ca39b5c4369cef2e2`; embedded `ARCHIVE-MANIFEST.json` SHA256 `d3477d75369d07a27449dca8d3f2a7de5dc2a935db1b186859e14d3fade9a997`. Member readback and separate native uploaded/not-uploading/no-conflicts plus cloud ID/name/size/parent confirmation pass; remote bytes were not redownloaded. [Terminal acceptance receipt](docs/artifacts/pr190-closeout-archive-receipt.json) also preserves the system-Python import failure before ZIP creation and the separately named successful helper using the existing qualified environment. No scientific work was affected. Restore with `unzip -n "$ARCHIVE" '<manifest member path>' 'ARCHIVE-MANIFEST.json' -d planning/pr190-closeout-restored` in a fresh ignored directory, after verifying the whole ZIP hash, then verify the selected member hash. Receipt/index-only changes after this source snapshot remain in Git; all local originals and prior immutable archives remain retained.

Stop evidence is preserved in `~/Local/Research-Cloud/PR-190-HU20-bucket-validation/resource-stop-20261007.zip`: **36,028,458 bytes /252 verified members**, SHA256 `77ebc688432145e4e7cff06b69be715385ebabaa01e6e06afc76cf4d8373336e`; member manifest `ARCHIVE-MANIFEST.json`, SHA256 `f81a1c894a33af9ec004248a111fcc6ae2b6c6f3f8bc6b9cf1b5e7511ea01319`. [Drive copy](https://drive.google.com/file/d/1ST9kX2SkhD_7KXVdFv0UTV_O2wokZSmA/view), ID `1ST9kX2SkhD_7KXVdFv0UTV_O2wokZSmA`, parent `1C0z2LwrqiUZgVwU_flkBwvxM0MV5LeDh`: current native uploaded/not-uploading/no-conflict acceptance and cloud ID/name/size/parent metadata match; [receipt](docs/artifacts/pr190-resource-stop-archive.json). No remote byte re-download is claimed. Restore selected members into a fresh ignored directory using Python `zipfile.ZipFile(archive).open(member)`, then verify their SHA256 against the embedded manifest. Completed native payloads remain separate immutable ZIPs referenced by archived `run/collect/<job>/result.json` receipts; verify each ZIP and its `response.jsonl` member before use. No source or synced file was deleted.

Preparation and pilot evidence are retained under `~/Local/Research-Cloud/PR-190-HU20-bucket-validation/`; exact input/archive hashes and restoration references follow in final closeout. The first storage admission failed and is retained. After fresh #149 merged status, native uploaded/not-uploading/no-conflict acceptance, Drive name/size/parent readback, exact archive-member/source SHA256 checks and open #188/#190 dependency review, **83 inactive retrieval02 SQLite/compact copies /3,229,764,359 logical bytes** were removed under the existing owner authorization. Measured reclamation: **3.021 GiB**; final free space at removal: **24.067 GiB**. [Every removed path, hash, archive member and dependency record](docs/artifacts/pr190-merged-pr149-cleanup.json). Durable receipt also resides at `~/Local/storage-cleanup-receipts/PR190-merged-PR149-20261007/`. Other roots, all active inputs, shared Git/source, metadata and synced files remain intact.

Restore exact removed members from [#149's canonical archive](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view), SHA256 `20ad0f67df71464e7c06bdb1cf63451c243944c14e732b83650861f8f9bbcaed`, native path `~/Local/Research-Cloud/M1-board-pooling/M4-main-06-20261005/hu20-board-pooling-m4-complete-20261005.tar.gz`. The receipt gives each `campaign/...` member and expected SHA256. Extract into a fresh ignored nonsynced directory using Python `tarfile.open(archive, "r:gz").extractfile(member)` (resolves tar hard links), then verify the member's recorded SHA256 before use. The archive and exact selected member hashes verified; no remote byte re-download is claimed. Cleanup verification, native/cloud acceptance and failed/new admission receipts will be included in PR190's final member-hashed research ZIPs.

# Research results index

## PR200 HU100 checkpoint learning curves — completed evidence, October 8, 2026

[PR200](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/200),
[protocol](docs/native-hu100-learning-curves.md), [report and exportable plots](docs/reports/native-hu100-learning-curves.md).
All five averages retained at **100,691; 1,001,382; 5,001,210; 10,001,922; 11,042,440 actual nodes**.
The last is the verified capacity stop, not its requested-1B filename. No training,
model changes, checkpoint selection or release claim. **122,880 distinct final
hands /477,296 actions**, plus 960 excluded pilot hands, all replay/reproduce.
Uniform plays once/opponent; exact rows are reused. Twenty final-minus-earlier
comparisons use Bonferroni FWER0.05: only final−100k against check_call improves,
**+75.90 [+16.79, +135.01] BB/100**; 19 inconclusive. Every checkpoint loses to
the three aggressive opponents; better coverage does not consistently improve
profit. The initial reviewed handoff left the PR unmerged; the owner subsequently
authorized merging. No further research or cleanup follows this task.

M4 isolated root `~/Local/hu100-learning-curves-20261008/`, ignored
`results/hu100-learning-curves/run-01/`. Scientific source
`51597e88706465c33dc51577890989aa303fefca`; continuation controller reviewed at
`7a23843776a98331bb604ee2f1cdbb0473dca3d5`, exact SHA256
`d7ab9e6cba5743356d54ef86ac069f1c9892368be64be1b6fb39ae7f6ae09641`.
Final root2026100820312, pilot2026100820311, 2,048blocks/opponent/policy.
An original terminal fresh-headroom refusal after four complete/reproduced
checkpoints remains immutable. A disclosed operational protocol amendment filled
only the never-started fifth panel, under the **same original absolute 30-minute
cap**, frozen count/schedule and unchanged guards; no passed science repeated.
All science closed at986.16s from the original start; primary archival verification/
copy also completed within that cap. No new scheduler or further run is scheduled.
[Combined receipts and retained failure](docs/reports/native-hu100-learning-curves-artifacts/scientific-closeout.json).

[PR200 Research-Cloud folder](https://drive.google.com/drive/folders/12azuxRTXRyEnpO-7tSb6TTOA4teHccMf),
[primary ZIP](https://drive.google.com/file/d/1aZ5HtPGEY0-qCtQkoJbbY122S6xxFISg/view),
`hu100-learning-curves-20261008.zip`: **1,224,575,891 bytes /910 verified members**,
SHA256 `ac79cc76209e95c4dc1c4a6d6903b645fdae4aeec4fd0d7b7f2cf5603ced1139`.
Embedded `ARCHIVE-MANIFEST.json.gz` decompressed SHA256
`177c4eb37a5cfe10f9d940d4a81599e6ec3e6ca560e0cae7755c42adb83cfdd9`,
compressed member SHA256
`83a3b9672b02b4bd250d5e73f5d0b37f818e1bc16e06b408f5d1524abd79d0f3`.
Every member size/SHA256 and native-copy whole hash passes. Native uploaded1/
uploading0/conflicts0 and independent connector ID/name/size/parent accepted.
Remote archive bytes **not downloaded**. [Local receipt](docs/reports/native-hu100-learning-curves-artifacts/archive-local-receipt.json)
· [upload acceptance](docs/reports/native-hu100-learning-curves-artifacts/archive-upload-acceptance.json).

All **five exact input average members, sizes/hashes/actual nodes, original
checkpoint provenance and retrieval commands**:
[model restoration index](docs/reports/native-hu100-learning-curves-artifacts/model-retrieval-index.json).
They are `research/inputs/average-{100000,1000000,5000000,10000000,11042440}.jsonl.gz`;
all snapshots also remain under `research/hu100-learning-curves/run-01/`.
The source training checkpoints restore through the unchanged [PR197 model index](docs/reports/native-recovery-hu100-artifacts/followup-model-index.json).
Input retrieval checked #197's current merged state and its accepted local synced
ZIP whole/manifest/selected-member hashes; headers independently verify actual
completed nodes. No historical originals or archives were changed.

Restore to a fresh ignored nonsynced directory: download the indexed ZIP, run
`shasum -a 256 LOCAL_DOWNLOADED_PR200_ZIP` and require the whole hash above, then
`python3 -m zipfile -e LOCAL_DOWNLOADED_PR200_ZIP results/retrieved/pr200/NEW_ROOT`.
Decompress `ARCHIVE-MANIFEST.json.gz`, verify both manifest hashes, then verify
needed member sizes/SHA256 before use. Exact source is
`research/exact-source/source-51597e8.tar`; extract it separately or check out that
pinned Git revision. Raw panel paths are
`research/hu100-learning-curves/run-01/{pilot,final,pilot-reproduction,final-reproduction}/{actual_nodes}/{opponent}/`.
Use the frozen original evaluator/auditor/reporting source, not an automatic
campaign relaunch. The immutable initial stop lives in `run-01/state.json`,
`run-01/guard/` and `run-01/closeout.json`; continuation has its own claims,
`continuation-guard/`, `continuation-complete.json` and `continuation-closeout.json`.
The exact standalone controller and review/qualification live under
`research/qualification/`. Full schedule, private streams, traces, pilot/final
stats and all source/qualification failures are retained. Originals remain and
no cleanup or synced-file eviction occurred. Later report/plots and independent
review/CI closeout are separate metadata; the primary science ZIP stays immutable.

Accepted [later metadata closeout ZIP](https://drive.google.com/file/d/1ZiaxnPine9S53FI9tOpipmxYP8sL6GIq/view),
`hu100-learning-curves-closeout-20261008.zip`: **587,007 bytes /31 verified members**,
SHA256 `ec5304e8e6d75318f673bb4717efac56b4331233eaec3e495642d9a02776a23f`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`feb80cc8acc725231a6e3602165f4004c2c2c0413242b418dac49f50cbc73133`.
[Local/native receipt](docs/reports/native-hu100-learning-curves-artifacts/closeout-archive-local-receipt.json)
· [connector/native acceptance](docs/reports/native-hu100-learning-curves-artifacts/closeout-archive-upload-acceptance.json)
· [independent evidence review](docs/reports/native-hu100-learning-curves-artifacts/evidence-review.json).
Report, plots/CSVs, review receipts, primary-upload acceptance, restoration index
and dated pre-final CI snapshot are included. This metadata-only seal/copy happened
after completed science, with AC/disk/pressure admission, and restarted no science.
The final PR head's live GitHub checks are authoritative. Restore this ZIP separately
to a fresh ignored directory; verify its whole SHA256 and embedded manifest/member
sizes/hashes as above. Neither archive nor any local original was removed.

## PR197 separately authorized HU100 playing baseline — complete, October 8, 2026

[PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197),
[protocol](docs/native-hu100-playing-baseline.md), [report](docs/reports/native-hu100-playing-baseline.md),
[compact scientific receipts/diagnostics](docs/reports/native-hu100-playing-baseline-artifacts/scientific-summary.json).
New evaluation source `009b5d92529792b0db32798407cf56c25416beb5`; free isolated M4
root `~/Local/native-recovery-hu100-20261008/results/native-hu100-playing-baseline/`.
**40,960 final hands /160,425 actions**, 2,048 duplicate blocks/opponent, paired
seat rotations, average versus the same native-menu uniform reference. All final
actions/settlements independently replayed; all hands/decisions fully reproduced.
Pilot320 hands/root2026100819711 excluded from final inference; finalroot2026100819712.
Cost-only frozen counts preceded outcomes; 142.59s science plus archive within the
fresh 30-minute cap. Whole-family10GiB/pressure/originalswapbaseline+.5GiB/disk15.5/AC
guards complete; no correctness/resource failure or retry. All four old timers
disabled. No new training, checkpoint selection, cleanup, release or merge.
Per-opponent BB/100 and block95% intervals, paired differences, street/opponent
known-positive/zero/missing rates, actions and latency are in the report/CSV.
Results are a development baseline only; no general-strength or release claim.
Existing training/recovery/capacity evidence and archives below remain unchanged.

[Existing Research-Cloud folder](https://drive.google.com/drive/folders/1D2f8JkP1oZYPmeph9AexXD5ZnSLgBqfI),
[baseline ZIP](https://drive.google.com/file/d/1QEHJJ_yPJbfHYkX1Taa3wM4RnblVPdK1/view)
`native-hu100-playing-baseline-20261008.zip`: **574,958,858 bytes /227 members**,
SHA256 `a2c6f3106bdff486ece8ed1cec53521c235fe3eaf0c9debbae8b901e5ea5421d`.
Embedded `ARCHIVE-MANIFEST.json.gz`: decompressed JSON SHA256
`3ae2ec4cd2d8b5b41fc3e1f81c5bc1df9d75840a916a243d06f2259e895aee61`,
compressed member SHA256 `82d3870d094cbdec5341250d8c3ede6cefdb5115e20ef17a5868164dcc1f63ff`.
All sizes/hashes read back locally; native uploaded1/uploading0/conflicts0 plus
independent cloud ID/name/size/parent accepted. No remote-byte download.
[Local receipt](docs/reports/native-hu100-playing-baseline-artifacts/archive-local-receipt.json),
[native receipt](docs/reports/native-hu100-playing-baseline-artifacts/archive-native-upload.json),
[cloud acceptance](docs/reports/native-hu100-playing-baseline-artifacts/archive-upload-acceptance.json).

The exact audited average snapshot is
`research/run-01/final/models/ffd53decdd4af5bffc0ae34e98144d43e27eef92a49033a7d4008615576a93be.json.gz`,
**79,195,090 bytes**, SHA256
`ffd53decdd4af5bffc0ae34e98144d43e27eef92a49033a7d4008615576a93be`.
It is byte-identical to the original final average member indexed below; native
`.json.gz` snapshot suffix does not change its streaming JSONL format. Training
checkpoint/source/audit identity remains the [15-asset model index](docs/reports/native-recovery-hu100-artifacts/followup-model-index.json).
Selected evaluation source tar: `research/exact-source/source-009b5d9.tar`,
**5,242,880 bytes**, SHA256 `06590f243cb1df7726fabf3035fb1383b2615f46c963b2fb28538dd2e8bbd9ef`.
Full source restores through Git commit `009b5d92529792b0db32798407cf56c25416beb5`;
requirements and installed engine/environment fingerprints are pinned in each manifest.
Per-opponent members: `research/run-01/final/{opponent}/` contains manifest,
schedule, explicit private-action seeds, hands, decisions, timing and report;
`research/run-01/final-audit.json` and `final-reproduction/` contain full independent
replay and deterministic reproduction. Pilot/reproduction, qualified/reviewed
source receipts, findings, claims, raw resource samples and source are retained.

Download the indexed ZIP into a **fresh ignored directory**. Run
`shasum -a 256 LOCAL_DOWNLOADED_HU100_BASELINE_ZIP` and require the whole hash above;
extract with `python3 -m zipfile -e LOCAL_DOWNLOADED_HU100_BASELINE_ZIP
results/retrieved/pr197/NEW_HU100_BASELINE_ROOT`. Run `shasum -a 256
results/retrieved/pr197/NEW_HU100_BASELINE_ROOT/research/run-01/final/models/ffd53decdd4af5bffc0ae34e98144d43e27eef92a49033a7d4008615576a93be.json.gz`
and require the model hash above. Check required member sizes/hashes against the
embedded manifest before use. Configuration still names the original retained
input path; any later explicitly authorized reproduction must place verified
bytes in that path within a fresh ignored checkout, preserving active originals.
These restoration commands do not authorize another execution or consume a retry.

Prior post-integration closeout JSON also accepted after its original copy's
premature missing-item-ID failure (no copy retry):
[native-recovery-hu100-final-integration-20261008.json](https://drive.google.com/file/d/1Em7UXm-Y5ONd2561_sR6txeYCt8Cmdbw/view),
**13,870 bytes**, SHA256 `e4a260ea3ab2205ad600314c8d170ef49bece1ceea1c79bd370bf05055c39974`,
same cloud folder; native uploaded/no pending/no conflicts and independent cloud
ID/size/parent accepted. It preserves original review11/green0efeaef CI/timers;
[acceptance and retained failure](docs/reports/native-hu100-playing-baseline-artifacts/prior-integration-upload-acceptance.json).
No prior archive was changed; all open PR roots and dependencies remain retained.

[Post-seal closeout supplement](https://drive.google.com/file/d/1pTsqGGQm-rGfDPglRW1o4c2GFDLOqmI5/view)
`native-hu100-playing-baseline-closeout-20261008.json`: **50,023 bytes /18 embedded
UTF-8 members**, SHA256 `2444c12171b99af52964c7e1aaf2535b092fceb3e043daa7fb398929cd58e503`.
Embedded `member_manifest_and_content` canonical JSON (sorted keys, separators
`,`/`:`) SHA256 `8955a409d9ecd2436fbd5488ee39c67507f6c52d99f1e7ee21c682bb8d48e413`;
every member's UTF-8 byte count/SHA256 read back locally. It retains the post-seal
archive guard/resources/logs, native/cloud acceptance, exact-source green CI,
review15, disabled timers and report/PR status snapshots. Own upload acceptance
and later documentation-head CI necessarily postdate its snapshots; final Git
receipts/index and current PR checks are authoritative. Native uploaded1/
uploading0/conflicts0 plus independent cloud ID/name/size/parent accepted; no
remote-byte download. [Local receipt](docs/reports/native-hu100-playing-baseline-artifacts/closeout-local-receipt.json),
[upload acceptance](docs/reports/native-hu100-playing-baseline-artifacts/closeout-upload-acceptance.json).
Download to a fresh ignored directory; `shasum -a 256 LOCAL_DOWNLOADED_HU100_BASELINE_CLOSEOUT_JSON`
must match the whole hash above. Read `member_manifest_and_content`: each `utf8`
string restores the named member only after its encoded UTF-8 length/SHA256 match
`bytes`/`sha256`, into a fresh destination preserving active originals.
Independent evidence review15 found no actionable P1/P2 at `b22de859a4787c91bd5c84e400da96466dd191fa`;
[review and scope limits](docs/reports/native-hu100-playing-baseline-artifacts/evidence-review.json).
[Full CI passed on executed009b5d9](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37755600562).
Required CI on the later final documentation head must pass before handback;
[current PR checks](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197/checks)
record that subsequent status. No merge.

[CI clock correction supplement](https://drive.google.com/file/d/1JgPvyX4DBy3QuebjA6Ekw2NACXUNEbYf/view)
`native-hu100-playing-baseline-ci-clock-20261008.json`: **39,804 bytes /13 embedded
UTF-8 members**, SHA256 `d58561fe00cebcbf99af7947f970e64f4aaf419bad1513958a085077005c8e3f`;
canonical `member_manifest_and_content` SHA256
`c06058675b6b1b0f1ef0a6db20d28be59d3ec605e8e5a68810d802754e35edd3`.
Every member locally read back; native uploaded1/uploading0/conflicts0 and
independent cloud ID/name/size/parent accepted, no remote-byte download. Raw
failed CI log/patch, guarded 48-test M4 qualification and independent review16
are retained. Test-only source `fa84b84c6a32caaf557cd3823b885f015bb7d524` freezes
one historical fixture's clocks; production code/guards/deadlines/data remain
unchanged from executed009b5d9. [Correction and archive acceptance](docs/reports/native-hu100-playing-baseline-artifacts/ci-clock-correction.json).
Restore like the closeout JSON above: download to a fresh ignored directory,
`shasum -a 256 LOCAL_DOWNLOADED_CI_CLOCK_JSON` must match the whole hash, then
verify each UTF-8 member's bytes/SHA256 before reconstructing it. Existing primary
and closeout archives are untouched. Snapshot CI status predates later final-head
checks; current GitHub checks are authoritative. No new science or merge.

## PR197 follow-up — recovery verified / HU100 measured capacity stop, October 8, 2026

[PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197),
[owner authorization](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197#issuecomment-6054725121),
[report](docs/reports/native-recovery-hu100.md),
[scientific receipts](docs/reports/native-recovery-hu100-artifacts/followup-scientific-closeout.json).
Final scientific source `c4058b7f14a6df85e01f8354de676091572ab1bc`; retained
HU20 training source `64318398fea423a9c43c0db8635a3724e05bd55f`, same binary.
One full comparison verified every IEEE state/current/average probability and
bounded recovery metadata. **HU100 all four pilot sets and atomic capacity stop
fully audited: 11,042,440 nodes /3,255,387 entries. No 1B milestone reached.**
Native entry ceiling followed measured 2× export/audit headroom; no hard resource
guard fired, no retry. Original failed campaign below remains unchanged.
M4 ignored root `results/native-recovery-hu100/followup-01`, separate terminal
`followup-state.json`. Qualification110 Python/7Rust/build/artifact, independent
source review9 and full CI passed; evidence review10 found no blocking P1/P2
findings at documentation source `288b3f104dd61d030dfd75b65f632fdf3a1f610a`.
[Review](docs/reports/native-recovery-hu100-artifacts/followup-evidence-review.json).

[Existing Research-Cloud folder](https://drive.google.com/drive/folders/1D2f8JkP1oZYPmeph9AexXD5ZnSLgBqfI),
[follow-up ZIP](https://drive.google.com/file/d/1kMXJIUUB6YkYphhHSKsRp3Xacz_Oxno2/view)
`native-recovery-hu100-followup-20261008.zip`: **722,130,867 bytes /164 members**,
SHA256 `cd2e3c197ec2adbf8e161b1aaca39eccff017fc9aaf4ff0cfe667122e186b16f`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`84fadd58a44f8dcdbfcf157beb262797ead77767c544324d14589f151c4abfe4`. All member sizes/hashes read back
locally; native uploaded1/uploading0/conflicts0 and independent cloud ID/name/
size/parent accepted. Remote bytes not downloaded.
[Receipt](docs/reports/native-recovery-hu100-artifacts/followup-archive-receipt.json).

All **15 checkpoint/current/average member paths, sizes/SHA256 and audit status**
are in [model index](docs/reports/native-recovery-hu100-artifacts/followup-model-index.json).
Pilot checkpoints: `research/followup-01/pilot-01/training/HU100-2026100601-{nodes}.json.gz`
(nodes100000/1000000/5000000/10000000). Capacity checkpoint:
`research/followup-01/growth-01/training/HU100-2026100601-1000000000.json.gz`,
**114,633,184 bytes**, SHA256
`234628a0390502f6b17f4bad3486c47c2a5aa7a297fc611e00287533ea1f4567`;
filename/requested1B is incomplete, **actual11,042,440 nodes** verified.
Final exports/audit: `research/followup-01/final-capacity-audit/`.
Exact source tar `research/followup-01/worker-source.tar`, binary `worker/hu20-trainer`;
all logs, continuous samples, setup/operator failures, reviews, qualifications,
claims/CLOSEOUT and measured capacity receipts retained.
HU20 input/model dependencies restore from the original accepted PR197 ZIP below;
exact member hashes also appear in `research/followup-01/retrieval-and-archive-provenance.json`.
Two archival metadata fields are superseded by the
[provenance correction receipt](docs/reports/native-recovery-hu100-artifacts/followup-provenance-corrections.json).

Download into a **fresh ignored directory**; run `shasum -a 256 LOCAL_DOWNLOADED_FOLLOWUP_ZIP`
and require the whole hash above. Extract with `python3 -m zipfile -e
LOCAL_DOWNLOADED_FOLLOWUP_ZIP results/retrieved/pr197/NEW_FOLLOWUP_ROOT`; verify
required member size/SHA256 against the embedded manifest/model index before use.
Original HU20/historical #182 restoration below remains unchanged. No model enters
Git. All science exited; timers disabled; no cleanup/merge/release/arena/paid
compute.

[Follow-up closeout ZIP](https://drive.google.com/file/d/16Cye_hUW6UyEbDpk112C0NC1O20mm_GC/view)
`native-recovery-hu100-followup-closeout-20261008.zip`: **330,439 bytes /31 members**,
SHA256 `2921af44703c8c1fe8ea03278b7011f73ded1c99305c221f6753a626c7673ccc`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`be42cb352154a81c2bde068118d591fa0de84550a696f5bd68c2c8f8aa3c7f6b`.
Every member read back locally; native uploaded1/uploading0/conflicts0 plus
independent cloud ID/name/size/parent accepted; no remote-byte download.
[Receipt](docs/reports/native-recovery-hu100-artifacts/followup-closeout-archive-receipt.json).
Members include `closeout/followup-evidence-review-round10.json`,
`closeout/followup-timers-final.json`, `closeout/worker-closeout.json`,
`primary-archive/followup-provenance-corrections.json` and primary archive
guard/upload metadata. Report/index/PR snapshots predate this ZIP own acceptance;
final Git receipt/index supersede them. Retrieve to a fresh ignored directory:
`shasum -a 256 LOCAL_DOWNLOADED_FOLLOWUP_CLOSEOUT_ZIP`, require the whole hash above,
then `python3 -m zipfile -e LOCAL_DOWNLOADED_FOLLOWUP_CLOSEOUT_ZIP
results/retrieved/pr197/NEW_FOLLOWUP_CLOSEOUT_ROOT`; verify needed member
size/SHA256 against the embedded manifest. No original archive was replaced.

## PR197 original campaign — native recovery / HU100 resource stop, October 8, 2026

[PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197),
[protocol](docs/native-recovery-hu100.md), [result](docs/reports/native-recovery-hu100.md).
Scientific source `64318398fea423a9c43c0db8635a3724e05bd55f`; isolated M4 checkout
`~/Local/native-recovery-hu100-20261008`, ignored campaign root
`results/native-recovery-hu100/`. Qualified build/7 Rust tests/81 Python fixtures/
artifact guard. **Reference 1B current/average fully verified and pinned; recovery
saved/exported to same endpoint, equivalence interrupted by RSS guard; HU100 NOT RUN.**
Comparator detected 5.555-GiB aggregate RSS; no retry/raised limit. All failures and
completed but unaudited files retained. No equivalence or HU100 growth claim.

[PR197 Research-Cloud folder](https://drive.google.com/drive/folders/1D2f8JkP1oZYPmeph9AexXD5ZnSLgBqfI),
[science ZIP](https://drive.google.com/file/d/1hNbCAU71BcYFxVBx2OSmqYtp0sXfhmcS/view)
`native-recovery-hu100-resource-stop-20261008.zip`: **1,380,462,715 bytes /108 members**,
SHA256 `4b4cc0e890756ee076a31737a3f5abc8af63d0f2a9065f28c3dc83baad38f88f`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`b5e506be66fb68700be6332155bc726835e2dc42e19fa4b746165477327aa3d5`.
Every member size/hash verified locally. Native uploaded=1/uploading=0/no conflicts;
independent cloud ID/name/size/parent confirms acceptance. No remote-byte download.
[Receipt](docs/reports/native-recovery-hu100-artifacts/archive-receipt.json).
Exact source tar `research/worker-source.tar`, binary `worker/hu20-trainer`;
all plans/qualifications/logs/resources/failures/retrieval input retained.

Restore into a fresh ignored directory after downloading the indexed ZIP and
verifying its whole hash: `python3 -m zipfile -e LOCAL_DOWNLOADED_ZIP
results/retrieved/pr197/NEW_ROOT`; check every embedded member size/SHA256 before
use. Required models are `research/reference-02/training/HU20-2026100601-{nodes}.json.gz`
(nodes 100000000, 500000000, 1000000000) and
`research/recovery-02/training/HU20-2026100601-1000000000.json.gz`;
exact sizes/hashes/status in [milestone summary](docs/reports/native-recovery-hu100-artifacts/milestone-summary.json).
Exports: `research/{reference-02,recovery-02}/{current.json.gz,average.jsonl.gz}`;
all byte/hash pairs in the embedded manifest. Only reference 1B has full export audit.

Historical input `research/inputs/historical-500M.json.gz`: **171,794,336 bytes**,
SHA256 `a5318cd586c0b170a68334e4236111faddabaf7f686c071958757db888afab47`.
Original retrieval checked #182 MERGED, verified Drive archive ID
`1V2bbJ9kf0_MTdqwfcoo__XnCMWjAEkbi` whole SHA256
`1c71bdc9373993be071c2c231e2e2cfa24be4d506a864ffc3e86eed3bb15e16f`,
manifest `356afb24d65857560e88da09d9ba9b8cc08eee0fbf36c7830f5ec72ab77e2db1`,
member `research/inputs/O-2026100601-500000000.json.gz`.
Command: `.venv/bin/python results/native-recovery-hu100/retrieve-parent.py` on M4;
receipt now archived at `research/inputs/retrieval.json`. No cleaned #182 original
or active other-agent dependency was changed. Independent final review round 5 found no blocking evidence findings.
[Closeout supplement](https://drive.google.com/file/d/14MOuquGcYRPyeI5PUDW2Tdbmg8bPPxR9/view),
`native-recovery-hu100-closeout-20261008.zip`: **99,275 bytes /24 verified members**,
SHA256 `41534b63cb6f1aecaab0e1c9d0082f5b3c5e353c9465c49e99bd33714138b7f6`,
embedded `ARCHIVE-MANIFEST.json` SHA256
`d6c5f21ba8d1854962e6c4dbad44d76f825bd3aa8d3f520c5109e99f6522f156`.
It contains derived summaries, review/timer receipts, main archive member index,
upload evidence, archival guard records and report/index snapshots. All local
member size/hash checks passed; native upload complete/no conflicts and independent
cloud ID/name/size/parent acceptance match. No remote-byte readback.
Restore with `python3 -m zipfile -e LOCAL_DOWNLOADED_CLOSEOUT_ZIP
results/retrieved/pr197/NEW_CLOSEOUT_ROOT`, after whole hash verification; verify
embedded members. [Supplement receipt](docs/reports/native-recovery-hu100-artifacts/closeout-archive-receipt.json).
Closeout completed at 08:35 Madrid; all owned processes exited and all four
campaign timers disabled. All open PR197 originals retained; no cleanup applied.
## v0.4.2 published stable Latest — October 8, 2026

[Release PR #198](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/198), branch `feature/v042-release`; [readiness and publication sequence](docs/releases/v0.4.2/READINESS.md). The owner authorized the release PR, reviewed green-check merge, exact-source tag and stable Latest. No new research, retraining, extraction or selection. The historical #188 unpublished package below remains unchanged.

Input retrieval on M4 uses the already-synced [canonical #188 ZIP](https://drive.google.com/file/d/1ismlfD-LKQfFqAAR3O5LN_6UVmebHChH/view), `~/Local/Research-Cloud/PR-188-HU20-O-10B-LBR/hu20-v042-lbr-complete-20261008.zip`. Whole SHA256 `7a36e20f5e4560ec98d14ecaa4a6bb1fe77a7ecf1a5aa710cd1e03e586b45a80`, embedded `ARCHIVE-MANIFEST.json` SHA256 `f5354027221c0750b086c3ff2553f6acdaded38b218a37963922f5e42544b702`. After confirming #188 merged, only the six `research/package/` members were copied into ignored nonsynced `~/Local/v042-release-20261008/source/results/v042-release/preparation/`, validating member sizes/SHA256 and the original standalone verifier against source `1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8`. [Retrieval receipt](docs/releases/v0.4.2/verification/retrieval-receipt.json). Remote archive bytes were not downloaded; this is fresh local archive/member verification of the accepted canonical input.

To reproduce retrieval on M4, verify the indexed ZIP and embedded manifest hashes, then run `unzip -n "$ARCHIVE" 'research/package/*' 'ARCHIVE-MANIFEST.json' -d results/retrieved/v042-release` in a fresh ignored directory and verify every selected member against that manifest. Run the extracted `verify_v042_bundle.py` with `--expect-source 1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8`. Never use `research/package-attempt-01/`. Model member `research/package/O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz`, **249,237,403 bytes**, SHA256 `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`.

The [M4 deterministic smoke](docs/releases/v0.4.2/verification/smoke-summary.json) and [resource/source receipts](docs/releases/v0.4.2/READINESS.md#verified-integration) pass. Release working files and runtime journals remain in the fresh M4 root above and ignored M1 `planning/v042-release/` / `results/v042-release/`. The final seven assets are published as stable Latest at [v0.4.2](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.2), release ID 406532927, tagged source `a53f167fa736a481339034d1a7235202d599418c`. [Publication record, exact asset hashes and retrieval verification](docs/releases/v0.4.2/PUBLICATION.md). Publication manifest SHA256 `0257d1c3d724a0b3a81747a4868c1f236858b4fcc2c4d234797988daa618d82c` binds that exact source and preserves #188 preparation provenance. Fresh authenticated draft and unauthenticated public seven-file downloads independently verify on M4; the supported runtime loads the freshly published default and independently audits every smoke action/settlement. Latest and all older public asset URLs verify. Retrieve all seven files with `gh release download v0.4.2 --dir results/retrieved/v042-publication`, then on M4 run the downloaded `verify_v042_bundle.py` with `--expect-source a53f167fa736a481339034d1a7235202d599418c --require-publication`. Use a fresh ignored nonsynced directory. Raw receipts/journals remain in the indexed own release roots, including `merged-source/results/v042-release/public-download-01/` and `public-runtime-smoke-01/`; no heavy M4 release job remains. Existing v0.4.0/v0.4.1 release assets remain pinned and available. No evidence deletion, archive eviction, other PR root changes or post-publication cleanup is part of this task.


## PR196 — native HU100 engineering preparation, October 7, 2026

[PR196](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/196),
[protocol](docs/native-hu100-preparation.md), [validation](docs/reports/native-hu100-preparation.md).
Preparation only: no campaign checkpoint/export, resource pilot, arena/benchmark,
large artifact download or research archive was produced. Transient deterministic
fixtures remain outside Git. Future campaigns require an owning PR and the
full Research-Cloud archive/member/upload/retrieval receipts below; there is no
new restoration link or deletion authorization from this preparation.

## M1 storage vacuum and upload preference — October 7, 2026

Removed **1,185 merged PR166/PR162 research copies /7.149 GB logical** and **2,230
inactive download-cache files /0.439 GB allocated**: **7.573 GB measured reclaimed**,
**31.022 GB free** at cleanup while PR190 continues writing. Open #188/#190 roots
and dependencies, all PR185 roots, primary board-pooling inputs, shared Git, active
source/helpers/caches, credentials, personal files and synced archives remain.

[Selection and restoration](docs/artifacts/m1-vacuum-20261007.md),
[full per-path receipt/restore helper](https://drive.google.com/file/d/1rl31gm8Zem_dHI_gJL-ek9pQyFp9YqdO/view)
(467,969 bytes; SHA256 `55fd3febc6eacb0373a26acd1e708c41ed055c735681473d51dd3d437f30697c`).
The owner accepts confirmed Drive uploads for cleanup without downloading archives
or repeating hash audits. Existing manifests and original path/size/modification
history were used; changed/uncertain files remain. Exact canonical Drive IDs,
members and existing hashes are in the receipt. Run its
`restore.py --original ORIGINAL_ABSOLUTE_PATH --out results/retrieved/NEW_FILE`.
No payload download/repeated hash audit, synced deletion/eviction or unattended work.
Earlier retention claims are superseded only for the exact receipted paths.

## M4 storage vacuum — October 7, 2026

Owner-authorized removal of **279 merged PR162 duplicate inputs /4.807 GB logical**
and **4,244 inactive Chrome/Homebrew/Java cache files /2.157 GB allocated** reclaimed
**6.951 GB measured**; free space rose from **64.467 to 71.419 GB**. Open #188/#190
roots/dependencies, all PR185 files, primary board-pooling inputs, shared Git/source,
Codex runtime cache, personal files and synced archives remain protected.

[Verification and restoration](docs/artifacts/m4-vacuum-20261007.md),
[full per-path receipt and restore helper](https://drive.google.com/file/d/1YXuVlKUvNrTqWBtWIEPd_QEVWCDbBTzG/view)
(495,976 bytes; SHA256 `556b679e391026da60c82f5669f44fb1c84e89a66f5b62983d95c1a3c7cc3b79`).
The [PR162 canonical archive](https://drive.google.com/file/d/1_6dRapLReZ9-sAMPaePNX-nehXowGwki/view)
and all 2,809 members freshly verify; current native/cloud upload acceptance agrees.
Each removed research path records its exact archive member/size/SHA256 in the receipt;
run its `restore.py --original ORIGINAL_ABSOLUTE_PATH --out results/retrieved/NEW_FILE`.
Sample restoration passed. No remote-byte download, synced deletion, eviction or
unattended cleanup. Earlier retained-original statements are superseded only for
these exact receipted paths.

New entries follow the [shared storage and archive-receipt contract](docs/artifact-storage.md).
The [tracked-payload audit](docs/reports/necessary-cleaning.md) identifies retained Git evidence;
its retention list is not cloud verification or deletion approval.

## PR188 — fresh v0.4.2 LBR confirmation passes narrowly

[Predeclaration](docs/hu20-v042-lbr-confirmation.md), [PR188](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188). M4 work root `~/Local/hu20-o-10b-lbr-20261007/`; light M1 checkout `/Users/dberweger/Local/v042-lbr-confirmation/`. All six #185 exports match their original manifest hashes; **34,826,546 keys** pass current-main `entries`/`visits`/`zero_mass` direct-JSON exactness, zero mismatches. [Per-model receipts and scientific-source hashes](docs/reports/hu20-v042-lbr-artifacts/). Only #186 compact storage differs from the unchanged #176/#185 scientific pipeline; all play/reporter/auditor files are pinned.

Excluded pilot root **202610077101**, final **202610077201**; no #185 or pilot hands pooled. Fresh **24,576 LBR blocks/model /294,912 LBR hands** give **−1.091512 [−4.965223, +2.782198] BB/100**, lower >−5 passes by 0.034777; half-width 3.873710 meets ≤5. All four v0.4.2 candidate checks pass when combined with #185. Final 299,520 plus excluded pilot 10,752 hands /**1,456,557 actions and settlements** independently replay. Zero incomplete LBR decisions. [Report](docs/reports/hu20-v042-lbr-confirmation.md), [independent audit](docs/reports/hu20-v042-lbr-artifacts/final-audit.json), [summary](docs/reports/hu20-v042-lbr-artifacts/final-summary.json).

Final completed **October 8 01:57:25 Madrid**, in 7h49m52s within the owner-approved 12-hour cap; original eight-hour admission stop and before-hands cache failure retained. At most three M4 workers, peak 1.024323 GiB, 3-GiB guard and ≥15-GiB disk floor passed. Scientific snapshot `0621302417589fc3823765e939e8ad05ff6d8391`; final bundle hash `8d9fd3ff6f17e602f495fdebaccf8eb156a6a02300312e0a102720c56c611d99`. Frozen science excludes all later main changes.

**Complete science/package archive:** [hu20-v042-lbr-complete-20261008.zip](https://drive.google.com/file/d/1ismlfD-LKQfFqAAR3O5LN_6UVmebHChH/view?usp=drivesdk) in [PR-188-HU20-O-10B-LBR](https://drive.google.com/drive/folders/1mv9v1VV2-Szfn8jNAjvpU4oWvRdqIkY2), synced native `~/Local/Research-Cloud/PR-188-HU20-O-10B-LBR/` on both Macs. **1,982,165,922 ZIP bytes /1,996 source members plus embedded manifest, all 1,997 readbacks verified**. Whole SHA256 `7a36e20f5e4560ec98d14ecaa4a6bb1fe77a7ecf1a5aa710cd1e03e586b45a80`; embedded `ARCHIVE-MANIFEST.json` SHA256 `f5354027221c0750b086c3ff2553f6acdaded38b218a37963922f5e42544b702`. [Archive receipt](docs/reports/hu20-v042-lbr-artifacts/archive-receipt.json), [current native upload acceptance](docs/reports/hu20-v042-lbr-artifacts/archive-native-upload.json), [independent cloud ID/name/size/parent](docs/reports/hu20-v042-lbr-artifacts/archive-cloud.json). Native uploaded=true, uploading=false, conflicts=false, exact bytes. Cloud metadata independently agrees. **Remote archive bytes were not downloaded**; full local archive/member readback and upload acceptance are distinct evidence.

Contains `research/pilot/` and `research/final/` raw play, plans, reports, independent audits, resource/launch logs; all frozen/current source and environment snapshots; freshness searches, control scripts, original eight-hour admission stop and before-hands cache failure; M1 coordination receipts; six exact model inputs. All input/checkpoint origin manifests and restoration dependencies remain pinned in [preflight](docs/reports/hu20-v042-lbr-artifacts/preflight.json) and #185's indexed archives. Later main changes never enter the frozen scientific source. All originals retained; no deletion, eviction or other PR root changes.

**Unpublished package:** canonical members `research/package/`, fixed first seed **2026100601** exact 10B export named `O10B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz`, **249,237,403 bytes**, SHA256 `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`. Five checksummed assets plus SHA256SUMS verify against preparation commit `1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8`; [receipt](docs/reports/hu20-v042-lbr-artifacts/package-verification.json), [readiness](docs/releases/v0.4.2/READINESS.md), [card](docs/releases/v0.4.2/MODEL_CARD.md), [notes](docs/releases/v0.4.2/RELEASE_NOTES.md). Publication approval false, approved release source null; no release/tag/default change. `research/package-attempt-01/` is retained incorrect-source-binding evidence, never the package to use. System-Python rehash failure is retained under `research/coordination/closeout-failures.json`; corrected pinned-Python closeout rechecks all 403 scientific files.

Retrieve the linked ZIP into a fresh ignored nonsynced directory on M4. Do not overwrite an active input. Verify its whole hash before extraction, then every extracted member's bytes/SHA256 against the embedded manifest:

```sh
mkdir -p results/retrieved/pr188
# Download the indexed ZIP into this directory, or use the already-synced ZIP.
shasum -a 256 results/retrieved/pr188/hu20-v042-lbr-complete-20261008.zip
unzip -n results/retrieved/pr188/hu20-v042-lbr-complete-20261008.zip 'inputs/policies/*' 'research/package/*' 'ARCHIVE-MANIFEST.json' -d results/retrieved/pr188/extracted
python3.11 results/retrieved/pr188/extracted/research/package/verify_v042_bundle.py results/retrieved/pr188/extracted/research/package --expect-source 1d80bd02c2e8acf2dc99b75f6e0349bb55a941d8
```

[All six model members, exact sizes/hashes and per-member retrieval commands](docs/reports/hu20-v042-lbr-artifacts/archive-models.json). Members under `inputs/policies/`:

| Seed | 1B member (bytes, SHA256) | 10B member (bytes, SHA256) |
| --- | --- | --- |
| 2026100601 | `O-2026100601-1000000000.average.jsonl.gz` (142,677,367, `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`) | `O-2026100601-10000000000.average.jsonl.gz` (249,237,403, `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`) |
| 2026100602 | `O-2026100602-1000000000.average.jsonl.gz` (143,137,499, `5a994484e4cbed5146c5f0d627ee12d0f22e8fc41ebb1f3628c80b112ba8d59b`) | `O-2026100602-10000000000.average.jsonl.gz` (251,732,642, `52de62633d8a56e17019be7b8db6b5e744043a40667e2352171f67119ac6f2ce`) |
| 2026100603 | `O-2026100603-1000000000.average.jsonl.gz` (142,318,596, `52794e8b2ce1a1c660cf9cf3514baf56256368cc7b57acbe12405349fb9d42f0`) | `O-2026100603-10000000000.average.jsonl.gz` (251,815,844, `b0e514d0f46e52de32fa4e183bd5b88ed259cf147f4af661f4cba034775a3b4f`) |

**Report/receipt supplement:** [hu20-v042-lbr-report-closeout-20261008.zip](https://drive.google.com/file/d/1UKhQLDGeKK_LykrioiNNLIG-VFfRAyWs/view?usp=drivesdk), **157,065 bytes /49 source members plus manifest, all 50 readbacks verify**, SHA256 `1580762a3f42bce05ee3bea0f086dfe9ca0b7d7b7754a52c400276de2de102f5`; embedded `ARCHIVE-MANIFEST.json` SHA256 `87ae60f877a0e390198315c1f20a39d4d21a3cfdb2d72a4e952bb865ce2a4fa0`. Same PR188 folder. Contains the updated report/index snapshot, protocol, package source/tests/card/notes/readiness/manifest/checksums, scientific summaries, first archive acceptance and coordination receipts. All 49 source hashes also match the completed M1→M4 transfer. [Receipt](docs/reports/hu20-v042-lbr-artifacts/report-closeout-archive.json), [native](docs/reports/hu20-v042-lbr-artifacts/report-closeout-native-upload.json), [cloud](docs/reports/hu20-v042-lbr-artifacts/report-closeout-cloud.json) accept exact ID/name/size/parent, uploaded/no pending/no conflicts. Remote bytes not downloaded. Extract `closeout/` into a fresh ignored path and verify embedded-manifest member hashes; models come from the complete science/package ZIP above.

[Terminal CI/review/merge receipts](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188#issuecomment-6052959257) are appended after green-check merge and sealed separately in the same folder. This dated science/package record does not claim a merge before it occurs.

## PR193 — local bot-vs-bot spectator validation

[Guide](docs/spectator.md), [compact verification](docs/reports/web-spectator-verification.json). All 25 retained spectator hands /141 decisions and 12 human hands independently replay through the pinned loaders; 102 focused tests pass. Reversed and same-model pairings, pause, step, reload and historical observations are checked. These are interface checks, not a strength comparison.

Retain the active worktree `/Users/dberweger/Local/deepcfr-web-spectator/` and its ignored `results/spectator-browser/` journals, screenshots, browser reports (including the interrupted first attempt), independent audits and the original `evidence-manifest.json` plus `integration-evidence-manifest.json`. The manifest records member sizes/SHA256, excluding the local access token and browser profiles. The owned server and isolated smoke browsers are stopped. No archival, upload acceptance or original removal is claimed; at requested cleanup, check the PR status and use `~/Local/Research-Cloud/PR-193-web-spectator/` within the [designated project folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s).

Models in ignored `models/` are the existing release downloads: v0.4.0 `B100M-HU20-current-seed-2026093001.json.gz`, SHA256 `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`; v0.4.1 `O1B-HU20-opponent-sampled-average-seed-2026100601.jsonl.gz`, SHA256 `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`. Retrieve with the exact `gh release download` commands in the [readme](readme.md#install-and-play), then run the two model verifiers; the checked-in pinned manifests record their published release URLs and manifest hashes. No model binaries enter Git.

## PR185 — HU20 O at 10B, audited direct gain; LBR safeguard inconclusive

**Final status (October 7):** matched-seed three-lineage 10B−1B **+3.50 [1.63, 5.37] BB/100**, better for every seed. LBR **+2.79 [−6.25, 11.83]** fails to establish lower >−5; no package, model promotion or release. All 1,022,976 final +60,672 excluded pilot hands /5,959,763 actions and 160 native metrics independently verify. [Report](docs/reports/hu20-o-10b.md), [owner packet](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6031857945), [audit/review proof](docs/reports/hu20-o-10b-artifacts/confirmation-review-validation.json). Science finished at 05:05 Madrid inside the nine-hour cap; posting timeout at 05:15 after owner-reported hotspot disconnection is preserved. Native recurring checks disabled, shell monitor terminal. No scientific restart or further samples.

**Final M4 archive:** [hu20-o-10b-confirmation-retry-02-final-9h-20261007.zip](https://drive.google.com/file/d/1o75pBJpeTsynTzMc0ZDQEDPeVk88Yopt/view?usp=drivesdk) in [PR185 Research-Cloud](https://drive.google.com/drive/folders/1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL), native `~/Local/Research-Cloud/PR-185-HU20-O-10B/`: **610 member-readback-verified files /1,764,022,787 ZIP bytes**; SHA256 `30044d57b7daf4b8eb2669b3320a54ac5e087c8161f8bebb47706e9ab795e1f6`, manifest `d1452a5b265eda1807e0dbccf0fca05421b692379c18e8704b85856ad9a09231`. [Receipt](docs/reports/hu20-o-10b-artifacts/confirmation-final-archive.json), [native uploaded/no uploading/no conflicts](docs/reports/hu20-o-10b-artifacts/confirmation-final-native-upload.json), [cloud ID/name/size/parent](docs/reports/hu20-o-10b-artifacts/confirmation-final-cloud.json). Contains complete final/excluded-pilot raw traces, independent audits, all 40 native boards/160 metrics, resources, source archive/patch, model exports or exact references, controls, owner packet, original STOP/outbox and archival provenance. No remote-byte re-download claimed; originals and synced files retained.

Retrieve the pinned ZIP using Drive ID `1o75pBJpeTsynTzMc0ZDQEDPeVk88Yopt` into a fresh ignored nonsynced directory; verify whole SHA256 above, then `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-confirmation-retry-02-final-9h-20261007.zip results/retrieved/pr185/final`. Verify every `ARCHIVE-MANIFEST.json` member before replay. Member `confirmation/owner-review-packet.json` SHA256 `56ab5846cb33897f85aa54a09d04271909907b17e24596ac8324dcff9dbe3c6b`; direct traces `confirmation/final/direct/run/direct-lineage-<1..6>.hands.jsonl.gz` and arena traces `confirmation/final/arena/run/<O|R>-<seed>.hands.jsonl.gz` have exact hashes in the embedded manifest and tracked independent audits. Restore unchanged models/prepared inputs through `confirmation/FINAL-RESTORE-REFERENCES.json`, native inputs via `confirmation/native-input-restoration.json`, and scientific source via `confirmation/source-523e4a34.tar` plus exact resource patch. Release model binaries remain outside Git.

**Terminal M1 support archive:** [hu20-o-10b-agent-monitor-closeout-20261007.zip](https://drive.google.com/file/d/1IQp-ceEUpFldwnDKxuyLdeVXunltTkRq/view?usp=drivesdk), Drive ID `1IQp-ceEUpFldwnDKxuyLdeVXunltTkRq`, **126 members /182,480 ZIP bytes**, SHA256 `e04c3e21d5791c76a30dfae4d2a2274be9cda7cde361905239f75b685dd551b6`, manifest `3ed6bbec56cef597bf30af8bdbd6ed3ee089fbe1f2737033f2f7be338a1b22a6`. [Receipt](docs/reports/hu20-o-10b-artifacts/confirmation-agent-monitor-archive.json), [native acceptance](docs/reports/hu20-o-10b-artifacts/confirmation-agent-monitor-native-upload.json), [cloud metadata](docs/reports/hu20-o-10b-artifacts/confirmation-agent-monitor-cloud.json). Members below `coordination/` retain monitor snapshots/alerts, SSH failure/bridge logs, manual support recovery, owner handoff, recurring-task prompt/ID/disabled receipt, AI checks, verified metadata retrieval and manually posted packet receipt. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-agent-monitor-closeout-20261007.zip results/retrieved/pr185/monitor-final`; verify its embedded manifest. Both support processes have terminated. No deletion, eviction, ongoing run or scheduled follow-on research.

**Final publication/acceptance snapshot:** [hu20-o-10b-final-report-closeout-20261007.zip](https://drive.google.com/file/d/1D8jaVpwITDOyJe0xHqp3rnO6189gUvO2/view?usp=drivesdk), **101 verified members /401,876 ZIP bytes**, SHA256 `8e596aff209c1284245e5ee360bf7f9367436460612bde021e25808ec95ef7c9`, manifest `f18af94579ef7d23aa4dec0e4d22ca1f131f7bb43b092b8f25d2c6c828682538`. [Receipt](docs/reports/hu20-o-10b-artifacts/confirmation-report-archive.json), [native](docs/reports/hu20-o-10b-artifacts/confirmation-report-native-upload.json), [cloud](docs/reports/hu20-o-10b-artifacts/confirmation-report-cloud.json). Retains dated final report/protocol/roadmap/index, chart, audit/model/plan metadata, Drive acceptance, manual packet posting/validation and corrected handoff. Retrieve by Drive ID `1D8jaVpwITDOyJe0xHqp3rnO6189gUvO2`, verify ZIP/member hashes, then `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-final-report-closeout-20261007.zip results/retrieved/pr185/report-final`. Members use `tracked/` and `closeout/` prefixes. Snapshot predates its own index entry and final-head CI. The archived local-validation test-count typo (10) is corrected to the actual 12 passing tests in [final validation](docs/reports/hu20-o-10b-artifacts/confirmation-local-validation.json); no test was skipped. Later CI/review/merge receipts remain in the same archive folder. Originals retained.

### Earlier PR185 snapshots (superseded for current status)

**Approved final freeze archive:** same PR185 Research-Cloud folder, `hu20-o-10b-final-9h-freeze-20261006.zip`, **24 verified members /114,436 ZIP bytes**, SHA256 `c7285c922ba0fcaca25ec369b8fd3369a639e5828e95d5935e3a171ad0fdfce8`, manifest `898ebb941ed8817b685713582e469896ff7f7c2416e70ae6174c71803be53f12`. [Receipt](docs/reports/hu20-o-10b-artifacts/confirmation-final-freeze-archive.json). Exact owner instruction, frozen plans/hash, posted-before-play receipt, reverified inputs/source, admitted control scripts, coordination validation and dated tracked-doc snapshots retained. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-final-9h-freeze-20261006.zip results/retrieved/pr185/final-9h-freeze` and verify every embedded-manifest member. Upload acceptance pending; originals retained. Current final play/audit traces will be sealed separately as `hu20-o-10b-confirmation-retry-02-final-9h-20261007.zip`; future completion is not claimed.

**T3 Nightly migration handoff:** owner requested app update to enable recurring agent checks. Durable M1 handoff `~/Local/hu20-o-10b-20261006/T3-NIGHTLY-HANDOFF.md` preserves the running final worker, bridge, shell monitor, frozen experiment and remaining closeout. Static handoff/pre-update health/UI evidence sealed in the same PR185 Research-Cloud folder: `hu20-o-10b-t3-nightly-handoff-20261007.zip`, **5 verified members /9,109 bytes**, SHA256 `21f1f83b4c74c177149f30b10e018beabbc657e80fdacf182ff4839b20772f40`, manifest `78b9ea49cd6e8ed54b2fe94608bcbfb9ace2c6fa7284baf16b54c5a81bda3d49`. [Receipt](docs/reports/hu20-o-10b-artifacts/t3-nightly-handoff-archive.json). Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-t3-nightly-handoff-20261007.zip results/retrieved/pr185/t3-handoff` and verify embedded member hashes. This archive records readiness to download, not successful installation. Upload acceptance pending; originals retained. Detached M4/M1 processes verified alive before update; agent should verify after migration.

**Owner-requested 30-minute health monitor:** [single updating PR comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6026108250); light M1 root `~/Local/hu20-o-10b-20261006/monitor-30m/`, read-only M4 health/plan checks until terminal status or nine-hour deadline. No scheduled AI wake-up, outcome inspection or scientific changes. Static setup/verified first snapshot ZIP in the same PR185 Research-Cloud folder: `hu20-o-10b-monitor-30m-setup-20261006.zip`, **9 verified members /8616 bytes**, SHA256 `a356dda42ff5583b4da71f6f152cc8797bcc89b0ecda57fd5c28031f4aff828b`, manifest `6cfd4b1df580da69e7109049ad549f847d0c595815cb5887e4687c12279a68ff`. [Receipt/start](docs/reports/hu20-o-10b-artifacts/monitor-30m-start.json). Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-monitor-30m-setup-20261006.zip results/retrieved/pr185/monitor-setup`, verify the embedded member manifest before use. Upload acceptance pending; originals retained. Active later snapshots/failures remain in the monitor root pending terminal archival; no future monitoring completion claimed.

**Owner-approved final frozen and running:** [approval](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6025914457), [counts/time/hash posted before play](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6025946569). Exact blind-sized design retained; only final admission/cap becomes **9 hours /32,400 seconds**. Arena root 202610066801, direct 202610067001; LBR 4,608, pressure 12,288, eleven others 256, all six direct pairs 65,536. **1,022,976 final hands**; excluded pilot 60,672; all 40 native boards and full independent replays/audits. Canonical plan SHA256 `5d41ada25f9b5cd46c9405a115c2b6d1a34bdfe53a42e5062e50ee8cbd5749a1`; [bundle](docs/reports/hu20-o-10b-artifacts/confirmation-final-plan.json), [owner instruction](docs/reports/hu20-o-10b-artifacts/confirmation-owner-final-approval.json), [source/models/native reverified](docs/reports/hu20-o-10b-artifacts/confirmation-final-preflight-verification.json). Expected finish October 7 **06:49 Madrid**, nine-hour deadline **08:38**, from 23:38:40 frozen start. Storage downtime excluded; one worker/8-GiB/≥15-GiB disk and all release rules/source unchanged. Owner packet follows all audits, no publication without explicit owner go. Earlier hold snapshots below remain historical.

**Final admission hold after complete blind pilot:** six arena arms, six direct pairs and native timing complete; scores uninspected. Proposed LBR 4,608 /pressure 12,288 /eleven others 256 /six direct pairs 65,536 duplicate blocks. Forecast **5 h 44 min nominal, 7 h 10 min including 25% headroom**, exceeds the prospectively agreed three-hour gate; no final plan/hash/play yet. Storage downtime already excluded. [Quote](docs/reports/hu20-o-10b-artifacts/retry02-confirmation-quote.json), [posted admission hold](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6025818676). Waiting for owner instruction; no active compute worker remains.

**Completed retry-02 pilot archive:** same PR185 Research-Cloud folder, `hu20-o-10b-confirmation-retry-02-20261006.zip`, **242 verified members /907,275,611 ZIP bytes**, ZIP SHA256 `5430046a6ea2bb67eeea9bb9f17a99649ef10037a4dfb6e66abd8825e4273895`, manifest `5f19c6d296170c93df30d2876047506a15a786f4bb67302a017212dd35ca5642`. [Receipt](docs/reports/hu20-o-10b-artifacts/retry02-pilot-archive.json), [archive post](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6025824801). Includes complete raw pilot traces, models/audits, scientific source archive/patch, native preparation/one timing result, full source/input verification, all controls/root searches/SD-cost quote and hold status. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-confirmation-retry-02-20261006.zip results/retrieved/pr185/retry02-pilot`, verify every embedded-manifest member, then follow `confirmation/RESTORE-REFERENCES.json` and `confirmation/native-input-restoration.json` for exact canonical #165/#176/#185 source/model/reference inputs. Upload acceptance pending; originals/open roots/synced files retained. Earlier running snapshots below are historical.

**Displayed analysed results:** [PR table/chart/seat splits/training and audit status](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6025778202). All four first-lineage contrasts and descriptive seat splits are taken directly from verified independent audits; no fresh pilot means/intervals/labels inspected. Exact posted body and immutable chart/audit inputs are sealed in the same PR185 folder, `hu20-o-10b-analysed-publication-20261006.zip`, **11 verified members /120,236 ZIP bytes**, SHA256 `192272fa13e2205e15a42178cf51636bf21f7ee9df20e124fdf2afce0369babe`, manifest `09b56e4e6864741700f56b3ae8ca2e822be1686970b5c2e00182205add8e2f38`. [Archive receipt](docs/reports/hu20-o-10b-artifacts/analysed-publication-archive.json), [value/markup validation](docs/reports/hu20-o-10b-artifacts/analysed-publication-validation.json), [dated pilot progress](docs/reports/hu20-o-10b-artifacts/analysed-publication-progress.json). A sub-megabyte M1 metadata archive is light coordination; all training/evaluation stays on M4. The first archival preflight ran before transfer completed and stopped before creating a ZIP; failure/correction notes retained. SCP later returned zero. Upload acceptance pending, originals retained. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-analysed-publication-20261006.zip results/retrieved/pr185/analysed-publication` and verify every embedded-manifest member. Primary remains **+3.98 [1.35, 6.61] BB/100**, one lineage; three-lineage release confirmation pending.

**Retry-02 preflight archive:** same PR185 Research-Cloud folder, `hu20-o-10b-confirmation-retry-02-preflight-20261006.zip`, **31 verified members /352,785 bytes**, ZIP SHA256 `27aaff7ce7120b859c5d10d56f096bab039f17f2e6ffe15180447a2a23679c1d`, manifest `c30a3277cc4eefffabef2f0af07ce71f711e7fefeaac57c1c6776b0531dff295`. [Receipt](docs/reports/hu20-o-10b-artifacts/retry02-preflight-archive.json). Stores new control scripts/predeclaration, full freshness/unreadable-path proof, all 203 restored-input member hashes, model/source/storage verification and dated tracked-doc/PR-body snapshots. Source/model/native data remain pinned to the canonical prior archives; no active trace is copied into this preflight ZIP. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-confirmation-retry-02-preflight-20261006.zip results/retrieved/pr185/retry02-preflight`, verify embedded manifest, then use `confirmation-retry-02/native-input-restoration.json` and prior entries for exact native/model members. Upload acceptance pending; originals retained. Active pilot/final traces will be sealed separately at completion or stop.

**Fresh confirmation restart:** [pre-pilot declaration](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6025392668), own M4 work root `~/Local/hu20-o-10b-20261006/confirmation-retry-02/`, controls `retry-02/`. Arena pilot/final 202610066701 /202610066801; direct 202610066901 /202610067001. All old stopped outputs remain immutable/excluded; no scores inspected. Six models and all 592 scientific files reverified, 203 native inputs member-hash verified, missing 1.141 GiB restored and matching inputs reused as immutable hard links. About 27.0 GiB free after restoration, **3.24 GiB extra after native copies, policy/request estimate, 3-GiB trace/archive reserve and ≥15-GiB floor**. [Preflight receipts](docs/reports/hu20-o-10b-artifacts/retry02-retry-storage-verification.json), [additional-file estimate](docs/reports/hu20-o-10b-artifacts/retry02-native-additional-space-estimate.json), [models/source verification](docs/reports/hu20-o-10b-artifacts/retry02-retry-input-verification.json). Pilots running; no final count/quote/hash freeze yet, no new playing-strength result or release package. All original statistical/resource/time/release checks stay binding. Earlier disk-stop/current-space descriptions below are dated snapshots.

**Status closeout supplement:** same PR185 Research-Cloud folder, `hu20-o-10b-confirmation-status-closeout-20261006.zip`, **33 verified members /134,621 bytes**, ZIP SHA256 `2474e1cf724d2435516e2877272db8fadb9704c9accc3c813069e78b4af06f03`, manifest `f2ffc15bf83435a4a4c2a077707b8ec161a6174b5f97bbe3af08e7392c05ee7d`. [Receipt](docs/reports/hu20-o-10b-artifacts/confirmation-status-closeout-archive.json). Includes dated report/protocol/index/roadmap, six audit/source/storage/cleanup/upload receipts, PR body/status and validation/CI handoff for `beb9bf7`. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-confirmation-status-closeout-20261006.zip results/retrieved/pr185/status-closeout` and verify every embedded-manifest member. [Posted current status](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6024617690). Upload acceptance pending; originals retained. CI for the later indexing commit is not assumed by this snapshot.

**Current confirmation status:** all three 10B seeds complete, both new 1B checkpoint/export hashes reproduce #165 exactly, all six full accumulator/current-policy audits pass. First-lineage direct **+3.98 [1.35, 6.61] BB/100**, better, independently audited. Conditional confirmation stopped on the ≥15-GiB free-disk guard during blind arena pilot **202610066301**: three complete arms and one partial, outcomes uninspected. Direct pilot **202610066501**, final freeze/matches, turn/river scores and release gates incomplete. No other-seed strength conclusion or v0.4.2 package. [Current report](docs/reports/hu20-o-10b.md), [source verification](docs/reports/hu20-o-10b-artifacts/confirmation-source-verification.json), [STOP](docs/reports/hu20-o-10b-artifacts/confirmation-pilot-stop.json). Preserve failed roots; fresh predeclaration/roots and measured quote required for retry. About 16.7 GiB free after archive; unchanged layout needs **≥23.35 GiB plus policy/request space**. [Storage calculation](docs/reports/hu20-o-10b-artifacts/confirmation-storage-plan.json). ETA withdrawn; PR stays open.

**Confirmation disk-stop archive:** [PR185 Drive folder](https://drive.google.com/drive/folders/1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL), native `~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-confirmation-disk-stop-20261006.zip`: **103 verified members /1,270,367,609 bytes**, ZIP SHA256 `c1e1bbd0603a697e94d44da4ac69961fcd1771aae74d9d1296192910b1fdb8ea`, manifest `8d08e0023fabb92df8b9d5319401b2d7958b36daabf72f9a37625b79b3d8585c`. [Receipt](docs/reports/hu20-o-10b-artifacts/confirmation-stop-archive.json). New 602/603 average/current exports, all six audits, exact #176 source archive/patch, complete/partial traces, helpers, logs, failures and cleanup evidence retained. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-confirmation-disk-stop-20261006.zip results/retrieved/pr185/confirmation-stop`; verify embedded manifest, then follow `research/confirmation/STOP-RESTORE-REFERENCES.json` for hash-pinned unchanged native/model members from canonical #176/#165/#185 archives. Native upload now accepted (exact bytes, no conflicts); independent cloud metadata acceptance pending. Originals retained; synced files untouched.

**New trained exports:** `research/confirmation/policies/O-2026100602-10000000000.average.jsonl.gz`, **251,732,642 bytes**, SHA256 `52de62633d8a56e17019be7b8db6b5e744043a40667e2352171f67119ac6f2ce`; seed 603 corresponding member **251,815,844 bytes**, SHA256 `b0e514d0f46e52de32fa4e183bd5b88ed259cf147f4af661f4cba034775a3b4f`. [Six model/checkpoint identities](docs/reports/hu20-o-10b-artifacts/confirmation-models.json), [preparation completion](docs/reports/hu20-o-10b-artifacts/confirmation-preparation.json); individual `confirmation-export-audit-<seed>-<nodes>.json` receipts record current/average/checkpoint hashes and every stored node. These are research exports, with no release approval.

**Seed 603 training complete:** **3,780.96 s (63.02 min), 1.461-GiB peak sampled RSS**, exit zero/no guard failure; [receipt](docs/reports/hu20-o-10b-artifacts/seed603-finished.json). 1B checkpoint hash `63bb2ddb5bb102be82b919b5caa0ef95c1bcd5d852fe35af1762291525db7ac8` reproduces #165. 10B checkpoint member `O-2026100603-10000000000.json.gz`, **387,869,638 bytes**, SHA256 `71e3683936e69a3b4a4b7c286b2a8526e1eedd02f68aba1b34ae04a45c5e1392`. Seven-member ZIP `O-2026100603-10000000000-milestone.zip` in the same PR185 folder, **387,875,447 bytes**, SHA256 `e259d84cb5aaad7e2aec33d797fa2e5c6275c04ef7757ff21c19f5d5f34245da`, manifest `a2f8352bb8657ddaddc6434d0d41847cb32877a85ec16f3817b221c4ec56cc70`. [Archive receipt](docs/reports/hu20-o-10b-artifacts/seed603-10b-archive.json). Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/O-2026100603-10000000000-milestone.zip results/retrieved/pr185/seed603-10B` and verify every member/checkpoint hash. All eight new milestone ZIPs and two training tails are member-verified; [completion receipt](docs/reports/hu20-o-10b-artifacts/training-archive-complete.json), [all ZIP names/bytes/SHA256/manifest hashes](docs/reports/hu20-o-10b-artifacts/training-all-archive-receipts.json). Each milestone ZIP contains the matching `O-<seed>-<nodes>.json.gz` member; restore with `python3 -m zipfile -e <receipt-archive-path> results/retrieved/pr185/<seed>-<nodes>` and verify its embedded manifest. Seed-603 2B checkpoint SHA256 `a935fc2403dafb3ca982745f23c762f2de6a226cfa2a00e4d1667c26ee4e966d`; 5B `bb08472a95b11e2dc5c06565d59f6e04df0d4dc082d0d839528ee495c769d050`. New milestone cloud acceptance not yet claimed. Older active-training snapshots below are historical.

**Preliminary publication supplement:** `hu20-o-10b-preliminary-publication-20261006.zip`, same PR185 Research-Cloud folder, **13 verified members /192,070 bytes**, ZIP SHA256 `c40c17076b044e79f8bfae93d39042b2ec1209c73a37374f37f0852c7aa82e59`; manifest `4391c76c968468729dacd4b761d2791ce1aa217d2cef105b5e3336a79451d2ac`. All member readbacks pass. Members include `research/publication/{audits.json,strength-vs-nodes.png,strength-vs-nodes.svg}`, reproducible plotting source/historical input JSON and dated report/index/roadmap/PR/CI handoff snapshots. [Published preliminary results/chart and seed-602 completion](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6021715754). Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-preliminary-publication-20261006.zip results/retrieved/pr185/publication` and verify embedded manifest. Upload acceptance pending; originals retained. Full new-head CI remains required; this snapshot does not claim future training, CI or confirmation completion.

**Seed 602 completion receipt:** seed 602 finished 10B, exit zero/no failure, **3,888.30 s (64.80 min)**, peak sampled RSS **1.330 GiB**; all 1B/2B/5B/10B milestones complete and 1B matches #165. Seed 603 subsequently completed with exact parity; see the current status above. [Finished receipt](docs/reports/hu20-o-10b-artifacts/seed602-finished.json). Seed-602 10B checkpoint member `O-2026100602-10000000000.json.gz`, **387,635,205 bytes**, SHA256 `9c7d90b1bfeda26bb81462c742a17d04450e34077b9eb3bc35039877e4ee139f`. Seven-member ZIP `O-2026100602-10000000000-milestone.zip` in [PR185 Drive folder](https://drive.google.com/drive/folders/1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL), **387,641,010 bytes**, SHA256 `122c4970149f1def9dc965d65ef4a8bacaba49101dfe9dddd60b663238348b06`; embedded manifest `7c7acbbb550aaaa3687e44979cc2b54c5a3963c29083210fdc5c85f79910af9c`. [All-member receipt](docs/reports/hu20-o-10b-artifacts/seed602-10b-archive.json). Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/O-2026100602-10000000000-milestone.zip results/retrieved/pr185/seed602-10B` and verify manifest/checkpoint hashes. New milestone upload acceptance pending; subsequent complete export audits pass, with no release promotion. Earlier active-training snapshots below are historical.

**Owner-approved amendment:** evaluation/audit now uses **8 GiB and one worker at a time**; training retains its original guard. [Pre-pilot amendment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6020980784). The complete unchanged 10B export audit passes all **7,227,377 keys** at 6.216-GiB sampled peak RSS, 71.15 seconds. [Audit](docs/reports/hu20-o-10b-artifacts/export-audit-10000000000.json), [resource](docs/reports/hu20-o-10b-artifacts/audit-10b-8gib.resource.json), [one-line RSS patch and source hashes](docs/reports/hu20-o-10b-artifacts/owner-approved-8gib-source.json). Candidate average remains SHA256 `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae`, current `af6c1755c7cb816dbdb07ccd32d2dbde82fea3c2ab0c9a8946ce8525195c9bd1`; both now verified against first-seed 10B checkpoint `54553c008231126c94ec455e89dcbcdb162aa46736c1a40161cf0d63d666ab49`. All frozen final and excluded-pilot independent audits pass. The first-lineage primary is **better**, +3.98 [1.35, 6.61] BB/100, achieved half-width 2.626. Three-lineage confirmation subsequently triggered and stopped on its disk guard; see the current status above. Earlier stopped attempts and preparation archives below remain immutable historical records; their unaudited/stopped descriptions are superseded for current status only.


**Final counts frozen before play:** four contrasts ×61,440 duplicate blocks ×two seats = **491,520 final hands**, plus 32,768 excluded pilot hands. Fresh final root 202610066201; pilot 202610066101. Canonical plan SHA256 `a242aeccb1a46214fb7014c1fd72e2d1e66d5d98df3b7e2794acf17904322ad4`; all runner plans/hash/RSS patch and exact model identities pinned. [Frozen plan](docs/reports/hu20-o-10b-artifacts/final-plan.json), [quote](docs/reports/hu20-o-10b-artifacts/quote.json), [pre-play comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6021112472). Primary projection ±2.610; 43.52-minute serial play/report/replay quote including 50% headroom. Final play took 14.42 minutes; all independent final and excluded-pilot audits completed within the frozen two-hour cap. All **524,288 hands /2,763,383 actions** replay and raw-chip arithmetic, intervals and labels agree. [Independent audits](docs/reports/hu20-o-10b-artifacts/audits.json), [report and updated chart](docs/reports/hu20-o-10b.md), [preliminary PR table](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6021666029). Secondary 2B−1B +3.61 [1.01, 6.22], 5B−1B +4.19 [1.56, 6.82], 10B−R1 +15.82 [12.89, 18.75], all better. One lineage only; no direct 10B−2B/5B comparison or transitive chart values.

**Completed run archive:** `hu20-o-10b-8gib-run-20261006.zip` in [PR185 Drive folder](https://drive.google.com/drive/folders/1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL), native `~/Local/Research-Cloud/PR-185-HU20-O-10B/`: **134 verified members /404,635,290 ZIP bytes**, SHA256 `a8e74b482d72e171d5002dc6f88a0239d1ed59fd0b89f28be65c8f4a8be83363`; manifest SHA256 `59f65431b81368c99389f4afd4b832a37966034d36053b4a28b6b505f0222049`. [Receipt](docs/reports/hu20-o-10b-artifacts/run-archive.json). Every member readback verifies; final/pilot raw traces, plans, reports, independent audits, failure/runtime receipts and source patch retained. Unchanged checkpoint/policy/source inputs restore via hash-pinned references in `closeout/RESTORE-REFERENCES.json` to the original preparation ZIP. Native Drive reports uploaded=true/uploading=false/no conflicts/exact size; independent cloud listing acceptance pending. No remote-byte re-download claimed.

Restore to a new ignored nonsynced root: `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-8gib-run-20261006.zip results/retrieved/pr185/run`; verify `ARCHIVE-MANIFEST.json` and restoration references. Raw members `research/final/<family>/run/direct-lineage-1.hands.jsonl.gz`; exact per-family SHA256s are in [audits](docs/reports/hu20-o-10b-artifacts/audits.json). Retrieve earlier source archive and apply the exact recorded RSS-only patch before replay. Later chart/report/PR/CI receipts are sealed separately. Originals and synced archives remain; no deletion/eviction or #166 touch.


The amendment is sealed in `hu20-o-10b-8gib-amendment-20261006.zip` under the same Drive folder: **18 verified members /39,472 ZIP bytes**, SHA256 `2bac43ef001ca803ab4d340169cfaf2e919d3612ff96eff9620cecbe003cc495`; manifest SHA256 `e415906c4fdecf100441ceca42144a5ad69ff54958e79217786324037fc2c07e`. All members pass readback. It contains the resource approval/protocol comment, unchanged successful full-audit output/resource receipts, revised orchestration, exact RSS-only source patch and full source hashes, model catalog and restoration references to the original preparation ZIP. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-8gib-amendment-20261006.zip results/retrieved/pr185/amendment`, verify the manifest, restore the original source archive, then apply the recorded one-line patch. Native/cloud acceptance for this new ZIP is not yet claimed; originals and earlier stopped-attempt archives remain.

### Original preparation snapshot and archive provenance

[PR #185](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185), [protocol](docs/hu20-o-10b.md), [dated preparation report](docs/reports/hu20-o-10b.md). Strength undecided: zero pilot/final poker hands. Separate unchanged 10B average/current native exports pass, but the full unchanged accumulator audit stops at 6.059 GiB; allocator-only retry also stops at 6.072 GiB. Both retain the 6-GiB guard; all failures are archived, no audit waived. 1B/2B/5B full audits pass. Conditional confirmation and v0.4.2 package are not triggered.

M4 root `~/Local/hu20-o-10b-20261006/`; M1 light coordination root same path; feature checkout M1 `~/Local/deepcfr-o-10b/`. Main source `cdc315f00a6d077af3a4d13c438103354d1b4b5b`, match source `8bbfc457`, exact native binary SHA256 `642a9cc96dabc6d4f74acf139338e4eb4d2ac22a1643b78db875eb230fc4bfbd`. All source checkpoints for seed 601 match #182’s indexed hashes. [Input/output/resource catalog](docs/reports/hu20-o-10b-artifacts/preparation-status.json). Reserved roots 202610066101, 202610066201, 202610066301, 202610066401, 202610066501 and 202610066601 have no matches in readable searched histories; unreadable M4 paths are retained. All remain unused by this campaign.

Destination [PR-185-HU20-O-10B](https://drive.google.com/drive/folders/1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL), native `~/Local/Research-Cloud/PR-185-HU20-O-10B/`. Preparation ZIP `hu20-o-10b-preparation-stop-20261006.zip`: **81 verified members /3,074,383,890 ZIP bytes**, SHA256 `1124a6503be29a03d916be44b92c10f07056a28f48ba1423cbb29a68bf1107d6`; embedded manifest SHA256 `9a14bba765a5548e05c596ab9af68376333162cfdbc4c6cc618d1972dac65f5c`. Every member passes size/SHA256 readback. It preserves exact inputs, exports, both complete Git source archives, native binary, commands/helpers, failed preparations and dated running-training snapshots. Expanded source/build caches are represented by the immutable archives and exact binary. [Receipt](docs/reports/hu20-o-10b-artifacts/preparation-archive.json). Native upload currently pending, uploaded=false/uploading=true/no conflicts; folder ID/parent verified independently. No remote archive metadata acceptance or cloud byte readback claimed.

Retrieve into a **new ignored, nonsynced** directory: `mkdir -p results/retrieved/pr185; python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/hu20-o-10b-preparation-stop-20261006.zip results/retrieved/pr185`. Verify every `research/<path>` size/SHA256 against embedded `ARCHIVE-MANIFEST.json`, then follow `research/RESTORE-README.md`. First-seed checkpoint members: `research/inputs/O-2026100601-<nodes>.json.gz`, hashes from #182 above. Average members: `research/policies/O-<nodes>.average.jsonl.gz`; current members: `research/policies/O-<nodes>.current.json.gz`. 10B average SHA256 `15736cc61a72baa1e6722b1566897917ffb6fdf8bea8874a82b5485fe95d4bae` remains **unaudited, not promoted**. Exact 1B average is released v0.4.1 hash `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`.

Training controller 72900 /initial native trainer 72901 continues sequential seed 602 then 603 on M4. **Seed 602 @1B passes byte parity**: 215,171,058 bytes, SHA256 `a0ef695d6807aa4554ba01b32d27f64c39ccdc72d03be5a21c99cf67e52ad716`; observed save 399.18 s, peak training RSS 1.330 GiB. Its archive `O-2026100602-1000000000-milestone.zip`: **7 verified members /215,176,734 bytes**, SHA256 `7c51abc6963d5a415ffa3a8a130245ee63a81d1357db11e92e685280afb9613b`; manifest SHA256 `f4048a0514380046c3189d662fec81259fad8b6b26b8819b631bf20fa648f16f`. Member `O-2026100602-1000000000.json.gz` has the matching checkpoint hash. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/O-2026100602-1000000000-milestone.zip results/retrieved/pr185/seed602-1B`, and verify its embedded manifest.

**Seed 602 @2B now archived:** checkpoint member `O-2026100602-2000000000.json.gz`, 263,087,436 bytes, SHA256 `7add3b1310374217d8844b545f4ab32b8519722b1bd509180801fe3c2ec182c9`; save observed at 796.25 s (13.27 min), peak training RSS 1.330 GiB. Seven-member ZIP `O-2026100602-2000000000-milestone.zip`, 263,093,158 bytes, SHA256 `1be8bed66ec43f8c697eaecad0c68140a2447a41bed8285bb81864ead53ebeea`, manifest SHA256 `f9ec046ae1cc8a274fdb1f69f37078378a1211d8e99920df1e5efa7888c81ebf`; all members verify. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/O-2026100602-2000000000-milestone.zip results/retrieved/pr185/seed602-2B`, then verify the embedded manifest. [Milestone note](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185#issuecomment-6020851881). Native 1B ZIP upload acceptance now passes; independent archive cloud metadata acceptance remains pending.

**Seed 602 @5B archived:** checkpoint member `O-2026100602-5000000000.json.gz`, 333,408,609 bytes, SHA256 `14651b481fb44c968f1039f69710344973f5b54a0a75fe5395f4ee319afd85ad`; save observed at 1,956.01 s (32.60 min), peak training RSS 1.330 GiB. Seven-member ZIP `O-2026100602-5000000000-milestone.zip`, 333,414,365 bytes, SHA256 `dfe132d93873203d17f980db4fcba6d7bfbcd70cec481293677171093356689a`; manifest SHA256 `1133cb9a0a06331d0fbdc1045fa8cdf12990b17eb1eec22b51d9859f8eb29d25`; every member verifies. Restore with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-185-HU20-O-10B/O-2026100602-5000000000-milestone.zip results/retrieved/pr185/seed602-5B`, then verify the manifest. Upload acceptance for this new milestone is not yet claimed. Training continues to 10B, then seed 603.

Archive helper 74478 seals each subsequent atomic checkpoint and stable training tail for this authorized run only, then exits; future completion/upload is not claimed by this entry. Look up `<seed>/status.json`, `milestone-<nodes>.json`, `finished.json` under the active M4 `training/` root; archive/member receipts are retained beside each eventual ZIP. First-seed 10B is already canonical under #182. PR stays open; retain this active root, inputs, all originals and synced archives. No #166 touch, deletion, eviction or scheduled future research. Closeout documents/PR/CI/upload receipts are sealed separately in the same folder.

## M4 uploaded-original cleanup — October 6 evening

Merged **#166/#181** eligible originals and nonsynced archive copies removed after current Drive upload acceptance, full archive/member SHA256 and open-PR dependency checks: **6,025 paths /13.779 GB logical**, **12.198 GB measured free-space gain**, **30.173 GB free immediately after cleanup**. Open **#185/#186** roots/dependencies, shared Git, sources, prepared inputs, credentials, synced archives and unarchived evidence remain. PR181 closeout/publication originals lacking an exact M4 transfer map remain; only mapped preparation copies were removed.

[Cleanup and restoration](docs/artifacts/m4-uploaded-original-cleanup-20261006-evening.md), [compact verification](docs/artifacts/m4-uploaded-original-cleanup-20261006-evening.json), [full per-path receipt/restore helper](https://drive.google.com/file/d/1k6fw_iM91tvTfGCgocbADpt-L0Cx8dF3/view) (**2,576,866 bytes**, SHA256 `125a71d61c6f19324e1b9d1f03bf4b72e974f2f38860256f1d18662621659ed3`). Each removed path records canonical Drive ID/URL, archive hash, exact member/hash/size and restoration command. Retrieve the receipt ZIP into an ignored nonsynced directory and run its `restore.py --original ORIGINAL_ABSOLUTE_PATH --out results/retrieved/NEW_FILE`; it checks archive/member hashes. Historical retention claims are superseded only for the receipted paths. M1 originals remain. No unattended cleanup.

## PR182 — O learning curve

**Later merged-original cleanup:** after checking #182 MERGED, fresh native/cloud upload acceptance, every archive/member/original SHA256 and current dependencies, removed **23 M4 originals /3,020,460,782 logical bytes (2.81 GiB)**. [Exact removed paths, Drive URLs, archive members/hashes and restoration receipt](docs/reports/hu20-o-10b-artifacts/merged-pr182-cleanup-receipt.json), [fresh cloud metadata](docs/reports/hu20-o-10b-artifacts/merged-pr182-cloud-check.json). Canonical archives remain unchanged. Source/shared Git, plotting environment and metadata retained; #185 has verified copies of all required inputs, and all open roots including #166/#185/#186 stay protected. This supersedes earlier original-retention snapshots only. No synced deletion or eviction. Restore an individual file by downloading its receipt-pinned archive, verifying its whole SHA256, extracting the exact recorded member into a fresh ignored directory and checking the recorded member SHA256.

[PR #182](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/182), [report and chart](docs/reports/hu20-learning-curve.md), [predeclared protocol](docs/hu20-learning-curve.md). One training lineage, seed 2026100601. O@1B−500M **+5.31 [2.71, 7.91]**, 1B−100M **+14.19 [11.56, 16.81] BB/100**, both better; primary half-widths 2.597/2.627 meet ≤3. O@100M−R1 **−7.41 [−10.16, −4.67]**, worse; O@500M−R1 **+7.68 [5.05, 10.31]**, better. All **598,016 final +32,768 pilot hands /3,132,269 actions** independently replay with raw-chip estimate/interval/label agreement. Free M4 only, no release or new match authorization.

M4 research root: `/Users/dberweger/Local/hu20-learning-curve-20261006/`; light M1 publication/retrieval root: the same path on M1. Trainer main source `3776305f9e43e584a1c221c217323a609ff5db2b`; scientific tools/dependencies exactly #175 execution source `8bbfc457`, with hash checks. Excluded pilot root **202610065101**, final **202610065201**. Canonical bundled final plan SHA256 `c4be553d00115d0ebe051c9adf66d2183b1b53b0b7469badcdf1887d558ac552`. Frozen counts and 48.22-minute conservative quote precede final play; actual admission through all audits 9.91 minutes, largest externally sampled process RSS 5.235 GiB. Three workers, 6-GiB RSS and ≥15-GiB disk guards pass. Root-search unreadable paths and corrected source-transfer/probe failures are retained.

Archive destination: [PR-182-HU20-learning-curve](https://drive.google.com/drive/folders/1wNthhMPoO0XGPeEZbMZmQVIIxF4tJra9), native path `~/Local/Research-Cloud/PR-182-HU20-learning-curve/`. Complete [experiment archive](https://drive.google.com/file/d/1V2bbJ9kf0_MTdqwfcoo__XnCMWjAEkbi/view): **172 members /2,053,533,686 logical bytes**, ZIP **1,899,498,972 bytes**, SHA256 `1c71bdc9373993be071c2c231e2e2cfa24be4d506a864ffc3e86eed3bb15e16f`; manifest SHA256 `356afb24d65857560e88da09d9ba9b8cc08eee0fbf36c7830f5ec72ab77e2db1`. Every member passes size/SHA256 ZIP readback. Native Drive accepts uploaded=true/uploading=false/no conflicts/exact size; independent cloud ID/name/size/parent match. [Archive receipt](docs/reports/hu20-learning-curve-artifacts/archive-receipt.json), [upload acceptance](docs/reports/hu20-learning-curve-artifacts/drive-upload-confirmed.json). No remote byte re-download is claimed. It preserves exact input/checkpoint/current/average copies, both source archives, original native binary, environment, all raw traces, plans, reports, audits, logs, failures and `research/RESTORE-README.md`. A clearly dated snapshot covers active scaled-training metadata; subsequent atomically completed checkpoint ZIPs and closeout receipts remain in the same folder. No future training completion is claimed by the experiment snapshot.

Retrieve into an ignored fresh directory: `mkdir -p results/retrieved/pr182; python3 -m zipfile -e ~/Local/Research-Cloud/PR-182-HU20-learning-curve/hu20-learning-curve-complete-20261006.zip results/retrieved/pr182`. Verify every `research/<path>` size/SHA256 against the ZIP's `ARCHIVE-MANIFEST.json` before use. [Exact model catalog](docs/reports/hu20-learning-curve-artifacts/models.json) records all exported hashes/checkpoint provenance, including O@1B `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d` and shipped R1 `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`. Average members are `research/policies/O-{100000000,500000000,1000000000}.average.jsonl.gz`; source checkpoints are `research/inputs/O-2026100601-{nodes}.json.gz`, pinned to #165's manifest and full accumulator/current audits. Extract archived `source-match-8bbfc457.tar` separately and follow RESTORE-README to rerun reporter and independent replay without changing plans.

**2B milestone landed:** [checkpoint ZIP](https://drive.google.com/file/d/11l_VCpSH35bXYRCfujSlzEmt9se11_gu/view), `O-2026100601-2000000000-milestone.zip`, **258,280,835 ZIP bytes /6 verified members**. ZIP SHA256 `b6cae8b85747cc088edf4dee06e84fa8f859592e32ff4205047cfac3eb59ae7e`; embedded manifest SHA256 `1079a5012b2119ff600c561cfbf287105ebd3da38fd890c0e9438c421011dcbb`. Member `O-2026100601-2000000000.json.gz`: **258,269,802 bytes**, SHA256 `aa91646d59614c3df1544c96bf74d0483d278d3c4e5c8e844ad0f30b1d910d5b`. Actual stdout records **2,000,000,352 nodes /4,158,342 iterations /5,101,104 entries**, 741.22 training seconds before save; complete checkpoint observed at **758.78 seconds**, sampled peak RSS **1.462 GiB**. Native Drive upload acceptance and cloud ID/name/size/parent verify; originals remain. Retrieve with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-182-HU20-learning-curve/O-2026100601-2000000000-milestone.zip results/retrieved/pr182/2B`, then verify the embedded manifest and checkpoint SHA256 before use. The unchanged CLI serializes `config.max_nodes=1B` in save metadata; actual work is pinned by the command and retained milestone stdout. [Milestone PR note](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/182#issuecomment-6018910955). The same run subsequently completed 5B/10B below; this checkpoint has not been matched or promoted.

**5B milestone complete:** [checkpoint ZIP](https://drive.google.com/file/d/1kmS7yRPee2lhA67CLX2w9yrj2EO4_Y5i/view), `O-2026100601-5000000000-milestone.zip`, **328,483,201 ZIP bytes /6 verified members**. ZIP SHA256 `d11d404eac31b3e58011e2a961e9b2c9604e415e064c682388ed759ca403c930`; manifest SHA256 `a5250834508a503c8b391f8dbc77749eb52ebb18de10e59e99b45cd1c3fe557f`. Member `O-2026100601-5000000000.json.gz`: **328,471,948 bytes**, SHA256 `2af7f04e094cc5a8975c22c230677ee65d4effff2b7dca3cfc7387b3bae14d0f`. Actual stdout: **5,000,000,043 nodes /9,971,555 iterations /6,315,718 entries**, 1,868.11 seconds before save; completed save observed at **1,890.58 seconds (31.51 minutes)**. [Member/archive receipt](docs/reports/hu20-learning-curve-artifacts/milestone-5000000000-archive.json), [PR note](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/182#issuecomment-6019270243). Retrieve with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-182-HU20-learning-curve/O-2026100601-5000000000-milestone.zip results/retrieved/pr182/5B`, then verify its embedded manifest and checkpoint SHA256.

**10B milestone complete:** [checkpoint ZIP](https://drive.google.com/file/d/1Oi26CAw3C3Y2KkBsY4GNzTgxRYL9cFkI/view), `O-2026100601-10000000000-milestone.zip`, **383,604,666 ZIP bytes /6 verified members**. ZIP SHA256 `a35a628159cc8d9810802a6f264e35c210c40e4f5fe6ce4a26594b94ad3f1433`; manifest SHA256 `b33d49ee2c63683bacd4b23d1e76a13e33e20a4f4c219105b054d0c8f942fa46`. Member `O-2026100601-10000000000.json.gz`: **383,593,179 bytes**, SHA256 `54553c008231126c94ec455e89dcbcdb162aa46736c1a40161cf0d63d666ab49`. Actual stdout: **10,000,000,355 nodes /19,538,759 iterations /7,227,377 entries**, 3,738.37 seconds before save; completed save observed at **3,765.27 seconds (62.75 minutes)**. [Member/archive receipt](docs/reports/hu20-learning-curve-artifacts/milestone-10000000000-archive.json), [PR note](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/182#issuecomment-6019888327). Retrieve with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-182-HU20-learning-curve/O-2026100601-10000000000-milestone.zip results/retrieved/pr182/10B`, then verify its embedded manifest and checkpoint SHA256.

**Scaled training finished:** `/Users/dberweger/Local/hu20-learning-curve-20261006/scaled-training/` on M4. Exact owner-requested single lineage completed 2B/5B/10B, exit code zero, no guard failure, **3,766.16 seconds (62.77 minutes)** total guarded duration and **1,569,980,416 bytes (1.462 GiB)** peak sampled RSS. Preemptive guards were 5.5-GiB sampled RSS /15.5-GiB free disk. [Final receipt](docs/reports/hu20-learning-curve-artifacts/scaled-finished.json). All three checkpoints remain research inputs: no new matches, inference-export audits or promotion; v0.4.2 is the owner's decision. The CLI's legacy `config.max_nodes=1B` save metadata does not represent actual completed work; exact command and stdout are retained.

The stable [training-tail ZIP](https://drive.google.com/file/d/1CMIGR7wugH6ttPiqlfNNYvWds9CF1vUd/view), `O-scaled-training-tail-20261006.zip`, has **23 verified members /17,098 ZIP bytes**, SHA256 `a92680a42b70816f4f95af653922994685837cd05ea0b04ba3ade30c53ceced4`, manifest SHA256 `c337c6c6c4c73292ae578cb815195e4de46d24f6c1dac6135279a308a3a3b146`. It stores complete final logs/metadata and verified references to the canonical checkpoint ZIP members. [Tail receipt](docs/reports/hu20-learning-curve-artifacts/scaled-tail-archive-receipt.json), [native/cloud upload proof for 5B/10B/tail](docs/reports/hu20-learning-curve-artifacts/scaled-upload-confirmed.json). Every member passes readback; native uploaded=true/uploading=false/no conflicts and cloud ID/name/size/parent match. No remote-byte re-download is claimed. Retrieve with `python3 -m zipfile -e ~/Local/Research-Cloud/PR-182-HU20-learning-curve/O-scaled-training-tail-20261006.zip results/retrieved/pr182/training-tail`, verify its embedded manifest, then restore checkpoint members from their separate canonical ZIPs. Originals and other agents' roots/dependencies, including #166, remain protected; no synced deletion/eviction or cleanup was performed.

## Verified local-copy cleanup — October 6, 2026

Owner-authorized merged-run cleanup removed 2,134 verified files on M1 and 1,572 on M4. Measured removal-batch reclamation: **27.88 GB M1 / 19.04 GB M4**; final free space **57.23 GB / 38.64 GB**. Streamed archive verification populated native cache while #182 training continued, so these are distinct measurements.

[Cleanup and retrieval guide](docs/artifacts/merged-research-original-cleanup-20261006.md) · [full path/member/SHA256 receipts](https://drive.google.com/file/d/1P_KUWSE-gM2SwMCCvkn6TFONBslc6qz7/view?usp=drivesdk). All six canonical archives have fresh whole-archive/member checks, native upload acceptance and matching current Drive metadata. The receipt maps every removed path to its Drive ID, archive SHA256, member, source SHA256 and retrieval command. Open #166/#182 roots, #165 original inputs/provenance, #175 historical policies, source/shared Git, changed or unarchived files and synced archives remain. Earlier retention statements are historical; this cleanup receipt records later removals. No unattended task or cache eviction.


## PR181 — v0.4.1 release preparation

[Release PR #181](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/181), [comparison/report/chart](docs/reports/v0.4.1-release.md), [readiness, screenshots and verification](docs/releases/v0.4.1/READINESS.md), [cleanup inventory](docs/releases/v0.4.1/CLEANUP.md). #176 and #179 are merged; #180's model-storage guidance is preserved. The owner selected the exact #165 first-seed O average. The owner-approved v0.4.1 default retains shipped v0.4.0 selection. All **10 real web hands / 83 actions / 37 bot distributions** independently verify, including both seats and exact 201-chip free sizing. No missing/zero-mass/unexpected fallback appears in these hands. All 92 captured final-run API responses are HTTP 200, with no browser/server errors in that checked interval. All five unpublished assets verify after a staged HTTP download from a clean checkout. **The owner gave the final chat go; [v0.4.1 release assets](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.1) bind the exact O1 model to the merged source. v0.4.0 remains available.**

Original M1 evidence: `/Users/dberweger/Local/v041-release-20261006/`; M4: `/Users/dberweger/Local/v041-release-pr181-20261006/`. The **40 retrieved M4 files all match source size/SHA256**, excluding the live access credential, which is retained locally but never archived or published. Public sanitized receipts are under `docs/releases/v0.4.1/verification/`; private SQLite journals/recorded positions remain in the research archive. The model provenance is #176's unchanged prepared export, SHA256 `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`, with original checkpoint/export and Drive restoration pinned in the PR176/PR165 entries. No re-extraction, fallback replacement, training or paid compute.

Archive: `~/Local/Research-Cloud/PR-181-v0.4.1-release/v0.4.1-release-preparation-20261006.zip`, in [deepcfr-research-results](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s). **58 source members / 639,695,992 logical bytes; 636,443,857 ZIP bytes**, SHA256 `2ad0a8a92a9812a13067fce242547494a00246803cafc49ef2b983e7685e3a0e`; embedded manifest SHA256 `253e54cac9c8705b454d55886ab25795ab3e983e59579e700a4ac5cb8299abb7`. Every ZIP member passes size/SHA readback. [Archive receipt](docs/releases/v0.4.1/verification/archive-receipt.json), [native Drive acceptance](docs/releases/v0.4.1/verification/drive-native-upload.json): uploaded=true, uploading=false, no conflicts, exact size. No cloud archive byte re-download is claimed. Final owner-go, merge/tag, approved assets, GitHub download verification, release and spectator-issue receipts are preserved in a separate publication ZIP in this same folder; prior unpublished packages remain unchanged. Restore to a fresh nonsynced folder with `unzip ARCHIVE -d NEW_DIRECTORY`, verify `ARCHIVE-MANIFEST.json`, and follow `RESTORE-README.md`. Full source/environment, pinned inputs, private replay data, screenshots, guarded-resource receipts, auxiliary corrections, unpublished package and downloaded bytes are retained. Final-source/CI/PR receipts go in a separate immutable closeout ZIP in the same folder. All originals remain; no synced file is deleted/evicted and #166's open roots/dependencies are untouched. No uncertain cleanup item is removed.

## PR176 merge closeout supplement

#176 merged at `3776305f9e43e584a1c221c217323a609ff5db2b` after required checks and complete full CI on reconciled head `f19f1dc59965974043b49dd03c55374ecb44dd91`. Final run [37468322518](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37468322518) is green; owner-approved comments were preserved through #179/#180 reconciliation. No bypass, history rewrite or force push. Merge supplement: `~/Local/Research-Cloud/PR-176-HU20-v041-O-confirmation/hu20-v041-O-confirmation-merge-20261006.zip`, **15 verified members / 329,786,732 ZIP bytes**, SHA256 `17e688a61107bc7c196c757339887e1f06cf1dac900e17e491f3347fe4697805`, manifest SHA256 `03945c0051889cc33724d217e7d8aee9ca52c9321d9c0bc939ca7fbd46251f27`. Native Drive acceptance rechecked: uploaded=true, uploading=false, no conflicts, exact size. Source reconciliation, validation, final checks and merge receipts are preserved; prior archives and all originals remain.

## PR176 — HU20 v0.4.1 O confirmation

[PR #176](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/176), [complete report](docs/reports/hu20-v041-o-confirmation.md), [protocol](docs/hu20-v041-o-confirmation.md), [unpublished readiness](docs/releases/v0.4.1/READINESS.md). All four prospectively declared gates pass: direct O−R1 **+10.50 [7.90, 13.10]**, bounded LBR O−R **+37.56 [27.75, 47.37]**, native pressure **+6.92 [0.39, 13.45] BB/100**, and no other panel upper bound below −20. Every O lineage beats shipped R1 individually. All **641,280 pilot/final hands / 3,517,178 actions** independently replay, and all 160 fixed-board native metrics/bootstrap arithmetic verify. O1 E **1.066 [1.002, 1.131]**, Q **0.523 [0.481, 0.577]**; shipped R1 E **2.715 [2.492, 2.932]**, Q **2.679 [2.325, 3.058]**. Free M4 only, 113.35 minutes through final evaluation/audits, no training or rental. v0.4.0 stays stable pending owner review and explicit chat publication approval.

Originals: M4 `/Users/dberweger/Local/hu20-v041-o-confirmation-pr176-20261006/`; verified M1 retrieval `/Users/dberweger/Local/hu20-v041-o-confirmation-20261006/`. Exact scientific source `523e4a347d012beb8cb523497d2e9a4cbb90035f`; candidate integration source `f23a609928bd970d6a7be6cb4053a4dff15fed36`. Arena final/pilot roots 202610062101/202610062001; direct final/pilot 202610062301/202610062201. Pressure 16,384 blocks; LBR 5,632; eleven others 256 each; direct 49,152 per pairing. Canonical bundled plan SHA256 `49a1d6a05e82eaf929c5f76a275c9e3ed90a5a99b129cbb2817dd4b69c2c9a1b`. Retrieved **770 M4 files / 7,554,372,581 bytes** all match their snapshot SHA256s; every pilot/final raw trace and native response matches its independent audit. Additional M1 documentation/helper receipts are separately included.

Archive: `~/Local/Research-Cloud/PR-176-HU20-v041-O-confirmation/hu20-v041-O-confirmation-complete-20261006.zip`. The embedded `ARCHIVE-MANIFEST.json` records member sizes/SHA256s; every member passes ZIP readback. It contains exact six policy exports/input manifests, source/environment, frozen plans, blind pilot traces, final raw actions/settlements, lock-only native requests/compact/locked policies/common references/responses/binary, all audits, resources, auxiliary failures/corrections, candidate package and restoration guide. The unpublished inference bundle excludes engine binaries and private journals. Archive receipts beside the ZIP give its hash and member count; later final PR/CI/upload/documentation receipts go in a separate member-hashed closeout ZIP. Restore to a fresh nonsynced directory and follow `RESTORE-README.md`; original absolute paths and relocation provenance are preserved. **All originals remain. No synced file is deleted or evicted.** Cloud acceptance is distinguished from local hash verification.


Verified main ZIP: **792 members / 8,038,176,698 logical bytes**, ZIP **3,389,594,093 bytes**. SHA256 `9f473aa22b16037549716e8fd938b4b47670ca89e29b398bf241b1448fe597a3`; embedded manifest SHA256 `6cd6cca672569b42c062837de823043c08d7fd8dfc1d3fb4587fe5628ec875fe`. Every member passes readback. Final documentation/PR/CI/upload receipts are retained in a separate member-hashed closeout ZIP in the same folder; original files remain.

## PR179 — HU20 zero-mass current fallback

[PR #179](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/179), [full report](docs/reports/hu20-zero-mass-fallback.md), [frozen protocol](docs/hu20-zero-mass-fallback.md). Merged #178’s optional current-policy export fallback shows **no detectable difference** against either original: T′−T **+0.18 [−2.19, +2.56]**, O′−O **−0.35 [−2.71, +2.02] BB/100**. Secondary T′−R1 **+5.46 [2.47, 8.46]** and O′−R1 **+6.48 [3.48, 9.48]** are better; T′−O **−0.99 [−3.36, 1.39]** is no detectable difference. All lineages are reported. Neither primary is better, so the conditional pressure check is not triggered. No release decision.

M1-only originals: `~/Local/hu20-zero-mass-fallback-20261006/`. Execution source `9a24fe6b43088026e14227934ec6599e017646bf`; merged native source `0a3765f34335b8ca8efea7d79e25c7fd19d94657`. Exact six checkpoints and original T/O/R1 policies match #165’s manifest. All six fresh exports (23,042,805 keys) pass full accumulator/current-policy audits. Final root `202610063501`, excluded pilot `202610063401`; five families ×36,864 paired deal blocks ×three lineages ×two seats = **1,105,920 final hands / 5,721,474 actions**. All final and **61,440 pilot hands / 318,419 actions** independently replay. Canonical final plan SHA256 `1c379fec6390931a8d5f88dbf9c0390d552ef4fa6c2f4575748dfdbfd740bd18`. All primary half-widths are **2.37–3.40**, meeting ≤4. Free compute only, no M4 jobs or RunPod, at most three workers, 6-GiB RSS guard and 8-GiB disk floor; no guard failure or deletion.

Archive: `~/Local/Research-Cloud/PR-179-HU20-zero-mass-fallback/hu20-zero-mass-fallback-complete-20261006.zip`. **272 research members / 4,581,406,376 logical bytes**, ZIP **4,581,478,163 bytes**, SHA256 `2c18809a712434865c12ae77dd4ca281f10422f77014d3534063b4b5d50ef9b1`. Embedded gzip manifest SHA256 `8b173f3d2430848534933dcd96c8e02c90cd9e09cbda2e073eec7a3dffbb3e2e`; every member passes ZIP size/SHA256 readback. The ZIP stores gzip members without redundant compression: exact checkpoint/policy inputs, all research outputs, frozen source/binary/environment, plans, raw pilot/final action/settlement traces, logs, preparation attempts, audits, protocol/comments and `RESTORE-README.md.gz`. Use the verified `source-9a24fe6-complete-verified.tar.gz`; earlier attempts are preserved losslessly with correction receipts. [Archive receipt](docs/reports/hu20-zero-mass-fallback-artifacts/archive-receipt.json.gz) records identity and retained originals. Later publication/CI/final docs are sealed in a separate member-hashed closeout ZIP in the same folder. Local ZIP/member readback is verified; remote cloud byte readback is not claimed. Never delete or evict synced files.

## PR174 Luna versus Shield — October 6, 2026

PR174 evidence remains locally retained; no archival or cleanup has been performed. The active M1 root `/Users/dberweger/Local/luna-shield-chrome/results/luna-shield-20261006/` retains isolated preflight evidence, the primary private SQLite/access token, server/resource logs, run receipt, actual final HTTP export/history, strict-reporter rejection, raw audit and native evidence builder. The independently launched player's original rollout remains in `/Users/dberweger/.codex/sessions/2026/10/06/`, session `01a11070-fb6b-7961-a30b-e904731fd0af`; do not publish its reasoning or unfiltered contents. The [sanitized report](docs/reports/luna-shield-chrome.md) records 400 replayed hands and Luna −82 BB, with the failed browser protocol and missing planned metadata retained. The [manifest](docs/reports/luna-shield-chrome/manifest.json) identifies public evidence and hashes private inputs without publishing them.

The policy hardlink at `luna-shield-chrome/models/O-2026100601.average.jsonl.gz` depends on #171's retained original `/Users/dberweger/Local/hu20-cfr-plus-pr171-20261006/policies/O-2026100601.average.jsonl.gz`. The active PR174 root and retained PR171 input dependencies remain protected. No archival, moving or cleanup was performed. Eventual cleanup destination: `PR-174-Luna-Shield/20261006/` under the [project research folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s), with private access controls for journals and transcripts, manifests and restoration provenance. This is a planned destination, not an upload or a deletion authorization.

## PR175 — HU20 CFR+ floor control

[PR #175](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/175), [full report](docs/reports/hu20-floor-control.md), [frozen protocol](docs/hu20-floor-control.md). Primary T−R1 **+8.12 [5.50, 10.73]**; direct shield−T **−14.26 [−16.90, −11.62]**; secondary O−R1 **+11.45 [8.83, 14.07] BB/100**. Every lineage matches its aggregate label. All 884,736 final plus 36,864 pilot hands and 4,581,025 actions independently replay, with raw-chip interval/label agreement. All primary half-widths are ≤4. Free M4 only, 10.46 minutes through final admission/play/report/audit. No training or release decision.

Research originals: M4 `/Users/dberweger/Local/hu20-floor-control-pr175-20261006/`, verified M1 retrieval `/Users/dberweger/Local/hu20-floor-control-20261006/`. Execution source `8bbfc457` (unchanged #173 direct tools); final root `202610061601`, excluded pilot `202610061501`; each family has 49,152 duplicate deal blocks. Canonical bundled plan SHA256 `7e1cf06d29b2988faf9c262f7c81bdf7c12c4c5e2b59c4da702046e9d4779036`. All ten exact input exports and all 18 retrieved raw traces match their manifest/auditor hashes.

Archive: `~/Local/Research-Cloud/PR-175-HU20-floor-control/hu20-floor-control-complete-20261006.zip`. The embedded `ARCHIVE-MANIFEST.json` pins every research member by size/SHA256, with every member verified by ZIP readback. It preserves input manifests and exact policy files, complete source, environment/native binary, pilot/final plans, raw actions/settlements, reports/audits, logs, preparation failures and `RESTORE-README.md`. The separate `archive-receipt.json` beside it records ZIP size/SHA256, manifest hash and verified member count; subsequent publication/upload receipts are closeout metadata. Restore into a fresh nonsynced folder, verify all members, extract `source-8bbfc457.tar`, and follow the archived independent-replay commands. Originals remain on both Macs; no synced file is deleted or evicted. Native upload status is recorded separately; no independent cloud byte readback is claimed.


Verified main ZIP: **147 members / 2,100,003,481 logical bytes**, ZIP **2,018,260,354 bytes**. SHA256 `f350afc81cb8090d24ca125413df8d58259e1d1787912193d13de291904a6962`; embedded manifest SHA256 `d7a4aaa73f704bdd586f32bb969d10bc88887a31647918c383e66435724af16f`. All members pass readback. Final publication/validation/upload receipts and updated report metadata are retained in a separate member-hashed closeout ZIP in the same folder.

## PR171 CFR+ / 0.4.0-shield — October 6, 2026 (audited failed release candidate)

[Implementation #171](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/171) merged at `60f516da6b24563a49e609636a31fa17ae719bde`. [Research #173](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/173), [frozen protocol](docs/hu20-cfr-plus-confirmation.md), [owner review packet](docs/reports/hu20-cfr-plus.md) and [complete diagnostics](docs/reports/hu20-cfr-plus-details.md). Three 1B-node lineages, nine 100M/500M/1B checkpoints and six native current/traverser-reach-average exports pass the floor-zero labels, complete-header, nonnegative-regret and source-hash checks. [Checkpoint/export manifest](docs/reports/hu20-cfr-plus-artifacts/checkpoints-manifest.json).

The frozen rule is not met: LBR +36.13 [17.55, 54.70], native pressure −22.82 [−31.83, −13.80] BB/100. New direct primary against shipped R1: **−8.61 [−11.69, −5.53] BB/100, worse**, all three candidate lineages individually worse; all nine secondary pairings retained. The scoped full-export turn/river comparison uses the identical pipeline and all 40 boards: E 2.715 [2.492, 2.932] → 0.916 [0.863, 0.970] BB; Q 2.679 [2.325, 3.058] → 0.326 [0.287, 0.363]. No release recommendation. **After reviewing the direct confirmation loss, the owner declined rc1 for this candidate. No release, tag or pre-release; v0.4.0 stays stable.**

Owned root on both Macs: `/Users/dberweger/Local/hu20-cfr-plus-pr171-20261006/`. Training source main `60f516d`; frozen panel/earlier direct evaluation `6c8b49a059c80bd942150f26bc25847aa830917d`; full-export native loss source `9334341c366c1d51200b9b06f03445f274966840`; fresh nine-pair direct source `c471678`. Source snapshots, native binaries, requests/references, environment, model hashes, guards, all logs and failed auxiliary attempts are retained.

| Evidence | Root / preparation | Size | Canonical plan SHA256 / verification |
| --- | --- | --- | --- |
| Frozen panels | `202610060801` | 411,648 hands | `0806a98d8d6dc1cd59125ff9d4c9d23ce8c862794777d482ad5c755b6308ee1a`; every action/settlement and interval independently verified |
| Earlier matched-pair exploratory direct | `202610060901` | 73,728 hands | `5a56af93eb25a8f99c7f9976b342de72e2c4aeb20d49f69489a3cc1b1cb9b5dd`; independently verified, kept separate |
| New shipped-R1 primary / nine-pair direct | `202610061101` | 884,736 hands | `116b6f049d687785836a8cea4d9657c3b020c83cf15839e84c8458d867bf863f`; every action/settlement and interval independently verified; maximum individual-pair half-width 3.44 BB/100 |
| Outcome-blind panel / direct pilots | `202610060701` / `202610061001` | 2,496 / 4,608 hands | Separate from confirmation; costs/variance inspected before sizing, all raw files retained |
| Scoped full-export turn/river | 40 frozen common #149 boards, fresh two-policy requests | 160 native metrics | Manifest SHA256 `6969cd3044cd852b1bcf370400064b2c7ddfec8e285b653acda9a0579d700082`; qualified binary `fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f`; metrics and paired bootstrap independently verified |

Total arena/direct confirmation and retained exploratory evidence: **1,370,112 hands / 6,612,090 actions** independently replayed. M4 transfer: **715 files / 9,486,820,764 bytes**, every source and retrieved SHA256 matches; both hosts' originals remain. Native-pressure descriptive partitions reconcile exactly to the frozen contrast; no translator and zero off-menu native-pressure bet sizes. Labels and the final sample were declared before scores; no outcome-driven extension.

The owner gives the tested CFR+ production averages the **friendly model name 0.4.0-shield**, short **0.4.0-s**, for future checks. It is not a software release version or Git tag. [Model identity](docs/reports/hu20-cfr-plus-artifacts/shield-model-identity.json) pins all three seeds and export SHA256s. The name refers to lower measured weakness on the tested probes; it does not claim lower full-game exploitability. The direct confirmation remains worse.

Verified destination: `~/Local/Research-Cloud/PR-171-HU20-cfr-plus/hu20-cfr-plus-complete-20261006.zip`, in the [Drive folder](https://drive.google.com/drive/folders/1QiQiGUu5EARluoId5dV4XHbpx5JDslWx). **1,065 source files / 14,320,364,078 logical bytes; ZIP 8,077,888,199 bytes**, SHA256 `8ec9dbee5fd392b058d8a5c33e00a81a414f95189562d8dd0f9e9c66348b982c`; embedded manifest SHA256 `d213bef5423b2b3a333e1f02ae51ab7fdbe0d0cb55448edc1b4b24f29cc06840`. Every source/member SHA256, size and ZIP readback passes. [Receipt](docs/reports/hu20-cfr-plus-artifacts/archive-receipt.json), [M4 transfer proof](docs/reports/hu20-cfr-plus-artifacts/m4-transfer-verified.json). The main ZIP records the research/report snapshot at `d1acba7`; later owner naming/publication decisions and final documentation/receipts are retained as closeout metadata. **Native Drive/FileProvider reports upload complete:** uploaded=true, uploading=false, no unresolved conflict and exact 8,077,888,199-byte size. [Native status](docs/reports/hu20-cfr-plus-artifacts/drive-native-upload.json). Current October 6 Drive metadata independently confirms archive ID `1kn-i1ZhOA1sftQPC0nYek1MI1XuJBOGS`, exact name/size/parent; see the cleanup record above for current acceptance and removals. No cloud byte re-download is claimed. Before reading/copying historical #149/#162/#165/#169 dependencies, owning PRs were checked as merged. Open PR work roots, all synced files and every original remain; no deletion or eviction.

## Find recent M1 runs — October 5, 2026

M1 uses the same `~/Local/Research-Cloud` shortcut and [project Drive folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s) as M4. Initial free space was **13.3 GB**. The primary poker checkout is about **23.2 GB**, including **19.7 GB of shared Git data**, which remains protected alongside unclassified `planning/` files. Open #146/#166 folders, worktrees and dependencies are excluded from archival.

| Merged PR | M1 originals, under `~/Local` | Drive archive / restoration |
| --- | --- | --- |
| [#149 — board pooling](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149) | `hu20-board-pooling-m4-retrieval-20261004`, `-02/-03/-04/-05-20261004`, compact `-06-20261005`; `hu20-board-pooling-engineering-20261004` | [M1 history folder](https://drive.google.com/drive/folders/13RWLf779O6DpKZQ2rBKuEg-PpHOH7xX8) · [310.7-MB supplement](https://drive.google.com/file/d/17woiEOQSDwHvyq6eNzLhm5U2cl0nCQan/view) plus [canonical M4 campaign](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view); exact per-file restore mappings; confirmed uploaded |
| [#162 — trainer bench](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/162) | `hu20-trainer-bench-monitor-20261005` | [M1 monitor folder](https://drive.google.com/drive/folders/1Kq-JMOj5u4Rlnvevvb8QXfWRb_nCSKzT) · [305-KB supplement](https://drive.google.com/file/d/1bC5YnXxAfvsmMJwM-5eD-bukXM6BTSZb/view) plus [canonical M4 bench](https://drive.google.com/file/d/1_6dRapLReZ9-sAMPaePNX-nehXowGwki/view); confirmed uploaded |
| [#164 — native trainer parity](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/164) | `hu20-native-gate-20261005` | [M1 native gate folder](https://drive.google.com/drive/folders/1MCnYU8wXQ0y2aTkWS9NVVaJ9aBDksRp3) · [1.34-GB whole archive](https://drive.google.com/file/d/13aotv71LxwK4QQ0x3XIpP4sz4m42K9B_/view); confirmed uploaded; pre-fix failures preserved |
| [#165 — native average / frozen arena](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165) | `hu20-native-average-20261005`, `hu20-v041-archives-20261005`, `hu20-v041-closeout-20261005` | [Existing PR-165 folder](https://drive.google.com/drive/folders/1IXMpL2f18EVb1FfoFTvAaOGIzdrRpeD2): campaign and audit/source archives already confirmed; fresh local archive hashes and cloud readbacks pass |

[Full M1 record](docs/artifacts/m1-drive-organization-20261005.md), [upload acceptance](https://drive.google.com/file/d/1JPuWeSbPIbMsHhXpyghuo0o6gaVyqKnP/view) and [retained prior Drive index](https://drive.google.com/file/d/1Zydk8goU3Hg4VUGbGPQ-pRCb1yGRpmR6/view) give exact hashes, original paths and restore commands. The three new archives preserve **14,614 source records / 42.18 GB logical bytes** using **1.65 GB of compressed archives**, with exact references to already accepted payloads. Each included payload and source hash was verified. Logical totals include repeated snapshots/hard links and do not measure reclaimable storage. Original files remain unchanged; no deletion or cache eviction.


## Find recent M4 runs — October 5, 2026

Both Macs use `~/Local/Research-Cloud` for [deepcfr-research-results](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s). The shortcut points to `~/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results`. Use the same archive folders from M1 or M4; the archives preserve whole runs, member manifests, failures, source provenance and restoration instructions.

| PR / experiment | M4 originals, under `~/Local` | Drive location / exact archive | Current status |
| --- | --- | --- | --- |
| [#149 — board pooling](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149) | `hu20-board-pooling-20261004`; `hu20-board-pooling-closeout-06-20261005` | [M1-board-pooling / M4-main-06](https://drive.google.com/drive/folders/1Ab8dBtC48zABbiO1AzDVS35GvmHxcLdz) · [16.80-GB whole archive](https://drive.google.com/file/d/1NNSkCO811USN6U9L2p6lBbti1-6Q9r2X/view) | Confirmed uploaded; staged SHA256 rechecked |
| [#162 — trainer bench](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/162) | `hu20-trainer-bench-20261005`; `hu20-trainer-bench-closeout-20261005` | [PR-162 / M4-run](https://drive.google.com/drive/folders/1mn_Qb3b_qVaFs1YR8a7f6nuytA_tX3Vq) · [1.75-GB whole archive](https://drive.google.com/file/d/1_6dRapLReZ9-sAMPaePNX-nehXowGwki/view) | Confirmed uploaded; staged SHA256 rechecked |
| [#163 — equity buckets](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/163) | `hu20-equity-buckets-20261005`; `hu20-equity-buckets-closeout-20261005` | [PR-163 folder](https://drive.google.com/drive/folders/1gwungsUq-b9uey8InGy0yOD2GbaMRhp-) · [1.99-GB whole archive](https://drive.google.com/file/d/1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T/view) | Confirmed uploaded; staged SHA256 rechecked; validation inputs retained |
| [#165 — native average / frozen arena](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/165) | `hu20-native-average-20261005` | [PR-165 folder](https://drive.google.com/drive/folders/1IXMpL2f18EVb1FfoFTvAaOGIzdrRpeD2) · [campaign](https://drive.google.com/file/d/1xenvcfa5QfhGutNe5xNtDhFLYRqKcyxI/view) · [audit/source](https://drive.google.com/file/d/1olH4RanIwqGhmPae7Pw8GtvIwRr0e14n/view) | Already confirmed; M4 raw results are preserved in the M1 campaign archive |
| [#166 — completed M4 river validation / pilots](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166) | `hu20-turn-search-river-20261005` | [PR-166 / M4-river-validation](https://drive.google.com/drive/folders/1KPsr-sitP-pn-xNdUt-i0WJV0S2-49_e) · [1.52-GB whole snapshot](https://drive.google.com/file/d/1AcdU_6DI-F4SABL1ell4GP82oa_Lskh6/view) | Open PR protected; previously copied completed-evidence snapshot uploaded; no further writes to its work root or dependencies |

[Organization and verification record](docs/artifacts/m4-drive-organization-20261005.md), [existing-archive acceptance receipt](https://drive.google.com/file/d/1s6pTmisHaD5wkSeMH6J8Sc1QSoFIkWbM/view), [river snapshot receipt](https://drive.google.com/file/d/1_l7jszyuwKg3dP9sbgHISNKXJmCrgSUJ/view) and [retained prior Drive index](https://drive.google.com/file/d/10zDyJkQgV6Xve0pQ7nMYkzv9dIf6lsS5/view) provide exact hashes, member counts and restore paths. Before further archival changes, check the owning PR’s current status. Open PR work roots and dependencies are protected; #166’s copied snapshot is a point-in-time backup only. Original local evidence and active inputs remain; this organization does not reclaim their disk space. Existing historical pending statements below describe their original staging time; the current status is in this table.


## PR165 v0.4.1 frozen arena — October 5 closeout

[Complete result](docs/reports/hu20-v041-arena.md), [all lineage/position and street tables](docs/reports/hu20-v041-arena-details.md), [independent audit](docs/reports/hu20-v041-arena-artifacts/audit.json). The **release rule is not met**: native-pressure O−R **+0.44 [−18.47, 19.34] BB/100** is inconclusive, not a measured regression; its lower bound fails > −10. LBR O−R improves **+30.83 [14.03, 47.63]** and the severe-scenario check passes. All 165,888 hands / 850,106 actions replay; every aggregate absolute estimate and all 52 contrasts independently match raw-chip arithmetic. No follow-up arena, release, tag or deletion.

Original campaign: M1 `/Users/dberweger/Local/hu20-native-average-20261005/` and M4 `~/Local/hu20-native-average-20261005/`, with frozen M1 runtime checkout `.claude/worktrees/deepcfr-repo-access-830ea6` and M4 `repo`. Runtime source `376763af4f52a5829f7aac463750e3c501c20cff`; canonical plan SHA256 `5d096197840097b93e56aca3ccbb480ac3335e9ef546c7faac358233ac18626f`, file SHA256 `3b98e9c467371b653a47d2ed4b7348525669a4ce0cc085a49d7779fe481ac053`. Both done markers and all twelve complete model results pass. M4's eight raw/result files are copied into M1's `arena/run/` with exact size/SHA256 [transfer proof](docs/reports/hu20-v041-arena-artifacts/m4-run-transfer.json); all twelve policy hashes pass on both hosts and all six checkpoint hashes match export provenance. M4's original files remain in place.

Both whole archives are **accepted uploaded** in [PR-165-HU20-native-average](https://drive.google.com/drive/folders/1IXMpL2f18EVb1FfoFTvAaOGIzdrRpeD2), through native Drive at `~/Local/Research-Cloud/PR-165-HU20-native-average/`. Native FileProvider reports uploaded=1, uploading=0, no unresolved conflict and exact document size; cloud metadata independently matches exact name/size/parent. [Upload receipt](docs/reports/hu20-v041-arena-artifacts/drive-upload-confirmed.json) preserves IDs and archive hashes. Source files, archive members and staged archive SHA256s verify with zero mismatches; no cloud byte re-download is claimed.

| Archive | Cloud file | Compressed bytes | Source members / bytes | SHA256 |
| --- | --- | ---: | --- | --- |
| `hu20-native-average-complete-20261005.tar.gz` | [campaign](https://drive.google.com/file/d/1xenvcfa5QfhGutNe5xNtDhFLYRqKcyxI/view) | 3,166,550,740 | 123 / 3,166,550,515 | `b5b43d83cdfb9e9b25679eef94b9350c3ef4a76f8c4dad4d076090bb2c6aa898` |
| `hu20-v041-audit-closeout-20261005.tar.gz` | [audit/source snapshot](https://drive.google.com/file/d/1olH4RanIwqGhmPae7Pw8GtvIwRr0e14n/view) | 156,392,346 | 35 / 156,681,242 | `dcfd4fbc1fa360dd765f6db0481353a64646d1893497d4d71b2060b4f89083c5` |

Separate verified archives remain at M1 `/Users/dberweger/Local/hu20-v041-archives-20261005/`. The campaign archive preserves every training checkpoint/policy, pilot, timing, raw hand, tail, worker/launch log, frozen plan, merged reporter output and native binary under `M1-hu20-native-average-20261005/`. Manifest SHA256 `094953b982c8995dd3b4e17668850725d2696f42f72c169ac70eb8e74f831f3c`; [receipt](docs/reports/hu20-v041-arena-artifacts/campaign-archive.json), [member manifest](docs/reports/hu20-v041-arena-artifacts/campaign-archive-manifest.json).

The closeout snapshot preserves the exact independent audit/renderer/archival scripts, full-precision audit, logs, environment, both hosts' policy hash checks, M4 metadata/launch logs, transfer receipts, owner closeout instruction, report drafts and a complete frozen Git source archive under `M1-closeout/`. Manifest SHA256 `a7d61294d4a9f20827d9f79f811258395af51151be590d44a5b66442f2a5cdda`; [receipt](docs/reports/hu20-v041-arena-artifacts/closeout-archive.json), [member manifest](docs/reports/hu20-v041-arena-artifacts/closeout-archive-manifest.json). This snapshot precedes final upload/PR publication receipts; those compact supplements and final reports are retained beside the whole archives in Drive and in this PR.

Restore each archive into a fresh local directory with `tar -xzf <archive> -C <fresh-directory>`; verify every extracted `ARCHIVE-MANIFEST.json` member SHA256 before analysis. The embedded `RESTORE-README.txt` maps directory prefixes to original paths. To reproduce the reporter, use the preserved source with `PYTHONPATH=. python -m scripts.evaluate_hu20_v041_arena report --plan <restored>/M1-hu20-native-average-20261005/arena/plan.json --out <restored>/M1-hu20-native-average-20261005/arena/run`. To reproduce the independent audit, run preserved `M1-closeout/audit.py --plan <same-plan> --run <same-run> --out <new-audit.json>` with that source on `PYTHONPATH`. All original campaign/checkpoint/policy inputs, retrieved records, source snapshot and separate local archives remain; no eviction or deletion is authorized by this closeout.

## PR162 HU20 trainer bench — October 5 closeout

[Report](docs/reports/hu20-trainer-bench.md): both folds completed 3M iterations and all 40 held-out roots scored the 12 frozen policies. Production/opponent-sampled Q=0.9514/0.7308 classifies a poor CFR fixed point in v1 under the unchanged rule. Runtime commit `74ba3202396128a3823b654d05d0b002b3c4ceed`; no native failure, resource breach or relaunch; $0 paid compute. [Independent bootstrap verification](docs/reports/hu20-trainer-bench-artifacts/independent-summary-verification.json) reproduces every E/Q estimate and interval, and [raw verification](docs/reports/hu20-trainer-bench-artifacts/RAW-VERIFICATION.json) checks every native metric, paired #149 reference and request/response/policy hash.

M4 originals remain at `/Users/dberweger/Local/hu20-trainer-bench-20261005/`. Whole verified archive: `/Users/dberweger/Local/hu20-trainer-bench-closeout-20261005/hu20-trainer-bench-m4-complete-20261005.tar.gz`, **1,745,796,815 bytes**, SHA256 `67d51196af43202d6e2ebff5222ccfd6a264c1d9d0eb4efdc7ccfa1fa78d3dd3`. The archive verifies **2,809 file members / 5,663,368,615 logical bytes**; manifest SHA256 `62aefd4d8b9dc3e4380612f772d513384c2049e6fc708d3bba1334230cf0a049`. It preserves smoke evidence, all checkpoints and recovery states, every raw native result/log/request, frozen source, the selected lineage's #149 inputs/references, monitoring snapshots/comments and the first archival path-error attempt. Git administration and reproducible caches/environments are excluded; full upstream #149 source and other lineages remain in the separately indexed complete PR149 archive. [Archive receipt](docs/reports/hu20-trainer-bench-artifacts/ARCHIVE-RECEIPT.json).

A separate hash-verified copy is staged through M4 native Drive desktop at `~/Local/Research-Cloud/PR-162-HU20-trainer-bench/M4-run-20261005/` in the [designated research folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s), with `ARCHIVE-MANIFEST-20261005.json`, `ARCHIVE-RECEIPT.json`, `PROVENANCE.json`, `RAW-VERIFICATION.json` and `RESTORE-README.txt`. [Staging receipt](docs/reports/hu20-trainer-bench-artifacts/drive-staging.json): staged bytes match, M4 free disk 24.89 GiB, **cloud upload was originally unconfirmed; October 5 native acceptance, current cloud name/size/parent and staged SHA256 checks now confirm the archive**. No original or synced payload was deleted or evicted. The retained failed staging directory is `/Users/dberweger/Local/hu20-trainer-bench-closeout-20261005.partial-1791202039/`.

Restore the whole archive into a fresh directory, verify its SHA256 and every manifest member, and use `PROVENANCE.json` plus `RESTORE-README.txt` to rebind absolute input paths only in a restored working copy. The `hu20-trainer-bench-20261005/` prefix maps to the original M4 work root; `inputs/` contains copied immutable dependencies and archival/monitoring provenance. M1 reporter input/verification and the separate compressed retrieval are retained at `/Users/dberweger/Local/hu20-trainer-bench-monitor-20261005/`; the [M1 retrieval receipt](docs/reports/hu20-trainer-bench-artifacts/retrieval.json) verifies the full archive SHA256 and all 2,809 file members with zero mismatches.

## PR163 full-deck equity buckets — October 5 closeout

M4 originals: `/Users/dberweger/Local/hu20-equity-buckets-20261005/`. Frozen builder commit `4ade72cc18e512ed9f11fa6afa83c701a55069c8`; all six `{flop,turn,river}-k{50,200}.bin` tables, `summary.json`, original `SHA256SUMS`, status/host/test/memory files and complete `build.log` are retained. All 16 original files / **2,768,097,096 bytes** independently SHA256-verified on M4 before rental termination; the builder's ten manifest entries pass. `verification/` retains the exact reader, sanity/retrieval scripts and receipts, supplemental full-file hashes and complete pod MCP responses. [Report and pending validation plan](docs/reports/hu20-equity-buckets.md), [retrieval](docs/reports/hu20-equity-buckets-artifacts/retrieval.json), [sanity](docs/reports/hu20-equity-buckets-artifacts/sanity.json), [termination](docs/reports/hu20-equity-buckets-artifacts/termination.json).

Whole archive: M4 `/Users/dberweger/Local/hu20-equity-buckets-closeout-20261005/hu20-equity-buckets-20261005.tar.gz`, **1,991,949,530 bytes**, SHA256 `03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca`. All **29 source members / 2,768,143,080 logical bytes**, archive members and retained originals match; manifest SHA256 `a5d64a7882da34840dad53905e4a03284f822487732ce691ecc82cdf1585bdef`. Restore by extracting the archive into a fresh directory, then verify every `ARCHIVE-MANIFEST.json` member plus the original `SHA256SUMS`; `RESTORE-README.txt` records the directory prefix.

A separate APFS copy is hash-verified and staged through native Drive desktop at `~/Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz` in [the PR163 archive folder](https://drive.google.com/drive/folders/1gwungsUq-b9uey8InGy0yOD2GbaMRhp-). Manifest, restore README and [staging receipt](docs/reports/hu20-equity-buckets-artifacts/drive-staging.json) accompany it. **Upload was originally unconfirmed; October 5 native acceptance, current cloud name/size/parent and staged SHA256 checks now confirm the archive.** Original table folder and separate local archive remain; no deletion or forced cache eviction occurred. M4 free space after staging: **32.505 GiB**, above 20 GiB. These tables are active inputs for validation proposed on #163, awaiting explicit owner approval. Pod `90gceq0t9mqz6q` is gone; the four older exited pods are unchanged. Estimated compute $0.148889. No table files enter Git; no validation, training, new rental or merge has run.

## October 4 organization and native Drive uploads

[Canonical repository index](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/main/RESULTS_INDEX.md). Repository-relative report links below resolve there; the Drive folder links work independently.

This is the existing research index. Its September inventory and October 2 receipts remain below; the current locations and upload states in this section supersede those earlier snapshots.

Both Macs use Google Drive desktop in streaming mode, with `~/Local/Research-Cloud` pointing to the same [research folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s). Whole archives and experiment folders go through the desktop app. The connector's observed 536,870,912-byte ingestion ceiling does not apply to this route; no research payload was split for the native uploads.

- **Organization verified:** 207 existing entries moved into [Historical-experiments](https://drive.google.com/drive/folders/1AIsc7LBHc7ziwrpOuwMuCPnvZpCJS1Zr), and five cleanup records into [Archive-receipts](https://drive.google.com/drive/folders/10cG2BTBZLcE7lO10TgBZ0Ud32quaf_BX). IDs and existing links are preserved. The [relocation receipt](docs/artifacts/native-drive-relocations-20261004.json) records every original parent, destination and verified item ID.
- **M4 upload queue:** nine closed scopes, **28,756 files / 96,459,366,205 logical bytes (89.83 GiB)**, hashed before relocation into the streamed folder. Includes closed training and flop/turn evidence. Local payloads remain under Drive management while uploading.
- **M1 upload queue:** four closed scopes, **24,637 files / 37,983,420,181 logical bytes (35.37 GiB)**, likewise hashed and staged. The first attempt stopped before moving any file because newly created cloud folders had not appeared locally; after those folders synchronized, staging completed. The failed attempt is retained.
- **PR136 upload queue:** the intact `hu20-500m-campaign-20261001-archive-20261002.tar.gz`, **23,509,934,825 bytes**, is uploading from M1. SHA256: `c0dbb3879eeae5f89be888ca3b29a6c571f6adb214c9d23c6095d0cab007636d`. The separate M4 originals remain until upload confirmation.
- **Already confirmed:** the 14,041,643,443-byte branching archive, the overnight archive, PR144's six archives and 18 earlier M4 archives. Their previous cloud IDs remain valid; the branching archive now lives under `Historical-experiments/`.
- **Pending means pending:** native staging is not cloud-upload confirmation. An hourly archival follow-up checks completion and remaining duplicate/input dependencies. The [compact staging receipt](docs/artifacts/native-drive-staging-20261004.json) records exact source paths, destination paths, counts and manifest names. Logical totals include separately retained evidence and do not measure unique or physical disk use.

### October 4 scheduled follow-up and cleanup

The index/organization change passed all CI and merged in [PR #151](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/151). The [cleanup progress receipt](docs/artifacts/drive-cleanup-progress-20261004.json) records the following subsequent actions:

- **M1:** removed 26 abandoned, unindexed Git temporary packs (6,362,104,120 bytes), 12 separately retained staging archives with fresh cloud ID/name/size readback and matching archive SHA256 (3,522,030,679 bytes), and four obsolete connector parts (2,097,152,000 bytes). The parts matched the corresponding bytes of the intact recovered archive. Total removed logical bytes: **11,981,286,799 (11.16 GiB)**. Valid Git packs/indexes and readable HEAD trees remain intact. Physical free-space changes also depend on APFS sharing and ongoing Drive cache activity.
- **M4:** removed **1,826 files / 6,418,060,050 bytes** from separately retained, cloud-confirmed legacy archive roots after checking each member hash, current stat identity, tracked files, symlink/input dependencies and open handles. The complete per-file restore receipt remains `~/Local/research-archive-preparation-20261002/m4-remaining-duplicate-cleanup-20261004.json`; original relative member paths and cloud archive IDs are retained. M4 had **42.54 GiB free** after this cleanup. No synced Drive payload was deleted.
- **PR136 upload incident:** Drive logs record a completed CancelUpload at `2026-10-04T08:24:40Z`; the initiating actor is unknown. The file was absent from the upload folder/cloud listing but retained intact in Drive desktop's `canceled_uploads` recovery directory. Its full 23,509,934,825 bytes rehashed to the original SHA256 above. It was moved back into the native upload folder at about `09:35 UTC`, without rebuilding research or making another archive. **Upload remains pending**, and M4 originals remain retained. The recovery receipt is included in the compact progress record; do not confuse queueing with cloud completion.

Native upload queues are still processing. The hourly follow-up remains active; no cache eviction, new science or rental was performed.

### Later October 4 cleanup

[PR #152](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/152) passed all checks and merged. Ten older blueprint experiment folders now use their existing `Historical-experiments/` cloud folders through links at the original M1 paths. Removed **149 separate local duplicate files / 3,547,485,700 logical bytes (3.30 GiB)** after exact cloud relative-name/size readback, unchanged September inventory checks, source hashes, tracked-file checks and open-handle checks. `blueprint-04-ops` code remains local. Full per-file cloud IDs/hashes are retained in Drive `Archive-receipts/M1-blueprint-duplicate-cleanup-20261004.json`; the [compact cleanup receipt](docs/artifacts/blueprint-drive-cleanup-20261004.json) records every affected scope. No synced payload was removed.

The first cleanup validation compared access time, which changed during its own hash read; it stopped before any payload change. Its receipt/helper were retained, and the corrected check uses size, modification time, inode and device. All subsequent removals completed.

At the hourly check M4 had **46.67 GiB free**. M1 had **16.36 GiB free** after cleanup while Drive was still filling its streaming cache; removed logical bytes do not promise equal physical reclamation. Both native upload queues are progressing, and PR136 remains pending with M4 originals retained. Hourly chat updates continue.

### Whole archives by PR and run — owner-approved update

The owner subsequently approved converting the pending small-file batches to **one whole `.tar.gz` per distinct run inside its existing PR/experiment folder**. This supersedes the earlier preference limited to unstaged experiments. The [archive transition receipt](docs/artifacts/run-archive-bundling-20261004.json) lists every planned archive and its original source.

- **M4:** 37 run/experiment archives are building. All nine sources passed membership/link/open-handle checks before relocation to `~/Local/research-native-originals-m4-20261004/`; old working paths point there. Their loose upload folders were withdrawn, with every original payload retained. Whole archives enter the unchanged PR parent folders only after source hashes and every archived member verify.
- **M1:** 22 archives are building from the four existing native source folders in place. The first attempt to export a large native folder blocked and appeared to hydrate it; that worker was stopped before the first rename completed. Sources remain unchanged. The replacement avoids folder export and reads only the selected members.
- **Packaging incident retained:** after verifying 26 M4 archives, the helper rejected tarfile’s automatic hardlink metadata for an existing regular hard-linked file. The resumed helper disables automatic inode encoding, applies explicit content deduplication and the same full member hash checks, skips completed archives and retains the failed partial. No source was removed.
- **One shared calibration archive:** M1 `m4-calibration-03-complete` and M4 `calibration-03` have **17,850 identical relative names/sizes/hashes / 29,491,888,105 bytes**. Upload `M4-closed-research--calibration-03-20261004.tar.gz` once from M4; preserve both source locations until upload acceptance. No second archive is made on M1. For M1 restoration, extract `M4-closed-research/calibration-03/` and point the old `m4-calibration-03-complete` run path at it; names below that prefix and all member hashes are identical.
- **Restoration:** each archive includes `ARCHIVE-MANIFEST.json` and `RESTORE-README.txt`. Extract with `tar -xzf ARCHIVE.tar.gz` into a fresh local directory, then verify member SHA256 entries. Exact duplicate contents inside an archive may use standard tar hardlinks, preserving all relative paths; treat restored evidence as read-only. Private credentials remain outside the archives.

Building or queueing an archive is **not upload confirmation**. M4 originals remain separately retained; M1 originals remain in their existing native folders pending archive acceptance and later safe reconciliation. Existing confirmed historical archives and PR136's whole upload are unchanged. Free-space guards reserve 8 GiB, and no cache is forcibly evicted. Hourly chat updates continue. [PR #153](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/153) passed all checks and merged before this update.

### October 4 12:30 UTC follow-up

[PR #154](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/154) passed all checks and merged. The [hourly progress receipt](docs/artifacts/drive-archive-progress-20261004-1230.json) distinguishes packaged files, exact cloud name/size matches and accepted native completion.

- **M1:** all **22 run archives / 1,687,920,071 compressed bytes** are built and every source/member hash verified. Originals remain in place. The earlier read stopped when FileProvider changed one timestamp by 30 ns; content SHA256, size and inode stayed identical. A retained incident proves the exact double-seconds-to-timespec conversion. The new helper permits only that exact timestamp transformation after content validation; other identity changes still stop it. The failed partial is retained. No scientific work reran.
- **M4:** **36/37 archives / 29,743,314,324 compressed bytes** are built. The last `day-paper-16k-2026091902` run waits for sufficient free disk, retaining its originals. The prepared resume helper has not launched. A separate **427,324,620-byte PR145 six-input archive** finished with all six member hashes verified; it is queued in the existing PR145 folder, with originals retained.
- **PR136:** the full **23,509,934,825-byte** archive is now visible in the correct cloud folder as [this exact item](https://drive.google.com/file/d/19mPzvR0fF1vHzr-cRMsueTlHiQtthOcB/view). Its local native item ID matches. The app still reports one pending file, and explicit native completion for this archive is unconfirmed; separate M4 originals remain protected.
- **Storage:** approximately **11–12 GiB free per Mac** during this check while native uploads/cache continue. No originals were removed. Accepted upload cleanup will provide headroom; no forced eviction or chunking is used. Hourly updates continue.

### Accepted uploads and safe cleanup — October 4 owner-requested check

This confirmation supersedes the pending states in the historical snapshots above. [PR #155](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/155) passed all checks and merged. The [accepted-upload receipt](docs/artifacts/drive-upload-confirmed-20261004.json) records every current archive ID, name, size, source hash and restore prefix.

- **All 22 M1 run archives accepted:** 1,687,920,071 compressed bytes; native desktop reports Up to date/Synced and the current cloud names/sizes match every verified archive.
- **All 38 M4 archives accepted:** the 37 run archives plus the separate six-file PR145 input archive, totaling 38,268,264,296 compressed bytes. Read-only native FileProvider metadata reports `isUploaded=1`, `isUploading=0`, no unresolved conflict and the exact document size for each item. Native item IDs match the cloud IDs and exact names/sizes. No cloud re-download was required.
- **PR136 accepted:** the intact 23,509,934,825-byte archive is [confirmed uploaded](https://drive.google.com/file/d/19mPzvR0fF1vHzr-cRMsueTlHiQtthOcB/view). Its earlier cancellation/recovery and all failed attempts remain documented. The separate M4 originals were removed only after upload acceptance and per-member/dependency checks: 1,218 files / 23,524,835,844 logical bytes.
- **M4 run originals cleaned:** 28,758 separate local files / 96,886,556,351 logical bytes removed after member SHA256, unchanged stat, tracked-file, symlink-dependency and open-handle checks. The 38 archive IDs and original paths remain in restore stubs and the [full removal journal](https://drive.google.com/file/d/1PE25ZTR-0fYepxNRn5rauhEWa16W0KOd/view). Keys, coding source, valid Git/environment data and cloud payloads were preserved.
- **M1 canceled duplicates cleaned:** five abandoned copies outside the synced folder, 84,502,367,174 logical bytes, matched already accepted PR136/branching/overnight archives by exact SHA256 and identity. Their receipts and restoration links are retained in [Archive-receipts](https://drive.google.com/drive/folders/10cG2BTBZLcE7lO10TgBZ0Ud32quaf_BX).
- **Measured physical free storage after cleanup:** M4 **106,754,224,128 bytes (99.42 GiB)**; M1 **52,697,604,096 bytes (49.08 GiB)**. Logical removed bytes can share APFS blocks, so these figures are actual filesystem measurements rather than a subtraction estimate.
- **Remaining work:** inventory smaller closed evidence outside this batch, preserve current input dependencies until restore mappings are established, and reconcile the four M1 native raw-backup folders with their accepted canonical archives. These managed cloud folders remain intact; no synced data was deleted to free cache. Hourly follow-up and chat updates continue.

### Remaining evidence and raw-backup organization — October 4 13:37 UTC follow-up

[PR #156](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/156) passed scope/full0/full1/test/GitGuardian and merged. The [remaining-evidence receipt](docs/artifacts/drive-remaining-evidence-20261004.json) records this follow-up.

The four redundant M1 raw folders now live under `Historical-experiments/`, named `PR148--M1-closed-research--raw-backup-20261004`, `PR148--M1-exact-turn-evidence--raw-backup-20261004`, `M1-board-pooling--closed-evidence--raw-backup-20261004` and `M1-alias-audit--closed-evidence--raw-backup-20261004`. IDs, sharing and contents were preserved. Cloud parent/name readback and native folder IDs verified all four moves; original working aliases now point to the new native locations. Canonical whole archives remain in their PR folders. No raw cloud payload was deleted or large folder exported. [Organization receipt](https://drive.google.com/file/d/1uIf5ZCn61A3WCXeLd9eNbcqHFzIowr5d/view).

- **PR129 parity evidence:** a verified whole archive of 91 members / 173,805,058 source bytes from the M1 platform pilot and RunPod retrieval, `PR129-platform-parity-20261004.tar.gz`, is [confirmed uploaded](https://drive.google.com/file/d/1utMZPG-zV4lMIwgI7ouw0l2EJyyjOoKC/view) in [PR-129-platform-parity](https://drive.google.com/drive/folders/1SHrAg6PQV5wdG-tV3-WEu9AeO5eil5ft). Compressed size 149,683,235 bytes; SHA256 `95f1099b2587d6002b32e08ac7aed8c67c0b1e90b93588a8f4ce4e5289580023`. Native uploaded/not-uploading/no-conflict/exact-size status and current cloud name/size confirm acceptance. Both original sources remain intact pending dependency-aware cleanup.
- **PR145 earlier retrieval snapshots:** the M1 retrieval snapshot contains 819 result files / 2,469,278,300 bytes without matching content in the newer accepted M4 result archive, spanning earlier flop and turn attempts. They remain protected. The complete intact flop run is now verified in a separate whole `PR145-flop-retained-attempts-20261004.tar.gz` (501 members / 565,822,969 source bytes; 18,600,998 compressed bytes; SHA256 `500b269afce4799d313a9c49788b367f4f49e310a8a04723ba7cc609a6b3dff6`) and queued in the existing PR145 folder; upload acceptance is pending. A separate verified `PR145-turn-retrieval-snapshot-20261004.tar.gz` preserves the earlier turn run (507 members / 1,919,529,239 source bytes; 108,091,354 compressed bytes; SHA256 `8dd3d8621d8628a1a04e01947c0abdaf3eeb136bb6c332ce4c3209b7e0374422`) and is queued with originals retained. The other 193 result files / 16,075,493 bytes match accepted archive members but also remain retained during complete restoration/dependency mapping. Tool/source, local retrieval metadata and six input files remain protected. The first shell background launch ended before its initial status or any archive/source change; the incident is retained and the detached packaging launcher succeeded.
- **Other-owner evidence and coordination incident:** PR149 had separately resumed on October 4. Its preparation stopped when archival removed the unopened third-lineage export from the shared M4 input directory. Open-handle and narrow process checks missed a future input declared in the active plan; the earlier owner statement that all science was closed was stale. Eighty board exports completed, but qualification/native solver/main jobs never started. The owning task restored all three exact average exports from the retained M1 copies into isolated M4 inputs and verified the hashes on both hosts. No file was lost and no scientific restart was performed by archival. The [incident/restoration record](docs/artifacts/shared-input-archival-incident-20261004.json) is preserved. Protect PR149 preparation, both dated retrieval folders and its declared inputs through the owning task's verified closeout. Before any further original cleanup, check active task manifests/approval/continuation state for every declared input, including files without open handles. Coding fixtures, private keys and valid Git/environments remain available.

Hourly checks continue upload acceptance, remaining inventory, restore mapping and safe separate-original cleanup. These pending archives do not change the accepted state of the earlier 22 M1 / 38 M4 / PR136 batch.

### Retrieval copies accepted and cleaned — October 4 14:37 UTC follow-up

[PR #157](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/157) passed all required checks with no findings and merged. The [retrieval cleanup receipt](docs/artifacts/drive-retrieval-cleanup-20261004.json) supersedes the pending states above.

- **PR145 snapshots accepted:** [retained flop attempts](https://drive.google.com/file/d/1jEmvLPJsmnjmh-zZSrVevA0ura17dGlK/view), 18,600,998 bytes, and [earlier turn retrieval](https://drive.google.com/file/d/1GNyipWYxTImoXI9gJpkIzMSbCeBngEOk/view), 108,091,354 bytes. Both have native uploaded/not-uploading/no-conflict/exact-size proof and current cloud name/size/parent readback. Their member manifests, SHA256 and restoration prefixes remain embedded.
- **Retrieval provenance accepted:** [whole metadata bundle](https://drive.google.com/file/d/1Z1JnP3CrveLDfVpTUsxxe-WAXBZPoJbk/view), 146,598 bytes, SHA256 `30040539276a5f57d5564de8d7c48a506b2dddf583cb6317ef8c46ff4462c355`, preserves the source inventory, verification, earlier cleanup records and build-turn log. [Validation provenance](https://drive.google.com/file/d/1ssb70KWXwUsf2qltsm8phum_leMsSjIJ/view), 1,326 bytes, SHA256 `d0e4f6c50f261f83e62516a66611b7f4c6c3ce79f6509e7fd06dc48ea5626828`, preserves four additional top-level validation/preflight logs, including the empty failed-attempt logs. Every archive member was verified; small original metadata/logs remain for restoration.
- **Local duplicate cleanup complete:** 1,099 closed PR129 parity and PR145 flop/turn files, **2,659,157,266 logical bytes (2.48 GiB)**, removed after accepted upload proof, source hash/stat, tracked-file, open-handle, external symlink and active-task future-input checks. Zero retained candidates. Each original result root has an `ARCHIVED-RESTORE-20261004.json` stub. [Compact receipt](https://drive.google.com/file/d/1wXZF-2AXJ5A7GvwdZgRJ1KePkBKN6K4E/view) and [complete per-file restoration journal](https://drive.google.com/file/d/1-qgLdo0RMbxhPovbnkymMRJEl-L7bsyD/view) identify exact source paths, member hashes and cloud archives. No synced payload was removed.
- **Measured free storage:** M1 **50,607,751,168 bytes (47.13 GiB)** after this cleanup; M4 **102,100,340,736 bytes (95.09 GiB)** at this scheduled check. Cache activity and the separate active campaign can change these figures.
- **Active PR149 remains protected:** its owning task has resumed from isolated hash-verified inputs; the third lineage is exporting. The earlier interruption/restoration record remains unchanged. Both M1 retrieval roots, original six input exports and the M4 campaign/isolated inputs remain protected through verified owning-task closeout. Archival does not launch or resume science.
- **Remaining inventory:** metadata found older result roots in the averaged-extraction worktree (1,385,757,798 bytes), postflop-replication (47,271,792), local-CFR (11,204,339), blueprint-search-followup (31,095,006) and fresh-install smoke/replay (3,163,375). Archive coverage, ownership and future dependencies must be checked before moving or removing these. Git, environments, default inputs, private credentials and unrelated projects remain available. Hourly checks/chat updates continue; the selected batch is complete, not the entire archival task.

### Older worktree evidence — October 4 15:38 UTC follow-up

[PR #158](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/158) passed all required checks with no findings and merged. The [worktree archival receipt](docs/artifacts/drive-worktree-archives-20261004.json) records the next admitted scopes and exact source/member identities.

Six previously uncovered M1 scopes are now accepted in Drive with exact cloud ID/name/size/parent readback after archive member verification:

| Whole archive | Cloud file | Bytes |
| --- | --- | ---: |
| PR105 M1 search-followup log supplement | [archive](https://drive.google.com/file/d/1x1Rzb0awwECqiDBTpu0vsft-bYiziyKO/view) | 939 |
| v0.4.0 fresh-install validation | [archive](https://drive.google.com/file/d/1NGbjkLSpOGPWlDjCFbLujCChF4vqyFbx/view) | 382,028 |
| PR116 M1 scaling supplement | [archive](https://drive.google.com/file/d/1loH7PI8KE8iVnVCEgJUKCxNRHHdd5WUi/view) | 546,185 |
| PR113 M1 TP20 resource preflight | [archive](https://drive.google.com/file/d/1XKIC3ZU_TkEuxZvpHSjVGJpAK69QyUAC/view) | 4,246 |
| PR113 M1 demo evidence | [archive](https://drive.google.com/file/d/19MasgLs-yY50TA9KsQv0xw1e4SJ53LtI/view) | 50,098,498 |
| PR112 M1 human smoke evidence | [archive](https://drive.google.com/file/d/1ydTgsD-KTrVYe5fqAt2SUIYXWnrL67oz/view) | 2,516 |

The 43-member scaling supplement and one-member search log supplement preserve M1 byte variants absent from earlier M4 archives; original failure/preflight/recovery provenance remains retained. The [accepted supplement receipt](https://drive.google.com/file/d/19VnUbGifCHbpb6ykVX_O1OmBKknLI51I/view) pins archive SHA256, parents and counts. Six separate staging archive duplicates outside Drive, 51,034,412 bytes, were removed only after accepted upload/hash/stat/open-handle checks. Demo/default and original training-parent inputs stay local for coding and future dependencies.

Cleanup removed **234 closed worktree files / 1,155,068,143 logical bytes (1.08 GiB)** from scaling, local-CFR and search-followup roots. Each matches either an already accepted historical archive member or the new exact supplement; current source hashes, stat identity, Git tracking, external symlinks, handles and PR149's future-input plan were checked. No retained candidates; restore stubs and the [complete per-file journal](https://drive.google.com/file/d/1oYI1s6iX1ktCKtbjtpmMvH4n3-8ck5_o/view) record the original namespace, SHA256 and cloud archive/member. [Cleanup receipt](https://drive.google.com/file/d/1WdO0dELyuqbhJ0PrhKPjsmukNVNO78_3/view). No synced Drive payload was removed.

Three additional M4 closed v0.4.0 validation projects are now whole member-verified archives in Historical-experiments and **accepted** by native uploaded/not-uploading/exact-size plus cloud metadata:

- [Final candidate validation](https://drive.google.com/file/d/1EvNvbDwG5eSyMIZwScOsSa16fm_9uDhE/view): 221 members / 45,748,783 source bytes; 41,553,141 compressed bytes.
- [Release validation](https://drive.google.com/file/d/1dznPFtSqhvw3arGSdaELUKi6VQ8aS2Y1/view): 303 members / 46,256,373 source bytes; 41,926,664 compressed bytes.
- [Benchmark candidate validation](https://drive.google.com/file/d/1NxYZfSkL93YqzRSPoew-oWDOonCuGsFU/view): 1,466 members / 117,258,316 source bytes; 88,929,227 compressed bytes.

All three M4 sources remain local pending closure/dependency/restore review. Reproducible browser/node/build dependencies were excluded; archived member counts differ from raw directory totals for that reason. Packaging worker 45080 has closed; do not duplicate these archives.

Measured M1 free space after worktree cleanup was **27,135,582,208 bytes (25.27 GiB)**, and M4 after release packaging **100,750,036,992 bytes (93.83 GiB)**. M1 DriveFS currently accounts for about 22.43 GiB allocated storage and the second PR149 retrieval about 4.39 GiB; automatic streaming cache reclamation remains gradual, with no forced offload.

PR149's owning task completed 120/120 exports, but lock-only qualification exceeded 5 GiB and stopped before main execution. Its 3,029 evidence files are retrieved/hash-verified; monitoring is paused awaiting memory engineering and explicit owner readmission. This is not permission for archival to delete its campaign, isolated/original inputs or any dated retrieval copy. Preserve all future dependencies through owning-task closeout. Remaining M4 inventory includes older primary blueprint roots (333,924,067 bytes, historical folders exist) and web smoke/benchmark evidence (22,238,751 bytes including browser dependencies); classify exact archive coverage and coding dependencies before further cleanup. Hourly follow-up remains active.

### Smaller M4 evidence — October 4 16:39 UTC follow-up

[PR #159](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/159) passed scope/full0/full1/test and GitGuardian with no findings and merged. The [small archive receipt](docs/artifacts/drive-m4-small-archives-20261004.json) records the new accepted files and cleanup proof.

Seven distinct pilot/web runs are now whole archives in Historical-experiments, **accepted uploaded** with native status and exact cloud ID/name/size/parent proof. They contain 991 source members / 21,973,467 source bytes and 6,390,066 compressed bytes. Browser dependencies, code, environments, private files and active research were excluded; original run files remain pending dependency/restore review.

| Whole archive | Cloud file | Bytes |
| --- | --- | ---: |
| `M4-blueprint-pilot-v1-diagnostic-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1alfEi4jF_e5W1L0hli0vESmr1QFpxRHF/view) | 60,710 |
| `M4-blueprint-pilot-v1-first-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1ukZ0ujrJj8Sws7MMg6UHRBWOoKGcETHq/view) | 41,987 |
| `M4-blueprint-pilot-v1-full-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1vWUg4DjHlXJVVDHZYbl8qmjXzK2pEwJt/view) | 60,409 |
| `M4-blueprint-pilot-v1-reproduced-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1b3vlEWLO-gZeTjSrw7_jSn1b-euy14l0/view) | 28,220 |
| `M4-blueprint-pilot-v1-resumed-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1XCNVAO5nLyXOUqixuiwl9uMaQsYEFSGz/view) | 60,460 |
| `M4-play-web-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1uFqaDsECgGfaXObsg0dNru3CLliNs81a/view) | 2,638,016 |
| `M4-play-web-benchmark-evidence-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1pp_OI0kYTBKN_xDo7qti4b99xDRj54ig/view) | 3,500,264 |

Every archive embeds `ARCHIVE-MANIFEST.json` and `RESTORE-README.txt`. Extract into a fresh directory; restore members beneath their original `Local/` project prefixes and verify the recorded SHA256. [Accepted receipt](https://drive.google.com/file/d/1aFqhov369NW_u096KjYroomLHbcQw3jp/view) and [native proof](https://drive.google.com/file/d/11bf8oPtR5fWhcDiWDiFuYyscVBi4mwzF/view) preserve all file IDs, hashes and counts. A read-only metadata helper encountered missing `os.listxattr` in macOS system Python; the retained incident uses `/usr/bin/xattr` instead, with no payload mutation.

Guarded cleanup removed **2,000 separate M4 originals / 542,265,399 bytes (0.51 GiB)**: 1,990 release-validation files matching the three previously accepted whole archives, and ten primary six-player slice files matching the existing exact Historical cloud files. No new duplicate slice archive was uploaded. Current source/member hashes, stable stat identities, Git tracking, external symlinks, open handles and current PR149 approval/future-input plans were checked before deletion. Excluded coding/browser files remain. Each root has an `ARCHIVED-RESTORE-20261004.json` stub. [Cleanup receipt](https://drive.google.com/file/d/1rTrA7DKpWWPNuTdTl3UIGMEkEEc-D4Y4/view) and [full per-file journal](https://drive.google.com/file/d/1PxmACpU9l_9V5YqMGhocEFCrOMST_cNX/view) map original paths to exact cloud IDs/members/hashes. Synced Drive payloads were untouched.

M1 measured free space is **26,401,132,544 bytes (24.59 GiB)**; M4 after the new packaging is **101,046,308,864 bytes (94.11 GiB)**. Cache reclamation remains gradual. PR149's owner-approved 7-GiB pilot now fits memory, but its forecast is at least 31.2 main hours against 17.9 available; the owning task stopped before main and awaits a compute decision. All 3,198 retrieved evidence files, its whole campaign, isolated/original inputs and every dated retrieval root remain protected. No archival-triggered science or restart is authorized.

Next: finish dependency/restore review for these small accepted originals, final closed-evidence inventory and routine documentation CI. Earlier snapshots below remain historical; hourly chat updates continue.

### Closed evidence inventory — October 4 17:39 UTC follow-up

[PR #160](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/160) passed all required checks with no findings and merged. The [inventory/closeout receipt](docs/artifacts/drive-final-inventory-20261004.json) classifies the remaining local material and preserves each new archive identity.

Four more whole files are **accepted uploaded**, with native uploaded/not-uploading/no-conflict/exact-size plus current cloud ID/name/size/parent proof:

| Whole archive | Cloud file | Bytes |
| --- | --- | ---: |
| `PR145-M1-hu20-exact-turn-report-20261003-main03-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1uF41GRo6_Jgz_HGSlVN_ljm1k8smzM4V/view) | 30,971,221 |
| `PR145-M1-hu20-exact-turn-report-20261003-main03-local-replay-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1YV3PqWObk8ZcYSKmOhgxe8GIrqSPrztT/view) | 13,287,815 |
| `M1-archival-provenance-snapshot-20261004.tar.gz` | [archive](https://drive.google.com/file/d/10Tii-_oNXmxa5jYWWvpfNdRjufxJcTdR/view) | 594,617,807 |
| `M4-archival-provenance-snapshot-20261004.tar.gz` | [archive](https://drive.google.com/file/d/1g7579f9m5wYVvBhIK_r8tMM58I7m1cnS/view) | 90,543,889 |

The two PR145 folders preserve six published main03 report files and three independent local-replay output files: 44,766,307 source bytes. Their nine source hashes were absent from the earlier native manifests. The two host-specific archival-provenance snapshots preserve 267 M1 members / 678,862,476 source bytes and 119 M4 members / 193,607,825 source bytes, including retained packaging failures/partials, helpers, catalogs and cleanup receipts. Snapshot manifests explicitly define their freeze; subsequent small receipts remain separate. All embedded source/member hashes verified. Private credentials are excluded. The intact 594,617,807-byte M1 snapshot uploaded through native Drive without chunking or the connector ingestion ceiling.

Cleanup removed **566 closed M4 pilot/web files / 15,223,811 bytes**, preserving **425 access-token/browser-profile files** for private/everyday dependencies. [Receipt](https://drive.google.com/file/d/1YZFuZPJpvot5hIFMGOy_GcsIm3MWHavC/view) and [full journal](https://drive.google.com/file/d/1qrTqWrPbrXt4FeD0UtAKmTK1lhQjaidk/view). Separately retained PR145 report copies and closed staging/partial archive bytes were removed only after accepted archive/member hash, fresh stat, tracked-file, symlink, handle and future-input checks: **13 M1 files / 625,968,177 bytes** and **one M4 partial / 75,280,207 bytes**. Every failed-partial byte remains in the accepted provenance snapshots, with original path/hash restore mappings. [M1 receipt](https://drive.google.com/file/d/1EpJ6UXK-5G8gad9e6VJjq1Om0F1RVpQS/view), [M1 journal](https://drive.google.com/file/d/1-h4G7pJY7YKoNczhNOlzieTvihVeuvoK/view), [M4 receipt](https://drive.google.com/file/d/1i4WrAv0mYUImnBKoNFw6b1JMna6DFXgO/view), [M4 journal](https://drive.google.com/file/d/1HYi1mwNNey1No6j9EsMkUZD-qSx8J1c8/view). No synced cloud payload was deleted.

The metadata-only sweep inspected 61 M1 and 35 M4 relevant scopes with zero access errors, excluding symlink payloads, Git/environments/build/browser caches, private credentials and unrelated projects. All identified **closed heavy evidence** in this inventory has accepted cloud coverage. Required A/default/C/average inputs and HU20/TP20/fresh-install fixtures remain local; source checkout reports/configs and the tracked `deepcfr-test` matrix dataset are coding material. Small current archive-management manifests/helpers/restore stubs remain available for lookup. This inventory is not permission to remove future inputs or an assertion that active research is closed.

**PR149 is now running its owner-authorized main phase** after all 61 qualification checks; its revised forecast is 15.54 hours against 16.69 available before reserve. This supersedes the earlier stopped-forecast snapshots. Its entire M4 campaign, every dated M1 retrieval root and every declared future/original input remain protected. The current inventory found about 19.3 GB logical bytes across four M1 retrieval roots; metadata counts overlap source copies and do not measure unique physical storage. Archival will not touch them before the owning task's verified closeout/canonical manifest.

After this cleanup M1 measured **27,178,016,768 bytes (25.31 GiB)** free and M4 **99,169,501,184 bytes (92.36 GiB)** free. Native cache and active PR149 writes explain changes between readings; no forced offload. Next: documentation CI and later protected-task archival only after verified closure. Hourly checks/chat updates continue while that authorized work remains.

### Current archive folders

Paths are relative to the research Drive folder. Upload status is a snapshot on October 4, not a promise of completion. The older folder manifests remain historical evidence; new whole archives embed member hashes and restoration instructions.

| Folder | Evidence / location within it | State |
| --- | --- | --- |
| [PR-113-TP20](https://drive.google.com/drive/folders/1bvceeB1paom93UQyGbbR29gXxco5_H3M) | `M4-results-20261004.tar.gz` | Confirmed uploaded |
| [PR-132-observation-reuse](https://drive.google.com/drive/folders/1l8T_GUwa2uJ8LfqmplrMglQx_3hDn1lC) | `M4-results-20261004.tar.gz`, `M4-CI-repair-results-20261004.tar.gz` | Confirmed uploaded |
| [PR-133-mature-CPU](https://drive.google.com/drive/folders/1xH2hQ8budf3n9tOhCxcMj3A4d5e9pllL) | `M4-initial-results-20261004.tar.gz`, `M4-six-lineage-results-20261004.tar.gz` | Confirmed uploaded |
| [PR-136-HU20-500M](https://drive.google.com/drive/folders/1Jjg9yvbPQ25nws_wW1IupyMYCfnNVFc_) | Whole 23.51-GB campaign archive and recovery manifest | Confirmed uploaded; separate originals cleaned |
| [PR-144-history-compression](https://drive.google.com/drive/folders/1EnCmKftt50pebTWVtu1MvTUQEaw_5hCV) | Six complete archives; `archive-manifest.json`, `SHA256SUMS` | Confirmed uploaded |
| [PR-145-exact-flop](https://drive.google.com/drive/folders/1ciOOpSaLHqvCSI8wzhWDQORtizfeCrxZ) | `M4-results-20261004.tar.gz`, including retained flop/turn attempts; `M4-exact-flop-inputs-20261004.tar.gz` | Confirmed uploaded |
| [PR-148-turn-calibration](https://drive.google.com/drive/folders/1pmu8GZww8SHQBkVgWM5a-Rn5txeERxC5) | `M4-closed-research--RUN-20261004.tar.gz`, `M1-closed-research--RUN-20261004.tar.gz`, `M1-exact-turn-evidence-20261004.tar.gz` | Confirmed uploaded; calibration03 shared once |
| [M4-closed-training](https://drive.google.com/drive/folders/1Nm2vyxf2GbEk9t30--vPluJ-IFrk-TbQ) | `results--RUN-20261004.tar.gz` from `deepcfr-training` | Confirmed uploaded |
| [M4-local-CFR-diagnostic](https://drive.google.com/drive/folders/1i9HAiFFXKTgS0FLzGUtwsrEeNLaCiQsT) | `results-20261004.tar.gz` | Confirmed uploaded |
| [M1-board-pooling](https://drive.google.com/drive/folders/13SUnP_d1ZVtcyJAp-oqRecg5Wv3SYIfy) | `closed-evidence-20261004.tar.gz` | Confirmed uploaded |
| [M1-alias-audit](https://drive.google.com/drive/folders/18cCPHR6EfXLUBGbwmNukxttySSUUj0Rf) | `closed-evidence-20261004.tar.gz` | Confirmed uploaded |
| [Historical-experiments](https://drive.google.com/drive/folders/1AIsc7LBHc7ziwrpOuwMuCPnvZpCJS1Zr) | Earlier inventory entries, retaining names and IDs | Organization verified |
| [Archive-receipts](https://drive.google.com/drive/folders/10cG2BTBZLcE7lO10TgBZ0Ud32quaf_BX) | Cleanup and restoration receipts | Organization verified |

Other existing PR/M4 archive folders remain at the research root. M4 original paths retain restore stubs after accepted-upload cleanup; restore the recorded archive into a fresh local directory before computation. M1 links resolve to their native raw-backup folders under Historical-experiments. These links are restoration conveniences, not independent backups. Private credentials, Git internals and reproducible environments/caches were excluded and retained locally. Ordinary coding checkouts and fixtures remain available.

Let streaming reclaim cache space gradually, as requested by the owner. No immediate forced offloading is needed. **Never delete within the synced folder to free local space:** that deletes the cloud copy too. Separately retained inactive originals may be removed only after the accepted desktop completion check and dependency review. Restore research into a local working directory and verify the manifest before future computation.

## Historical October 2 archive and local cleanup

The owner accepted Google Drive desktop's **Successfully uploaded / Synced** status as the upload acceptance check. Inactive copies have now been removed: **8,722 files / 41,249,100,807 logical bytes (38.42 GiB)** across the primary checkout, PR144 evidence and temporary archive staging. The full removal/restore receipt is `LOCAL-CLEANUP-20261002.json` in the research Drive folder; the [compact cleanup record](docs/artifacts/local-cleanup-20261002.json) preserves totals, archive hashes, paths and exclusions.

- **PR144:** six complete archives in [PR-144-history-compression](https://drive.google.com/drive/folders/1EnCmKftt50pebTWVtu1MvTUQEaw_5hCV). Download `archive-manifest.json` and `SHA256SUMS` from that folder; verify the selected archive, then extract its member path under a fresh restoration directory. The manifest maps every archive prefix to its original working root. Failed/partial attempts and all closed raw evidence remain in these archives. The original local retrieval paths in the [scientific report](docs/reports/dr2x2-ac-comparison.md) now require this restoration step.
- **Preserved locally:** original A inputs, six final C checkpoint/current files, six A/C average exports referenced by symlinks, working blueprint trees, tracked reports, private credentials and other agents' work.
- **Still uploading:** `branching-comparison.tar.gz` (14,041,643,443 bytes). Its original remains local until the desktop confirms completion.
- **M4:** read-only serial archival transfer started for 27 closed research result roots, including PR136, into separate PR/root folders. Uploads are **pending**, and every M4 original remains intact. Active exact-flop/turn experiment and its input root are excluded. Durable local progress: `/Users/dberweger/Local/research-archive-preparation-20261002/m4-archive-status.json`.
- **Git housekeeping:** 411 abandoned, unindexed temporary Git files were removed after checking age and open handles. All four repository HEAD trees remained readable. Free disk was about **120 GiB** immediately afterward; ongoing archive staging changes that figure. No committed Git pack, branch or scientific source was removed.


Large untracked training/evaluation artifacts are indexed here so a PR can link to their location without checking them into Git. Working files may live wherever convenient. During cleanup, archive them to the owner-designated [Google Drive folder](https://drive.google.com/drive/folders/188bEt6i0RHqegCCdvpf3wPzUiRw78N2s).

## Current archive state

The September 30 catalog remains the original inventory: **8,117 files / 208 top-level entries / 48,465,009,999 logical bytes**. October 2 cleanup removed completed inactive entries after owner acceptance of Drive upload completion. The [catalog](docs/artifacts/results-catalog.json) now marks each removed or retained top-level entry. The October 4 section records subsequent completed branching upload and native staging; retained input dependencies still require review before duplicate cleanup.

- Original root: `/Users/dberweger/Local/deepcfr-texas-no-limit-holdem-6-players/results`.
- Streamed destination: `/Users/dberweger/Library/CloudStorage/GoogleDrive-dberweger2017@gmail.com/My Drive/deepcfr-research-results`.
- The original inventory names are retained. Current legacy paths gain the `Historical-experiments/` prefix where recorded in the October 4 relocation receipt; existing PR folders remain at the root.
- [Machine-readable catalog](docs/artifacts/results-catalog.json) records exact bytes/files and report references. Logical sizes include duplicate retained archives/extracted copies; they do not measure physical APFS storage.

## Cleanup and retrieval policy

1. Keep active checkpoints, recovery slots and other agents’ input dependencies in their working location. Coordinate before cleanup.
2. Preserve every attempt, failed/partial output, source/config/model identity, manifest/hash and exact retrieval command. Add the PR/report and archive relative path to this index. Never replace scientific evidence with a success-only archive.
3. Let Drive upload. Under the owner's October 2 decision, accept the desktop's Successfully uploaded/Synced status, check exact cloud names/sizes and preserve archive member hashes and restoration evidence. A placeholder or filename alone is insufficient. Remove only owner-authorized inactive local copies; keep pending uploads and active dependencies.
4. Before analysis, restore the needed immutable files into a working directory, verify their hashes and source/model identities, and keep restored heavy computation on the authorized host. Do not train or audit inside a streamed folder.

## Recent work outside the initial batch

These are the original working roots, outside the initial primary-checkout batch. October 4 staging places #132/#133 evidence in the PR folders above and preserves old paths as cloud links. #134 input dependencies remain local pending review.

| PR | Retained location / evidence |
| --- | --- |
| [#149](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/149) | M1 `/Users/dberweger/Local/hu20-board-pooling-20261003`: outcome-blind census/index (400,510,976-byte SQLite), three restored checkpoint members (~390 MB) with [restoration hashes](docs/reports/hu20-board-pooling-artifacts/checkpoint-restoration.json), companion, nine Mac fixture checks, quote screenshots and original 769,095-byte and revised 771,213-byte external-source archives. External AGPL checkouts `/Users/dberweger/Local/hu20-board-pooling-tool` and `hu20-board-pooling-tool-v2`; [fingerprint](docs/reports/hu20-board-pooling-artifacts/external-tool.json). Original stored-average inputs and #145 fixtures remain under `/Users/dberweger/Local/hu20-m4-archive-20261002`; no new M4 access. [Preparation](docs/reports/hu20-board-pooling-preparation.md); $5 ceiling approved then rental deferred; revised held-out split/13 Mac gates retained; Linux qualification/main outcomes pending. Active dependencies retained, not in the primary Drive batch. |
| [#132](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/132) | M4 `/Users/dberweger/Local/trainer-observation-reuse-20260930/results/observation-reuse-m4-20260930`; manifest/retrieval in `docs/reports/trainer-observation-reuse-m4.md` on that PR. |
| [#133](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/133) | M4 `/Users/dberweger/Local/runpod-mature-cpu-six-20260930/results`; complete and failed/partial mature CPU attempts; manifests/report on that PR. Preserve active extra-control dependencies. |
| [#134](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/134) | M1 `/Users/dberweger/.codex/worktrees/main-isolated/deepcfr-texas-no-limit-holdem-6-players/results/hu20-stackoff-20260930` (~43 MiB) and `results/hu20-stackoff-inputs` (1,017,051,279 bytes); report/compact evidence on that PR. New Guy reports all owned workers closed and inputs retained. |

## Initial batch table of contents

**Historical inventory; current removed/retained states are in the machine-readable catalog and October 2 cleanup record.** Report links identify associated research. Associated PRs identify the latest report-changing PR found in main history; they do not assert which run originally produced a file. Legacy/unmapped entries are retained for later identification.

| Entry in original inventory | Logical bytes | Files | Report / associated PR |
| --- | ---: | ---: | --- |
| `.DS_Store` | 32,772 | 1 | Unmapped legacy/local evidence |
| `analyze-fitting.py` | 1,348 | 1 | Unmapped legacy/local evidence |
| `arena-dev-1` | 578,287 | 6 | Unmapped legacy/local evidence |
| `arena-final-control` | 4,320,079 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-final-control-replay` | 4,319,658 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-final-smoke` | 684,241 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-final-smoke-replay` | 684,214 | 6 | [arena-validation](docs/reports/arena-validation.md); [#44](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/44) |
| `arena-sensitivity-20260915` | 4,323,714 | 6 | Unmapped legacy/local evidence |
| `arena-sensitivity-replay-20260915` | 4,323,594 | 6 | Unmapped legacy/local evidence |
| `audit-readiness.py` | 6,455 | 1 | Unmapped legacy/local evidence |
| `audit-representation.py` | 1,355 | 1 | Unmapped legacy/local evidence |
| `audit_snapshot_readiness.py` | 7,300 | 1 | Unmapped legacy/local evidence |
| `blueprint-04-ops` | 13,368 | 2 | Unmapped legacy/local evidence |
| `blueprint-history-fixed-coverage-m4` | 168,896,207 | 10 | Unmapped legacy/local evidence |
| `blueprint-history-summary-learning-m4` | 520,841,454 | 22 | Unmapped legacy/local evidence |
| `blueprint-history-summary-v1-m4` | 684,819,417 | 19 | Unmapped legacy/local evidence |
| `blueprint-m4-slice-v1-continued-m4` | 256,135,764 | 5 | [blueprint-m4-slice-v1](docs/reports/blueprint-m4-slice-v1.md); [#95](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/95) |
| `blueprint-m4-slice-v1-m4` | 76,866,163 | 5 | [blueprint-m4-slice-v1](docs/reports/blueprint-m4-slice-v1.md); [#95](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/95) |
| `blueprint-pilot-v1-m4` | 198,609 | 14 | [blueprint-pilot-v1](docs/reports/blueprint-pilot-v1.md); [#91](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/91) |
| `blueprint-runpod-check-v1` | 1,839,316,262 | 53 | [blueprint-runpod-sizing-v1](docs/reports/blueprint-runpod-sizing-v1.md); [#97](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/97) |
| `blueprint-search-m4-smoke-20260924` | 137,076 | 7 | Unmapped legacy/local evidence |
| `blueprint-search-m4-smoke-final-20260924` | 137,667 | 7 | [blueprint-postflop-search-m4-smoke](docs/reports/blueprint-postflop-search-m4-smoke.md); [#103](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/103) |
| `blueprint-search-m4-smoke-pinned-20260924` | 137,081 | 7 | Unmapped legacy/local evidence |
| `branching-comparison-provisional.json` | 225,760 | 1 | Unmapped legacy/local evidence |
| `branching-comparison.tar.gz` | 14,041,643,443 | 1 | [holdem-branching-online](docs/reports/holdem-branching-online.md); [#79](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/79) |
| `branching-handoff-history.md` | 9,691 | 1 | Unmapped legacy/local evidence |
| `branching-handoff.md` | 1,251 | 1 | Unmapped legacy/local evidence |
| `branching-host-gate.json` | 1,869 | 1 | Unmapped legacy/local evidence |
| `branching-operations-local` | 72,238 | 24 | Unmapped legacy/local evidence |
| `branching-pilot-first` | 59,322,226 | 20 | Unmapped legacy/local evidence |
| `branching-pilot-first-resumed` | 38,757,526 | 16 | Unmapped legacy/local evidence |
| `branching-pilot-second` | 76,793,883 | 20 | Unmapped legacy/local evidence |
| `branching-pilot-second-resumed` | 45,502,176 | 16 | Unmapped legacy/local evidence |
| `branching-pilot-verification.json` | 5,136 | 1 | [holdem-branching-pilot](docs/reports/holdem-branching-pilot.md); [#78](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/78) |
| `branching-rental.json` | 1,297 | 1 | Unmapped legacy/local evidence |
| `branching-retrieved` | 1,502,864,548 | 507 | Unmapped legacy/local evidence |
| `build-fitting-report.py` | 3,137 | 1 | Unmapped legacy/local evidence |
| `build_strategy_report.py` | 6,776 | 1 | Unmapped legacy/local evidence |
| `card-diversity` | 1,368,388 | 3 | [holdem-multistreet-preliminary](docs/reports/holdem-multistreet-preliminary.md); [holdem-card-diversity-admission](docs/reports/holdem-card-diversity-admission.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-card-diversity-runtime-amendment](docs/reports/holdem-card-diversity-runtime-amendment.md); [#81](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/81); [#82](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/82); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `card-diversity-campaign` | 127,485,011 | 29 | [holdem-card-diversity](docs/reports/holdem-card-diversity.md); [#82](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/82) |
| `card-diversity-idle-calibration` | 1,350,148 | 2 | Unmapped legacy/local evidence |
| `card-diversity-retry` | 1,368,390 | 3 | Unmapped legacy/local evidence |
| `check-fitting.py` | 796 | 1 | Unmapped legacy/local evidence |
| `collect-readiness.py` | 3,347 | 1 | Unmapped legacy/local evidence |
| `collection-performance` | 288,750 | 32 | [holdem-collection-performance](docs/reports/holdem-collection-performance.md); [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [#66](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/66); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67) |
| `collector-branching` | 16,955,607 | 4 | [holdem-collector-branching](docs/reports/holdem-collector-branching.md); [#77](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/77) |
| `core-v1-baseline` | 19,364,403 | 6 | Unmapped legacy/local evidence |
| `cpu-pilot-local-v1` | 5,827,614 | 35 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-remote-v1` | 14,166,605 | 130 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-remote-v1.tar.gz` | 2,062,422 | 1 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-rental.json` | 793 | 1 | Unmapped legacy/local evidence |
| `cpu-pilot-replays` | 3,113,433 | 6 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `cpu-pilot-replays.tar.gz` | 3,057,126 | 1 | [cpu-pilot](docs/reports/cpu-pilot.md); [#49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49) |
| `decision-errors` | 18,943,875 | 2 | Unmapped legacy/local evidence |
| `frozen-fitting` | 1,440,341,426 | 86 | [holdem-persistent-critic](docs/reports/holdem-persistent-critic.md); [holdem-frozen-fitting](docs/reports/holdem-frozen-fitting.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-policy-comparison](docs/reports/holdem-policy-comparison.md); [#72](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/72); [#74](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/74); [#76](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/76); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `frozen-fitting-supervisor.log` | 0 | 1 | Unmapped legacy/local evidence |
| `historical-validation` | 3,734,237 | 8 | Unmapped legacy/local evidence |
| `historical-validation-replay` | 3,734,238 | 8 | Unmapped legacy/local evidence |
| `holdem-baseline-resumed-v1` | 67,894,936 | 73 | [holdem-baseline](docs/reports/holdem-baseline.md); [#61](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/61) |
| `holdem-baseline-v1` | 94,276,325 | 93 | [holdem-collection-performance](docs/reports/holdem-collection-performance.md); [holdem-variance](docs/reports/holdem-variance.md); [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [holdem-baseline](docs/reports/holdem-baseline.md); [#61](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/61); [#66](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/66); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67); [#68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68) |
| `launch-fitting.sh` | 1,229 | 1 | Unmapped legacy/local evidence |
| `launch-readiness.sh` | 926 | 1 | Unmapped legacy/local evidence |
| `local-training-proposal-20260918` | 133,431,692 | 23 | Unmapped legacy/local evidence |
| `longer-05-handoff.md` | 24,044 | 1 | Unmapped legacy/local evidence |
| `longer-05-integrity-second.log` | 318 | 1 | Unmapped legacy/local evidence |
| `longer-05-integrity.json` | 310 | 1 | Unmapped legacy/local evidence |
| `longer-05-integrity.log` | 369 | 1 | Unmapped legacy/local evidence |
| `longer-05-rental.json` | 732 | 1 | Unmapped legacy/local evidence |
| `longer-05-retrieved` | 992,906,569 | 231 | Unmapped legacy/local evidence |
| `longer-05-summary.json` | 82,825 | 1 | Unmapped legacy/local evidence |
| `longer-05-verification.tar.gz` | 10,578,536 | 1 | [holdem-longer-training](docs/reports/holdem-longer-training.md); [#73](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/73) |
| `longer-05.tar.gz` | 6,623,622,300 | 1 | [holdem-longer-training](docs/reports/holdem-longer-training.md); [#73](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/73) |
| `longer-05.tar.gz.sha256` | 94 | 1 | Unmapped legacy/local evidence |
| `longer-calibration` | 49,494,973 | 25 | [holdem-longer-calibration](docs/reports/holdem-longer-calibration.md); [#70](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/70) |
| `longer-calibration-tests.log` | 672 | 1 | Unmapped legacy/local evidence |
| `longer-calibration-verification.json` | 1,570 | 1 | Unmapped legacy/local evidence |
| `longer-calibration-verification.log` | 49 | 1 | Unmapped legacy/local evidence |
| `longer-calibration-verified` | 49,493,734 | 24 | Unmapped legacy/local evidence |
| `longer-calibration.log` | 826 | 1 | Unmapped legacy/local evidence |
| `longer-calibration.tar.gz` | 16,997,428 | 1 | [holdem-longer-calibration](docs/reports/holdem-longer-calibration.md); [#70](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/70) |
| `m4-overnight-20260919` | 1,014,177,619 | 459 | [holdem-continuous-m4](docs/reports/holdem-continuous-m4.md); [#88](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/88) |
| `m4-overnight-20260919.tar.zst` | 9,399,210,638 | 1 | [holdem-continuous-m4](docs/reports/holdem-continuous-m4.md); [#88](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/88) |
| `m4-overnight-20260919.tar.zst.sha256` | 144 | 1 | Unmapped legacy/local evidence |
| `m4-review` | 3,676 | 1 | Unmapped legacy/local evidence |
| `multistreet-calibration-optimized` | 422,056 | 3 | Unmapped legacy/local evidence |
| `multistreet-calibration-original` | 421,467 | 2 | Unmapped legacy/local evidence |
| `multistreet-campaign.tar.gz` | 20,793,652 | 1 | [holdem-multistreet-campaign](docs/reports/holdem-multistreet-campaign.md); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85) |
| `multistreet-handoff.md` | 14,085 | 1 | Unmapped legacy/local evidence |
| `multistreet-integration` | 1,303,943 | 11 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `multistreet-integration-plan.json` | 6,322 | 1 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `multistreet-integration.log` | 52,967 | 1 | Unmapped legacy/local evidence |
| `multistreet-operations-local` | 46,882 | 23 | Unmapped legacy/local evidence |
| `multistreet-pilot` | 9,796 | 2 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `multistreet-process-smoke-1` | 19,758 | 12 | Unmapped legacy/local evidence |
| `multistreet-process-smoke-1.json` | 486 | 1 | Unmapped legacy/local evidence |
| `multistreet-process-smoke-4` | 19,754 | 12 | Unmapped legacy/local evidence |
| `multistreet-process-smoke-4.json` | 486 | 1 | Unmapped legacy/local evidence |
| `multistreet-reference-stability.json` | 378,831 | 1 | [holdem-multistreet-campaign](docs/reports/holdem-multistreet-campaign.md); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85) |
| `multistreet-retrieved` | 87,765,231 | 2713 | [holdem-multistreet-campaign](docs/reports/holdem-multistreet-campaign.md); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85) |
| `multistreet-source-310258a.tar.gz` | 2,321,096 | 1 | [holdem-multistreet-pilot](docs/reports/holdem-multistreet-pilot.md); [#83](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/83) |
| `neural-convergence-kuhn-11` | 33,130,869 | 11 | Unmapped legacy/local evidence |
| `neural-convergence-kuhn-29` | 33,205,100 | 11 | Unmapped legacy/local evidence |
| `neural-convergence-kuhn-47` | 33,201,917 | 11 | Unmapped legacy/local evidence |
| `neural-convergence-kuhn-summary` | 36,210 | 2 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-11` | 50,966,099 | 9 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-11-paused` | 41,587,271 | 8 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-29` | 92,894,296 | 12 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-47` | 93,137,412 | 12 | Unmapped legacy/local evidence |
| `neural-convergence-leduc-summary` | 36,387 | 2 | Unmapped legacy/local evidence |
| `neural-fitting-v1` | 4,121 | 2 | Unmapped legacy/local evidence |
| `neural-kuhn-v1` | 68,573 | 3 | [neural-validation](docs/reports/neural-validation.md); [#47](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/47) |
| `neural-leduc-refit-v1` | 4,771 | 2 | Unmapped legacy/local evidence |
| `neural-leduc-v1` | 60,300 | 3 | [neural-validation](docs/reports/neural-validation.md); [#47](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/47) |
| `neural-readiness-handoff.md` | 1,598 | 1 | Unmapped legacy/local evidence |
| `neural-readiness-retrieved` | 818,445,367 | 312 | [neural-readiness](docs/reports/neural-readiness.md); [#55](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/55) |
| `neural-readiness.tar.gz` | 138,850,141 | 1 | [neural-readiness](docs/reports/neural-readiness.md); [#55](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/55) |
| `neural-strategy-refit-v1` | 9,785 | 2 | [neural-convergence](docs/reports/neural-convergence.md); [#48](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/48) |
| `opponents-validation` | 850,470 | 6 | Unmapped legacy/local evidence |
| `opponents-validation-replay` | 850,531 | 6 | Unmapped legacy/local evidence |
| `persistent-critic` | 515,168,017 | 29 | [holdem-persistent-critic](docs/reports/holdem-persistent-critic.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-river-reference](docs/reports/holdem-river-reference.md); [#75](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/75); [#76](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/76); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `plot_strategy_report.py` | 3,400 | 1 | Unmapped legacy/local evidence |
| `policy-comparison` | 45,668,305 | 46 | [holdem-multistreet-preliminary](docs/reports/holdem-multistreet-preliminary.md); [holdem-next-training-analysis](docs/reports/holdem-next-training-analysis.md); [holdem-policy-comparison](docs/reports/holdem-policy-comparison.md); [#74](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/74); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85); [#87](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/87) |
| `policy-comparison-supervisor.log` | 0 | 1 | Unmapped legacy/local evidence |
| `readiness-audit.json` | 9,881 | 1 | Unmapped legacy/local evidence |
| `readiness-cleanup.json` | 470 | 1 | Unmapped legacy/local evidence |
| `readiness-collector.json` | 342 | 1 | Unmapped legacy/local evidence |
| `readiness-collector.log` | 0 | 1 | Unmapped legacy/local evidence |
| `readiness-collector.pid` | 6 | 1 | Unmapped legacy/local evidence |
| `readiness-launch-manifest.json` | 9,712 | 1 | Unmapped legacy/local evidence |
| `readiness-provider-stop.py` | 2,062 | 1 | Unmapped legacy/local evidence |
| `readiness-runtime.json` | 3,813 | 1 | Unmapped legacy/local evidence |
| `representation` | 49,446,619 | 23 | [hu20-native-reopening-preliminary](docs/reports/hu20-native-reopening-preliminary.md); [holdem-multistreet-preliminary](docs/reports/holdem-multistreet-preliminary.md); [holdem-longer-training](docs/reports/holdem-longer-training.md); [strategy-fitting](docs/reports/strategy-fitting.md); [#52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52); [#73](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/73); [#85](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/85); [#115](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/115) |
| `representation-run.log` | 833 | 1 | Unmapped legacy/local evidence |
| `river-reference` | 17,154,184 | 12 | [holdem-river-reference](docs/reports/holdem-river-reference.md); [#75](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/75) |
| `river-reference.log` | 3,306 | 1 | Unmapped legacy/local evidence |
| `sampled-full-tests.log` | 672 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot` | 310,756,128 | 159 | [holdem-sampled-pilot](docs/reports/holdem-sampled-pilot.md); [#69](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/69) |
| `sampled-pilot-run.log` | 827 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot-verification-initial` | 3,515,806 | 5 | Unmapped legacy/local evidence |
| `sampled-pilot-verification-initial.log` | 240 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot-verification.log` | 68 | 1 | Unmapped legacy/local evidence |
| `sampled-pilot-verified` | 60,962,178 | 49 | Unmapped legacy/local evidence |
| `sampled-pilot.tar.gz` | 138,909,769 | 1 | [holdem-sampled-pilot](docs/reports/holdem-sampled-pilot.md); [#69](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/69) |
| `sampling-comparison` | 909,657 | 18 | [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67) |
| `sampling-comparison.tar.gz` | 73,982 | 1 | [holdem-sampling-comparison](docs/reports/holdem-sampling-comparison.md); [#67](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/67) |
| `sampling-variance` | 2,527,826 | 33 | [holdem-variance](docs/reports/holdem-variance.md); [#68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68) |
| `sampling-variance.tar.gz` | 450,685 | 1 | [holdem-variance](docs/reports/holdem-variance.md); [#68](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/68) |
| `selfplay_from_3000_eval_3500_to_6000.csv` | 1,927 | 1 | Unmapped legacy/local evidence |
| `selfplay_from_3000_eval_3500_to_6000.json` | 4,504 | 1 | Unmapped legacy/local evidence |
| `setup-fitting.sh` | 590 | 1 | Unmapped legacy/local evidence |
| `snapshot-operations-local` | 2,316 | 5 | Unmapped legacy/local evidence |
| `snapshot-preview-manifest.json` | 10,444 | 1 | Unmapped legacy/local evidence |
| `snapshot-preview-report.json` | 265,099 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness-audit.json` | 30,993 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness-handoff.md` | 1,563 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness-retrieved` | 3,202,663,274 | 360 | [snapshot-readiness](docs/reports/snapshot-readiness.md); [#65](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/65) |
| `snapshot-readiness.sha256` | 103 | 1 | Unmapped legacy/local evidence |
| `snapshot-readiness.tar.gz` | 1,869,868,238 | 1 | [snapshot-readiness](docs/reports/snapshot-readiness.md); [#65](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/65) |
| `snapshot-smoke-v1` | 1,218,915 | 85 | Unmapped legacy/local evidence |
| `standard_phase1.csv` | 2,575 | 1 | Unmapped legacy/local evidence |
| `standard_phase1.json` | 7,180 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_1500_4000.csv` | 1,794 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_1500_4000.json` | 4,371 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_3000.csv` | 1,787 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_3000.json` | 4,364 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_5000.csv` | 1,794 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_2500_5000.json` | 4,371 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_early.csv` | 1,350 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_20k_early.json` | 2,913 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_top.csv` | 1,545 | 1 | Unmapped legacy/local evidence |
| `standard_phase1_top.json` | 3,615 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity-calibration-copy` | 370,183 | 2 | Unmapped legacy/local evidence |
| `strategy-capacity-handoff.md` | 8,389 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity-remote-v1` | 947,223,673 | 584 | [strategy-capacity](docs/reports/strategy-capacity.md) |
| `strategy-capacity-remote-v1.tar.gz` | 180,879,752 | 1 | [strategy-capacity](docs/reports/strategy-capacity.md) |
| `strategy-capacity-rental.json` | 1,027 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity-summary.json` | 11,356 | 1 | Unmapped legacy/local evidence |
| `strategy-capacity.png` | 169,765 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-handoff.md` | 5,328 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-input-audit.json` | 5,398 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-inputs` | 13,644,908 | 37 | [strategy-fitting](docs/reports/strategy-fitting.md); [#52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52) |
| `strategy-fitting-inputs.tar.gz` | 13,361,710 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-remote.tar.gz` | 41,128,008 | 1 | [strategy-fitting](docs/reports/strategy-fitting.md); [#52](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/52) |
| `strategy-fitting-rental.json` | 1,413 | 1 | Unmapped legacy/local evidence |
| `strategy-fitting-retrieved` | 123,573,053 | 1194 | Unmapped legacy/local evidence |
| `strategy-fitting-smoke` | 1,333,925 | 27 | Unmapped legacy/local evidence |
| `summarize-representation.py` | 2,435 | 1 | Unmapped legacy/local evidence |
| `summarize-sampled-pilot.py` | 3,165 | 1 | Unmapped legacy/local evidence |
| `tabular-reference-v1` | 596,913 | 10 | [tabular-validation](docs/reports/tabular-validation.md); [#46](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/46) |
| `tabular-reference-v1-replay` | 596,916 | 10 | [tabular-validation](docs/reports/tabular-validation.md); [#46](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/46) |
| `tensorboard-calibration-check` | 8,676 | 1 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_1500_to_4000_step500` | 2,446,589 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_2500_3000` | 2,209,823 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_2500_to_5000_step500` | 2,265,302 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_2500_to_5000_step500_10k` | 5,925,211 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_500_to_3000_step500` | 2,227,387 | 6 | Unmapped legacy/local evidence |
| `tournament_phase1_20k_early` | 2,305,659 | 6 | Unmapped legacy/local evidence |
| `tournament_selfplay_from_3000_3500_to_6000_100` | 1,218,995 | 6 | Unmapped legacy/local evidence |
| `tournament_selfplay_from_3000_3500_to_6000_10k` | 5,460,669 | 6 | Unmapped legacy/local evidence |
| `update-readiness-docs.py` | 5,786 | 1 | Unmapped legacy/local evidence |
| `update_snapshot_docs.py` | 6,681 | 1 | Unmapped legacy/local evidence |
| `verify-longer-archive.py` | 2,795 | 1 | Unmapped legacy/local evidence |
| `verify-sampled-pilot-initial.py` | 2,520 | 1 | Unmapped legacy/local evidence |
| `verify-sampled-pilot.py` | 2,544 | 1 | Unmapped legacy/local evidence |
| `write-readiness-report.py` | 11,093 | 1 | Unmapped legacy/local evidence |
| `write-representation-report.py` | 5,622 | 1 | Unmapped legacy/local evidence |
| `write_snapshot_report.py` | 11,092 | 1 | Unmapped legacy/local evidence |
| `write_strategy_summary.py` | 11,610 | 1 | Unmapped legacy/local evidence |

## HU20 turn-search development (PR #148)

- External AGPL play/quality harness: `/Users/dberweger/Local/hu20-turn-search-tool`; upstream is a symlink to the unchanged pinned `/Users/dberweger/Local/hu20-exact-flop-tool/upstream`. Restore with commit 9d1509fe5077d019825f833eed04b16d342dfda1. Source/binary fingerprints: `docs/reports/hu20-turn-search-artifacts/external-inventory.json`. Never vendor these sources into MIT.
- M1 references, requests, responses, logs, profiles and receipts: `/Users/dberweger/Local/hu20-turn-search-20261002/m1-parity-reference-01` through `-08`; `-04` retains the failed inserted-wager fixture. Current reference is `-08`; thread comparisons are `m1-thread-parity-t1`, `-t4`, `-t6` under the same parent. Compact counts/hashes: `docs/reports/hu20-turn-search-artifacts/m1-parity.json`. Full profiles, the parent `status.md`, old references and both build logs remain retained. The M4 phase is indexed below; no rental has started.
- Inputs remain the preserved six exports in `/Users/dberweger/Local/hu20-m4-archive-20261002/hu20-exact-flop-check-inputs`; they remain other agents' dependencies.
- Keep all originals and failed builds/requests. Archive to the existing designated Drive destination with complete manifests and symlink retrieval provenance at cleanup. No local deletion is authorized.

- HU20 #148 M4 admission/pilot and future Part A: `/Users/dberweger/Local/hu20-turn-search-20261003` on M4; isolated MIT checkout `repo`, source 75cbda7 for the 156-hand pilot. Includes immutable admission, six input hashes, published #145 report copy, cumulative `budget.json`, initial helper startup failure, pilot log/launch, complete `pilot-01` hand archives/manifests and independent `pilot-01-verification.json`. Local transfer-verified copies: `/Users/dberweger/Local/hu20-turn-search-20261002/m4-pilot-01` and `m4-pilot-metadata-01`; helper/report provenance is retained in the same parent. Compact prospective count/power proof: `docs/reports/hu20-turn-search-artifacts/part-a-timing-freeze.json`. No artifact deletion or paid allocation is authorized by this entry.

- HU20 #148 final Part A: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/part-a-01`, pushed source 5e4de63, 77,052 complete hands. Local full transfer-verified copy `/Users/dberweger/Local/hu20-turn-search-20261002/m4-part-a-01` includes original summary/manifest, replay, post-replay console failure and selection verification. Compact publication: `docs/reports/hu20-turn-search-part-a.md` and `hu20-turn-search-artifacts/part-a-result.json`. Preserve the original journal and attempts. External M4 harness uses a dedicated pinned upstream clone; build setup symlink failure/relocation and build/parity evidence remain at the same M4 root. No cleanup, paid allocation or original #145 binary replacement.

- HU20 #148 reference-bound calibration: full immutable ranges `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-references-01.json` (19,002,241 bytes), proof `reference-freeze-proof-01.json`, external native sources/build logs/failures and `m4-parity-01` remain under the same M4 root. Compact index/settings are committed; all 144 selected-base references and original report/range/result hashes remain retrievable. Original #145 bytes/binary remain unchanged. No evidence eviction is authorized.

- HU20 #148 stopped calibration-01: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-01` and full M1 copy `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-01`. All 94 manifest members (46,619,321 bytes) independently size/SHA verified. Guard/admission/launch/clock snapshots, 17 timeout rows, interrupted eighteenth request, logs and TensorBoard remain retained. Publication: [stopped curve](docs/reports/hu20-turn-search-calibration.md) and `docs/reports/hu20-turn-search-artifacts/calibration-01-stopped.json`. Family RSS guard stopped at 4.341 GiB against 4.25 GiB; processes exited, no swap growth, cumulative journal 5740.47 seconds. Recurring continuation is paused; [proposed readmission](docs/hu20-turn-search-resource-readmission.md) requires owner restart approval. No qualified setting, evidence eviction, automatic restart or rental.

- HU20 #148 owner-approved calibration-02: prospective settings `configs/diagnostics/hu20-turn-search-calibration-readmission-02.json`, source 2d761ca, fresh admission `docs/reports/hu20-turn-search-artifacts/calibration-02-admission.json`. Raw/preliminary admission and launch helpers retained at M4 `/Users/dberweger/Local/hu20-turn-search-20261003`; no new outcomes precede this freeze. Original stopped calibration-01 is retained separately, with no timing stitching or budget reset.

- HU20 calibration-02 is running from tested source 4105419, worker 18889/sidecar 18890. Final fresh admission snapshot was pushed at a5f4d48 before outcomes; preliminary snapshot is retained. Compact source/settings/admission/PID/clock provenance: [launch evidence](docs/reports/hu20-turn-search-artifacts/calibration-02-launch.json). Complete requests/responses/curve/status/TensorBoard/clock remain at M4 root `calibration-02`; local launch/helper/admission/validation evidence in `/Users/dberweger/Local/hu20-turn-search-20261002`. Original calibration-01 and #145 evidence stay untouched; no cleanup or paid allocation.

- HU20 #148 owner-approved clean stop of calibration-02: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-02`, complete M1 copy `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-02-owner-stop`. All 1,752 manifest members / 1,614,849,743 bytes independently size/SHA verified. The full 238-row curve, requests/responses/receipts/manifests, TensorBoard, original incomplete summary, SIGTERM reason, owner stop request and append-preserving journal are retained. [Budget report](docs/reports/hu20-turn-search-calibration-02-budget.md), complete committed screen curve, stop/verification and timing-only forecast identify zero final rows, 18 unattempted reduced screen rows and no selected setting. Processes exited, swap unchanged, cumulative 12,053.465 seconds charged. The forecast-blocked settings proposal cannot launch; heartbeat PAUSED pending owner budget/scope decision. No calibration-01 stitching, evidence eviction or paid allocation. Earlier running/launch entries are historical.

- HU20 #148 approved retained-screen final preparation: `configs/diagnostics/hu20-turn-search-calibration-final-30-only-03.json` and `docs/reports/hu20-turn-search-artifacts/calibration-03-approved-forecast.json` prospectively bind the original 238-row screen hash, all six timing-selected configurations and 1,728 final coordinates. Three-hour river reserve enforced; 120-second continuation requires separate approval. Original calibration-02 remains immutable; new final evidence will use M4 `calibration-03`. No launch or cleanup is implied by this preparation entry.

- HU20 #148 calibration-03 resumed final: M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-03`, actual source 93d55eb (full CI passed), worker 87830/sidecar 87831. Fresh admission 70e445d preceded final outcomes. [Launch provenance](docs/reports/hu20-turn-search-artifacts/calibration-03-launch.json) pins settings/binary/source/admission/clock; full launch/helpers, charged preflight, CI proof and local status retained at `/Users/dberweger/Local/hu20-turn-search-20261002`. All 238 prior screen rows are copied with provenance; no screen rerun, no calibration-01 stitching or evidence cleanup. Expected 1,728 final coordinates, three-hour river reserve enforced. Preserve full raw requests/responses/shared receipts/curve/TensorBoard and original append-preserving journal. No paid allocation or automatic 120-second fallback.

- HU20 #148 completed calibration-03: all 1,728 final coordinates, no strict/relaxed thirty-second qualifier, all workers exited without guard failure. [Complete report](docs/reports/hu20-turn-search-calibration-03.md), [all 1,966 final/retained-screen rows](docs/reports/hu20-turn-search-artifacts/calibration-03-full-curve.jsonl) and [result/verification/journal](docs/reports/hu20-turn-search-artifacts/calibration-03-result.json) preserve every coordinate, law, receipt, timeout and exclusion. Raw M4 `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-03`: 15,292 manifest members / 29,486,803,777 bytes independently size/SHA verified. Complete M1 retrieval `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-03-complete`, all 15,292 members independently size/SHA verified on M1; [retrieval proof](docs/reports/hu20-turn-search-artifacts/calibration-03-retrieval-verification.json) records final cumulative 42,294.351 seconds and retained TensorBoard hashes. The local publication/verifier helpers, admission, original journal snapshots and TensorBoard remain indexed there. Heartbeat PAUSED for owner decision; separate 120-second proposal/quote remains unapproved. Preserve originals and prior attempts; no eviction, rental, automatic restart or budget reset. Earlier running entries are historical.

PR149 revision 3 active M4 evidence: `/Users/dberweger/Local/hu20-board-pooling-20261004`, own checkout and external hash-verified Mac tool. Retrieve to M1 with member manifest/hash verification at closeout; retain in the dated folder for the existing PR149 Drive archive destination. No deletion or CloudStorage manipulation; no RunPod rented, $5 unused.

PR149 M4 revision-3 mandatory disk stop, October 4: verified M1 copy `/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-20261004`, 2,418 files / 1,084,490,993 bytes, zero member mismatches; [retrieval receipt](docs/reports/hu20-board-pooling-artifacts/m4-stop-retrieval.json). M4 dated original remains untouched, including nine complete exports, tenth partial, clock/failure and separate AGPL sources/binaries. Archive both to the existing PR149 Drive destination after dependency review; no cloud completion is claimed and no deletion is authorized. No solver values or paid cost. Original average inputs are still active dependencies for owner-authorized continuation.

PR149 second preparation stop, October 4: verified M1 copy `/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-02-20261004`, 2,623 files / 4,697,713,515 bytes, zero mismatches; [receipt and isolated input restoration](docs/reports/hu20-board-pooling-artifacts/m4-stop-02-retrieval.json). Preserve the first immutable verified copy/receipt too. All 80 complete exports, both clocks/failures, separate AGPL sources/tools and restored source inputs are retained in the dated M4 campaign. **Active dependency: `/Users/dberweger/Local/hu20-board-pooling-20261004/inputs-restored-02` contains all three frozen average exports; retain it for owner-authorized resumption.** The shared original input alias was archived during preparation; it remains a restoration pointer and was not modified by this task. Restore provenance uses the retained M1 `hu20-m4-archive-20261002` copy; all original source hashes match on both Macs. Archive to the existing PR149 Drive destination after dependency review; no cloud completion or deletion authorization is claimed. Qualification/main unrun; heartbeat paused; $0 paid compute.

PR149 qualification RSS stop, October 4: third verified M1 copy `/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-03-20261004`, **3,029 files / 10,125,291,134 logical bytes / 7,296,037,751 unique-inode bytes**, 82 verified hard-link groups, zero mismatches; [receipt](docs/reports/hu20-board-pooling-artifacts/m4-stop-03-retrieval.json). Preserve both older verified copies and immutable receipts. Dated M4 campaign retains all 120 complete compact exports, prepared-02 recovery dependencies, isolated source inputs, every failed attempt and external AGPL tools. First real V4 and converged pilot are descriptive qualification evidence; fresh lock-only failed the 5-GiB RSS ceiling and main never started. Heartbeat paused, $0 paid cost, no hypothesis decision. Archive to the existing PR149 destination only after dependency review; no cloud completion or deletion authorization is claimed. All source/recovery dependencies remain valuable for separately admitted bounded-memory continuation.

PR149 owner-approved 7-GiB worker readmission retains `prepared-03` and all three old retrieval copies as active dependencies. [Readmission-04 receipt](docs/reports/hu20-board-pooling-artifacts/m4-resume-04.json) rechecks 253 preparation/source members and all 120 native request/compact pairs; no solver result reuse. New `qualification-04`, `guard-qualify-04`, per-pilot resource records and approval/clock append remain in the dated M4 campaign for fresh hash-verified retrieval at closeout. No archiving, deletion, cloud completion or additional resource waiver is implied.

PR149 readmission-04 forecast stop: fresh M1 `/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-04-20261004`, **3,198 files / 10,473,469,109 logical bytes / 7,644,215,726 unique-inode bytes**, 82 verified hard-link groups, zero mismatches; [receipt](docs/reports/hu20-board-pooling-artifacts/m4-stop-04-retrieval.json). All three older copies' manifest members reverified. The new copy uses rsync `--link-dest` for unchanged third-copy members, without changing their contents; preserve all copies/receipts and M4 originals. First fresh solve/lock fit 7 GiB, but forecast lower bound 31.182 hours exceeds 17.899 available before reserve; first replay interrupted, other pilots/main unattempted. Retain all 120 exports, isolated inputs, prepared-02/-03 recovery dependencies, new approval/resource/forecast-stop records and partial replay. Heartbeat paused; no inference, deletion/cloud completion or additional budget authorization; $0 paid cost. Existing PR149 Drive destination remains the eventual archive target after dependency review.

PR149 engineering amendment: external M1 `/Users/dberweger/Local/hu20-board-pooling-engineering-20261004` retains the fresh AGPL source/build, zero-CFR tests, parsing benchmark and copied compact M4 receipts. New M4 binary `pooling-engineering-05-mac`, source archive, exact pilot-0 reruns in `engineering-parity-05`, approval/clock append and `qualification-05` remain in the dated campaign. [Fingerprint](docs/reports/hu20-board-pooling-artifacts/engineering-amendment-05.json); [exact parity](docs/reports/hu20-board-pooling-artifacts/engineering-parity-05.json). Preserve all four prior verified copies/receipts and original binaries, prepared exports and isolated inputs. Compact new receipts are locally retained; full fresh retrieval is pending closeout. No cloud completion, deletion, extended allowance or rental is implied; eventual archive remains the existing PR149 destination.


PR149 engineering-05/main preemptive memory stop: fifth verified M1 copy
`/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-05-20261004`, **3,476 files /
12,124,413,687 logical bytes / 9,295,160,304 unique-inode bytes**, 82 hard-link
groups, zero mismatches; [verification](docs/reports/hu20-board-pooling-artifacts/m4-stop-05-retrieval.json).
All four older immutable copies and manifest members reverified; retain them
and the dated M4 original unchanged. New qualification-05 passes 61 gates;
main-05 retains one atomic collect and one partial, zero common boards.
External M1 `hu20-board-pooling-engineering-20261004/real-pool-preflight` holds
three data-only statistics fixtures, eager/streamed byte-identical policy files
and capacity receipts; synthetic split metadata is engineering-only. New
bounded driver repair remains on M1, not deployed/readmitted to M4. Native
engineering source/binary, archive, prepared exports, isolated inputs and
clock/stop/admission receipts remain valuable dependencies. Heartbeat PAUSED,
owned work stopped, $0 paid cost. Existing PR149 Drive destination remains the
eventual archive target after dependency review; no upload/cloud completion,
delete authorization, automatic restart or extra allowance is implied.


PR149 main-06 completed October 5: **120 solves, 120 locked evaluations, 18 replay checks and all 40 common boards**, no main exclusions or failures. [Frozen report](docs/reports/hu20-board-pooling.md) and [resource/response verification](docs/reports/hu20-board-pooling-artifacts/m4-main-06/resources.json) retain D=0.1943 [0.1550, 0.2359] and passing 5% missing-key coverage. The complete raw campaign has **5,740 file members / 56,262,193,754 logical bytes / 53,432,940,371 unique-inode bytes**, plus symlink/hard-link provenance. Git administration alone is excluded; every working source and exact runtime commit e0c91d25 are retained. It is packed losslessly at M4 `/Users/dberweger/Local/hu20-board-pooling-closeout-06-20261005/hu20-board-pooling-m4-complete-20261005.tar.gz` (**16,802,192,861 bytes**, SHA256 `20ad0f67df71464e7c06bdb1cf63451c243944c14e732b83650861f8f9bbcaed`); manifest SHA256 `d5608ba75a785dfc5633f96dc27246d1b0c7781e478e67bc31972163565fca65`.

A separate APFS clone is hash-verified and staged through native Drive desktop at `M1-board-pooling/M4-main-06-20261005/` in the [existing PR149 archive folder](https://drive.google.com/drive/folders/13SUnP_d1ZVtcyJAp-oqRecg5Wv3SYIfy); **cloud upload was originally pending; October 5 native acceptance, current cloud name/size/parent and staged SHA256 checks now confirm the archive**. Fresh M1 `/Users/dberweger/Local/hu20-board-pooling-m4-retrieval-06-20261005` contains the hash-verified 240-record compact reporter input and exact scientific reporter replay; the full compressed archive and **all 5,740 file members now hash-verify with zero mismatches**, preserving 82 hard-link groups. [Closeout receipt](docs/reports/hu20-board-pooling-artifacts/m4-main-06/retrieval.json) records every member/archive check, all five prior-copy checks and the recovered transport interruption. Verification streamed the lossless archive without expanding the 56-GB campaign. All five older immutable M1 retrieval copies reverify with zero mismatches. Preserve every original, isolated input, prepared-02/-03 recovery dependency, old stop receipt and the external AGPL tools; no deletion/eviction authorization. $0 paid, no rental, no training or merge.

PR149 final compact report, resource/replay checks, independently verified retrieval receipt, reporter replay and current roadmap/index are also staged as nine hash-verified documents under `M1-board-pooling/M4-main-06-20261005/report-closeout/`; [staging manifest](docs/reports/hu20-board-pooling-artifacts/m4-main-06/closeout-staging.json). This supplements the immutable whole raw archive; it does not claim cloud completion or authorize deletion.

- HU20 #166 stages 1–3, October 5: river evidence remains on M4 `~/Local/hu20-turn-search-river-20261005/river-01`; terminated pilot evidence (parity, every solver request, timing hands, manifests and failures) remains on M1 `~/Local/hu20-turn-search-pilot-20261005/evidence`. Formal quote preparation, raw authenticated MCP catalog/billing replies and local read helper are retained at M1 `~/Local/hu20-turn-search-quote-20261005`; no credentials are written there. Compact [quote/report/reproduction](docs/reports/hu20-turn-search-arena-quote.md) is committed with all timing-input sizes/hashes. The $18.25 three-pod 3090 proposal remains unapproved; CPU5 exceeds the $25 ceiling. No production allocation, archival/cleanup authorization, or changes to the four old exited pods.

- HU20 #166 quote revision 2: [revised report](docs/reports/hu20-turn-search-arena-quote-02.md), quote/profile inventory and immutable MCP/compression reads bind the fixed request-SHA lowest-1% sampling policy. Local measurement helpers/raw replies remain at M1 `~/Local/hu20-turn-search-quote-20261005/revision-02`. No existing profile body was deleted: revision measurements only read the original pilot; compression streamed to a byte counter. Future new production pods retain every hand/request/receipt/response plus SHA/size of every generated profile, and only sampled bodies; user-requested unsampled deletion follows durable hashing on those pods. Lossless compressed chunks preserve exact member bytes/hashes without raw expansion on the Macs. The new $6.03 mean/$16.25 maximum proposal remains unapproved and stock-conditional. Original quote and all original evidence remain retained; no general Mac cleanup authorization.

- HU20 #166 owner-approved launch preparation: M1 `~/Local/hu20-turn-search-arena-20261005` retains the explicit chat approval record, approved plan/quote, fresh catalog reads and an immutable 434 MiB copy of the M4 pilot transfer bundle. Approved evaluator/solver source remains `6a5ff44`; separate operational scripts add fail-closed checkpoint admission, actual-host preflight and streaming evidence verification. No original Mac evidence is changed. Every paid creation/readback and provisioning-clock charge will be recorded before dependent work.

- HU20 #166 first admission rejection: `m589v6gr99fdab` (two RTX 3070s, $0.26/hour) had 37.4 quota CPUs but only 48,999,997,440 cgroup RAM bytes. No solver/arena ran. Compressed admission evidence/member manifest/stream verification/termination/readback are retained at M1 `~/Local/hu20-turn-search-arena-20261005/retrieved/m589v6gr99fdab`; creation/catalog/quote receipts, ledger and the new rental-plan request are in its parent. Newly owned pod terminated; old exited pods untouched. Owner now requires the exact next rental plan before any further allocation.

- HU20 #166 stage-4 mixed-fleet handoff/cost gate: M1 `~/Local/hu20-turn-search-arena-20261005/stage-4-mixed` retains owner-named pod readbacks, raw v1/v2 SSH measurement transcripts, actual admission records, empty initial billing response, sleep-charge exclusion, ownership handoff and actual-fleet quote. CPU quotas 17.85/18.7/18.7/13.6/10.2 require eight slots; full five-pod maximum $26.14 exceeds $25. Alternative four-pod/six-slot quote $21.22 awaits owner approval. No solver, parity or arena outcomes exist from this attempt. All five named transferred pods remain active pending decision; old exited pods unchanged.

- HU20 #166 stage-4 approved mixed-fleet execution: [launch record](docs/reports/hu20-turn-search-arena-stage-4.md). M1 `~/Local/hu20-turn-search-arena-20261005/stage-4-mixed` retains the owner handoff/approvals, 5080 termination and 404, actual v1/v2 admissions, immutable input/operations hashes, fresh quote/ledger and PR status comments. M4 `~/Local/hu20-turn-search-arena-20261005/stage-4-mixed` owns the durable supervisor, one-time per-pod dispatch receipts, authenticated remote-controller snapshots and complete actual-host gates; future retained chunks and stream/member verification belong under `retrieved/<pod-id>`. Coordinates 0–3 began at 16:14 UTC after both 3070s passed checker and 96/96 exact replay; coordinates 4–5 are reserved for separately admitted 3080 Ti hosts. Full worker-count remains six. Science source/green CI is `6a5ff44`; evaluation is in progress, with no strength or completion claim. No scientific restarts, new rentals, original Mac evidence changes, or historical-pod mutations.

- HU20 #166 stage-4 stopped closeout (October 5): [final report](docs/reports/hu20-turn-search-arena-stage-4.md) supersedes the historical running entries above. M4 `~/Local/hu20-turn-search-arena-20261005/stage-4-mixed/retrieved/<pod-id>` retains every compressed evidence chunk, archive/member SHA and size manifest, sealed stream-verification receipt and termination/readback. Its parent retains exact 500-decision controller counts (86 timeout fallbacks), partial 13,560-hand summary, spend estimate, raw posted billing, final list-pods and closeout/report receipts. M1 retains copied final metadata at the same relative path. All five owned pods terminated; protected historical pods and original Mac evidence unchanged. Frozen protocol incomplete; no full strength contrast or restart. Posted billing $0.957235 through 16:00 UTC; full provisioning estimate $1.587550 gross/$1.068661 cap-counted after sleep exclusion, final invoice pending. No destructive local evidence cleanup.

- HU20 #166 stage-4 archive/posted-billing refresh: `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/stage-4-closed-M4-20261005.zip` (541,375,924 bytes, SHA-256 `d4ff034cd19d6e6fd8db39313ceba16ff76031d86bc5bc8b74dfea87a87defc6`) and `stage-4-closed-M1-20261005.zip` (6,372,654 bytes, SHA-256 `aee43e7c031cf5b2cac8b7f67b6ee6367e95fc196d11227fb4dbf11482de8777`) contain the raw chunks, sealed manifests, all partial evidence, original report failure and successful closeout. Embedded archive inventories and destination/member verification receipts accompany the ZIPs. Credentials/caches excluded; originals preserved. New read-only MCP billing: $1.170743 posted gross/$0.651854 after owner sleep exclusion, including final bucket through 17:00 UTC. Hardware-independent proposal metadata and billing refresh will be retained in a separate research ZIP. No stopped-run hand is reused.

- HU20 #166 prospective fixed-work revision 3: [protocol/quote](docs/hu20-fixed-work-arena-proposal.md) and compact [timing, proposed plan and quote](docs/reports/hu20-fixed-work-arena/) retain the full 464-request stage-4 timing index, immutable 24-request completed/censored sample, six-thread cost-only CPU measurements, exact macOS scientific-profile comparison, latest MCP offers and posted-billing refresh. Full request/profile/log bodies, excluded default-thread diagnostic and original RSS-only raw-hash comparison failure are zipped as `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/fixed-work-proposal-M4-20261005.zip` (353,579,275 bytes, SHA-256 `e4987d936053eb8276120eff235e21c42356cee33ab181590408e7ef2031fc0b`, 1,768 verified members). A separate M1 proposal ZIP retains the source snapshot, isolated AGPL harness bundle/build, MCP replies, approval proposal and publication receipts. Originals remain preserved; no stage-4 hand is reused. No paid host allocated; owner chat approval and each actual-host parity/retention/cost gate remain pending.

- HU20 #166 revision-3 publication/archive seal: [approval comment 6000059021](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-6000059021) binds source `249a79b`, proposed plan canonical SHA256 `8a4e939f7058e0a85d281ef1a069e90bf0b63465f3f115c0f095ac888216e42e` and quote-file SHA256 `ffea2f314a39a1d6372e37feb6c0b0b42f6cb26df5418faf7cef5d7d2fce5a95`. Final M1 proposal ZIP is `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/fixed-work-proposal-M1-final-20261005.zip` (1,644,611 bytes, SHA256 `8b2ea740094ef8cd9453cf8eb8acfe05a776ff3a9f209718293536ce468d556b`, 35 verified members), containing source snapshot, AGPL source bundle/build identity, raw MCP offers/billing, quote, plan, PR comment/body and publication/CI receipts. The prior preliminary M1 proposal ZIP remains preserved. Both M4 ZIPs were copied and every archive/member reverified at this destination; originals preserved. [Linux CI](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37352006086) is in progress on the exact proposed scientific source. Await chat approval; no rental or arena launched.


- HU20 #166 revision-3 approved stock attempt: [report](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-6000829086) supersedes the prior pending-approval status. Owner chat approval, green scientific/operational CI, 120 exact macOS references and sealed source-bound bundle are retained on both Macs at `~/Local/hu20-turn-search-arena-20261005/revision-3-run-01/`. RunPod rejected both approved-shape placement attempts; zero pods created/$0 new spend/no arena started. Every request/error receipt and fresh empty list-pods are retained. `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/fixed-work-approved-stock-attempt-M4-20261005.zip` is 1,479,190,279 bytes/SHA256 `af02f9c59ac85b9741cf1fbc69ff98c98c883757e675c0cc36a48082a49f7f32`, 2,303 verified members; corresponding M1 ZIP is 155,949 bytes/SHA256 `de12d548a8e61c31a45eead05501e2a430e913727fdc0cc63394e1c123a273ae`, 74 verified members. Embedded manifests and verification receipts accompany both, with originals preserved and credentials excluded. The M4 ZIP's relative source `.` means the exact M4 run root above. No stock guarantees, Linux parity claim, automatic retry, alternate fleet or old-hand reuse.


- HU20 #166 equivalent-search owner closeout: [handoff](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/166#issuecomment-6001549575), [compact seal](docs/reports/hu20-fixed-work-arena/owner-closeout-20261005.json). Owner requested all work closed until tomorrow. All large budget-qualified placements failed without creation; admission-only 3070 `hewb8i6o4eji98` was the sole new pod. Before resource admission/build/parity/arena began, its controller/setup evidence was retrieved as a lossless compressed chunk and every member hash verified, then deletion/404 and fully paginated empty list-pods confirmed. No new hands/decisions; provisioning-clock new upper charge $0.004278, cumulative cap spend $0.670942 pending posted billing. The $20 ceiling/$16 dispatch rule and fixed-50/zero-fallback science persist; newest operational CI was cancelled on closeout and must be green before future dispatch. Both Macs retain `~/Local/hu20-turn-search-arena-20261005/revision-3-run-02/` provenance and compact closeout; M4 also retains frozen bundles/references. Originals preserved, credentials/caches excluded, all four historical protected pods untouched. Research ZIPs, each verified against every member and accompanying embedded manifest, are in `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/`:
  - `fixed-work-equivalent-closeout-M1-20261005.zip`: 217,363 bytes, SHA256 `5257f0e723cbda3013a898f551ee7ae32ccd0db18c7dcb9e08f510bd6bd2f625`, 114 verified members.
  - `fixed-work-equivalent-closeout-M4-20261005.zip`: 1,699,648,752 bytes, SHA256 `e1243c2cb6a98f6ebf32a79122cc22aa15665ad1b34c85c908d0ba78add51517`, 1,653 verified members.
  No automatic retry, schedule, background evaluation, merge or promotion. Fresh stock/quote, current source CI and actual-host exact parity/retention/cost admission are prerequisites for an owner-resumed attempt.

- HU20 #166 completed attempt 2: [frozen report](docs/reports/hu20-fixed-work-arena/attempt-2-closeout.md), [compact primary/sensitivity results](docs/reports/hu20-fixed-work-arena/attempt-2-results.json) and [archive seal](docs/reports/hu20-fixed-work-arena/attempt-2-archive.json). All 82,944 hands / 20,736 complete lineage pairs independently native-replayed; one defect-affected LBR pair excluded only in sensitivity, five guarded automatic partition restarts, zero missing final pairs. Full evidence from all three pods verified on M1 before MCP deletion/404/all list pages; all pods gone. Whole research root `~/Local/hu20-turn-search-arena-20261005/revision-3-run-03/` archived without deleting originals: `~/Local/Research-Cloud/PR-166-HU20-turn-search-arena/attempt-2-full-closeout-M1-20261006.zip`, **5,645,638,363 bytes**, SHA256 `cac0d2a766a8d0276820662112b68bb12ffe3688b8597faa0f63d43a829b3a0e`, **1,433 member files** all verified against the embedded manifest. Nested pod archives preserve 125,156 independently verified members, solver inputs/profiles, controller histories, failures, interrupted streams and resume logs. Exact approved/launched/closeout source snapshots and the frozen input bundle are included. Authentication files excluded. A separate `attempt-2-publication-seal-M1-20261006.zip` (**250,264 bytes**, SHA256 `32fe458e412c3a1500165f0900cb40e446b0e3e07cd5e66bf7d22c291dc76c12`, **23 verified members**) and adjacent verification receipt retain subsequent report publication/CI/timer-disable metadata; its files overlay the earlier snapshot. Native Drive upload confirmation remains pending; no local evidence removal, synced-folder deletion, eviction or unrelated PR cleanup is authorized. Earlier stage-4 and attempt-1 evidence remain archived/excluded.

## M4 originals removed after verified upload — October 5, 2026

Owner-authorized cleanup removed 3,399 hash-verified merged #149/#162/#163 original result files and separate local compressed archive copies. M4 free space rose from 19.2 to **69.3 GB**, approximately **50.0 GB reclaimed**. Canonical Drive archives remain; #149 preparation/input/source/tool folders, #162 input/source folders, shared Git and open #166/#169 work remain protected. M1 and #165 originals were unchanged. Historical retention entries are superseded only for the exact removed paths in the [cleanup receipt](https://drive.google.com/file/d/1nldLqO9B2Ln4E-jCLfE3UseCqK2G7exX/view). [Report and restore details](docs/artifacts/m4-original-cleanup-20261005.md); [receipts and previous index](https://drive.google.com/drive/folders/1fI2S4LaPJqMTlxEfSgA5SBX_43G5ZJFF).

Future merged research originals may be removed under the owner's authorization once matching Drive uploads verify and dependency review passes. Open PR roots and dependencies remain protected; no synced file deletion, forced offloading or unattended cleanup is authorized by this record.

## PR169 M4 scoring originals removed — October 6, 2026

PR169 is **merged**. Its completed [scoring archive](https://drive.google.com/file/d/17X9Fpi1gNL0BVU24iN6xGBuoIMG8_Jpa/view) preserves all 54 exports, 40 held-out roots, input/reference copies, raw outputs, failures, plots and pinned source 6fd63e0. ZIP size 1,869,797,825 bytes; SHA256 `198301d2c14a8abf35a1567c241e96254294c10941ebca7efc96ae92c15170c7`. Current native upload acceptance, cloud name/size/parent and fresh local archive/member/source hashes verify. [Archive folder](https://drive.google.com/drive/folders/1j9i0_HYXfiDjJYmwqvIxdkicPowWcYhT).

Under the owner's cleanup authorization, 676 exact manifested M4 files and the separate local ZIP were removed: **4.84 GB reclaimed; 51.0 GB free** at cleanup. Open #166/#171, shared inputs, M1 originals and all synced archives remain untouched. Historical originals-retained claims are superseded for the exact removed paths in the [cleanup receipt](https://drive.google.com/file/d/1S5wkafKgADZvhsR3cOBDjYGy9njIhfvG/view). [Report and restore details](docs/artifacts/pr169-original-cleanup-20261006.md); [receipt folder and prior shared index](https://drive.google.com/drive/folders/1xeOUW53rNxh2_op2-CYjUEwSi6989cJZ).

### October 6: research models excluded from Git and forensic duplicates cleaned

The owner requested that unreleased model binaries live in Research-Cloud and be retrieved only when needed. `.gitignore` now covers local planning, serialized model formats and research archive bundles. AGENTS.md requires hash-verified retrieval into ignored working directories and explicit owner approval for exact-path release model exceptions. Source, reports and retrieval manifests remain available in the repository.

The M1 forensic copies `planning/forensics/scratch/c4k.pt` and `c16k.pt` were created locally on September 20, 2026, after merged [PR #89](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/89). Neither path has commits in the checked branch/tag/remote/T3-ref history. T3 checkpoint staging repeatedly read the unignored files, hit its 30-second VCS timeout and left partial packs. Equivalent staging with local exclusions now completes in 1.4 seconds.

| Former local copy | Canonical Drive archive | Member below archive root | Bytes | Model SHA256 |
| --- | --- | --- | ---: | --- |
| `planning/forensics/scratch/c4k.pt` | [results--day-current-4k-2026091902-20261004.tar.gz](https://drive.google.com/file/d/1s-D-48RWoLx1RfO6ES38QRCyfDfU2Ues/view) | `results/day-current-4k-2026091902/scenario-0-seed-2026091902/pinned/1024/training-1024.pt` | 688,892,004 | `634af5aeb746c6d945f23a19c99af83113a0c2973a4af298abe4715c62b4cc87` |
| `planning/forensics/scratch/c16k.pt` | [results--day-current-16k-2026091902-20261004.tar.gz](https://drive.google.com/file/d/1d9ynZM5Ssx-hcHQM9VPRQye9GoT6V-1R/view) | `results/day-current-16k-2026091902/scenario-0-seed-2026091902/pinned/1024/training-1024.pt` | 783,479,055 | `96a96cc9ab9a7769ad676616c44cf0b9fde99a493b71247a52cb2ec9d57eba1f` |

Both canonical archives and the selected members were freshly rehashed, their embedded manifest entries matched, native Drive reported uploaded/no uploading/no conflicts, and cloud ID/name/size/parent checks passed. PR #89 was rechecked as merged. No tracked references were found across 46 registered worktrees, no selected model was open, and the only local filename references were historical forensic output JSON. Two local duplicates (1,472,371,059 bytes) were removed; the accepted archives and historical analysis files remain available.

Removed 82 additional Git-reported garbage packs (33,710,801,880 bytes), each older than two minutes, unchanged and without open handles. Full Git integrity and reachable-object checks passed before and after cleanup. No valid packs, refs, primary index, active PR roots or synced payloads were deleted. Temporary pack count is zero; M1 measured about 51 GB free after both cleanups. Matching local model exclusions protect existing project checkouts on both Macs while the repository rules land; other repositories retain their prior settings.

[Model cleanup receipt and original-path restoration commands](https://drive.google.com/file/d/1XgRadPHdyCnp17SUg-9t7zDIZa5P-TjB/view), [temporary-pack cleanup receipt](https://drive.google.com/file/d/11pRy40cfqt38Q781JyLO1rAsFoc_yHaa/view). [Retrieval commands and archive hashes](docs/artifacts/research-model-storage-20261006.md) show how to fetch the models into ignored `results/retrieved/pr89/` and verify them. The shared index retains the same Drive ID. This does not schedule cleanup or offload synced files.
