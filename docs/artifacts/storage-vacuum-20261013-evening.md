# Uploaded cache cleanup — October 13 evening

The owner requested storage again while #265's frozen HU200 lineage was active.
This pass released **32.176 GB**: nine uploaded archive
caches and one inactive Rust debug build directory. Verified free space at
2026-10-13T17:30:03.448138+00:00: **M1 59.940 GB /M4 73.163 GB**.
M1 released 0.341 GB; M4 released 31.835 GB.
Free-space observations also reflect other running work and are distinct from
the per-path allocation deltas credited only to this pass.

## Safety and preservation

Current owning PRs were confirmed merged. Each archive was uploaded, not
uploading, conflict-free, unpinned, not downloading, not paused/excluded/trashed,
and not held open. Exact local path/size/inode/time and cloud ID/name/size/parent
checks preceded normal `FileManager.evictUbiquitousItem` release. Fresh cloud
readback and local same-size placeholders confirm all nine identities remain.
No archive download or repeated payload/member hash audit was performed.
There was no synced deletion, forced offload or unpinning.

The #265 agent already had authorization to clean inactive research originals;
this pass therefore removed **no research originals**. Its cleanup is separate
and is not credited here. The active M1/M4 frozen root, #250 source lineage,
#262/#263 engineering/evaluation inputs, #163 tables, canonical Slumbot inputs,
final trainer executables and shared Git were protected.

M1 removed only `~/Local/overbet-correctness/native/hu20-trainer/target/debug/`,
**321,769,472 allocated bytes**, after confirming #258 merged, a real directory
inside the checkout, unchanged directory mtime and no active/open build use.
Retained source is commit `b7213567b3a46ff893eba61149ccca9106142865`; Rust debug
output rebuilds from the retained checkout/toolchain with
`cargo test --manifest-path native/hu20-trainer/Cargo.toml`. Source, research
payloads and the final release executable were not removed.

## Archive restoration

Each cache-only path restores by opening/downloading its unchanged Drive file.
Research retrieval still requires the existing index's whole/member hash checks
into fresh ignored nonsynced storage before use. No model or archive was removed
from Drive. The following byte deltas exclude remaining provider bookkeeping.

| Mac | Local cache path | Unchanged Drive file | Allocated bytes released |
| --- | --- | --- | ---: |
| M1 | `~/Local/Research-Cloud/PR-258-overbet-correctness/overbet-correctness-M1-20261013.zip` | [overbet-correctness-M1-20261013.zip](https://drive.google.com/file/d/19b0sbvtXjZJwtWUhhXptc5IMYXVDJSI8/view?usp=drivesdk) | 10,219,520 |
| M1 | `~/Local/Research-Cloud/PR-258-overbet-correctness/review-fixes-M1-20261013.zip` | [review-fixes-M1-20261013.zip](https://drive.google.com/file/d/1LbvsBFfYw7VQNDFbmiI6vg7np6zdq8Pk/view?usp=drivesdk) | 3,670,016 |
| M1 | `~/Local/Research-Cloud/PR-264-slumbot-account-benchmarks/slumbot-account-benchmarks-20261013.zip` | [slumbot-account-benchmarks-20261013.zip](https://drive.google.com/file/d/1H679E0MqQe-X-QmKR6GZBTtfbVGvEj0U/view?usp=drivesdk) | 2,957,312 |
| M1 | `~/Local/Research-Cloud/PR-261-hu200-k50-752m-slumbot-benchmark/hu200-k50-752m-slumbot-benchmark-20261013.zip` | [hu200-k50-752m-slumbot-benchmark-20261013.zip](https://drive.google.com/file/d/1eGy7bXGzBQbbSbSDrhIYRkZCyESpRqpV/view?usp=drivesdk) | 1,916,928 |
| M4 | `~/Local/Research-Cloud/PR-246-hu100-k50-2b/hu100-k50-2b-completed-M4-20261012.zip` | [hu100-k50-2b-completed-M4-20261012.zip](https://drive.google.com/file/d/1TR5HaryZUk8dpH1Jq6RPzctPxv2ILPmg/view?usp=drivesdk) | 13,402,865,664 |
| M4 | `~/Local/Research-Cloud/PR-260-hu100-k50-v2/hu100-k50-v2-evidence.zip` | [hu100-k50-v2-evidence.zip](https://drive.google.com/file/d/18H0Ve-4Uvzi4RLA0pFVZKGFhd-yQkkmv/view?usp=drivesdk) | 10,465,689,600 |
| M4 | `~/Local/Research-Cloud/PR-246-hu100-k50-2b/hu100-k50-2b-averages-M4-20261012.zip` | [hu100-k50-2b-averages-M4-20261012.zip](https://drive.google.com/file/d/14qFPvcNiJYorBapxb7QW9bxOyn1mxbIn/view?usp=drivesdk) | 3,488,718,848 |
| M4 | `~/Local/Research-Cloud/PR-260-hu100-k50-v2/hu100-k50-v2-terminal-average.zip` | [hu100-k50-v2-terminal-average.zip](https://drive.google.com/file/d/1QNswSktLznNQq7NwBS9jyACeJFXmkJ6q/view?usp=drivesdk) | 2,363,019,264 |
| M4 | `~/Local/Research-Cloud/PR-254-hu100-k50-vs-v052/hu100-k50-vs-v052-M4-20261013.zip` | [hu100-k50-vs-v052-M4-20261013.zip](https://drive.google.com/file/d/1uf4oknvw5Vr7fuTAEVYGgLr7CLS_n-d8/view?usp=drivesdk) | 2,114,748,416 |

[Full private receipt](https://drive.google.com/file/d/1CFS_jPYS8vpoiFv49W4KNdsNRSkrVcAK/view?usp=drivesdk), ID `1CFS_jPYS8vpoiFv49W4KNdsNRSkrVcAK`,
**26,988 bytes**, SHA256 `8a431cf112236d90f8ff707d8f2d1f75b9d265f04c9376ff1e30b1f9af86e8d5`.
Independent metadata readback confirms its name/size and Research-Cloud parent
`188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. Exact before/after stats, native acceptance,
cloud identities and build-cache restoration provenance are preserved there.
No application shutdown, research/default/release change or scheduled cleanup.
