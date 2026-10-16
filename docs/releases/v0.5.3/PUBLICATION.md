# v0.5.3 publication verification

**Published October 16, 2026 at 09:33:39 Madrid (07:33:39 UTC), stable Latest.** [Release](https://github.com/dberweger2017/CFR-NLHE-6max/releases/tag/v0.5.3) · [package PR #298](https://github.com/dberweger2017/CFR-NLHE-6max/pull/298) · [release-docs PR #302](https://github.com/dberweger2017/CFR-NLHE-6max/pull/302).

The owner authorized this release in chat on October 16. Tag `v0.5.3`, the release target and `release-manifest.json` all name **`74f1be71c1b0f4851dac5e701862e22a1290da23`**, the merge of #302. #302 changed only the three shipped docs (release notes, model card, install guide), which still described #298's owner-review bundle. All four checks on that commit passed; its first CI run was cancelled by a later push to main and was rerun on the same commit. Release ID **414617811**; the Latest API returns it with draft and prerelease both false.

The model is PR272's frozen 5B average, unchanged: seed **2026100601**, 5,000,006,023 cumulative nodes, iteration 1,794,658, 97,290,578 entries, **3,196,291,551 bytes**, SHA256 **`c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`**. It ships as three ordered parts of at most 1 GiB, together with its flop, turn and river K50 tables. There was no retraining, re-export or seed selection for this release.

## Verification

1. **Runtime:** #298's real M1 smoke used this model and these tables at source `ffa9ff7d`. It covered:
   - two fresh CLI loads, each about 10 minutes, with a peak CLI child RSS of 1.74 GB;
   - 37 completed human and spectator hands plus one partial, all independently replayed;
   - restart before a pending bot decision, and refusal of a wrong model, table or identity.

   Between that source and the release commit, the runtime and bundle code are unchanged: the only code added is #297's separate blend scripts, which the runtime doesn't import.
2. **Build:** a clean detached checkout of the release commit built the bundle with `--release v0.5.3 --publication-approved`.
   - The three model parts, three tables and the verifier are byte-identical to #298's reviewed inventory.
   - The bundled standalone verifier passed under `python3.11 -I -S` with `--expect-source 74f1be71… --require-publication`, and `SHA256SUMS` checks.
3. **Draft:** all twelve assets were uploaded to a draft release with no retries.
   - Each was downloaded back with authenticated GitHub downloads and hashed as it streamed. All twelve matched the built bundle byte for byte.
   - To save M1 disk, the downloads were hashed while streaming rather than stored a second time.
4. **Public:** after publication, all twelve assets were downloaded again from their public URLs without credentials and hashed the same way. All twelve matched.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `flop-k50.bin` | 12,867,944 | `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111` |
| `INSTALL.md` | 2,570 | `697e5d77578a33eea8b32ecff33aedaaa29173ca694a7e347b0b7e7a5025b5b8` |
| `MODEL_CARD.md` | 4,420 | `a63ea023144a552aa1ba4ea439c2fd9d1beaea9d72e0952a022d4243079807f7` |
| `O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-001` | 1,073,741,824 | `3a0a95117efe3fafc1c99a93563683de57f976ec776ee0b755daf682af5a66c7` |
| `O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-002` | 1,073,741,824 | `9ef0c480cc7ffea65a22c01d986b4ad6522060445d855a5e73734fcec3160592` |
| `O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz.part-003` | 1,048,807,903 | `f4171fa4309ada3df445cf0f70ff333296f4be4ff285eec88c1f51e49089f0e6` |
| `release-manifest.json` | 3,676 | `86a6cd999db08b1af5ccf1551b12db4b4fb7e80f1ab23917c3b397c940b707db` |
| `RELEASE_NOTES.md` | 2,757 | `f4337ffa4daa3d7cf17142a89502a87cc3c3990710bae5df0a086d6615c30c33` |
| `river-k50.bin` | 1,231,562,564 | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |
| `SHA256SUMS` | 1,066 | `7600feb9be20b9f67ce5a72be619aeaf7d55ebab0b68f0553384cf341198df12` |
| `turn-k50.bin` | 139,600,524 | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |
| `verify_hu100_bundle.py` | 19,347 | `e43e44e1a6fe1264ddfd74646eaf0ff4a31e94d318fe70ce62086dd4851717d7` |

The built bundle is kept on the M1 at `~/Local/v053-publication/assets/`.

## Older releases

v0.5.2, v0.5.1, v0.5.0, v0.4.2, v0.4.1 and v0.4.0 remain public and unchanged. `--hu100-release` loads v0.5.0–v0.5.3 bundles; #298 confirmed that old manifests rebuild byte-identically and v1 trace hashes are unchanged. The HU20 table's default stays v0.4.2.
