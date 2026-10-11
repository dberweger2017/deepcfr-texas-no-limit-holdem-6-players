# v0.5.0 publication verification

**Published October 10, 2026 at 14:08:27 Madrid (12:08:27 UTC), stable Latest.** [Release](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.5.0) · [release PR #227](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/227).

The owner authorized this release in chat. Tag `v0.5.0`, the release target and `release-manifest.json` all name **`b9c9bd161422e87371a2641ae4c60304c66eed79`**, the merge of #227 after all five checks passed. Release ID **408924208**; the Latest API returns it with draft and prerelease both false.

The model is #207's export, unchanged: seed **2026100601**, 1,000,002,065 nodes, **1,173,264,021 bytes**, SHA256 **`47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9`**. It was copied from the canonical M4 original, whose hash matched before the copy. There was no retraining, re-extraction or seed selection.

## Verification

1. **Build:** a clean detached checkout of the merge commit built the bundle with `--publication-approved`. The bundled standalone verifier passed with `--expect-source b9c9bd16… --require-publication`.
2. **Draft:** all seven assets were uploaded to a draft release, downloaded back with authenticated GitHub downloads into a fresh directory, and verified again. They were byte-identical to the built bundle.
3. **Public:** after publication, all seven assets were downloaded again from their public URLs without credentials, and the standalone verifier passed on them. They were byte-identical again.
4. **Runtime:** the release smoke ran on the M1 at the release PR's head, with these model bytes. Two real CLI loads took 236–243 s; 8 human hands and 2 spectator hands were played, with a restart before a pending bot decision. Independent replay of every journal passed, and the HU20 table and a changed inference identity were rejected. Peak RSS was 3.7 GB. The merge added only a storage-cleanup roadmap note, so the smoke wasn't repeated, to keep the M1 free for #225's guarded scoring.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `INSTALL.md` | 1,956 | `a9c00d3fded741fe79ab5ec3442496d327ae9bb0b7975dbaf839d782550e7379` |
| `MODEL_CARD.md` | 4,550 | `8286ac696e3ce3538265b6e1922bcac4cd0f8c596b92734595a8a8b43636ad3c` |
| `O1B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 1,173,264,021 | `47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9` |
| `release-manifest.json` | 2,466 | `9a7a7209087e5e8ec7d93ab06f382233777cefcbbe0e905f48ea845468cee729` |
| `RELEASE_NOTES.md` | 1,506 | `26f50a8653383463d9cd8a5a7c48a962d1159f223d349b6198d23be6c4c81fe8` |
| `SHA256SUMS` | 542 | `c3241e24ee63252003f15e024d2de1996ec342756ca639c473352e09abf95412` |
| `verify_v050_bundle.py` | 7,238 | `575f5c7ac975933e9afc9ab0fc6a344846b9788cb427b3cd29c051f9c91b9042` |

## Older releases

v0.4.2 (7 assets), v0.4.1 (5) and v0.4.0 (4) remain public and unchanged. Each one's first asset returns HTTP 200. The HU20 table's default stays v0.4.2; v0.5.0 runs as its own table with `--v050`.
