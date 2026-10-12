# v0.5.2 publication verification

**Published October 12, 2026 at 17:03:26 Madrid (15:03:26 UTC), stable Latest.** [Release](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.5.2) · [release PR #240](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/240).

The owner authorized this release in chat. Tag `v0.5.2`, the release target and `release-manifest.json` all name **`3a3bc4218bab96637d6a27a69b73383ee8bb196c`**, the merge of #240 after all five checks passed. Release ID **410675053**; the Latest API returns it with draft and prerelease both false.

The model is #226's export, unchanged: seed **2026100601**, 4,000,000,174 nodes, iteration 3,716,073, **2,121,415,396 bytes**, SHA256 **`0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0`**. It was restored from #226's accepted archive on the M1: whole-archive SHA256 `ea58899645688f3fb391c12b501dc0941754878e96eec165b835e1847b4105b1` and manifest `5dad61abfe120225592fb3b29f11bb812febc6ca438635ccb1d333b9ea4dd664` matched their index entries, and the member's bytes, hash and header matched the release pins. There was no retraining for this release, no re-extraction and no seed selection.

## Verification

1. **Runtime:** the release smoke ran on the M1 with this model at #240's code. Two real CLI loads took 395 s each; 8 human and spectator hands were played, including an off-menu 550 raise, an all-in and a restart before a pending bot decision. Independent replay of every journal passed, and the HU20 table and a changed inference identity were rejected. Sampled peak RSS was 4.0 GB. Only the release docs changed between that run and the merge, to record these numbers.
2. **Build:** a clean detached checkout of the merge commit built the bundle with `--release v0.5.2 --publication-approved`. The bundled standalone verifier, `verify_hu100_bundle.py`, passed with `--require-publication`.
3. **Draft:** all seven assets were uploaded to a draft release, downloaded back with authenticated GitHub downloads into a fresh directory, and verified again. They were byte-identical to the built bundle. The model's first upload attempt timed out on GitHub's side (HTTP 408) before any asset existed under its name; the retry completed.
4. **Public:** after publication, all seven assets were downloaded again from their public URLs without credentials, and the standalone verifier passed on them. They were byte-identical again.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `INSTALL.md` | 2,144 | `c0d0a2fcd7e6fe724b041881c15b360f8869d2fb01ece35c6ad9e3e706954fe0` |
| `MODEL_CARD.md` | 6,055 | `21819e27ef83aabb1bf2b287510dd224dbab6d1193180e1b22a657ec468b86c9` |
| `O4B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 2,121,415,396 | `0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0` |
| `release-manifest.json` | 2,549 | `0cd861e293f84c3aaedc036546634decbc1fe888d433070768e527cc4c14d51b` |
| `RELEASE_NOTES.md` | 1,519 | `55a3a7f09045039813bdab116fc82ac1f70bf4bc565e55c7fe679285e63bdae6` |
| `SHA256SUMS` | 543 | `38badc1bfc70d727191fcbd26584527e0ea68b4aecc3ca187403a214e2a3c0b3` |
| `verify_hu100_bundle.py` | 11,821 | `e27a621be82e533dc5bdc29646fec6944b26e501cbc1bd871859b84f7e3bd817` |

## Older releases

v0.5.1 (7 assets), v0.5.0 (7), v0.4.2 (7), v0.4.1 (5) and v0.4.0 (4) remain public and unchanged. `--hu100-release` loads v0.5.0, v0.5.1 and v0.5.2 bundles, and the rebuilt v0.5.0 manifest is byte-identical to the published one. The HU20 table's default stays v0.4.2.
