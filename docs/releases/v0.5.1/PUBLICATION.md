# v0.5.1 publication verification

**Published October 11, 2026 at 11:52:11 Madrid (09:52:11 UTC), stable Latest.** [Release](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.5.1) · [release PR #233](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/233).

The owner authorized this release in chat. Tag `v0.5.1`, the release target and `release-manifest.json` all name **`dca7547e6eca82ce77f3e42230b82fabc1c29a4c`**, the merge of #233 after all five checks passed. Release ID **409575446**; the Latest API returns it with draft and prerelease both false.

The model is #223's export, unchanged: seed **2026100601**, 2,000,000,460 nodes, **1,603,422,956 bytes**, SHA256 **`e6f79ccac39a651352424e05382681f4d1994bb570b025f692556d170f87e9ae`**. #223's original had been removed from the M4 after archival, so the release copy is #226's byte-identical regeneration from the seed. #226 fully audited it, and its hash matched before and after the copy. There was no retraining for this release, no re-extraction and no seed selection.

## Verification

1. **Build:** a clean detached checkout of the merge commit built the bundle with `--release v0.5.1 --publication-approved`. The bundled standalone verifier, `verify_hu100_bundle.py`, passed with `--require-publication`.
2. **Draft:** all seven assets were uploaded to a draft release, downloaded back with authenticated GitHub downloads into a fresh directory, and verified again. They were byte-identical to the built bundle.
3. **Public:** after publication, all seven assets were downloaded again from their public URLs without credentials, and the standalone verifier passed on them. They were byte-identical again.
4. **Runtime:** the release smoke ran on the M1 with this model at the release PR's code. Two real CLI loads took 304–305 s; 8 human hands and 2 spectator hands were played, with a restart before a pending bot decision. Independent replay of every journal passed, and the HU20 table and a changed inference identity were rejected. Peak RSS was 3.8 GB. Only documentation changed between that run and the merge.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `INSTALL.md` | 2,121 | `0d1cfd3048f51eaec59efd18d2377ede70eff43dad98811197c3dc68a86ce086` |
| `MODEL_CARD.md` | 5,402 | `a321b57698a436d119bfbba0dae8bbd8a274b91cbc30617f8aeca586c22f4c0e` |
| `O2B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz` | 1,603,422,956 | `e6f79ccac39a651352424e05382681f4d1994bb570b025f692556d170f87e9ae` |
| `release-manifest.json` | 2,466 | `47185da27bd8589b249938fd52c58b57975423f56c1e32037d35b26f8f765df6` |
| `RELEASE_NOTES.md` | 1,524 | `8d53769145155acb5f123ea484a6fc782c998f044322d811ba38362f67c43bfe` |
| `SHA256SUMS` | 543 | `d3541636c1ecdb8cb2c5137266c8d5ab1bff4e30657c7d59c4b546f1510689b7` |
| `verify_hu100_bundle.py` | 10,186 | `fa488ebe7b12a1e03ffbbfe8ba7ca78f7a57c06857b9e67f1546300a8e47f662` |

## Older releases

v0.5.0 (7 assets), v0.4.2 (7), v0.4.1 (5) and v0.4.0 (4) remain public and unchanged. `--hu100-release` loads both v0.5.0 and v0.5.1 bundles, and the rebuilt v0.5.0 manifest is byte-identical to the published one. The HU20 table's default stays v0.4.2.
