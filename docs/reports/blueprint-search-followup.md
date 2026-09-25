# Saved-blueprint search follow-up: final report

The [frozen protocol](../blueprint-search-followup.md) compared corrected public-range search with the original #103 search on three checkpoints from one training lineage. All six planned benchmark runs completed on September 25, 2026. The primary 58M scripted comparison **did not show an improvement**: corrected minus original was **−6.07 BB/100 (95% CI −55.36 to +43.22)**. Corrected search itself lost **−380.02 BB/100 (95% CI −474.72 to −285.31)** against the scripted pool. This is one lineage and one evaluation schedule, so it does not establish independent-seed robustness.

## Paired results

Each checkpoint used 1,024 scripted and 256 random six-seat rotation blocks per arm. The table reports corrected search minus original search, with 95% intervals over independent blocks. All six intervals include zero.

| Checkpoint | Scripted difference, BB/100 | Corrected scripted profit, BB/100 | Random difference, BB/100 |
| --- | ---: | ---: | ---: |
| 5.83M | +19.02 [−41.02, +79.06] | −432.61 [−536.56, −328.66] | +50.39 [−27.49, +128.27] |
| 12M | +39.89 [−10.86, +90.64] | −367.73 [−468.26, −267.20] | +12.13 [−86.85, +111.10] |
| 58M | **−6.07 [−55.36, +43.22]** | **−380.02 [−474.72, −285.31]** | −12.08 [−91.46, +67.30] |

The secondary corrected-search comparison on the *same* scripted schedule gives 58M minus 5.83M **+52.59 BB/100 [−20.06, +125.25]**. The 12M minus 5.83M contrast is +64.88 [−1.93, +131.69]. Both are inconclusive. On the random schedule, the corresponding contrasts are +119.61 [−32.01, +271.23] and −24.10 [−150.79, +102.58]. Do not treat the earlier #103 positive search result as directly comparable: it used a different schedule and a different baseline policy.

## Validity and resources

Each checkpoint completed 15,360 hands across the two comparisons, with zero failed hands and zero invalid actions. The 5.83M and 12M runs used the 16-GiB M4 under a 10.5-GiB RSS guard. The 58M run used an eight-vCPU, 64-GB AMD EPYC 9575F RunPod CPU host under a 50-GiB RSS guard and two-hour runner limit. Runtime from manifest to final result was about 13.6, 13.4, and 13.6 minutes, respectively. The RunPod account balance moved from $9.70 to $9.43 during the rental; the pod was stopped after artifact retrieval and shows $0.00/hour.

| Checkpoint | Peak process RSS | Scripted searches / fallbacks | Scripted search p95 | Scripted range lookups: trained / untrained / off-tree / off-menu | Scripted continuation lookups: trained / untrained / off-tree |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5.83M | 5.08 GiB | 5,001 / 1 | 0.120 s | 901,045 / 2,305,306 / 317,856 / 182,880 | 56,456 / 1,644,830 / 448,429 |
| 12M | 6.49 GiB | 4,878 / 2 | 0.120 s | 954,330 / 2,189,958 / 284,544 / 167,136 | 99,108 / 1,607,020 / 421,389 |
| 58M | 45.73 GiB | 4,741 / 11 | 0.127 s | 934,498 / 2,108,414 / 299,520 / 168,384 | 200,693 / 1,440,720 / 425,076 |

The 58M random arm made 430 searches with zero fallback, p95 0.166 seconds, and peak RSS 45.64 GiB. Its scripted fallback rate was 11/4,741 attempts; no fallback caused an invalid hand. The 58M continuation's trained-lookup share increased to 9.7% from 2.6% at 5.83M, yet no paired strength improvement was established. Those counts describe internal search calls, not an independent sample of poker decisions.

## Artifacts and verification

The machine-readable [comparison report](blueprint-search-followup.json) contains the full results and telemetry. Reproduce it with `python -m scripts.report_blueprint_search_followup --5m results/search-followup/5m --12m results/search-followup/12m --58m results/search-followup/58m --out results/search-followup/report.json`. The script verifies every file against each run's checksum manifest, verifies each hand digest, checks clean source and identical schedules/search settings, then computes checkpoint contrasts by paired block. It produced identical bytes on the M4 and local Mac.

| Checkpoint | Checkpoint SHA-256 | Result checksum manifest SHA-256 |
| --- | --- | --- |
| 5.83M | `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a` | `3aca86b267519e445acdbb646017368b004fc76f8e8f51e0e75fb5780c6b5a92` |
| 12M | `c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845` | `fd5aeeaa7d56e5ffe2334ae2f8893ed6ccbf1c8154b7e7b2db99ebdc73d0619d` |
| 58M | `1b7d9ef0a6f111ac82d99f802cf7ac7bc5c8685ff6a2f214cb918f7ef9361bc2` | `071b3e8f8723dfaeb6603d4e9df32af489e7ee3c20c1c84c57de81d2300c3fae` |

All raw hand rows, manifests, TensorBoard events, and logs are retained in `~/Local/blueprint-search-followup/results/search-followup/` on the M4 and in the ignored local `results/search-followup/` directory. The complete three-checkpoint `campaign-archive.tar.gz` is on both machines with SHA-256 `f5887e862dd3ee7a86dfe0397432a81196952432ae635f79ec84ba4af11a892b`. The original 58M pod archive is SHA-256 `b0817aedb6e2dd4a2cce90af64dce6ec957b4cc66a665cb2c6277fd8a6072fcf`. Run manifests record source commit `b21a840042b60a162ee3faba034a36b3cefef6ff` and package/native-engine hashes. TensorBoard events were emitted every 64 completed rotation blocks and mirrored from the paid host before shutdown.

## Decision

The predeclared primary improvement criterion requires the scripted paired interval's lower bound to exceed zero; it failed. Larger checkpoints also did not show a clear paired gain, while 58M requires roughly 46 GiB for evaluation and the agent still loses to scripted opponents. **Do not promote corrected search or fund another long high-RAM blueprint run on this evidence.** Keep the corrected implementation available for diagnosis. The next playing-strength investigation should use retained hands and lookup telemetry to find where scripted opponents exploit the policy, then declare a targeted change and fresh evaluation schedule before more paid scaling. This is a recommendation from this comparison, not a new campaign authorization. v0.5 remains unqualified.
