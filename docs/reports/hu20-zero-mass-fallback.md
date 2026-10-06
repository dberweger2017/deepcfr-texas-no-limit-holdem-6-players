# M1 zero-mass current fallback comparison

The six #165 1B checkpoints were re-exported from merged #178 with `--zero-mass current`. T′ and O′ differ from the exact original T and O averages only at stored zero-mass keys. Positive-mass averages, training and missing-key behavior are unchanged. No release decision.

| Contrast | Three-lineage BB/100 [95% interval] | Predeclared label |
| --- | ---: | --- |
| Primary A: T′ vs T | +0.18 [-2.19, +2.56] | no detectable difference |
| Primary B: O′ vs O | -0.35 [-2.71, +2.02] | no detectable difference |
| Secondary: T′ vs shipped R1 | +5.46 [+2.47, +8.46] | better |
| Secondary: O′ vs shipped R1 | +6.48 [+3.48, +9.48] | better |
| Secondary: T′ vs O | -0.99 [-3.36, +1.39] | no detectable difference |

T’s fallback has no detectable direct difference from its original export. O’s fallback has no detectable direct difference from its matched original. These are direct pairwise results for the retained policies; nondetection does not establish equivalence or general poker strength.

## Every retained lineage

| Contrast | Training seed | BB/100 [95% interval] | Label |
| --- | --- | ---: | --- |
| Primary A: T′ vs T | 2026100601 | -0.03 [-3.40, +3.34] | no detectable difference |
| Primary A: T′ vs T | 2026100602 | -0.46 [-3.86, +2.94] | no detectable difference |
| Primary A: T′ vs T | 2026100603 | +1.04 [-2.32, +4.40] | no detectable difference |
| Primary B: O′ vs O | 2026100601 | +0.27 [-3.07, +3.62] | no detectable difference |
| Primary B: O′ vs O | 2026100602 | -1.62 [-4.99, +1.75] | no detectable difference |
| Primary B: O′ vs O | 2026100603 | +0.30 [-3.03, +3.64] | no detectable difference |
| Secondary: T′ vs shipped R1 | 2026100601 | +7.67 [+3.87, +11.47] | better |
| Secondary: T′ vs shipped R1 | 2026100602 | +5.34 [+1.55, +9.13] | better |
| Secondary: T′ vs shipped R1 | 2026100603 | +3.37 [-0.36, +7.11] | no detectable difference |
| Secondary: O′ vs shipped R1 | 2026100601 | +8.55 [+4.78, +12.33] | better |
| Secondary: O′ vs shipped R1 | 2026100602 | +6.38 [+2.61, +10.15] | better |
| Secondary: O′ vs shipped R1 | 2026100603 | +4.52 [+0.78, +8.25] | better |
| Secondary: T′ vs O | 2026100601 | -0.78 [-4.12, +2.57] | no detectable difference |
| Secondary: T′ vs O | 2026100602 | -1.79 [-5.17, +1.59] | no detectable difference |
| Secondary: T′ vs O | 2026100603 | -0.39 [-3.74, +2.95] | no detectable difference |

## Frozen method and audit

Labels were declared in [#179 before the pilot](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/179#issuecomment-6015294307): better if lower bound >0, worse if upper bound <0, otherwise no detectable difference. All intervals are nominal Student-t 95%, conditional on three retained training lineages. Seats and lineages are averaged inside independent deal blocks; no multiplicity-adjusted claim. One BB is 100 chips, so mean net chips per hand equals BB/100.

The #175 direct runner, loader, duplicate deals/seats, separate action streams, reporter and independent auditor have unchanged source hashes. A storage adapter gzips only metadata and logs. Final root **202610063501** is disjoint from excluded pilot **202610063401**. Counts come from outcome-blind maximum block SDs across each primary’s three lineages and aggregate, targeting ≤3.5 BB/100 with headroom for requested ≤4. All secondary families use the larger primary count. No sample extension.

| Family | Paired deal blocks | Final hands |
| --- | ---: | ---: |
| Primary A: T′ vs T | 36,864 | 221,184 |
| Primary B: O′ vs O | 36,864 | 221,184 |
| Secondary: T′ vs shipped R1 | 36,864 | 221,184 |
| Secondary: O′ vs shipped R1 | 36,864 | 221,184 |
| Secondary: T′ vs O | 36,864 | 221,184 |

Canonical bundled final plan SHA256 `1c379fec6390931a8d5f88dbf9c0390d552ef4fa6c2f4575748dfdbfd740bd18`. Achieved primary aggregate/lineage half-widths range **2.37–3.40 BB/100**, all meeting the requested ≤4. Counts and roots remain frozen.

Every **1,105,920 final hands / 5,721,474 actions**, plus **61,440 pilot hands / 318,419 actions**, independently replay and reproduce all aggregate, lineage and position estimates and labels. Pilot arithmetic was audited only after counts froze; pilot scores were never used to size or select outcomes.

M1 only, free compute, at most three workers, unchanged 6-GiB worker guard and 8-GiB free-disk stop floor. Startup-inclusive final play quote **24.5 minutes**, independent replay/report allowance **8.1**, **49.0** with 50% headroom; two-hour final cap. Completed final play: **29.49 minutes**; final independent report/replay: **4.48 minutes**. All outputs, failures and partials remain compressed; no file was deleted or evicted. The owner independently freed M1 space during preparation. No M4 compute or RunPod.

## Export identity

All six checkpoint SHA256s and original T/O/R1 sizes and hashes match #165’s frozen manifest. Shipped R1 is `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`. Native release source is merged main `0a3765f34335b8ca8efea7d79e25c7fd19d94657`; relevant native/loader files match exactly. Each `cfr_average.audit` checks every checkpoint accumulator, emitted probability, visit count, stored regret/current policy and coverage.

| Export | Audited keys | Zero-mass keys | Export SHA256 |
| --- | ---: | ---: | --- |
| Tprime-2026100601 | 3,350,938 | 2,593,211 | `2cb66b0b04b9edc57283f72fb486e72624cdd50a332e7f4f8f044bcc8681bbca` |
| Tprime-2026100602 | 3,375,797 | 2,617,480 | `6e8dc452be781924b2ab4948615c3c6feb5aa6387be776b85afec8846beb5504` |
| Tprime-2026100603 | 3,345,507 | 2,589,249 | `45546b602ffb14b90fb81c7f4e8a49539702f734b90d59d1def572a290eaeeb8` |
| Oprime-2026100601 | 4,319,080 | 1,389,710 | `d0cf1cb61733f0382af8828687f04647314749d1e02fcbd8e7986cfd8b547973` |
| Oprime-2026100602 | 4,343,088 | 1,410,789 | `6384553230241a604d792bb8df48051fdf29cc26e9e4569f344d4f53180616ed` |
| Oprime-2026100603 | 4,308,395 | 1,385,523 | `b91105467956d6e7064ad5641f52c06e6248a72f5672c9a768e02453aa31014e` |

## Conditional native pressure

Neither primary aggregate was labeled better, so the conditional pressure check was not triggered.

## Validation and archive

Review of #178 found no unresolved findings; both full CI shards, test aggregation and GitGuardian were green before merge. Local validation: 28 native parity/bench tests, 165 diagnostic tests, Rust tests, 16 existing measurement tests and two real-arena pressure replay tests pass. Retained attempts include an initial concurrent Git-ref refresh failure (corrected before verified merged build), native tests initially skipped before the explicit built rerun, and the small synthetic fixture corrected to the arena’s inference minimum. No poker run was repeated for an outcome.

The complete member-hashed ZIP, exact checkpoint/policy inputs, frozen source/binary/environment, plans, pilot/final raw traces, logs, failures, audits and restoration instructions are under `~/Local/Research-Cloud/PR-179-HU20-zero-mass-fallback/`. Archive/member readback receipts and hashes are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md). Originals stay; no synced files are deleted or evicted.
