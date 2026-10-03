# RunPod quote — approval pending

Observed in the signed-in RunPod console on October 3, 2026, 07:51–07:54 UTC.
No deployment was submitted. Availability is not reserved.

| Item | Quoted configuration / cost |
|---|---|
| CPU | 5-GHz Memory-Optimized, 2 vCPU, 16 GB, **High** availability |
| Compute | $0.13/hour |
| Container disk | 100 GB, $0.014/hour, ephemeral; no network volume |
| Combined | **$0.144/hour** from itemized charges (UI rounds to $0.14) |
| Expected wall | Preliminary 12–20 hours, subject to native Linux pilots |
| Hard rental clock | **24 hours total**, including build, qualification and failed compute |
| Retrieval/shutdown reserve | Last **2 hours included**; production stops before hour 22 |
| At the quoted rate | $1.728–$2.880 for 12–20 hours; $3.456 for 24 hours |
| All-in authorization requested | **$4 USD maximum**, no credit purchase, no replacement rental |
| Transfer | $0 provider fee; conservatively reserve time for transfer |

[RunPod billing](https://docs.runpod.io/accounts-billing/billing) states no data
transfer fees and per-second compute/storage billing. The console labels its
checkout per millisecond; use the hourly breakdown for this conservative bound.
The $4 cap leaves $0.544 beyond the exact 24-hour subtotal for fees/rounding/tax;
abort before payment if any mandatory charge would exceed it. Rate ceiling
$0.15/hour including disk. Recheck this same offer before deployment. Larger
3-/5-GHz memory shapes were unavailable at the final check; earlier transient
availability vanished when selected. No substitution beyond this quote.

Use two independent solver workers, one Rayon thread each, `nice`, **4 GiB
pre-allocation arena allowance, 5 GiB owned RSS per worker, 11 GiB aggregate**.
The aggregate must additionally be <=80% measured available cgroup/host memory.
If two workers do not pass admission, stop and report; do not quietly serialize,
increase budgets, change menu or rent more capacity. Preserve >=20 GiB disk
free. The recorded #145 2-BB limped stored-average roots had 465-second median
and 493-second maximum end-to-end time on M4 with two Rayon threads, maximum
4.493 GiB RSS and 1.834 GiB compressed API estimate. Those timings include more
projection/secondary evaluations than this pipeline and are not a Linux
benchmark. The wall estimate remains tentative until the fixed pilots.

Native Linux qualification precedes production. Forecast remaining main time
as 1.5 × slowest fixed pilot wall × 240 root solves / 2 workers. Start the main
campaign only if that estimate fits the unreset remaining clock minus the
retrieval reserve. Failure to fit is a stop and an owner decision, not a smaller
outcome-selected corpus or a budget extension.

Scope includes the native external Rust build, portable macOS/Linux fixture
parity, three 20,000-deal real-export V4 pilots, shared-codebook preparation,
120 collection jobs, eligible-root deterministic replay/relocking, all raw
retrieval and hash verification, and termination of the owned pod and ephemeral
storage after verification. No training, promotion, M4 use or merge. No network
volume or long-lived storage charges. Preserve all failed/partial evidence.

The local approval screenshots are
`/Users/dberweger/Local/hu20-board-pooling-20261003/runpod-quote-full.jpg`
and `runpod-quote-checkout.jpg`; they are not published because the console header
includes private account information. The tracked JSON quote records the offer
and screenshot hashes without account details.
