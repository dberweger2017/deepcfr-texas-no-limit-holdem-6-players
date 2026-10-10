# K50 bench scoring: M1 admission stopped

**Not admitted; no scientific outcome.** The separately requested scoring attempt on October 10, 2026 stopped during its first host inspection, before restoring inputs, running a timing pilot or launching the lock evaluator. The M1 was on battery and total system swap was 4,183.62 MiB (about 4.387 GB), above the frozen 3,000,000,000-byte ceiling. No scores were produced or inspected, and no scientific command was retried.

The fresh isolated checkout is `~/Local/hu20-equity-bench-scoring`, branch `feature/hu20-equity-bench-scoring`, based on current main `ff984da9a16bf1e76e891fab239d0785188a93af`. Host identity was arm64 MacBookPro17,1, eight cores and 16 GiB RAM. Disk showed 26 GiB available and system free memory 51%; a worker family and kernel pressure level were not measured because admission had already failed. The [admission receipt](hu20-equity-bench-scoring-artifacts/admission.json) preserves the raw power, swap and headroom readings, their timing limitation and the intended guards.

The owner's M1 authorization retains a 7 GiB whole-family ceiling, normal pressure, at least 15% system free, total swap at most 3,000,000,000 bytes, AC power and at least 16 GiB free disk. Owner use of the M1 is compatible with these guards only while they pass. The instruction to stop and report any guard breach blocks this attempt; no process was stopped to make room and no limit was relaxed. The existing overnight launcher also identifies the M4 explicitly, so it was not invoked on this M1.

## Frozen question remains open

The [#222 protocol](../hu20-equity-bench.md), exports, roots, folds, references and matched checkpoints (4,000,977 /3,855,889) remain unchanged. The original [#222 stop](hu20-equity-bench.md) stays preserved. This is a separate unsuccessful admission, not a continuation or retry of its science. Accepted #222 and #163 input archives were located but not copied, opened, extracted or hashed; [their restoration index](hu20-equity-bench-artifacts/model-input-index.json) remains authoritative.

There is no complete pilot timing quote to publish and no 40-root E/Q table, learning curve, paired bootstrap or trained-policy gap to the #190 K50 witness (E=0.3917). No pass, fail or inconclusive result can be assigned. The stopped admission rules out nothing about K50's trained quality and does not support advancing to full-game step 4. Roadmap item 4.3 remains unresolved.

The single immediate next step is for the owner to make the M1 eligible (AC power and total swap below the unchanged ceiling), then explicitly authorize a new admission. This attempt stays stopped; it will not watch the host or resume automatically.

## Closeout

The [independent scoring/source review](hu20-equity-bench-scoring-artifacts/source-review.json) confirms refusal of admission. The separate [end evidence review](hu20-equity-bench-scoring-artifacts/evidence-review.json) independently verifies the archive and upload receipts with no findings. The 8,641-byte, seven-payload-member [administrative archive](https://drive.google.com/file/d/1We8DP3Jbz3SPk2HbKgr5fgNJou8r46Ix/view) passes local manifest/member readback and separate native/cloud upload acceptance ([receipt](hu20-equity-bench-scoring-artifacts/archive-receipt.json), [acceptance](hu20-equity-bench-scoring-artifacts/cloud-upload.json)). Remote archive bytes were not downloaded. Only admission and administrative evidence are archived; there are no scores, derived scientific outputs or restored input copies to archive or delete. Native trainer code, defaults and releases are unchanged. No M4, paid compute or other PR cleanup was used.
