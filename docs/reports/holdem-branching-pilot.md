# Second-decision training: local cost pilot

Both arms of the [committed pilot](../holdem-branching-training.md) completed, with no failures, retries or rental spend. This is an engineering and cost result, not a playing-strength comparison. [Compact measurements](holdem-branching-pilot.json) retain the source/environment manifests, all iteration timings, coverage, artifact hashes and tiny evaluation results.

| Measurement | First decision | First two decisions |
| --- | ---: | ---: |
| Total run seconds | 48.86 | 51.90 |
| Mean iteration seconds | 8.11 | 7.66 |
| Mean collection seconds | 2.74 | 2.52 |
| Mean fitting seconds | 5.12 | 4.84 |
| Mean replay/admission seconds | 0.22 | 0.28 |
| Peak process RSS during training, GB | 0.664 | 0.825 |
| Final checkpoint, MB | 35.33 | 42.08 |
| Final checkpoint serialization, seconds | 8.18 | 9.39 |

The eight iterations completed in 100.75 seconds including checkpointing and evaluation. Each arm used seed 701, four iterations, 192 roots per iteration, width 32, 64 fitting steps and the production replay capacity of 4096 per role. The two arms differ only in collector depth. Their later policies and visited trees differ, so a slightly shorter mean candidate iteration is not a fixed-workload speedup.

## Coverage remains visible

Each row has 192 scheduled role/root cases; branch records are correlated.

| Iteration | First: roots with postflop | Second: roots with postflop | First: postflop records | Second: postflop records |
| --- | ---: | ---: | ---: | ---: |
| 1 | 47 | 90 | 67 | 240 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 5 | 0 | 6 | 0 |

Neither arm has established useful postflop coverage after these few fits. Four iterations cannot rank the approaches, but they show why the longer comparison must keep coverage visible. The final style and random evaluations have only eight paired blocks, intentionally below the arena's confidence-interval threshold. Their raw outcomes remain saved; no arm was selected by its score.

## Recovery and validation

All 631 repository tests pass, including both sampler modes in 4/5/6-player recovery and fresh-process continuation, exact expected gradients including later sampled decisions, corrupted expansion labels, reservoir/root normalization and pre-change collector fingerprints. A telemetry-only follow-up emits explicit zero street counts so TensorBoard does not retain a misleading last nonzero value when a street disappears.

Both actual final boundaries were resumed in separate CLI processes without another optimizer step. Final training checkpoint, average export, both evaluation reports and both raw outcome files reproduce byte for byte for each arm. All six catalogued artifact hashes are verified, including iteration-two checkpoints. The verification record is retained at `results/branching-pilot-verification.json`; source artifacts and resumed copies are under `results/branching-pilot-{first,second}*`.

Measured revision: `010334e65c1420d444684dfac0020b99ca58032d`; integration and pilot protocol committed at `b093e2b`. The raw timing data retains sparse street maps as originally measured; explicit zero telemetry was added afterwards and does not change learning or these measurements.

## Remote planning decision

Early iteration costs suggest roughly 65–70 minutes of pure training for 512 iterations on this local CPU, before growing replay/archive cost and evaluation. This is not a remote ETA. Six seed/arm processes require a concurrency and memory check on the selected host; early sub-1-GB process RSS does not bound later checkpoint serialization peaks.

Proceed to a declared Runpod CPU comparison within a **four-hour / $3.50 total rental cap**, below the remaining $6.13 CPU authorization. A 16-vCPU / 32-GB, advertised 5-GHz compute instance currently quotes $0.56/hour plus disk. Reserve enough ephemeral disk for all checkpoints, evaluations and retrieval archives; include its actual price in the cap. Use a short host calibration before campaign seeds and stop if measured throughput or memory makes the limits untenable. GPU spend is separate and unused.

The following campaign must fix seeds, workload, paired evaluation, stopping rules and interpretation before launch. No production model is promoted by this pilot.
