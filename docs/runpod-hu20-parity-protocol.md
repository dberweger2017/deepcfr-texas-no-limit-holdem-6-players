# RunPod HU20 cross-platform pilot

## Scope and freeze

The owner authorizes this small CPU rental separately from the running M4
posterior audit. The [plan](../configs/blueprint/runpod-hu20-parity.json) fixes
one seed, the unchanged uncapped HU20/K1/current-policy recipe, a from-zero
1M completed-node target and a 500k-node recovery boundary. Complete outer
iterations may overshoot both boundaries. No playing-strength evaluation,
additional seed, longer campaign or model promotion is part of this pilot.

The merged source base is `7d31a39c80cc72deb772336ce251f3c4bdcd46c9`;
the pinned engine remains `5db20e3d5d6862b32a7402035c1340b622d3b005`.
The pilot wrapper is an additive harness: it does not change trainer,
abstraction, engine, learning or random streams. Record the executed harness
revision, Python version, engine origin/build hashes and zlib versions.
Use Python 3.11 and one trainer worker on both platforms. Install only the
pinned engine and focused-test dependencies needed by this standard-library
trainer; Torch/GPU fitting is not exercised.

## Work and exact comparison

1. Train from zero through the first completed iteration at/above 1M nodes.
   Save the first complete state at/above 500k without interrupting training.
2. In a fresh process, load that midpoint, verify save/load identity, and
   continue to the same 1M completed-work stopping rule. This repeated suffix
   is recovery validation, not another independent seed or useful new work.
3. Save training and final-current artifacts at 1M. Record the next iteration's
   derived deal/action seeds and initial Python action-RNG state hashes. This
   trainer owns no persistent RNG across completed iterations: its complete
   traversal continuation is determined by config seed and iteration counter.
4. Execute one extra complete iteration in each path and save it for a next-
   iteration recovery check. Label those nodes as validation work.
5. Require exact equality of every uncompressed checkpoint record and current
   export, including all information keys, menus, regrets, strategy
   accumulators, visits, table/config identity and iteration. Check all
   non-timing iteration-report fields and completed counts across platforms.
   No tolerance is allowed for meaningful trainer state. Retain the first
   differing record on failure; stop before any longer training.

Record compressed SHA-256, uncompressed SHA-256, sizes and gzip headers
separately. Different compression bytes cannot establish a trainer difference;
identical semantic payloads cannot explain a transport difference by themselves.
If bytes differ, identify the actual header/compressor cause. Do not normalize
away an unexplained numerical or state difference.

The first comparison may use the retained first fixed-seed B preflight from
#115 (seed `2026093011`, 1,000,389 nodes / 2,626 iterations / 118,978 entries),
whose checkpoint, export and next-iteration recovery files are immutable.
Those are a historical reference, not a fresh current-source M4 run. Its
trainer/artifact/game source is unchanged; the abstraction's later optional
history-label override defaults to absent during training. A fresh current-
source M4 reference still runs **after** the active audit releases M4, before
declaring the requested current-source cross-platform result complete.

### Owner-approved reference host amendment — September 30

Before any fresh macOS reference training, the owner explicitly authorized
using the M1 for #129. The fresh reference now runs on the AC-powered M1
instead of the queued M4. The idle M4 queue is cancelled before performing
any reference training; the active #128 M4 audit stays untouched. This is a
narrow exception to the earlier travelling-M1 restriction for this fixed
pilot and its verification, not authorization for a larger campaign.

Keep the original harness revision `e18f0079a14addc90938acca8c30795e8af09691`,
Python **3.11.14**, pinned engine, seed, configuration, node boundaries and
all comparison requirements unchanged. Use one worker, a 3-GiB process RSS
cap, 10.5-GiB aggregate owned-job ceiling, 0.5-GiB swap-growth ceiling and
8-GiB free-disk floor. Limit this complete reference/verification to one
hour, with the existing 30-minute ceiling per training path. No playing
outcomes are evaluated. Record AC and resource measurements.

Label the result precisely: fresh current-source **Linux versus M1** parity,
plus exact agreement with the retained historical M4 reference. Do not claim
a fresh current-source M4 run or infer M4 throughput from the M1 measurement.

## Resources and billing

Choose the cheapest available sensible CPU offer after checking live prices.
One worker; no GPU. A 2-vCPU/4-GB offer is sufficient for the retained 0.331-GiB
1M M4 preflight, allowing considerable runtime/build headroom. Use at least
20 GB container disk so the 8-GiB free-space guard can pass after setup.
No persistent/network volume is required. Trainer peak RSS ceiling is 3 GiB;
record platform and owned-job RAM, swap and disk. Stop on nonfinite state,
failed iteration, guard failure or parity difference; retain partials.

Maximum paid rental duration is two hours including setup/transfers; maximum
cost is **$0.50**, within the earlier remaining CPU authorization and the
owner's new explicit pilot authorization. The run itself has a 30-minute
ceiling and should finish much sooner. Arm a provider shutdown watchdog
independent of the SSH training session. Retrieve and verify all artifacts
before terminating the pilot pod; no billable storage is intentionally kept.
Record the final provider cost or explicitly distinguish a rate × elapsed
estimate from settled billing.

M1 does only edits/Git/transfers/brief network control. It never loads policy
or checkpoint data, trains, builds the engine or runs tests. M4's existing
scientific job and deadline remain untouched. Linux can compare historical
M4 files while that job runs; fresh M4 training/large checks wait for release.

## Commands on Linux or free M4

```bash
python -m pytest -q tests/test_hu20_platform_pilot.py
python -m scripts.hu20_platform_pilot run \
  --plan configs/blueprint/runpod-hu20-parity.json --out results/platform-pilot/direct
python -m scripts.hu20_platform_pilot run \
  --plan configs/blueprint/runpod-hu20-parity.json --out results/platform-pilot/resumed \
  --resume results/platform-pilot/direct
python -m scripts.hu20_platform_pilot compare \
  --left results/platform-pilot/direct --right results/platform-pilot/resumed \
  --out results/platform-pilot/resume-comparison.json
```

Publish one draft PR with measured throughput, memory/save/export costs,
full-state/transport parity, independent resume evidence, hashes/retrieval
commands, instance/live rate/final cost, and a conditional three-seed runtime
and resource estimate. A short prefix does not establish late-table throughput,
safe multi-process scaling or poker strength. No subsequent run is authorized.
