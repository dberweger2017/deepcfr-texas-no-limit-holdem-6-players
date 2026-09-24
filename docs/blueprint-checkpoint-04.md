# Checkpoint 0.4: monitored blueprint campaign

## Decision

Continue the original six-player, 100 BB blueprint seed `2026092402` from iteration 8,733, 5,834,622 entries, 45,157,550 traversal nodes, and checkpoint SHA-256 `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`. The recipe and abstraction remain fixed. Only the operational entry ceiling rises to 100 million so RAM, work, or time can determine the stopping point. The goal is to learn whether profitable random-opponent play improves as the table grows; scripted-opponent validation reveals whether gains generalize. This is not a v0.5 qualification run.

Use one 3 GHz memory-optimized RunPod CPU pod with 8 vCPU and 64 GB RAM, one trainer worker, and a 50 GiB conservative parent-plus-worker RSS guard. Verify the pod's live entitlement and price before starting. The owner-reported September 24 quote is $0.44/hour for CPU plus disk. The first continuous segment has a 24-hour wall cap (at most $10.56 of CPU at that quote, plus disk); it may stop sooner on RAM, disk, or correctness. Do not silently extend paid compute beyond this segment. Select enough disk for checkpoint snapshots and keep at least 10 GiB free. Retain artifacts and verify hashes on the M4 before terminating a disposable pod. Record the actual CPU model, vCPU entitlement, memory limit, disk, sustained nodes/second, peak RSS, and settled cost.

## Measurements

The campaign configuration is [checkpoint-04-campaign.json](../configs/blueprint/checkpoint-04-campaign.json). At the source checkpoint, the first crossing of 6, 12, 24, and 36 million entries, and the final stop, save immutable training snapshots. Also overwrite a recoverable latest checkpoint at least hourly. At each measured snapshot, evaluate the current regret-matched policy **from the paused in-memory table**, without exporting a duplicate full policy. This leaves the training table intact and avoids the current export-memory spike. The comparison arm is a uniform choice over the same abstract action menu.

- Random validation: 1,024 independent blocks, 6,144 candidate hands, fixed split and deals at every checkpoint. The `random` policy chooses legal action kinds, including minimum and all-in raises. Report candidate BB/100, 95% interval, and paired difference from uniform.
- Scripted validation: 512 independent blocks, 3,072 candidate hands against the existing tight-passive, loose-aggressive, and pot-pressure pool. Report the same profit measures. These estimates are directional, not a strength gate.
- For each benchmark, count **actual candidate decisions** by street, trained versus uniform-fallback lookup, and fold/check/call/raise choice. Chart preflop and flop trained fraction and sample count prominently; retain turn and river counts too. Do not infer strength from lookup coverage alone.
- Record total traversal nodes, iteration, table entries, RSS, checkpoint size/time, nodes/step-second, and wall time. TensorBoard reads append-only progress and evaluation logs. A private TensorBoard server exposes the dashboard through SSH tunneling.

The validation deals are reused for trend comparison. Do not repeatedly choose a final model from these deals and call the result held out. A separate 4,096-block random **test** schedule is frozen in the campaign configuration. Run [confirm_blueprint_04.py](../scripts/confirm_blueprint_04.py) only after choosing the final checkpoint; it verifies the declared checkpoint hash before opening those deals. A positive trend means improvement over broad work intervals, not that every noisy checkpoint increases. Checkpoint 0.4 is earned only if the final fresh random result has a positive lower 95% bound and the validation series supports improvement; otherwise retain the campaign as evidence of the architecture or resource limit.

## Execution and recovery

Install the pinned project and monitoring requirements. Use [checkpoint-04-training.json](../configs/blueprint/checkpoint-04-training.json) and [train_blueprint_campaign.py](../scripts/train_blueprint_campaign.py):

```sh
python -m scripts.train_blueprint_campaign \
  --plan configs/blueprint/checkpoint-04-training.json \
  --campaign configs/blueprint/checkpoint-04-campaign.json \
  --resume SOURCE_CHECKPOINT \
  --out RUN_DIRECTORY \
  --workers 1 \
  --max-wall-seconds 86400 \
  --max-rss-gib 50 \
  --min-free-gib 10 \
  --checkpoint-seconds 3600

python -m scripts.monitor_blueprint --run RUN_DIRECTORY --logdir TENSORBOARD_DIRECTORY
tensorboard --logdir TENSORBOARD_DIRECTORY --host 127.0.0.1 --port 6006
```

Run the monitor and TensorBoard server separately while training proceeds. For a continuation, use the previous `checkpoint.json.gz` as the source, a **new** output directory, and pass `--starting-nodes` from the previous `result.json`. This preserves cumulative work on the chart. Training exits with code 2 after a guarded stop; that is a recoverable boundary. A failed evaluation or illegal action is a correctness failure. Before any resumption, verify the checkpoint hash, inspect the stop reason and resource trend, and preserve the earlier output directory.

After choosing the final checkpoint, run the sealed test once on a host that can load that checkpoint:

```sh
python -m scripts.confirm_blueprint_04 \
  --campaign configs/blueprint/checkpoint-04-campaign.json \
  --checkpoint CHOSEN_CHECKPOINT \
  --expected-sha256 CHOSEN_CHECKPOINT_SHA256 \
  --out FINAL_TEST_DIRECTORY
```

If random profit remains flat while entries and trained lookup rates grow, inspect action choices and the learned strategy. If entries grow but held-out lookup rates stay flat, prioritize a better public-history abstraction or coverage mechanism. If profit rises at the memory limit, continue the same seed on a larger-RAM host after sizing the next segment. Changes to the learning recipe create a new comparison, not an invisible continuation of this seed.

## Exploratory mixed-table comparison

After choosing the final checkpoint and completing its sealed random test, play one separate 3,000-hand six-seat cash table. Seat two independent copies of the chosen current blueprint policy, two independently seeded random policies, one tight-passive scripted policy, and one loose-aggressive scripted policy. Use deal seed `2026092411`, which is separate from the validation and sealed-test schedules. Rotate the button one seat each hand. Reset all players to 100 BB at the start of every hand so all six participate throughout; add each hand's net result to a cumulative ledger. Report each player's cumulative net BB at 1,000, 2,000, and 3,000 hands, plus the two-player group totals. The ledger must sum to zero at every hand.

This is a descriptive comparison with one lineup and one deal seed. It is neither a tournament with eliminations nor an independent strength gate. Put its table and, if useful, a cumulative-balance plot at the end of the campaign report. Retain the JSON with balances every 100 hands and the chosen checkpoint hash. Run it on the existing pod only after training has stopped and the final checkpoint is fixed; do not start another rental or extend the training segment for it.

```sh
python -m scripts.blueprint_mixed_table \
  --checkpoint CHOSEN_CHECKPOINT \
  --expected-sha256 CHOSEN_CHECKPOINT_SHA256 \
  --out MIXED_TABLE_DIRECTORY
```
