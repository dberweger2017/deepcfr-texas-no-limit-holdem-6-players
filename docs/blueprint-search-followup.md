# Search follow-up on saved blueprint checkpoints

## Question and comparison

Does correcting the public-range search improve six-player play against the scripted style pool, and does the larger trained blueprint help when both checkpoints use the same search? The frozen checkpoints are the 5,834,622-entry M4 source (`94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`), the 12-million-entry snapshot at iteration 18,455 (`c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845`), and the 58,015,659-entry final checkpoint (`1b7d9ef0a6f111ac82d99f802cf7ac7bc5c8685ff6a2f214cb918f7ef9361bc2`). They are one training lineage, so this comparison cannot establish independent-seed robustness.

The corrected search draws opponent holdings from their joint card-compatible range, uses exact action probability when an observed action is on the abstract menu, and keeps continuation mode active after its depth boundary. The original PR #103 behavior remains an explicit benchmark arm. Both arms use 12 worlds, 96 range candidates per opponent, four continuation styles, and a one-second decision limit. Internal lookups in the corrected arm report trained, on-tree-untrained, and off-tree-history counts separately for range inference and continuation play. Search remains a sampled rollout, not a multiplayer CFR solver.

## Frozen evaluation

The [M4 plan](../configs/blueprint/postflop-search-followup-m4.json) and [pod plan](../configs/blueprint/postflop-search-followup-pod.json) use identical schedules. Against the scripted style pool, run 1,024 independent six-seat rotation blocks per arm and checkpoint. Against random opponents, run 256 blocks per arm and checkpoint as a regression check. The primary contrast is corrected versus original search with the 58M checkpoint against scripted opponents. A second paired contrast compares corrected search on 58M and 5.83M on the same schedule. Keep all failed hands in the validity denominator. Report paired BB/100 with 95% intervals by independent block, absolute BB/100, invalid actions, search fallbacks, internal lookup counts, p95 search latency, peak RSS, runtime, and verified artifact hashes. The 0.4 sealed random test is historical and will not be rerun.

Run the 5.83M checkpoint on the 16-GiB M4 first, alone, under a 10.5-GiB process guard. Try the saved 12M snapshot on the M4 only if it loads and stays under the same guard; otherwise move it to the pod without swapping the M4. A 64-GB RunPod pod is needed for the 58M checkpoint. Use a 50-GiB RSS guard, two-hour runner wall cap, disk guard, and approximately $3 rental ceiling including retrieval. Save source revision, plan, checkpoint hashes, every hand result, and stop reason. Retrieve and verify artifacts before ending billing. No new blueprint training is part of this PR; the owner chose existing checkpoints.

Pass `--tensorboard` to the comparison runner to write events under each output directory's `tensorboard/` folder. It logs cumulative candidate, original-search, and paired BB/100 with 95% intervals, search attempts/fallbacks, p95 search time, and peak process RSS every 64 completed rotation blocks (384 hands per arm), plus the final block. Point TensorBoard at the parent directory containing the checkpoint runs to see all three. These repeated intermediate estimates are monitoring only; the frozen full-schedule comparison determines the result. For a private M4 dashboard, run TensorBoard there with `--host 127.0.0.1` and forward its port through `ssh -L` from the local Mac.

This is one evaluation campaign on a draft PR. The paired scripted contrast must have a positive lower 95% bound to claim a measured search improvement. A non-positive or inconclusive result is retained, not rerun on the same schedule. The comparison decides whether further blueprint training is worth paying for; it does not by itself qualify v0.5 or promote a default player. Keep this PR draft until code, runs, artifact verification, and report are complete, then review and merge the same PR.

## Results

Pending.
