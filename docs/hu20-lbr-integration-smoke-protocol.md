# B100M native/fast LBR integration smoke

I run one sequential M1 process using the already local v0.4 fixed-first-seed
inference export, SHA-256
`4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`.
Verify compressed bytes before loading; load once, read-only. No training
checkpoint, other models, new campaign, M4 computation or paid compute.

Freeze eight fresh paired blocks, root `202610030201`, split `test`; use
`stream_seed(root, 'test', 'deal', 2, block)` and both target seats, with button
`block % 2`. Target action stream is
`stream_seed(root, 'test', 'action', 'target', block, rotation)`; LBR stream is
`stream_seed(root, 'test', 'opponent', 'lbr', block, rotation)`.

Compare native LocalBestResponse and explicitly selected
RankedCachedLocalBestResponse on identical actual target play, per-hand streams
and LBRConfig(chance_samples=4, max_seconds=5). Both use the existing default
restricted cap-2 response menu; the target retains its native-reopening menu.
Reset hand-local probability cache and ranker/reference rank caches before each
hand. Run native then fast for each coordinate, without parallel workers.

Record exact actions, value arrays, completed batch counts, posterior/RNG states,
actual target RNG, public-event digest and native chip outcome. Require equality
of semantic telemetry (timing fields naturally differ). All requested batches
must complete; deadline truncation is an explicitly failed integration condition,
not permission to choose easier seeds or hide mismatches. Replay exact recorded
actions independently through the native engine.

Report load time separately, total/decision algorithm wall times and their ratio,
per-hand measurements, observed street coverage, fallback exposure and process
peak RSS. Preserve raw rows and output hashes, including failures. Eight paired
blocks are an integration smoke, not a statistical performance study, playing
strength campaign, or a new proof of the #126 ranker.
