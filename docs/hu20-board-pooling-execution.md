# Board pooling execution handoff

The [protocol](hu20-board-pooling-protocol.md) and [live quote](hu20-board-pooling-runpod-quote.md)
are reviewable in draft #149. **No pod exists for this task; approval is pending.**
This file is an execution recipe, not authorization to provision compute.

## Retained local inputs

- Three original average exports:
  `/Users/dberweger/Local/hu20-m4-archive-20261002/hu20-exact-flop-check-inputs`.
  Use only the three `stored-average` rows in the frozen plan; verify every SHA.
- External AGPL source archive and original/new harnesses:
  `/Users/dberweger/Local/hu20-board-pooling-20261003/external-source.tar.gz`.
  [Archive fingerprint](reports/hu20-board-pooling-artifacts/external-source-archive.json)
  and [source/toolchain fingerprints](reports/hu20-board-pooling-artifacts/external-tool.json).
  Never extract it into the MIT checkout. Preserve its license and upstream pin.
- #145 original small fixture:
  `/Users/dberweger/Local/hu20-m4-archive-20261002/hu20-exact-flop-check-20261001/results/turn-check-20261002/compact-validation-01`.
  Transfer its request and compact export unchanged; the validator rewrites
  temporary host paths and records original hashes.
- #145 recorded native river/payoff/MES fixtures: sibling
  `river-gates-tool-03`. Transfer request JSON plus matching response directories,
  source/result gates and their hashes.
- Restored original checkpoints and receipt:
  `/Users/dberweger/Local/hu20-board-pooling-20261003/checkpoints`.
  The solver-free companion already finished on M1 in 15.56 seconds. Member
  hashes match all three #141 pins. The 23.5-GB archival container's manifest hash
  was retained, not independently rehashed; each extracted member was verified
  before use. No M4 contact or lower-milestone substitute occurred.
- Mac pooling singleton evidence:
  `/Users/dberweger/Local/hu20-board-pooling-20261003/mac-singleton-01`.
  [Nine passing checks](reports/hu20-board-pooling-artifacts/mac-singleton-gates.json)
  against the original qualified binary. Raw requests/responses are retained.

## After explicit quote approval

Deploy only the quoted CPU/disk shape and retain its pod ID, image digest, rate,
deployment/billing start time and provider details. The external source is
installed under an unrelated path, e.g. `/workspace/hu20-board-pooling-tool`;
the MIT checkout can be `/workspace/hu20-board-pooling-repo`. Use native x86_64
Linux, Rust 1.96.0, Python >=3.10, pinned numpy/scipy and the repo's exact pokers
revision. No torch/GPU/model training dependency is needed by these diagnostics.
Record actual package versions and complete build logs. Use `--locked` and the
same legacy compiler flag already recorded by #145; never edit upstream.

Create an external `approval.json` from the tracked quote, recording the human
approval, pod ID, **original** rental start and deadline epochs (start+86,400),
unreset baseline swap, binary/plan hashes and later qualification file/hash.
The quote supplies the frozen worker/RSS/disk limits. `owner_approved_quote`
must remain false until the human approves it. This file is not checked into
Git with credentials or used to reset a spent clock.

Every paid preparation and qualification command runs under
`python -m scripts.guard_board_pooling --approval ... --out NEW-GUARD-DIR -- COMMAND`.
Archive the guard's baseline/resources/completion or failure. No automatic
restart after a guard or scientific gate fails. A local controller must retain
the original pod ID and rental deadline, monitor the guarded stage, and perform
retrieval/termination within the quote; exiting the solver alone does not stop
provider billing. Verify that controller access/retrieval work before launching.

1. Run `scripts.build_board_pooling_tool --tool-root EXTERNAL --out NEW-BUILD.json`.
   It verifies pinned sources/upstream and builds both new and original reference
   binaries natively. Record Linux/CPU/glibc/toolchain/binary fingerprints so #148
   can reuse the external tool. No agent messages are required for that reuse.
2. Run `scripts.prepare_board_pooling --plan configs/diagnostics/hu20-board-pooling.json
   --inputs INPUTS --out NEW-PREPARED --memory-gib 4` from the MIT checkout.
   This replays all roots, performs 100,000 full-key comparisons, computes the
   one shared equity codebook and exports native trees/ranges/actual-key tables.
   The disk-backed policies avoid simultaneous multi-gigabyte model loads.
3. Run `scripts.qualify_board_pooling` with both Linux binaries, the transferred
   tiny fixture and recorded river fixtures, frozen plan and prepared directory.
   It compares macOS payoff/MES/EV fixtures, checks singleton relocking, runs
   three real-export 20,000-deal V4 checks and fixed native resource/convergence
   pilots. Preserve every result, including pilot losses. These are not the
   main readout. Mark qualification passed only when all checks actually pass.
4. Record qualification hash and binary hash in the approval record. Run
   `scripts.run_board_pooling` with plan, prepared inputs, new result directory,
   binary and approval record. The driver checks free memory/CPU and a
   1.5×slowest-pilot time forecast before production. It interleaves all 120
   board/lineage jobs, collects exact-key sufficient statistics, freezes the
   common support mask, pools once per lineage, and replays/relocks each root.
   Every replay must reproduce its original equilibrium and statistics.
5. On completion **or mandatory stop**, retrieve the whole owned result tree,
   build a remote member SHA manifest, and verify every retrieved member.
   Include builds, exporter, qualification, guards and controller/billing logs.
   Hash compact requests/policies/codebook and every raw response. Avoid per-hand
   production dumps. Confirm pod termination and no remaining owned volume.
6. Run `scripts.report_board_pooling` locally on the hash-verified results.
   Publish report/compact JSON/hashes, exclusions, failures, signed placement,
   conditional board intervals and actual billing evidence; retain unavailable
   posted billing explicitly. Update ROADMAP and PR body. Keep the PR draft.

The report helper refuses primary classification for incomplete campaigns or
insufficient common three-export coverage. If a quote forecast or guard stops
the campaign, preserve the fixed corpus and all spent compute; do not quietly
change sample size, cap, thread count, key scheme or source after outcomes.
