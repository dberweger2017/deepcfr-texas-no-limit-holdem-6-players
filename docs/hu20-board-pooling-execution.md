# Board pooling M4 execution — revision 3

Owner authorization October 4 replaces the RunPod plan. No rental exists; $5
unused. Read the [protocol](hu20-board-pooling-protocol.md) and tracked M4 budget.

Fresh campaign: `/Users/dberweger/Local/hu20-board-pooling-20261004` on M4, with
its own checkout. Python: `/Users/dberweger/Local/deepcfr-training/.venv/bin/python`.
Inputs: `/Users/dberweger/Local/hu20-board-pooling-20261004/inputs-restored-02`,
verify all three stored-average hashes. The shared original alias was archived;
use this isolated dependency folder. External tool stays outside checkout; copy/hash-verify
the retained Mac v2 and reference binaries/source receipt. No new M1 solves.

Record an external approval.json with source/binary/plan fingerprints, observed
swap baseline, shared append-preserving clock path, original start and unreset
24-hour deadline. One worker, six threads; 5-GiB worker / cache-inclusive 8-GiB
family ceiling; 20-GiB disk floor. Use `scripts.guard_board_pooling` for every
stage; it reuses #148's admission and RunBudget. Detached jobs survive SSH.
Use `ssh -n m4` in loops. No deletion or CloudStorage changes.

1. Guarded prepare: `scripts.prepare_board_pooling --memory-gib 4`, all forty
   boards, three immutable disk-backed average policies, K100k, frozen codebooks.
2. Guarded qualify: `scripts.qualify_board_pooling`, retained singleton/native
   payoff/MES fixtures plus three fixed real-export 20k-deal V4 checks and
   convergence/memory/time pilots. Six native threads fixed. Linux parity not run.
3. Post pilot resource/results and conservative forecast on #149 before main.
   Update qualification hash in external approval only after all gates pass.
   `scripts.run_board_pooling` refuses forecast beyond the remaining reserve.
4. Guarded main: 120 collect solves, opposite-half fits, fresh lock-only passes
   (plus covered-context sensitivity) and 18 unchanged same-thread replay jobs.
   Stop owned work on any gate/guard failure; no automatic restart.
5. Retrieve/hash-verify all attempts, prepare final report and compact JSON,
   update roadmap and draft #149. No hypothesis decision if incomplete/common
   mask insufficient or missing-key coverage exceeds 5%. Report every outcome.
   Preserve raw evidence under the dated folder for RESULTS_INDEX Drive archive.

October 4 second owner readmission uses `approval-resume-03.json`,
`guard-prepare-03` and fresh `prepared-03`. Recovery checks the pinned second
retrieval manifest (SHA-256
`38710bfd86d9d18c73b93e059ef95bc67b0d501597a276406957507c7595d8f5`),
old index/source hashes, regenerated codebooks, menus, ranges and card features.
Only identical completed artifacts are shared by immutable hard links into the
fresh directory; old attempt evidence is never rewritten. Regenerate K and
features before accepting reuse. The missing third-lineage exports are newly
computed. This is deterministic exporter recovery, not changed science or
solver-result reuse. `continue_board_m4_03.py` queues `qualification-03` only
after successful guarded preparation; it never starts main or restarts failure.
The original clock retains both failed attempts (1871.3424450419989 seconds).
