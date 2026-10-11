# Fresh M4 scoring of the frozen K50 bench

The owner authorizes a new M4 attempt of #222's frozen scoring, starting all
40 roots again after the current M4 HU100 campaign is finished. This is a
separate attempt, not a resumption of #222, #225 or #230. Their failure latches,
raw partials and archives stay unchanged. No previous complete result is copied
into this attempt. Restore exact exports rather than retrain.

## Evidence status and fixed science

The owner subsequently requested analysis of the nine completed M1 roots; that
[partial readout](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/230#issuecomment-6101289186)
is now known. This M4 attempt **cannot claim outcome blindness or an independent
confirmation sample**. It completes the same deterministic benchmark with
already inspected nine-root evidence. The new pilot is timing-only, and new
partial scores remain uninspected until all 40 roots and the evaluation guard
complete. Neither prior observations nor pilot outcomes change the run.

Use [#222's protocol](hu20-equity-bench.md) exactly: the 40 #149 turn roots,
two frozen folds, B500M seed 2026093001 ranges, exports from seed 202610050001,
matched K50 iterations 4,000,977 / 3,855,889, unchanged references, labels,
menus and history. All 21 policies per fold are scored in one native lock-only
pass per root, with six threads, nice 10 and the original 4 GiB arena. v1 uses
v1 labels; equity uses #190's global K50 adapter via native slot `eq50-fit0`.
The reused `run_hu20_equity_bench.evaluate` and `.report` functions are unchanged.

Apply the frozen primary to all 40 roots: pass requires matched K50
opponent-sampled E minus v1 at 3M to have its paired 95% bootstrap upper bound
below −0.10 BB **and** a negative K50 minus v1 10M point. Fail if the matched
lower bound is above −0.10 BB; otherwise inconclusive. Draw 2,000 paired boards
with Python Random(202610050002), sorted roots and percentile indices 49/1949.
Report all exported E/Q against B/L/P, learning curves and context gaps to the
full-corpus #190 global K50 witness E=0.3917. No sample, checkpoint, threshold
or bootstrap adjustment; no optional additional scoring on a weak result.

## Queue and admission

The M4 is Apple M4 /10 cores /16 GiB. The M1 is used only to prepare/review code;
all restoration, pilot and scoring execute in the isolated M4 checkout
`/Users/dberweger/Local/hu20-equity-bench-scoring-m4`, branch
`feature/hu20-equity-bench-scoring-m4`, from current main at preparation.
Do not launch heavy preparation or science while the HU100 campaign owns M4.
Use the current campaign PR226's terminal merged record, and a fresh process
inventory showing no other research worker/controller, as the release gate.
If another campaign takes the machine, continue waiting. Do not stop processes.

Keep the whole-family 7 GiB ceiling, creation-verified process ownership,
normal macOS pressure, at least 15% free memory, 9 GiB scoring admission proxy,
AC and greater than 16 GiB free disk. The owner's explicit 10,000,000,000-byte
total system swap amendment remains in effect. Pass `--machine m4` and
`--swap-ceiling-bytes 10000000000`; default M1 behavior stays unchanged. The
M4 guard has its own advisory lock. Availability is a metadata preflight,
not a scientific launch: wait for all admission conditions before claiming a
phase. Any actual guard breach or exactness mismatch stops the new attempt
permanently, with issue/SOMA, archive and end review; no scientific retry.

## Execution, once after release

Use the M4's qualified Python 3.11 research environment. Before restoration,
check the live merged status of the owning input PRs #222, #190 and #163.
Sources and exact member pins are in
[model-input-index.json](reports/hu20-equity-bench-artifacts/model-input-index.json)
and [RESULTS_INDEX](../RESULTS_INDEX.md). The archives are already in the M4's
Research-Cloud folder; retrieval must verify whole/member hashes, never use
unverified copies or extract into a synced folder. A fresh ignored base is
`results/equity-bench-scoring-m4-20261010`; do not reuse an older base.

With `PY` the absolute Python path and `BASE` that ignored root, execute:

```sh
PY -m scripts.guard_hu20_equity_scoring --machine m4 --swap-ceiling-bytes 10000000000 --out BASE --name restore -- PY -m scripts.restore_hu20_equity_scoring --base BASE
PY -m scripts.guard_hu20_equity_scoring --machine m4 --swap-ceiling-bytes 10000000000 --out BASE --name prepare -- PY -m scripts.score_restored_hu20_equity_bench prepare --base BASE
PY -m scripts.guard_hu20_equity_scoring --machine m4 --swap-ceiling-bytes 10000000000 --out BASE --name pilot --scoring -- PY -m scripts.run_fresh_hu20_equity_scoring pilot --base BASE
PY -m scripts.run_fresh_hu20_equity_scoring quote --base BASE
```

One independent source review must be clear before the pilot; its file SHA256s
are rechecked by the fresh controller. Restore #222's exports, adapters and
references, #190's exact qualified evaluator, and all three #163 K50 tables;
verify every extracted member and alias. Preparation changes only absolute
file paths. The wrapper retains exclusive pilot/evaluate/report claims and
refuses any prior results or continuation seed.

After the complete guard-clear pilot, post the **full** timing/resource/storage
quote on this new PR before evaluation. Include all 40 new roots, zero reused
roots, 21 policies/fold, native pass shape, pilot source, measured peak, guard
extrema, 40× measured pilot time, 25% schedule allowance and 30 minutes for
closeout. Retain 1 GiB output-original and 1 GiB derived-ZIP planning allowances
and the 16 GiB disk floor. Do not infer scores from the pilot. Save the actual
comment URL and SHA256 of `full-scoring-quote.json` in `BASE/quote-posted.json`.
Then launch exactly once:

```sh
PY -m scripts.guard_hu20_equity_scoring --machine m4 --swap-ceiling-bytes 10000000000 --out BASE --name evaluate --scoring -- PY -m scripts.run_fresh_hu20_equity_scoring evaluate --base BASE
```

Use metadata-only progress checks (completed direct result paths, guard receipts,
resource metadata and elapsed time); never read result JSON, worker logs or raw
native responses while pilot/evaluate are active. Use T3 scheduling rather than
active polling. Disable the science timer on terminal failure or full completion.
Only when all 40 fresh results and the evaluation guard are clear, run guarded
`run_fresh_hu20_equity_scoring report --base BASE`, then inspect and publish
numeric results. A child return code alone never qualifies a guarded operation.

## Closeout

One independent end evidence review reproduces all 40-root metrics, frozen
classification, plots, exact inputs/source and guard/archive provenance. On
pass, recommend full-game K50 lineages with learning curves and direct v0.4.2
matches. List missing Python `load_training`/`AveragePolicy`, web runtime,
turn-search range handling and LBR support, per
[native-equity-bucket-keys](reports/native-equity-bucket-keys.md). On fail or
inconclusive, record the finite-bench implication without a full-game claim.

Archive new scores, derived outputs, raw responses, failures and reproduction
source in a member-hashed ZIP in
`~/Local/Research-Cloud/PR-<new-number>-hu20-equity-bench-scoring/`. Do not duplicate
#222's archived inputs. Verify local member readback and native plus independent
Drive metadata acceptance; index exact restoration and archive receipts. Delete
only this attempt's own extracted copies after acceptance and dependency review;
never delete inside synced folders or alter other PR roots. Preserve originals
on any uncertainty. Update roadmap item 4.3 and one short Current position entry
when this attempt lands. Artifact checker before every commit; merge only green
checks with no findings, unwatch before final handoff. No training, native trainer
edits, paid compute, release, defaults or other-PR cleanup.
