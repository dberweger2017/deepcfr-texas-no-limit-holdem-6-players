# B100M faithful LBR kernel profile — September 30, 2026

## Result

The frozen, outcome-free 24-case M4 profile completed. **Seven-card hand
ranking is the dominant measured cost inside the cached bounded LBR calls.**
In the profiled warm calls, `hand_value` and its descendants consumed 6.521
of 7.509 seconds of `choose_action` cumulative time (**86.8%**). The ranker
entered its five-card evaluator 1,029,000 times: 21 five-card combinations
for each of 49,000 recorded seven-card `hand_value` calls. The next largest
individual function self times were the five-card evaluator (1.970 s), its
rank generator (1.724 s), `sorted` (0.845 s) and string rank lookup (0.344 s).
Range and saved-policy query work was much smaller in these warm calls.

This is a **function profile**, not a claimed whole-campaign speedup. The
profiled calls repeat the same case after a cold call, so they warm both the
shared saved-policy query cache and Python's separate `hand_value` LRU cache.
On river, that repeated-case rank cache removed all recorded ranker calls in
the warm profile. Rank time is therefore understated for fresh river boards,
and the profile alone cannot predict how much a new ranker would shorten the
full 58,047-call inventory. `cProfile` also adds overhead. No optimized
ranker was implemented or benchmarked in this PR.

| Street | Frozen cases | Profiled `choose_action` | `hand_value` subtree | Share |
| --- | ---: | ---: | ---: | ---: |
| Preflop | 6 | 2.868 s | 2.583 s | 90.1% |
| Flop | 6 | 1.533 s | 1.307 s | 85.3% |
| Turn | 6 | 2.887 s | 2.631 s | 91.1% |
| River | 6 | 0.222 s | 0 s after identical-case warming | Not representative |

All 24 calls completed their requested comparison batches in both phases.
The cold calls took 6.187 seconds in total; the *profiled* warm calls took
7.509 seconds, so those numbers must not be divided into a speedup. The
three model loads took 39.940 seconds. The entire successful attempt took
54.150 seconds; sampled peak process RSS was 4,287,234,048 bytes, swap stayed
at 761.38 MiB, and free disk after completion was about 37 GiB. All 60,065
saved-policy misses in the cold phase became 60,065 hits and zero new misses
in the repeated warm phase. This confirms that the new profile exercised the
intended shared-query cache; it does not imply the whole workload achieves
100% cache hits.

## Provenance and retained attempts

- Protocol: [frozen profile protocol](../hu20-lbr-kernel-profile-protocol.md).
  The [24 selected case IDs](hu20-lbr-kernel-profile-selection.json) were
  committed at source `4835a7840f8f2bafcc58b8c992882e30661c74b0` before
  M4 execution. Their source is PR #121's 336-case corpus with case digest
  `1559b5aac31016a9db92cfc2b319c28abf110446866d50dba0f45126eacadc4e`.
- Focused M4 check before profiling: `python -m py_compile
  scripts/profile_lbr_kernel.py` and `pytest -q tests/test_cached_lbr.py`
  (**8 passed**). The script verified the committed case list, original raw
  hashes, public-prefix digests and three B100M model specs at run time.
- Attempt 1 failed immediately because its raw-directory argument pointed to
  the #116 archive with different filenames. It ran no selected case and is
  retained as [attempt-1-result.json](hu20-lbr-kernel-profile-artifacts/attempt-1-result.json),
  SHA-256 `c3e0d414f5eb78a979c66ba30d48681f1ba7225be8683309c3b6f10f006fe037`.
  The frozen selection and original one-hour deadline were unchanged.
- Attempt 2 used the actual retained #117 curve archive and completed 24/24
  cases. Its compact [result.json](hu20-lbr-kernel-profile-artifacts/attempt-2-result.json)
  has SHA-256 `24c19f9a704891144829940c2e1e98c198430839dae19e36d09d24b1f5eef545`.
  The full 808,088-byte `attempts.jsonl` has SHA-256
  `5363a8ee329b556b4bcc96c923d877d6710c183579b7979db592c977207b9c11`.
  It remains on the M4 at
  `/Users/dberweger/Local/hu20-lbr-kernel-profile-pr123/results/hu20-lbr-kernel-profile-m4-20260930-attempt-2/attempts.jsonl`.
  Retrieve with `scp m4:/Users/dberweger/Local/hu20-lbr-kernel-profile-pr123/results/hu20-lbr-kernel-profile-m4-20260930-attempt-2/attempts.jsonl .`
  and verify with `shasum -a 256 attempts.jsonl`.
- The M4 ran one heavy process on AC power. It was released after PID 39218
  exited; `/tmp/DR_RESEARCH_M4_COORDINATION.txt` records the claim, failed
  path, corrected attempt and release. No poker returns, posterior-conditioned
  range, target conditional action value, training or paid compute was run.

## Recommended next step: exact ranker test, then the conditional audit

The next **one** engineering experiment should be a separately selectable
exact seven-card evaluator in the bounded LBR chance loop. Leave the engine's
showdown behavior, saved policies, LBR range update, chance draws, action
menu, utility reduction order and five-second soft-time check unchanged.
Implementing a direct seven-card evaluator is justified by the measured
hotspot, but this profile does **not** prove that it will accelerate the full
audit. The experiment is an option to lower a repeated scientific cost, not a
new poker strategy or a prerequisite imposed by a ten-hour limit.

1. **Freeze a correctness and timing protocol before running the candidate.**
   Use the existing 336 real #121 LBR cases, the same model/source hashes and
   independent deterministic hand fixtures spanning every rank category,
   wheel/straight-flush edges, full-house and two-pair ordering, ties and suit
   permutations. Include a controlled near-five-second timer fixture because
   no real #121 case was timer-limited. Retain the old `hand_value` as oracle.
   Freeze distinct-call timing coordinates and seeds without looking at LBR
   values or poker returns.
2. **Require exact behavioral equivalence first.** Compare the new and native
   rank tuples on the fixtures, then every ordered LBR menu, chosen action and
   raise-to, posterior range support, zero-evidence record, requested and
   completed batch count, and per-action chip value on the 336 cases. Keep the
   existing `1e-10`-chip absolute tolerance for the value vectors. Any
   mismatch or changed timer completion blocks a drop-in equivalence claim;
   preserve the failed case instead of relaxing the tolerance. No hidden
   opponent cards or future deck may enter the lookup.
3. **Measure whole cost after equivalence.** Use fresh, *distinct* calls from
   #121's frozen 3,591 first-sample and 672 incremental-sample timing subset,
   not the identical-case warm repeats in this profile. Compare native and
   candidate in separate fresh processes; include model loads, cache behavior,
   all four streets, CPU/wall time, peak owned-job RSS, swap and disk. Recompute
   the 58,047-call one-sample inventory and the four-sample, 24-decision,
   96-world posterior-audit cost with the 1,728-second historical value proxy,
   1,200-second control/report reserve and at least 1.25× headroom. The
   present cached projection is about **19.29 hours**, not a measured full
   run; this profile supplies no new whole-cost estimate.
4. **Proceed to the scientific measurement without manufacturing a speed
   gate.** If the ranker passes and saves meaningful whole-job time, use it
   in a newly frozen posterior-conditioned audit. If it fails or saves little,
   use the already validated cached LBR. Keep #119's 24 outcome-blind
   coordinates, four likelihood samples per hypothetical holding, 96 paired
   conditional worlds with the disjoint selection/evaluation halves, suit
   controls, river reference and original model identities. Amend the old
   two-hour feasibility stop *before* opening any values; choose one new
   absolute safety deadline and preserve every partial/failure. The owner's
   later guidance explicitly allows a run longer than ten hours. The audit,
   rather than this profile, must determine whether high-visit local errors
   persist under the reconstructed attacker range.

**Host decision.** Use the refrigerated M4 for ranker correctness and fresh
timing: one heavy process, 10.5 GiB RSS ceiling, 0.5 GiB swap-growth guard and
8 GiB free-disk floor, with `/tmp/DR_RESEARCH_M4_COORDINATION.txt` checked
before each phase. For the scientific audit, the M4 is a credible default:
the prior cached projection is roughly 19.29 hours and measured memory is far
below its guard. That is now an acceptable duration, not an automatic stop.
If freeing the M4 would materially help concurrent work, run an *outcome-free*
paid-CPU pilot first: one and then four independent workers on a quoted
CPU-only host with at least 64 GiB RAM, checking per-worker throughput,
cross-platform native equivalence, aggregate memory, artifact transfer and
shutdown/recovery. Four copies of the observed 4.29-GB peak alone are about
17.2 GB; 64 GiB leaves substantial operating and model-load margin. Choose
the paid host only from its measured end-to-end wall time and current exact
price, not assumed core scaling or an old account balance. No price, balance,
paid worker or cross-platform throughput has been verified in this attempt.

**Training decision.** Neither the hotspot nor a faster evaluator diagnoses
the remaining −73 BB/100 LBR weakness. No training intervention is selected
until the posterior-conditioned values and controls finish. Bring that
evidence and one mechanism-specific training proposal to the owner before
starting any training. No training, paid host or follow-on audit was launched
as part of this report.
