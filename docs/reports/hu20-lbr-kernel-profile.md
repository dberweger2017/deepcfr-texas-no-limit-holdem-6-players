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

## Next decision

**One next engineering test:** make a separately selectable exact seven-card
ranking implementation for the LBR chance loop, while retaining the current
`hand_value` as the oracle. Freeze a rank fixture that covers all hand
categories, ties and suit permutations; compare the new evaluator against the
old one on every sampled holding/board from PR #121's equivalence corpus.
Then require identical native LBR actions, requested/completed batch counts,
range support and per-action chip values at the existing `1e-10` tolerance
before benchmarking speed. Timer-limited real cases remain an explicit gap:
the existing 336-case corpus had none, so add a controlled timer-boundary
fixture before calling the new executor a general drop-in replacement.

If that equivalence gate passes, benchmark fresh *distinct* calls and
startup-inclusive full cost. The owner's later guidance removes ten hours as
a scientific feasibility cutoff: a longer M4 audit or paid CPU workers may be
reasonable. Select the host only after measuring the new per-call cost and,
for a paid option, a bounded independent-worker throughput/RAM pilot and
current price. **No training intervention is selected by this profile.**
