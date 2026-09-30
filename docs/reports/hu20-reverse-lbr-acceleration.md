# Faithful reverse-LBR likelihood acceleration: M4 result

Draft [PR #121](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/121) is an outcome-free engineering follow-up to [#119](hu20-b100-posterior-feasibility.md). It leaves the native bounded `LocalBestResponse` and all saved B100M models unchanged. **The selectable shared-query cache reproduced native behavior on the frozen validation corpus, but it did not make the planned posterior-conditioned experiment feasible on the M4.** No posterior distribution, target conditional action value, poker return, training result, or model promotion was generated.

## What was frozen and checked

The [protocol](../hu20-reverse-lbr-acceleration-protocol.md) and [336-case corpus](hu20-reverse-lbr-acceleration-artifacts/corpus.json) were committed before optimized timing. Cases came from #119's 24 trained B100M decisions, all seed × street × actual-position cells, using only the retained #117 public action prefixes and the target's own cards. Hypothetical attacker holdings were selected by hash rank and used independent fixed LBR seeds. The selected-list digest remained `578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322`; the case digest was `1559b5aac31016a9db92cfc2b319c28abf110446866d50dba0f45126eacadc4e`. Neither terminal payoff nor the actual hidden attacker hand selected cases or entered the simulated LBR observation.

The candidate is a separate `CachedLocalBestResponse`. It inherits native `choose_action`, range updates, zero-evidence handling, chance sampling, soft-time checks, first-index tie resolution and chip-value arithmetic. It shares only deterministic saved-target `distribution(replay(public history, acting seat, hypothetical acting pair))` results across LBR instances for the **same immutable source object**. The cache key contains the full public event tuple, seat and hypothetical pair. It does not cache RNG, a sampled deck, the LBR posterior or an action decision. The production native path is unchanged.

| Correctness check | Result |
| --- | --- |
| Frozen real cases | **336/336**, zero native-versus-cached mismatches |
| Native versus cached action, exact raise-to and ordered menu | All matched |
| Requested/completed chance samples, range support, zero-likelihood record | All matched; all 336 completed their requested samples |
| Per-action LBR chip-value vector | All elements within the frozen absolute `1e-10` chip tolerance; matched fresh-process output digests were identical |
| Repeated execution | Cached repeats matched; a separate native process and a separate cached process reproduced every frozen case digest |
| Coupled cyclic suit permutation | Native/permuted and native/cached-permuted comparisons all passed |
| Synthetic references | Preflop and river, free and facing states, zero-evidence preservation and forced first-index tie tested separately |

The real corpus covered three seeds, preflop/flop/turn/river, both positions, and observed LBR checks/calls/folds/raises. Its simulated LBR decisions also included all four action kinds. The smallest real best-versus-next value margin was 0.1105 chip; no real near tie occurred. A forced exact tie checked first-index selection. No real case hit a soft-time limit or zero-likelihood update. The latter is covered by a small reference test; actual timer truncation was **not** exercised. The expanded M4 focused suite passed **8/8**. The [machine-readable result](hu20-reverse-lbr-acceleration-artifacts/m4/publication-final/report.json) retains the counts.

## Speed, resource use and full-workload estimate

Native and cached executors each ran the same 336 cases in a fresh process and loaded the same three B100M sources. Their output digests matched exactly. The algorithmic timing compares only LBR calls; the end-to-end timing includes model/checkpoint loading and corpus setup.

| Matched corpus | Native | Cached | Speedup |
| --- | ---: | ---: | ---: |
| Sum of 336 LBR call wall times | 81.91 s | 65.83 s | **1.24×** |
| Whole fresh process | 120.67 s | 104.03 s | **1.16×** |

The cache hit rate on this corpus was 88–89% by seed. A separately frozen, hash-ranked larger timing set used **3,591 first-sample calls** (6.19% of #119's 58,047-call one-sample inventory) plus **672 independent second-sample calls**. It took 1,138.60 s including loading/setup, reached 99.1% cache hits, and used one CPU core nearly continuously (measured call CPU/wall ratio ≈1.00). The extra samples reused cached target queries but remained close in cost to their first samples. These are algorithmic LBR action simulations, not a posterior or value calculation.

The larger sample's street-weighted means project **13,156 s / 3.65 h** for all 58,047 one-sample calls. For additional samples, the calculation conservatively takes at least the first-sample mean for each street; four samples project **52,624 s / 14.62 h for likelihood alone**. This is an extrapolation from 6.19% of one-sample calls, not a measured full inventory. Repeated calls, cache growth and the independent controls may behave differently outside this subset; the per-street p95-style stress calculation is much longer and is not a confidence interval.

| Future frozen design | Likelihood projection | Total with historical value/control allowance and 1.25× headroom |
| --- | ---: | ---: |
| 4 LBR samples × 96 paired worlds | 14.62 h | **19.29 h** |
| 4 × 192 | 14.62 h | **19.89 h** |
| 4 × 384 | 14.62 h | **21.09 h** |
| 8 × 96 | 29.24 h | **37.56 h** |

The total uses #119's **1,728-second** historical allowance for paired 96-world uniform/posterior values, scales it linearly with worlds, reserves **1,200 seconds** for the suit/independent-river/report controls, then applies 1.25× headroom to all components. The control reserve is not a fresh exact-reference benchmark. Even the mean-based minimum is well outside a bounded M4 window. The first/extra-sample timing and per-street full counts are in the [result JSON](hu20-reverse-lbr-acceleration-artifacts/m4/publication-final/report.json).

Peak sampled RSS was **4.11 GB** in the larger run, below the 10.5-GiB limit. Swap stayed at 761.38 MiB before and after every measured phase; at least 41.2 GB remained free, above the 8-GiB floor. The cache's shallow dictionary measurement was about 5.2 MB per seed in the larger subset; this excludes referenced Python objects and does not prove full-workload memory safety. Every heavy phase used one M4 process. The research clock began at **07:10:31 UTC** and retained its **10:10:31 UTC** ceiling; all measurements and the M4 seal finished before it. The M1 did only edits, Git and compact transfers.

## Decision and retained artifacts

**The cache is faithful on the frozen cases, but its 1.24× algorithmic gain is insufficient.** A 99.1% hit rate alongside nearly full single-core use indicates that repeated target-policy lookup is not the dominant remaining cost for this workload. This does not establish where *within* the LBR's own range/chance work the time goes. There is no scientific posterior-conditioned result to interpret and no new training-mechanism conclusion.

**One next experiment:** profile and accelerate the unchanged LBR's per-holding range/chance kernel in a separate, bounded, outcome-free M4 pilot, then require exact native equivalence on this same corpus and a new whole-workload cost gate. Preserve RNG draw order, soft-time behavior, value arithmetic and first-index tie-breaking. Do not launch the posterior-conditioned value audit until a separately authorized plan meets the four-sample/96-world minimum with controls and headroom.

**M4 versus paid compute:** the next exact-kernel engineering pilot belongs on the M4. The current evidence shows a CPU-bound single-process implementation and no measured parallel scaling or specific frozen paid-host run that warrants rental. The RunPod balance is unknown; no spend was made. Paid compute is not a substitute for this correctness gate.

The [compact M4 artifacts](hu20-reverse-lbr-acceleration-artifacts/m4) include result summaries, the original clock, focused-test logs and [17-file SHA-256 manifest](hu20-reverse-lbr-acceleration-artifacts/m4/publication-final/manifest.json). Copied compact files and the frozen corpus matched the manifest hashes. Full per-case attempts and process logs remain on M4 at:

`/Users/dberweger/Local/hu20-reverse-lbr-pr121/results/hu20-reverse-lbr-m4-20260930`

To retrieve and verify the full archive from an SSH-capable machine:

```sh
rsync -a -e 'ssh -o HostName=100.122.216.94' \
  m4:/Users/dberweger/Local/hu20-reverse-lbr-pr121/results/hu20-reverse-lbr-m4-20260930/ \
  ./hu20-reverse-lbr-m4-20260930/
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('hu20-reverse-lbr-m4-20260930')
manifest = json.loads((root / 'publication-final/manifest.json').read_text())
for original, record in manifest['files'].items():
    path = root / Path(original).relative_to(manifest['retained_m4_root'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
print('verified', len(manifest['files']), 'files')
PY
```
