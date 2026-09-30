# B100M posterior-conditioned decision audit: feasibility stop

Draft [PR #119](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/119) is a separate, dependent diagnostic after sealed #117. The [protocol](../hu20-b100-posterior-protocol.md) was committed before selection or likelihood timing. **No posterior-conditioned or uniform conditional action-value outcome was generated.** The outcome-free M4 preflight found that faithfully marginalizing the unchanged bounded LBR over every compatible attacker holding cannot fit the initially authorized roughly two-hour heavy-compute allowance. The protocol's stop rule was applied; no world budget, holding set, action model, or likelihood count was reduced to obtain an answer.

## What completed

The selector used the three retained B100M files from #117's fresh bounded-LBR curve, root `202610010300`, with three seeds and 512 paired blocks each. It read public coordinates, trained lookup and visit count, actual button, and public action traces. It did not use terminal profit, LBR action values, hidden attacker cards, or future deck. The curve's *aggregate* playing returns had already been opened in #117, so this is a new decision set and fresh intended conditional root, **not** a sealed new playing schedule. #117's 60 primary coordinates and four card-collision roots were excluded by coordinate. Every seed × street × actual button/big-blind position cell was filled at the frozen **300-visit** threshold; the 1,000-visit threshold did not fill all cells. All 24 selected decisions were trained. The selected-list digest is `578b8b67f21a15d2961e88cf498aeeb64831c9827e578e5ce1db62f85d43e322`.

The new [target-perspective posterior implementation](../../src/diagnostics/reverse_lbr.py) starts with uniform compatible LBR holdings and computes each observed LBR action's likelihood by replaying its public prefix under every hypothetical holding and fresh bounded-LBR RNG. It does **not** reuse `LocalBestResponse.update()` as the target's posterior; that method infers the target holding from target actions. It has explicit zero-evidence reporting rather than an invented softmax. M4 focused tests passed **12/12** in 7.21 seconds, including synthetic reverse-posterior normalization and invariance when the actual hidden opponent holding or future deck changes while the target-visible state is fixed. The M4 preflight loaded all three B100M exports through `Target`, which verified current-policy/checkpoint hashes, game identity, entries, and probabilities against the retained checkpoint lineages.

The timing phase then ran the unchanged LBR on 132 hash-chosen hypothetical-holding/prefix cases across all seeds and streets. It discarded predicted actions and match indicators. It wrote only elapsed time, RSS and resource data. It did **not** generate a posterior distribution, local action returns, uniform/posterior gap, suit-control outcome, or independent river result. The raw per-call measurements, frozen decision, selected coordinates, failure-free process log, machine-readable [result](hu20-b100-posterior-feasibility-artifacts/report.json), and [manifest](hu20-b100-posterior-feasibility-artifacts/manifest.json) are retained.

## Measured cost and stop decision

There are **58,047** hypothetical-holding × observed-LBR-action calls for one internal RNG sample across the 24 selected decisions. The 132 timed calls took 0.0648–0.5242 seconds each (mean 0.2263; median 0.1721). The outcome-free preflight used the slowest observed call within each street, multiplied by that street's full call count. It projected **25,597 seconds / 7.11 hours for one sample** and **102,389 seconds / 28.44 hours for four** before conditional-value worlds. This is intentionally conservative and is a planning projection, not measured total runtime. An overall-mean extrapolation for four samples is about **14.6 hours**; even extrapolating the fastest observed call to every call gives about **4.18 hours** before values. Those alternative extrapolations are not throughput guarantees, but show that the stop is not an artifact of choosing only the slowest call.

At the freeze, **6,991 seconds** remained under the conservatively recorded two-hour M4 deadline. The protocol also reserved about 1,728 seconds for paired 96-world uniform/posterior values and 1,200 seconds for controls and reporting, with 1.25× headroom. No candidate in the declared 1/2/4/8-likelihood-sample × 96/192/384-world grid fit. The minimum four-sample primary estimator was especially far outside the allowance. The research clock was recorded conservatively from the first focused M4 tests and was neither reset nor extended.

Peak sampled RSS was **3.16 GB** (about 2.94 GiB), below the 10.5-GiB guard. Swap stayed at 0.744 GiB before and after timing; M4 had approximately 39 GiB free disk, above the 8-GiB floor. One heavy M4 process ran at a time. The M1 performed only source edits, Git, compact transfers and status reads; no paid compute or training was used. The retained M4 root is:

`/Users/dberweger/Local/hu20-b100-posterior-pr119/results/hu20-b100-posterior-m4-20260930`

The final manifest covers six compact attempt files, and its SHA-256 is `c82ff5834f0aaa98417ceb34264d4c513c3872dfbf8f809ec46d2ce94037478e`. The copies in [the artifact directory](hu20-b100-posterior-feasibility-artifacts) were checked against every manifest entry. To retrieve the original attempt from an SSH-capable machine:

```sh
rsync -a -e 'ssh -o HostName=100.122.216.94' \
  m4:/Users/dberweger/Local/hu20-b100-posterior-pr119/results/hu20-b100-posterior-m4-20260930/ \
  ./hu20-b100-posterior-m4-20260930/
shasum -a 256 ./hu20-b100-posterior-m4-20260930/manifest.json
```

## Answers and one next experiment

| Requested question | Result from this attempt |
| --- | --- |
| Did posterior conditioning change the local gaps? | **Unknown.** Neither range's new values were run. |
| Do high-visit trained keys still show credible local errors? | **Unknown under the posterior.** All 24 selected keys have ≥300 visits, but selection and timing are not value evidence. #117's uniform-range evidence remains separate. |
| Did all suit-permutation controls pass? | **Not run.** No control outcome was opened. |
| Did independent river references agree? | **Not run.** No reference outcome was opened. |
| What training mechanism is supported? | **None newly established.** Do not infer card/history abstraction or more-nodes benefit from this stopped attempt. |
| Exactly one next experiment | **An outcome-free faithful-LBR likelihood acceleration benchmark on M4.** Batch/cache the unchanged LBR calculation across hypothetical holdings without changing its action contract or RNG law. For fixed public prefixes, hypothetical holdings and seeds, compare every optimized action to the present native implementation; include suit-permuted prefixes. Then repeat the same cost projection. Require a measured common four-sample/96-world plan with the declared controls and headroom before scheduling conditional values. This is an implementation and timing experiment, not training. |
| M4 or paid parallel compute? | **M4 for that benchmark.** The measured bottleneck is runtime, not RAM, and no validated faster executor or specific frozen paid-host workload exists. Do not rent from this result. |

The conservative projection would require roughly a **35× likelihood speedup** to fit four samples plus the value/control reserve and 1.25× headroom in the remaining envelope; the overall-mean extrapolation implies roughly **18×**. These are engineering targets for an **outcome-free** pilot, not performance claims. If faithful acceleration fails, a later proposal may compare a larger authorized M4 window with a specific paid CPU/RAM option and its current price. No new compute budget or follow-on run is authorized here.
