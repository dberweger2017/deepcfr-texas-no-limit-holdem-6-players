# HU200 frozen-table playing comparison

[PR268](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/268)
· [Predeclared protocol](../hu200-frozen-evaluation-protocol.md)

**More training on the same table made this lineage stronger against its 752M
parent.** The final 20B average wins the direct duplicate match by
**+76.95 [+71.93, +81.96] BB/100**. The predeclared Slumbot-confirmation rule passes:
the primary lower bound is above zero and none of the seven scripted difference
intervals has an upper bound below −20 BB/100. This recommends an external
confirmation, not a release or a claim about Slumbot strength.

**We did not detect further progress from 5B to 20B against the same parent.**
The paired overlapping-block difference is
**+0.09 [-8.36, +8.54] BB/100**. This is an inconclusive
learning-curve result, not proof of convergence and not a direct 20B-versus-5B
duel. Both models clearly improve over 752M.

## Fixed models and game

Only cumulative training changed. HU200, #163 K50 cards, menu v1, seed
2026101302, R=1, linear CFR and the opponent-sampled lifetime average remained
fixed. #265 exactly reproduced #250 before freezing all **85,373,907 keys**;
its sorted-key SHA256 is
`0318ed5894960151e5e55f8fde538bdfe1eda1d3ec496877eaf735f34ef4bf3c`.
Every chosen current/average export passed all retained audit rows. This
campaign verified the full model and table hashes, exported headers, parent
hashes, actual counters, frozen start iteration and unchanged sorted-key digest.
[Exact input pins and restoration provenance](hu200-frozen-evaluation-artifacts/inputs.json).

| Average | Actual nodes | Iterations | Bytes | SHA256 |
|---|---:|---:|---:|---|
| #250 terminal | 752,593,175 | 493,496 | 2,221,370,011 | `60bd37c979e018f11f136370f468cc41b8e1435550531707e89d45f28709ec7c` |
| #265 frozen 5B | 5,000,000,031 | 3,748,464 | 2,756,501,598 | `43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512` |
| #265 final 20B | 20,000,002,032 | 16,670,844 | 3,164,563,975 | `9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6` |

#250's target-named `1000000000.average.gz` is the 752M cap terminal, not a
reached 1B model. The final really reached 20B; no earlier final was substituted.
The [merged #265 report](hu200-k50-frozen.md) and
[model index](hu200-k50-frozen-artifacts/model-index.json) retain the full audits,
checkpoint parents and archive locators. The baseline was restored from its
accepted average-only ZIP under the detached guard, with whole ZIP, embedded
manifest and extracted member checks. Candidates and tables were read-only
indexed originals; their files and existing caches were not modified.

Evaluation used merged #263's HU200 execution/audit functions at source
`cc98b63858ed508ccff3491fc86c9e5f8bca87f6`. Each policy had its own schema/table
identity. Translation was **off**, with uniform missing-key and zero-mass play;
the newer optional check/call missing-key fallback was not selected. Each hand
reset to 200 BB, blinds 50/100 chips, no rake/ante. Cards, evaluator seeds and
opponent private information never entered policy observations.

## Counts, pairing and direct results

All requested final counts fit; **no reduction** was used. The cost-only pilot
played exactly 1,024 primary direct blocks and 64 paired blocks per script;
only timing, memory and paired SDs informed sizing. Three caches were built
serially without hands, taking 365.90, 376.33 and 369.05 seconds. Their explicit
unsynced M4 path was `results/hu200-frozen-evaluation/policy-cache`. Each later
contrast ran in its own process, with at most two distinct policies mapped.
The primary direct timing was the declared 5B cost proxy; no extra pilot hands
were played for that pair.

The [canonical plan](hu200-frozen-evaluation-artifacts/canonical-plan.json),
SHA256 `4c5079b481d316d953f8c89aecd1a9a89692ab7ba2d4277ff658fd5799f780c7`,
was posted in the PR description and exactly read back at **01:23:09 Madrid**,
before final hands. Its 1.25× padded remaining-science quote was 9,892.66 seconds
(2.748 hours), with the fixed 09:30 science end and 10:00 closeout deadline.
[Posting/readback receipt](hu200-frozen-evaluation-artifacts/plan-posted.json).
Roots were reserved against current main and every other open PR before pilots;
[reservation](hu200-frozen-evaluation-artifacts/root-reservation.json).

| Contrast | Independent blocks | Primary hands | Difference, nominal paired Student-t 95%, BB/100 |
|---|---:|---:|---|
| Final 20B − 752M, primary direct duplicate | 262,144 | 524,288 | **+76.95 [+71.93, +81.96]** |
| Frozen 5B − 752M, descriptive direct duplicate | 131,072 | 262,144 | +77.12 [+69.51, +84.73] |
| Final − 5B return against 752M, overlapping shared blocks | 131,072 | No additional hands | +0.09 [-8.36, +8.54] |

A direct block plays one physical deal twice, swapping model seats; the block
average is the independent observation. Both direct contrasts intentionally
share root `20261014091503` and rung `frozen-lineage`. The descriptive final−5B
learning curve pairs the first 131,072 final blocks with all 5B blocks. Final's
return against 752M on that subset is +77.21 BB/100; its full-sample primary
mean is +76.95, so subtracting the two full-sample means would not reproduce
the paired learning-curve estimate. No additional model pair was played.

Intervals are nominal 95%, with no multiple-comparison correction, exactly as
predeclared. They condition on these pinned models, one training lineage and
this internal benchmark; they do not measure training-seed variation or
unrestricted-game convergence.

## Seven-opponent paired panel

Each opponent used **16,384 independent blocks /65,536 primary hands**:
final and baseline in both seats on shared physical deals and logical-role
private action streams. Different policy actions can consume different later
draws. Each interval below uses block-level paired differences, not independent
hands. All rates and intervals are BB/100.

| Script | 752M absolute, 95% | Final 20B absolute, 95% | Final − 752M paired, 95% |
|---|---|---|---|
| random | +95.13 [+23.37, +166.90] | +114.57 [+39.78, +189.37] | +19.44 [-20.21, +59.09] |
| check/call | +98.00 [+89.94, +106.07] | +162.82 [+150.27, +175.38] | +64.82 [+52.42, +77.22] |
| tight aggressive | +47.84 [+31.75, +63.93] | +73.37 [+62.47, +84.28] | +25.53 [+10.06, +41.00] |
| loose aggressive | +112.74 [+81.16, +144.33] | +117.27 [+93.48, +141.06] | +4.52 [-27.45, +36.50] |
| 1.5× pot | -256.57 [-290.40, -222.73] | -259.97 [-293.36, -226.59] | -3.41 [-37.17, +30.36] |
| half pot | -9.54 [-30.73, +11.66] | +19.49 [+3.34, +35.64] | +29.03 [+6.60, +51.46] |
| matched 1× pot | +55.84 [+33.03, +78.64] | +77.28 [+61.32, +93.24] | +21.44 [-0.66, +43.55] |

Check/call, tight-aggressive and half-pot paired gains clear zero at nominal
95%; the other four paired differences are inconclusive. All seven upper
bounds exceed −20 BB/100, passing the severe-regression part of the fixed rule.
Random, loose-aggressive and 1.5× pot intervals still include losses worse
than −20 BB/100. The predeclared upper-bound criterion passes despite that
remaining uncertainty; absence of a regression is not established.

**1.5× pot pressure still beats both models badly:** final is −259.97
[−293.36, −226.59] BB/100. Its difference from 752M is inconclusive. The half-pot
and matched 1× pot controls keep #263's thresholds, bluffing, raise frequencies
and two draws per decision fixed; they vary sizing only. They are scripted
controls, not approximations of Slumbot's strategy.

## Coverage and reached-key visits

[Complete per-contrast/model/street coverage](hu200-frozen-evaluation-artifacts/coverage.csv)
retains exact decision counts and all four visit bands for **every** chosen
model. Missing-key and zero-mass percentages use all model decisions. Visit
band percentages below use **known reached decisions**, including known
zero-mass keys and excluding missing keys. The CSV also preserves the original
all-decision visit-band counts; missing keys are not mislabeled as known
zero-visit keys. These are decision-weighted reached-key statistics, not the
row-weighted stored-table distribution. Rounding can hide very small nonzero
shares; the CSV has unrounded fractions. The Git copy uses LF line endings;
the original archived CSV bytes are retained unchanged.

The 752M and 20B rows below come from the primary; 5B rows come from its direct
match against 752M. Different policies reach different histories, so these
coverage changes do not isolate a causal effect on a fixed set of states.

| Model | Street | Decisions | Missing | Zero mass | Known visits 0 | 1–9 | 10–99 | 100+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 752M | preflop | 566168 | 0.000% | 0.000% | 0.00% | 0.05% | 0.99% | 98.96% |
| 752M | flop | 371419 | 0.000% | 0.001% | 0.01% | 0.17% | 2.30% | 97.52% |
| 752M | turn | 239981 | 0.007% | 0.007% | 0.11% | 1.15% | 10.90% | 87.84% |
| 752M | river | 167616 | 0.062% | 0.067% | 0.59% | 4.60% | 31.65% | 63.16% |
| 5B | preflop | 274354 | 0.001% | 0.000% | 0.00% | 0.01% | 0.36% | 99.63% |
| 5B | flop | 187109 | 0.001% | 0.000% | 0.00% | 0.02% | 0.60% | 99.38% |
| 5B | turn | 122807 | 0.018% | 0.002% | 0.01% | 0.20% | 2.53% | 97.26% |
| 5B | river | 83359 | 0.084% | 0.014% | 0.02% | 1.09% | 8.30% | 90.59% |
| 20B | preflop | 545842 | 0.000% | 0.000% | 0.00% | 0.00% | 0.09% | 99.90% |
| 20B | flop | 378372 | 0.002% | 0.000% | 0.00% | 0.00% | 0.18% | 99.81% |
| 20B | turn | 251909 | 0.011% | 0.000% | 0.00% | 0.04% | 0.96% | 99.00% |
| 20B | river | 172482 | 0.069% | 0.001% | 0.01% | 0.33% | 4.15% | 95.51% |

Known direct river decisions with 100+ visits rise from **63.16%** at 752M to
**90.59%** at 5B and **95.51%** at final. That denser retained support accompanies
the large parent gain, but does not establish a measurable 5B-to-final gain.
Freezing does not add unsupported histories. Against 1.5× pot, missing-key
shares remain **32.05%/48.59%/58.42%** on baseline flop/turn/river and
**33.02%/48.99%/59.42%** on final. Both still use uniform fallback there. The
matched 1× pot panel has near-zero misses; the half-pot panel retains **1.02%,
3.46%, 7.23%** final flop/turn/river misses. These findings leave action/history
support as a separate limitation.

## Full verification, time and guards

All **1,245,184 final primary hands** independently replayed and exactly
reproduced **before any readout**: 524,288 primary, 262,144 learning curve and
458,752 scripted. Reproduction adds the same number of executed hands. Every
menu/key, action stream, public transition and settlement passed; failed hands
were not rerun. [Global verification receipt](hu200-frozen-evaluation-artifacts/all-final-verified.json),
[complete result summary](hu200-frozen-evaluation-artifacts/summary.json).

#265 explicitly released compute at 00:40:27 Madrid. At 01:00 the live release
statement, absence of matching processes and both free locks passed. This
campaign launched detached in M4 tmux at **01:02:08**, acquired the shared lock,
and finished science at **02:18:49**. The initial archive/guard closeout exited
at **02:18:55**, well before the fixed deadline. No part of science, sampling,
stopping or initial archiving depended on an M1 SSH session staying awake.

The **916 five-second scientific readings** are all retained. Sampled
family peak was **12.765 GiB**, minimum free memory
**72%**, minimum free disk **23.818 GiB**,
maximum swap growth **0 bytes** above the 1,418,786,242-byte campaign baseline,
and pressure was normal in every reading. Maximum observed sample gap was
**5.250 seconds**. The archive guard adds two readings.
There were **no sampled breaches, sustained stops or scientific failures**.
[Closed resource summary](hu200-frozen-evaluation-artifacts/resource-summary.json).

The unchanged guard stops only after 60 continuous observed seconds above
13 GiB family RSS, critical pressure, below 8% free memory or above 4 GB swap
growth; disk below 16 GiB stops immediately. Observation gaps reset sustained
timers. Owned PIDs are checked by creation time before any signal. No #265 or
other campaign processes were signaled.

## Slumbot confirmation preparation

The rule passes. The final average is copied and hash-verified on db-ocean at
`~/deepcfr-slumbot/inputs/model-frozen/hu200-k50-frozen-20b.average.gz`;
[source/transfer receipt](hu200-frozen-evaluation-artifacts/server-copy-receipt.json).
The direct detached M4 transfer took **299.64 seconds**, with verification
before the partial was atomically renamed. The existing server checkout
`5d0ed66ee6d59eb9d9eae859f3c4e5ac4c10c0ea` and venv were unchanged. Its K50
tables passed all three full hashes. Offline cache preparation and checks took
**754.76 seconds**, including ten seconds observing coexistence with the
existing 752M player. The existing Slumbot client's local loader then opened
the verified cache in **11.95 seconds**, with no service calls, policy actions
or hands. [Cache receipt](hu200-frozen-evaluation-artifacts/cache-receipt.json),
[client compatibility](hu200-frozen-evaluation-artifacts/client-verification.json).

The prepared, syntax-checked launcher is
`~/deepcfr-slumbot/run-frozen-20b.sh`, SHA256
`95e6f20552a826b012c2ce0a29088259530b930ae5586aa58867555c05696c02`.
It requests **5,000 hands**, fresh policy seed **20261014026801**, account
**`deepcfr-k50-frozen-20b`**, and the separate `~/deepcfr-slumbot/cache-frozen`
cache. It retains the existing launcher's limits of 200 abandoned and 50
rejected hands. The target CLI lacks `--cache-read-only`; the compatible
launcher uses the already verified cache, whose arrays are mapped read-only.
[Reviewable launcher](hu200-frozen-evaluation-artifacts/run-frozen-20b.sh),
[preparation receipt](hu200-frozen-evaluation-artifacts/launcher-receipt.json).
**It was not started.** The owner creates the named account, supplies its
credentials and starts the launcher. Existing `slumbot-*` sessions and
`~/deepcfr-slumbot/runs` were untouched; no account or credential was accessed.

The existing 752M player was present in every server reading; the held-load
observation measured its coexistence with the loaded frozen policy. Two caches
contain **5,115,547,961 bytes each**; with the shared K50 tables, their backing
files total **11,615,126,954 bytes /10.817 GiB**. This exceeds observed free and
available RAM, although file-backed pages can be reclaimed. No new live play
was used to measure concurrent throughput.

| Server memory observation | GiB |
|---|---:|
| Physical RAM | 15.621 |
| Two cache arrays plus shared table backing | 10.817 |
| Available RAM before cache preparation | 7.706 |
| Minimum available RAM during cache preparation | 3.046 |
| Peak new-process RSS during preparation | 4.825 |
| Held frozen process RSS after load | 0.268 |
| Existing 752M process RSS at that held observation | 0.330 |
| Available RAM after warm client load | 7.803 |
| Raw free RAM after warm client load | 0.184 |

The **136 retained server readings** use five-second waits, with maximum
observed gap **5.809 seconds**. Swap was already nearly full before preparation
(172,032 bytes free out of 4,294,963,200) and reached zero free during it; the
new process's sampled swap was zero. RSS is distinct from file backing and can
count shared pages in both processes. These measurements qualify the offline
handoff and describe the paging headroom; they do not certify concurrent-play
speed. [Server memory summary](hu200-frozen-evaluation-artifacts/server-memory-summary.json).

## Evidence and validation

The scientific ZIP is **971,526,585 bytes /229 members**, whole SHA256
`884cfb06fe9af9f70e8dc26d18f16750f103554ff63c3145bf984e0f0bcdd2d3`;
embedded `ARCHIVE-MANIFEST.json` SHA256
`1390563a4858e04ea8dccd020bde9f479a603f26e41b1419e1881774e29f7d00`.
Every member passed local size/SHA256 readback. Native uploaded1/uploading0/
no-conflicts and independent cloud ID/name/size/direct-parent confirmation accept
[the scientific archive](https://drive.google.com/file/d/1EwsMAG3ZdX7SVW1SIigpf8dgr5Vgd02_/view?usp=drivesdk)
in [PR268's folder](https://drive.google.com/drive/folders/1Dl6kCUsCh_4Y7SXIYMEg9zSmvln-n6Md).
[Local receipt](hu200-frozen-evaluation-artifacts/archive-receipt.json),
[native acceptance](hu200-frozen-evaluation-artifacts/native-upload-acceptance.json).
Remote archive bytes were not downloaded or reverified. Models and tables are
indexed dependencies, not duplicated in this ZIP or committed to Git. All
originals and caches remain; no synced deletion, eviction or forced offload.

The closeout supplement is **153,967 bytes /25 members**, whole SHA256
`3be0d7e75116da0b0511ce0e63518cbe769409862c07e23c85dd2f2448542565`,
embedded manifest SHA256
`ad58731d46553cd7329cd7c1be17fc8dc31b99542e45621b8a000b5e433cf0b8`.
It preserves the complete closed archive guard, upload receipts, every server
preparation reading, offline compatibility/transfer receipts, launcher/helper
sources and administrative failures. [Local closeout readback](hu200-frozen-evaluation-artifacts/closeout-archive-receipt.json).
Native uploaded1/uploading0/no-conflicts and independent ID/name/size/direct
parent metadata confirm [the closeout upload](https://drive.google.com/file/d/1Zv1zSZZDwTm1o2gKircuWG3Qv62e-6ZY/view?usp=drivesdk).
[Native receipt](hu200-frozen-evaluation-artifacts/closeout-native-upload-acceptance.json),
[cloud confirmation](hu200-frozen-evaluation-artifacts/closeout-cloud-upload-acceptance.json).
Both archives are accepted. Remote archive bytes were not downloaded or
reverified. Restoration and separate acceptance are recorded in RESULTS_INDEX.md. The initial
checkout refspec error was fixed before science and preserved in the input
configuration. Server setup/CLI capability checks and the metadata helper's
import-path failure were corrected before any live play or archive write;
[all administrative failures](hu200-frozen-evaluation-artifacts/admin-failures.json)
and the original failed helper log remain in the supplement. There were no scientific retries, failed-hand replacement,
outcome-dependent extension, count changes, new training, account creation,
release, tag, default change or merge.

Before the campaign, 35 focused admission/guard/HU200 replay/cache tests and all
five exact-source CI checks passed. Closeout validation repeats the 35 focused tests, checks staged repository
storage and whitespace, and reviews the source/result/receipt diff against the
predeclared protocol. No correctness findings remain open. Final-head CI is
checked on the PR before handoff; the owner decides the merge. The owner decides the merge.
