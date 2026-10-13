# K50 HU200 752M: missing-key check/call fallback

The test does **not show a direct improvement**. The candidate’s return against the plain model
is **−4.43 [−14.27, +5.42] BB/100** in 131,072 fresh duplicate blocks. The
predeclared rule fails, so no Slumbot confirmation or new account is proposed.
Random and the half-pot bettor improve; pot pressure is inconclusive. The default
remains uniform.

The play-time hypothesis is that checking when legal, otherwise calling, **only
at missing keys** improves #250's unchanged 752,593,175-node average. Stored
positive-mass and zero-mass keys retain their exact distributions. Translation
is off; there is no training. The default remains uniform.

The rule is exactly #251's `diagnostic_action(view, 'call')` in
[`scripts/pot_pressure_diagnostic.py`](../../scripts/pot_pressure_diagnostic.py).
Each decision still consumes one sampler draw. Metadata is sparse for backward
byte compatibility: an omitted `missing_key_fallback` always means `uniform`.
Check-call policy descriptions, model specs, decision traces and Slumbot journals
identify the option; applied decisions are counted by street. Analysis separates
missing keys from stored zero-mass keys.

## Descriptive Slumbot split

Read-only, independently chain/settlement-verified completed hands, no all-in
adjustment. The M1 sample comprises #261's three parts (186 + 228 + 4,586 hands).
The dedicated-account sample is a fixed, byte/hash-anchored prefix of db-ocean's
live `k50-752m-20261013-1218` journal; that run remains active and untouched.
Intervals are nominal Student t 95% over independent completed hands within each
selected group. Unfinished/abandoned decisions are excluded from hand groups.

| Sample | Missing-key hands | Net chips | BB/hand [95%] | No-missing hands | Net chips | BB/hand [95%] |
|---|---:|---:|---:|---:|---:|---:|
| M1 #261 | 191 | −296,150 | −15.51 [−27.61, −3.40] | 4,809 | −107,800 | −0.22 [−0.55, +0.10] |
| Dedicated account prefix | 829 | −1,354,750 | −16.34 [−22.13, −10.55] | 22,323 | −377,300 | −0.17 [−0.35, +0.01] |

Hands touching a miss on each street (columns overlap when one hand misses on
several streets), followed by the first-miss partition:

| Sample | Preflop | Flop | Turn | River | First miss: preflop/flop/turn/river |
|---|---:|---:|---:|---:|---|
| M1 #261 | 46 | 67 | 81 | 86 | 46 / 43 / 49 / 53 |
| Dedicated prefix | 151 | 236 | 346 | 449 | 151 / 157 / 241 / 280 |

The completed M1 hands contain 320 missing decisions and one stored zero-mass
decision. The dedicated prefix contains 1,312 missing decisions and two stored
zero-mass decisions. These categories are distinct; a stored key with no average
mass is not a missing key.

**This is selection-biased:** missing keys arise in unusual, big-pot lines. It
motivates the test but does not measure the fallback. Both samples played uniform
fallback; their historical outcomes cannot be attributed to check-call.

[Per-part hashes, identities and streets](hu200-missing-key-fallback-artifacts/m1-reanalysis.json),
[dedicated-prefix record](hu200-missing-key-fallback-artifacts/server-reanalysis.json).

## Default identity and evaluation protocol

Main `c53f7935` versus the branch, fixed 32-block HU100 and HU200 matches, root
202610130701, rung `identity`: all decompressed hand and decision SHA256 values
match. Gzip envelope timestamps/output names are nondeterministic and excluded;
all canonical trace bytes are identical. A separate 32-block /64-hand HU200 match
uses the **actual 752M model and pinned K50 tables**, root 2026101309051, main
`c53f7935ebbc45180e179e5a3aea74af48bff783` versus scientific source
`134719693a3204dedb0c92ba7b59f5add563837e`: hand SHA256
`a92f72d673df506be41479db2ddd7b3890d78a9a995e97920c93afa720c547e8`, decision
SHA256 `4f236b8cdba2f8f9a9d06c5b54f59aaa00bf9e22689506e4274a52f6647fefc8`
on both. [Real-model identity receipt](hu200-missing-key-fallback-artifacts/real-hu200-default-identity.json).
Only hashes, never outcomes, were read from this separate identity check. [Exact hashes and fixtures](hu200-missing-key-fallback-artifacts/default-identity.json).

The paired decision rule is predeclared: recommend a 5,000-hand owner-started
Slumbot confirmation only if the direct paired nominal lower 95% bound is above
zero and no scripted paired nominal upper 95% bound is below −20 BB/100. Report
all seven scripted contrasts and fallback shares by street. No new account is
proposed before the rule passes.

A cost-only pilot uses 1,024 direct duplicate blocks and 64 blocks per scripted
opponent. It exposes time, memory and paired SDs without outcome inspection.
Target final counts are 131,072 direct blocks and 16,384 per scripted opponent;
one reduction is permitted before outcomes only if projected wall time exceeds
12 hours at up to four workers. The canonical plan was published and read back from [#267](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/267)
before final admission. Scientific source `134719693a3204dedb0c92ba7b59f5add563837e`,
direct root 2026101309001, panel root 2026101309002; the plan keeps the target
counts, totaling 720,896 primary hands plus exact reproduction. The cost-only
pilot completed in 97.39 seconds; the conservative four-worker projection,
including a second serial replay for readout, is 3.795 hours, below 12 hours.
No counts were reduced. [Pilot costs/SDs/guard summary](hu200-missing-key-fallback-artifacts/pilot-quote.json).

All server jobs run detached in our own tmux sessions at nice 10. Every resource
reading is retained at five-second intervals. Stop at available memory below
2 GB for 60 seconds, swap growth above 1 GB, disk below 20 GB, invalid hands,
failed reproduction or the declared wall cap. Model/tables and the verified
cache under `~/deepcfr-slumbot` are opened read-only; both policies share the same
loaded arrays with private fallback settings and samplers. Every final hand
must replay and reproduce exactly before aggregate readout. The owner decides
merge; no default model, translation or release change follows this experiment.

## Paired results

All 720,896 primary hands replayed and exactly reproduced before readout. Direct
uses two seat rotations per independent block; each scripted block uses those
same deals/streams for both arms (four hands). Intervals use block means, never
individual seats or hands. Direct and each panel have nominal Student t 95%
intervals; the panel's seven-comparison Bonferroni intervals are also shown.

| Opponent | Blocks | Direct win rate or scripted difference BB/100 [nominal 95%] | Seven-panel corrected 95% |
|---|---:|---:|---:|
| Direct duplicate | 131,072 | −4.43 [−14.27, +5.42] | — |
| Random | 16,384 | +131.50 [+59.76, +203.25] | [+33.02, +229.98] |
| Check/call | 16,384 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| Tight aggressive | 16,384 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| Loose aggressive | 16,384 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| 1.5× pot pressure | 16,384 | +6.92 [−27.12, +40.95] | [−39.80, +53.63] |
| Half-pot pressure | 16,384 | +36.12 [+19.93, +52.31] | [+13.90, +58.35] |
| Matched 1× pot pressure | 16,384 | 0.00 [0.00, 0.00] | [0.00, 0.00] |

The direct row is the candidate’s net win rate against the plain model, as in
#263; the plain model’s net is its opposite. Scripted rows are candidate minus
plain against that fixed opponent.

The direct lower bound is −14.27, which does not clear zero. No scripted upper
bound is below −20 BB/100, so the scenario safeguard passes; both conditions were
required. **No Slumbot confirmation is recommended under this rule.** There is no
new account or launcher. This does not establish that check-call is weaker: the
direct interval includes zero and small gains.

| Scripted opponent | Plain BB/100 | Check-call BB/100 |
|---|---:|---:|
| Random | +131.54 | +263.04 |
| Check/call | +103.86 | +103.86 |
| Tight aggressive | +67.52 | +67.52 |
| Loose aggressive | +137.67 | +137.67 |
| 1.5× pot pressure | −227.93 | −221.01 |
| Half-pot pressure | −2.68 | +33.44 |
| Matched 1× pot pressure | +58.29 | +58.29 |

The random and half-pot gains also clear the seven-panel corrected intervals.
The 1.5× pot bettor still beats both variants by over 220 BB/100. Four controls
never encounter a missing key; their arms reproduce identical hands and their
contrasts are exactly zero. Those zero contrasts measure no fallback exposure.

## Missing-only exposure by street

Cells give missing-key decisions /all model decisions on that street (share).
Candidate missing decisions are exactly the recorded check-call applications.
Stored zero-mass decisions are separate in the linked coverage data and never
trigger the option. Counts describe the paths each arm actually reached;
changing an earlier missing decision can change later exposure.

| Check-call arm | Preflop | Flop | Turn | River |
|---|---:|---:|---:|---:|
| Direct duplicate | 0 /271,402 (0.000%) | 1 /168,389 (0.001%) | 8 /103,983 (0.008%) | 61 /66,563 (0.092%) |
| Random | 5,431 /32,509 (16.706%) | 2,102 /11,542 (18.212%) | 775 /4,497 (17.234%) | 318 /1,782 (17.845%) |
| Check/call | 0 /32,768 (0.000%) | 0 /28,634 (0.000%) | 0 /28,634 (0.000%) | 0 /28,634 (0.000%) |
| Tight aggressive | 0 /19,064 (0.000%) | 0 /6,367 (0.000%) | 0 /3,289 (0.000%) | 0 /1,748 (0.000%) |
| Loose aggressive | 0 /27,639 (0.000%) | 0 /14,300 (0.000%) | 0 /9,653 (0.000%) | 0 /6,953 (0.000%) |
| 1.5× pot pressure | 1,639 /22,000 (7.450%) | 3,225 /8,581 (37.583%) | 4,193 /6,397 (65.546%) | 4,712 /5,588 (84.324%) |
| Half-pot pressure | 226 /22,488 (1.005%) | 302 /8,591 (3.515%) | 367 /4,804 (7.639%) | 477 /2,960 (16.115%) |
| Matched 1× pot pressure | 0 /22,058 (0.000%) | 0 /8,076 (0.000%) | 0 /4,353 (0.000%) | 0 /2,308 (0.000%) |

| Plain arm | Preflop | Flop | Turn | River |
|---|---:|---:|---:|---:|
| Direct duplicate | 0 /271,976 (0.000%) | 0 /168,204 (0.000%) | 8 /104,178 (0.008%) | 65 /66,735 (0.097%) |
| Random | 5,431 /32,509 (16.706%) | 2,102 /11,542 (18.212%) | 775 /4,497 (17.234%) | 318 /1,782 (17.845%) |
| Check/call | 0 /32,768 (0.000%) | 0 /28,634 (0.000%) | 0 /28,634 (0.000%) | 0 /28,634 (0.000%) |
| Tight aggressive | 0 /19,064 (0.000%) | 0 /6,367 (0.000%) | 0 /3,289 (0.000%) | 0 /1,748 (0.000%) |
| Loose aggressive | 0 /27,639 (0.000%) | 0 /14,300 (0.000%) | 0 /9,653 (0.000%) | 0 /6,953 (0.000%) |
| 1.5× pot pressure | 2,083 /22,468 (9.271%) | 2,467 /7,780 (31.710%) | 1,900 /4,082 (46.546%) | 1,289 /2,153 (59.870%) |
| Half-pot pressure | 279 /22,582 (1.235%) | 162 /8,464 (1.914%) | 186 /4,602 (4.042%) | 198 /2,656 (7.455%) |
| Matched 1× pot pressure | 0 /22,058 (0.000%) | 0 /8,076 (0.000%) | 0 /4,353 (0.000%) | 0 /2,308 (0.000%) |

The direct candidate applies the fallback only **70 times in 610,337 decisions
(0.01147%)**, including 61 river decisions. This is a very low-exposure
self-play test: it does not show a gain under the required direct gate, while the
half-pot and random panels test many more interventions. These results support
scenario-specific gains, not a general or Slumbot strength claim. No sample was
extended or rerun after outcomes.

[All contrasts, arm intervals, coverage and replay counts](hu200-missing-key-fallback-artifacts/paired-results.json).

## Resources and validation

Pilot 97.39 s; final play/replay/reproduction 42.06 min; final guarded
readout 14.38 min; science plus readout 56.44 min.
At most four workers, launched with `nice -n 10`. Final evaluation retained
475 guard readings, with minimum available memory
7.378 GB, maximum swap growth
0.233 MB, and minimum free disk
95.069 GB. Readout retained 170
readings: memory ≥7.909 GB, swap growth ≤0.520 MB,
disk ≥95.062 GB. No evaluation guard breach, invalid hand,
failed replay or reproduction, or scientific retry.

The initial descriptive server pass preceded the guard controller. The exact
same byte-anchored prefix was then independently reverified under guards,
reproducing the complete analysis exactly; both passes and source provenance
are retained. No pilot or final evaluation was unguarded. This disclosed setup
omission does not resample journals or alter the final comparison.

[Costs and guard summaries](hu200-missing-key-fallback-artifacts/final-costs-and-guards.json),
[validation and setup deviation](hu200-missing-key-fallback-artifacts/validation.json).
Local testing: 2,058 full-suite passes /113 skips /54 subtests, followed by all
three clean-source admission checks passing after committing the report edits.
The focused scientific-source suite passes 134 tests; later analysis/cache CLI
checks pass. CI builds native code and runs both full shards. The final-head CI
record is captured at PR handoff.

## Evidence storage

Server traces and closed analyses remain at
`/home/davide/deepcfr-eval/missing-key-cc/`. A member-hashed, fully read-back ZIP
contains plans, summaries, all primary/reproduction traces, every guard reading,
analyses, validation/setup records and source patches. Shared model/table/cache
inputs and original Slumbot journals are excluded and retain their indexed
restoration/prefix provenance. Storage receipt and whole/member hashes are in
RESULTS_INDEX. The M1’s reachable Wi-Fi path reported neither metered nor constrained status;
the ZIP was transferred with its whole SHA256 preserved into
`~/Local/Research-Cloud/PR-267-hu200-missing-key-fallback/`. Native uploaded1 /
uploading0 /zero unresolved conflicts and independent cloud ID/name/size/parent
readback accept the [Drive archive](https://drive.google.com/file/d/1gDOgSgoEbAdWjUplAK2hgwCd4KwuJeUB/view?usp=drivesdk),
375,510,415 bytes, SHA256 `04c3f863b8a450b41dbabf2980a3a7207b62bf70647056d8129132088256c5bd`.
All 282 member sizes/hashes passed readback; embedded `ARCHIVE-MANIFEST.json`
SHA256 `50c6bc8d2e92446928886d6977afb93624506f643446c72a2bb1c44573dccfed`.
[Upload/transfer receipt](hu200-missing-key-fallback-artifacts/cloud-upload-acceptance.json).
Remote Google Drive bytes were not downloaded; this is separate from the complete
server ZIP readback and matching whole hash after M1 transfer. Post-archive
upload/guard/CI receipts remain compact Git lifecycle records. All server/M1
originals are retained; nothing is deleted, evicted or force-offloaded.

Archive setup had two failures before writing any ZIP: an unavailable import
path, then a syntax error from shell quoting while fixing that path. The complete
corrected helper was transferred as a file and retried only for archival; both
failed commands, guard receipts and original helpers are retained.
No scientific hand or result was rerun.
