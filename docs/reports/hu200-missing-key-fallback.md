# K50 HU200 752M: missing-key check/call fallback

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

**This is selection-biased:** missing keys arise in unusual, big-pot lines. It
motivates the test but does not measure the fallback. Both samples played uniform
fallback; their historical outcomes cannot be attributed to check-call.

[Per-part hashes, identities and streets](hu200-missing-key-fallback-artifacts/m1-reanalysis.json),
[dedicated-prefix record](hu200-missing-key-fallback-artifacts/server-reanalysis.json).

## Default identity and evaluation protocol

Main `c53f7935` versus the branch, fixed 32-block HU100 and HU200 matches, root
202610130701, rung `identity`: all decompressed hand and decision SHA256 values
match. Gzip envelope timestamps/output names are nondeterministic and excluded;
all canonical trace bytes are identical. [Exact hashes and fixtures](hu200-missing-key-fallback-artifacts/default-identity.json).

The paired decision rule is predeclared: recommend a 5,000-hand owner-started
Slumbot confirmation only if the direct paired nominal lower 95% bound is above
zero and no scripted paired nominal upper 95% bound is below −20 BB/100. Report
all seven scripted contrasts and fallback shares by street. No new account is
proposed before the rule passes.

A cost-only pilot uses 1,024 direct duplicate blocks and 64 blocks per scripted
opponent. It exposes time, memory and paired SDs without outcome inspection.
Target final counts are 131,072 direct blocks and 16,384 per scripted opponent;
one reduction is permitted before outcomes only if projected wall time exceeds
12 hours at up to four workers. Canonical final source, counts and fresh roots
will be published in the PR body before final admission.

All server jobs run detached in our own tmux sessions at nice 10. Every resource
reading is retained at five-second intervals. Stop at available memory below
2 GB for 60 seconds, swap growth above 1 GB, disk below 20 GB, invalid hands,
failed reproduction or the declared wall cap. Model/tables and the verified
cache under `~/deepcfr-slumbot` are opened read-only; both policies share the same
loaded arrays with private fallback settings and samplers. Every final hand
must replay and reproduce exactly before aggregate readout. The owner decides
merge; no default model, translation or release change follows this experiment.

Paired results and final guard/storage receipts are pending.
