# Frozen HU200 v0.5.5 blend: normalized-export stop

**Stopped before building a blend or playing any hand.** The requested prerequisite
is false: these average exports store normalized action probabilities, rather
than raw action accumulators. The owner explicitly instructed: “If the exports
are already normalized, stop and report before choosing anything else.” No
alternative formula has been selected.

Source inspected: current main `98658f7b` (full revision in the evidence receipts).
The required frozen, confirmation, translation and evaluation-harness reports,
repository workflow and artifact-storage contract were read before execution.
Input SHA256 verification and bounded header/row inspection on the 16-GiB M4
are being recorded; no training, union scan, merge, cache construction, pilot,
final play, Slumbot request or other machine's research compute is authorized by
this stopped attempt.

## Export and weighting semantics

`native/hu20-trainer/src/export.rs`, function `export_in`, reads each checkpoint
row as `[key, names, regrets, accumulated, visits]`. It computes
`total = fsum(accumulated)` and writes the average-export row as
`[key, names, p, total, visits]`, where `p[a] = accumulated[a] / total` for a
positive-mass row. Zero-mass rows use `1 / len(names)` with the selected uniform
fallback. The extraction label is
`normalize-lifetime-iteration-opponent-sampled-accumulator-v1`.

`src/blueprint/average.py`, `AveragePolicy.__init__`, validates that `p` is a
probability vector with sum one and loads it directly into inference storage.
`src/diagnostics/cfr_average.py` independently uses the same representation.
Thus the third field is normalized; the fourth is a key-local accumulated mass.

`native/hu20-trainer/src/cfr.rs`, `Options::weights`, returns `(t, t)` for linear
CFR. In `Traversal::visit`, opponent-sampled averaging adds `t * policy[a]` on
each sampled opponent-node visit, before sampling its action. There is no
own-reach multiplier in that branch. The per-key `total` is the sum over actions
of those accumulated updates; it is not a seed-wide total or the traverser
visit count. The nominal iteration-weight schedule through iteration `T` sums
to `T * (T + 1) / 2`; that schedule alone does not describe which sampled visits
a particular retained key received. No seed-wide normalization denominator has
been chosen for a blend.

The normalized probabilities and stored row mass permit a numerical estimate
of positive-mass raw accumulators by multiplication. That would require a
revised interpretation and is not performed under this instruction. A uniform
zero-mass probability vector still represents zero accumulator mass.

## Requested campaign, not run

The requested single rule was `sum_s accumulator_s(key, action) / W_s`, with
missing keys contributing zero, uniform zero-mass rows, and a distinct output
identity. The requested fixed evaluations were 131,072 direct duplicate blocks
against each of the three seeds and 16,384 blocks against each of seven scripts,
with all four models paired on common deals, translation 512/128, independent
replay and full reproduction before outcome readout. Those counts are retained
as scope, not represented as completed evidence.

Because the semantic stop precedes construction, union/overlap counts, export
and cache capacity projections, merge/cache resource peaks, played missing-key
shares and poker results are **not measured**. There is no capacity admission
or playing-strength conclusion. No release, tag, default change or merge.

## Evidence status

Initial ETA: approximately 15 minutes for the stopped-preflight record, input
checks and small archive/readback, subject to Drive upload acceptance. Final
receipts will replace this provisional status. Existing research inputs and
caches are retained; no cleanup is needed for a bounded read-only preflight.
