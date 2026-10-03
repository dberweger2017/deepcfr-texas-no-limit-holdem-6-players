# HU20 board pooling preparation

No new main solver outcomes and no paid compute. The corpus and thresholds
were committed in 69b7c02 before new fixture solves. Draft PR #149 retains the
prospective [protocol](../hu20-board-pooling-protocol.md),
[pending quote](../hu20-board-pooling-runpod-quote.md) and
[execution recipe](../hu20-board-pooling-execution.md).

The 3,000-deal, seed-1 average census selected a limped pot with flop check-check:
135 reaches on 135 unique boards, ahead of the qualifying alternatives. The
min-raised/check-check line had 39 boards and did not meet the predeclared 40
minimum. All 10,814 earlier-street lookups hit retained keys. Forty boards were
sampled without replacement with recorded seeds. No turn action or payoff was
used to choose the line or boards. Every new root has a 2-BB pot.

## Solver-free companion

The top-20 Set A/B key union contains 20 keys, all present in all three original
500M checkpoints. Streaming the original hash-verified checkpoint members took
15.56 seconds on M1 without solver or training. Across the 60 key/lineage rows,
visits range 18–12,905, median 3,172.5. Pairwise retained-average TV has median
0.1045 and maximum 0.8786. The raw
[companion](hu20-board-pooling-artifacts/companion.json) retains signed regrets,
normalized lifetime averages, average mass, current regret-match distributions,
all named-action comparisons and missing-key status.

Each selected key has positive-excess diagnostic contexts on 1–4 distinct #145
root boards. This narrow archived context set is not the historical distribution
of boards updating the trainer, which checkpoints do not retain. High visits
alone do not establish convergence, and disagreement alone does not identify
whether card pooling or trainer coverage caused the loss.

## Engineering validation

The external new harness preserves the original AGPL upstream and tool. It
collects sufficient statistics indexed by actual v1 hashes and a genuinely
shared equity codebook, then maps action probabilities by menu name. Linux RSS
uses `ru_maxrss` in KiB converted to bytes; macOS retains byte units.

The Mac tiny turn fixture passed nine comparisons against #145's qualified
binary: deterministic replay and both-seat blueprint, singleton full-v1 and
equity-50 losses. The source-verifying build script also rebuilt both external Mac harnesses in
9.20 seconds for the reference build; repeating the nine comparisons against
that freshly rebuilt reference passed. The [build receipt](hu20-board-pooling-artifacts/mac-build-verified.json)
and [repeated gate evidence](hu20-board-pooling-artifacts/mac-singleton-rebuilt-gates.json)
retain their different binary fingerprints. The focused Python suite plus
existing flop/turn diagnostic tests passed **36 tests**. Independent tests cover contradictory strategies on
two physically different boards sharing a real information key, shared equity
labels, named-action alignment, disk-policy inference, paired board intervals
and an incomplete campaign's inability to classify. Linux parity, real-export
V4 and resource pilots remain pending paid approval, not claimed passed.

Retained preparation incidents: the initial compiler rejected legacy upstream
implicit raw-pointer references; the same recorded #145 compiler flag resolved
it without upstream edits. A JSON macro syntax error in the new external module
was corrected before fixture execution. The first two-board test accidentally
compared a trips board with an unpaired board and had no equal descriptor; it
was replaced with two fixed unpaired fixtures. A one-off shell inspection used
the system Python without pokers and was rerun in the existing venv. No failed
main values, training or resource guard bypass occurred.

The [external manifest](hu20-board-pooling-artifacts/external-tool.json) pins
source hashes, Rust/Cargo lockfile and the new Mac binary. The separate AGPL
source archive is 769,095 bytes, hash
776e59c7f3b4692c958b10e0d5339d75f279fe63316039155021b4ad8d3c0305.
Only Python, documentation and JSON provenance are committed to this MIT repo.
The M4 was not contacted and no other job was altered.

## Conditional training recommendation

No training is proposed as part of this diagnostic. If board-blind v1 stays
close to the per-root witness, first audit native trainer updates, sampling
coverage and strategy averaging under the unchanged key; use a separately
authorized paired 100M, three-lineage A/B for one concrete correction. If the
shared v1 witness loses most of the gap and shared equity-50 retains a
substantial advantage, investigate a versioned equity abstraction with its own
coverage/memory preflight before the same size pilot. An expensive projected
witness alone does not prove an abstraction lower bound. Intermediate or
incomplete results retain the split or uncertainty rather than forcing either
training direction.

Use matching training/evaluation seeds, current and stored-average extraction,
a fixed fresh opponent panel including bounded LBR, paired uncertainty and
resource/key-occupancy reporting. #143's finer-card pilot and #144's history
compression comparison both retained negative strength point estimates; do not
combine their changes into an uncontrolled follow-up or extend v1 to another
long campaign solely because its visit counts are high. Freeze that follow-up
protocol and obtain its own compute approval before starting it.
