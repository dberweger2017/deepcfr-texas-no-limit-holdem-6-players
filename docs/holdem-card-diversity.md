# Nested-board card-diversity diagnostic

## Question

Does reliable-target transfer improve when training covers more river boards, and
does a small own-hand/board feature control explain an improvement beyond the
scaled baseline? This is a finite regression diagnostic. It does not measure
full-game strength or authorize a production model.

## Frozen matrix

The four fixed model arms are `original`, `scaled`, `cards`, and `features`.
`original`, `scaled`, and `cards` reuse the preceding diagnostic architectures.
`features` uses the scaled input stream and adds normalized `hand_value` tuples
for the visible seven-card hero holding plus the visible five-card board. The
feature extractor accepts only those seven cards and rejects overlaps. It does
not receive opponent holdings, hidden worlds, range support, Q values, or
range-conditioned equity. Categories are divided by 8 and tie-break ranks by
14; no fitted normalization is used.

Every arm uses seeds 907, 911, and 919, Adam at learning rate 0.001, batch 32,
gradient clipping at 1.0, and exactly 1,024 updates. Metrics are recorded at
steps 0, 128, 512, and 1,024. The 16-board arm contains 192 training contexts;
the nested 48-board arm contains 576. Both therefore use the same optimizer
updates but different effective exposure (about 171 versus 57 passes through a
context). This is a board-coverage comparison, not an equal-epoch comparison.

The shared range, two public continuation profiles, exact target weighting, and
decision-cost metrics are unchanged from the reliable-target diagnostic.

## Data and sealed comparisons

The generator creates 48 training, 8 validation, and 8 test board groups from
seed 20260917. It rejects every canonical suit-equivalent board from the prior
24-board diagnostic and keeps each complete board group in one split. Each board
has six compatible hero holdings and two betting situations, giving 768 total
contexts. Validation and test groups are shared by both training arms.

The primary validation/test comparisons are fixed before fitting:

1. `scaled` with 48 training boards versus `scaled` with 16, for board diversity.
2. `features` with 48 training boards versus `scaled` with 48, for the visible-card mechanism.

The other original/cards curves are descriptive. A primary candidate must lower
validation decision cost by at least 0.02 BB and 10% on every seed, stay within
0.02 relative regret RMSE of its baseline, and have training relative RMSE at
most 0.20. The sealed test evaluates only the predeclared primary arms once;
there is no outcome-driven replacement.

## Cost and checks

Calibration uses seed 809, two training contexts, and eight steps per arm. With
a 50% projection allowance, exact reference generation has a 3,600 second hard
cap and fitting has a 1,200 second hard cap. No rental, adaptive extension, or
campaign fit is implied by this protocol.

The preparation and diagnostic command is:

```bash
python -m scripts.check_card_diversity \
  --plan configs/holdem/card-diversity.json \
  --out results/card-diversity
```

Run admission calibration alone with `--calibrate`. After a completed run,
`--verify` reloads every weight in a fresh process, checks hashes and target
split bookkeeping, confirms that test targets never enter fitting curves, and
recomputes validation and sealed-test comparisons.
