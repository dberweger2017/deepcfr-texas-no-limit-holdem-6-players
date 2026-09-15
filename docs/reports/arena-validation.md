# Arena validation protocol

Declared before running the control experiment:

- Use `configs/arena/sensitivity.json`: 128 independent four-player fixed-stack blocks, 100 BB, validation root seed 49021. Run every seat rotation in both arms, for 1,024 hands total.
- Candidate checks/calls; baseline folds whenever legal. Opponents are selected from checks/calls and folds using the saved opponent stream.
- Acceptance: all hands complete legally, accounting reconciles, and the approximate 95% Student-t interval for the paired improvement in BB/100 excludes zero on the positive side.
- This is a deliberately easy infrastructure control. It is not a competitive poker benchmark or evidence about a trained model.
- A two-block smoke run and a comparison of an identical policy against itself must remain inconclusive. Reproducing a saved run must give identical deterministic hand records; timings are excluded.
- If the control is inconclusive or fails, retain the result. Do not extend this budget after inspecting the interval. Diagnose the protocol or implementation in a separate declared experiment.

## Results — September 15, 2026

The control passed on the declared 128-block budget. All 1,024 scheduled hands completed, with no failed hands or invalid actions.

| Measure | BB/100 | Approximate 95% interval |
| --- | ---: | --- |
| Checks/calls candidate | 36.95 | [28.94, 44.96] |
| Folding baseline | -31.35 | [-33.79, -28.91] |
| Paired improvement | **68.29** | **[61.86, 74.73]** |

The two-block smoke plan completed 124 hands across six-player fixed stacks, five-player unequal stacks, and four-player bankroll sessions. All three comparisons were **inconclusive**, with confidence intervals withheld. Unit tests also verify identical-policy comparisons remain inconclusive, rotations do not inflate sample size, and failures suppress every strength estimate.

Both saved runs reproduced with byte-identical hand records and matching report values excluding timings. The complete suite passed **181 tests**; targeted Ruff checks and the diff whitespace check passed. CI also runs and reproduces the smoke plan.

The first control ran at `37710c9` and reproduced successfully. Review then removed private scenario labels from public hand identifiers and made reports name their policy inputs. The same fixed control schedule was re-executed at clean revision `5f4832cb6e4520c1cc711e8989b7a623e0296f82`; its estimates are unchanged. This is the same data schedule, not additional independent confirmation or an expanded budget.

The final local run took about 2.6 seconds inside the runner. These tiny built-in-policy timings are not a neural inference or training throughput measurement.

[Compact provenance and result hashes](arena-validation.json) are committed. Full local bundles are retained under ignored `results/arena-final-control`, `results/arena-final-control-replay`, `results/arena-final-smoke`, and `results/arena-final-smoke-replay`. Recreate the plans using the recorded implementation and environment; the strict reproduction command intentionally rejects a different source fingerprint or engine binary.

This validates schedule execution, reproduction, conservative small-sample handling, and sensitivity to an intentionally easy difference. It does not validate a competitive opponent pool, old checkpoints, the CFR algorithm, or general playing strength.
