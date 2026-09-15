# Benchmark opponents and frozen-model validation

## Declared checks

These plans and acceptance checks are recorded before running their results:

- `configs/arena/opponents-smoke.json`: two blocks per scenario, four-/five-/six-player unequal stacks and a six-seat bankroll session. Require legal completion, chip conservation, and exact reproduction. Every comparison must remain inconclusive at this sample size.
- `configs/arena/historical-smoke.json`: two blocks for each six-player 20/100/200 BB scenario, 72 scheduled hands across both arms. Compare the archived self-play iteration 5900 against phase-one iteration 100, using the fixed style pool. Require legal completion and exact reproduction from bundled model bytes. No strength gate or checkpoint promotion is part of this check.
- `configs/arena/core-v1.json`: the predeclared 128-block, 100 BB style-control baseline at six, five, and four players, 3,840 scheduled hands. Record every scenario and its interval, including disappointing or inconclusive results. This checks the broader pool; it does not select a trained model or justify changing the default policy.
- Run the full regression suite. Require hidden-world invariance for frozen four-/five-/six-player test networks; unchanged global random streams and model weights; strict shape/metadata/hash validation; legal action mapping; and restoration of CPU settings after errors.

The two archived checkpoints do not record a training seed. Report that provenance as unknown. Different iteration numbers or files are not evidence of independent training runs. Synthetic checkpoint fixtures only test the transport of seed metadata and are not trained benchmarks.

Full suites for 20/200 BB, unequal stacks, and persistent bankroll sessions are committed as declared follow-up plans. Their larger campaigns are not needed to accept this infrastructure PR. Do not expand a budget after inspecting results to turn an inconclusive interval into a passing one.

Results will be appended after these checks finish.
