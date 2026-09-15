# Benchmark opponents and frozen-model validation

## Declared checks

These plans and acceptance checks are recorded before running their results:

- `configs/arena/opponents-smoke.json`: two blocks per scenario, four-/five-/six-player unequal stacks and a six-seat bankroll session. Require legal completion, chip conservation, and exact reproduction. Every comparison must remain inconclusive at this sample size.
- `configs/arena/historical-smoke.json`: two blocks for each six-player 20/100/200 BB scenario, 72 scheduled hands across both arms. Compare the archived self-play iteration 5900 against phase-one iteration 100, using the fixed style pool. Require legal completion and exact reproduction from bundled model bytes. No strength gate or checkpoint promotion is part of this check.
- `configs/arena/core-v1.json`: the predeclared 128-block, 100 BB style-control baseline at six, five, and four players, 3,840 scheduled hands. Record every scenario and its interval, including disappointing or inconclusive results. This checks the broader pool; it does not select a trained model or justify changing the default policy.
- Run the full regression suite. Require hidden-world invariance for frozen four-/five-/six-player test networks; unchanged global random streams and model weights; strict shape/metadata/hash validation; legal action mapping; and restoration of CPU settings after errors.

The two archived checkpoints do not record a training seed. Report that provenance as unknown. Different iteration numbers or files are not evidence of independent training runs. Synthetic checkpoint fixtures only test the transport of seed metadata and are not trained benchmarks.

Full suites for 20/200 BB, unequal stacks, and persistent bankroll sessions are committed as declared follow-up plans. Their larger campaigns are not needed to accept this infrastructure PR. Do not expand a budget after inspecting results to turn an inconclusive interval into a passing one.

## Results

Completed on September 15, 2026 at clean revision `af0912430b5d4cd3219da98ffec9f3f48c85ad1f`, after the protocol was committed. The [compact JSON record](benchmark-validation.json) preserves exact plans, source/schedule/outcome hashes, environments, policy provenance, timings, and all scenario comparisons. Raw bundles remain under the local ignored `results/` paths named there; they are not part of a fresh checkout.

| Check | Completed / scheduled hands | Failed hands / invalid actions | Result |
| --- | ---: | ---: | --- |
| Style smoke | 180 / 180 | 0 / 0 | Exact reproduction; all comparisons inconclusive |
| Archived-model smoke | 72 / 72 | 0 / 0 | Exact reproduction from snapshots; all comparisons inconclusive |
| Core control | 3,840 / 3,840 | 0 / 0 | Candidate exceeds check/call in each declared scenario |

Both smoke replays produced byte-identical hand records and identical report contents except performance. Their two-block budgets intentionally withhold intervals. The archived self-play-minus-phase-one differences were +492.75, +114.08, and +19.17 BB/100 at 20, 100, and 200 BB respectively. **These small samples do not establish which checkpoint is stronger.** Neither archive records a training seed; no independence or promotion claim follows.

### Core control, 100 BB

`tight_aggressive` versus `check_call`, against `styles-evaluation-v1`, 128 independent blocks per scenario:

| Players | Candidate BB/100 | Baseline BB/100 | Paired improvement BB/100 | 95% interval for improvement |
| --- | ---: | ---: | ---: | --- |

| 6 | 83.33 | -904.36 | 987.69 | [557.67, 1417.71] |
| 5 | 44.34 | -894.77 | 939.10 | [557.32, 1320.89] |
| 4 | 19.24 | -526.99 | 546.22 | [251.64, 840.81] |

These large differences mostly show how costly unconditional calling is against aggressive controls. They are not expected win rates against competent humans or evidence for a trained policy. Every predeclared scenario is retained, with no sample extension or model change after inspecting the outcomes. The larger depth/session campaigns were not run.

### Regression evidence and limits

The final local suite passes **205 tests**. New checks cover legal varied-style play, named-pool purpose enforcement, hidden-world invariance for four-/five-/six-player frozen fixtures on preflop/flop/turn decisions, legal model decoding, unchanged global random streams and frozen weights, hash/metadata/shape/non-finite rejection, model snapshot replay after removing the original fixture, tamper detection, and restoration of Torch runtime settings after errors. CI also runs and reproduces the style smoke.

A subsequent review guard rejects a registry belonging to a different plan before output creation (`a186d22`). That guard is covered by the 205-test result and does not change successful policy decisions. Strict source fingerprints mean the above experiment bundles must be reproduced with their recorded source revision and environment, not the later guard revision. Test fixtures verify the current implementation's reproduction path independently.

Archived networks use fixed six-player absolute-seat inputs. Sessions and different player counts are rejected; there is no padding, seat-zero remapping, or opponent-model fallback. The original training rules and learning algorithm remain unvalidated. CPU runner timings in the JSON exclude preflight model loading/hashing/snapshotting and are not GPU throughput measurements. No training or paid compute was used for this PR.
