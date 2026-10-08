# Why HU100 coverage did not remove aggressive-opponent losses

**Two different problems remain.** Tight_aggressive and loose_aggressive losses
are mainly in hands with positive-mass coverage throughout. Pot_pressure instead
exposes histories outside the current training menu's abstract support; more
training with that unchanged menu cannot populate those keys. Coverage measures
availability of an average, not convergence or a good response to an opponent.

This is a diagnostic of [#200's](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/200)
existing sample, not a new strength test or causal attribution. All five
checkpoint policies, five opponents and **102,400 candidate hands /396,161 actions
/177,171 target decisions** are included, wins and ties as well as losses.
Uniform's reused arms are excluded from these associations. All original
evidence remains unchanged. [Prospective protocol](../native-hu100-diagnosis.md).

## Missing keys: support, size and encoding

Every actual action, menu, key, legal bound, pot, actor and settlement passed
Python replay and native parity: **zero mismatched hands**. Lookup classes and
probabilities match the hash-verified averages. There is no demonstrated
native/Python encoding or menu correctness issue in this sample.

Across all checkpoints, the **28,636 missing decisions** split into 18,415 on
observed all-menu paths, 323 after off-menu histories whose key appears in
another verified checkpoint, 23 with supported alternate-menu witnesses, and
9,875 from histories/menus excluded by exhaustive supported-token search.
These are decision counts, not distinct-key/history counts. **Zero unresolved**
cases occurred; the implementation retains that class when a bound is reached.
Search inspected 605 public signatures, using 0.56 seconds of its 120-second cap.
Off-menu opponent actions were counted independently; they were never used
alone to declare a key unreachable.

At the final **11,042,440-node** capacity checkpoint:

| Opponent | Missing decisions | Supported but absent | Unsupported history/menu |
|---|---:|---:|---:|
| random | 1,056 | 15 | 1,041 |
| check_call | 71 | 71 | 0 |
| tight_aggressive | 29 | 29 | 0 |
| loose_aggressive | 92 | 92 | 0 |
| pot_pressure | 918 | 8 | 910 |

Tight/loose aggressive opponents made **zero off-menu actions** in the complete
final sample. Pot_pressure made 1,004; its 1.5-pot sizing generates many excluded
histories. Yet six of its missing decisions after off-menu actions have supported
alternate paths, and two have exact observed menu paths. Random's open jams are
another support gap; random nevertheless wins money for the target in this sample.
Support gaps therefore do not imply losses by themselves.

## Where the losses occur

These disjoint **hand** contributions use every panel's 4,096-hand denominator
and add to its BB/100. “Any zero” means at least one zero-mass decision and no
missing key; “no decision” includes opponent folds before the target acts.

| Final opponent | All positive | Any zero, no missing | Ever missing | No target decision | Total |
|---|---:|---:|---:|---:|---:|
| tight_aggressive | −71.42 | −9.94 | −6.71 | +22.61 | **−65.47** |
| loose_aggressive | −351.00 | +3.00 | −14.72 | +13.60 | **−349.12** |
| pot_pressure | +9.19 | −2.39 | −149.54 | +20.06 | **−122.68** |

All-positive-mass hands number 2,213, 2,897 and 1,894 respectively. Fixing missing
lookups alone cannot explain the tight/loose aggressive losses. Conversely,
pot_pressure's missing-key exposure is a strong descriptive priority; those
harder trajectories are selected by play, so −149.54 is **not** an estimated
benefit from adding support.

Positive-mass decisions still have limited traverser visits, especially late:

| Opponent / street | Decisions | Median visits | Visits <10 | Median average mass | Mean raise probability |
|---|---:|---:|---:|---:|---:|
| tight / preflop | 2,417 | 63 | 5.8% | 242,825 | 48.0% |
| tight / turn | 313 | 13 | 41.9% | 148,381 | 54.1% |
| tight / river | 151 | 7 | 53.6% | 79,705 | 49.5% |
| loose / preflop | 3,576 | 27 | 16.0% | 241,994 | 49.7% |
| loose / turn | 966 | 10 | 49.8% | 118,147 | 55.5% |
| loose / river | 629 | 4 | 73.0% | 69,898 | 59.1% |

Mass is an iteration-weighted opponent-sampled accumulator, **not** a sample
count; positive mass can coexist with zero traverser visits. Large mass alone
does not establish a well-learned strategy. Nor are all covered losses confined
to low visits: tight's 327 flop decisions with ≥100 visits have mean final payoff
−3.42 BB per decision. No convergence or plateau claim follows from this lineage.

In positive-mass cells, larger pots associate with worse final payoffs in **both
positions**. Tight's <4-BB versus ≥64-BB cells average −0.51 versus −45.45 BB on
BTN/SB, and −6.66 versus −53.68 BB on BB; loose's corresponding means are −2.75
versus −29.15 and −7.38 versus −41.56. These are decision-weighted final hand
payoffs, with repeated decisions repeating an outcome. They are not street
losses, additive contributions, action values or causal evidence of overraising.
The [complete cells](native-hu100-diagnosis-artifacts/decision-cells.csv) retain
every checkpoint/opponent/street/position/pot, visit and mass band, fold/passive/
raise probability, wins, ties, losses and distinct-hand counts, including small cells.

## Replayed examples and recommendation

Four illustrative paths were selected after the full readout by deterministic
first occurrence, not used as an inferential sample. Their [exact paths](native-hu100-diagnosis-artifacts/failure-examples.json)
and an alternate witness passed **five native fixtures /33 decisions**, zero mismatches.

- Tight, block 2/rotation 1: BB 84o faces a 36-BB preflop pot after raises to
  3/9/27 BB. Its positive-mass key has seven visits, mass 79,844 and 98.17% raise
  probability; it raises to 81 BB, then calls a jam and loses 100 BB. This
  illustrates a learned sparse branch, not a proven action-value error.
- Loose, block 3/rotation 0: BB QTs takes the same 3/9/27-BB prefix, then jams.
  The key has one visit, mass 54,883, 56.19% raise probability; payoff −100 BB.
- Pot_pressure, block 1/rotation 1: its flop raise to 10 BB after a 1-BB bet
  produces an exhaustively unsupported key; uniform fallback folds, payoff −3 BB.
- Pot_pressure, block 728/rotation 1: a missing turn key after off-menu sizes
  has an all-menu witness at **127 BB**, versus the observed **148 BB** pot.
  Both payloads hash to `e007cacc5c1ac12ee298721e58b18151`. Exact amounts can differ
  while history tokens, card descriptor, actor flags and menu names coincide.

Ranked next steps, requiring their own prospective experiment:

1. **Action/history support:** address pot_pressure's demonstrated menu-support
   gap, with explicit action-support/translation semantics and rules parity.
   Additional unchanged-menu training cannot cover its 910 excluded final decisions.
2. **More training, after memory engineering:** test whether higher visits improve
   the covered aggressive branches at fixed support, with matched snapshots and
   held-out comparisons. Sparse late streets make this plausible, not proven;
   #197 stopped at table capacity, not a useful convergence endpoint.
3. **Further investigation:** if covered losses persist with adequate visits,
   distinguish averaging/early-policy carryover, card abstraction and exact
   price/stack information omitted by this key. This sample cannot separate them.
4. **Encoding fixes:** low priority; no demonstrated discrepancy warrants one.

## Verification, resources and restoration

M1 `MacBookPro17,1`, 16 GiB, normal pressure, 76% free and 30.29 GiB disk free
at admission. Derived limits: **4 GiB whole-family RSS, 24.29 GiB disk floor,
original 1,610.50-MiB swap baseline/+0.25 GiB, normal pressure/AC**. One sequential
worker and a single 1,800-second absolute cap; science reserves 240 seconds for
archive readback. Timing-only pilot: 800 candidate hands /3,321 actions; frozen
full-sample quote **976.27 seconds**, with 1,520.68 seconds remaining.
Primary replay took 84.53 seconds; pilot/retrieval, final/native parity,
independent cell/hand arithmetic and **36 focused tests** finished 398.31 seconds
after admission. Archive readback and guard closeout finished at **838.17 seconds**
under the same cap. Across 85 science/examples/seal samples: peak family RSS
0.301 GiB, swap growth zero, ≥27.11 GiB free disk, ≥75% system free, all normal
pressure/AC. These are sampled measurements, not transient peak bounds. No training or policy
change, budget extension, prior-evidence modification, cleanup or merge.

Primary scientific source `16ad96c34a449cb021e004b9231e38573837421c`; independent
source review closed all findings at `a5a1399`, followed only by test regressions.
Supplemental example/export source is preserved with exact hashes in the archive.
Independent final evidence review and archive acceptance are recorded in the
[artifact index](../../RESULTS_INDEX.md). Source review found and corrected
diagnostic denominator/recount/launcher issues before execution; no game fix was
demonstrated. Focused regressions cover those issues and supported off-menu aliases.

Reproduce from a fresh ignored root using the indexed five exports and traces:
`python -m scripts.diagnose_native_hu100 analyse --inputs RESTORED/inputs --out NEW_OUTPUT`,
then run the pinned native binary's `parity NEW_OUTPUT/native-fixtures.jsonl` and
`python -m scripts.diagnose_native_hu100 verify --out NEW_OUTPUT` under a separately
admitted budget. The existing run's one-use launcher is not automatically restarted.
`scripts/report_native_hu100_diagnosis.py` regenerates the report tables.
Whole input ZIP and 205 selected members verify; native Drive hydrated the
initially dataless archive on read. The original retrieval's false remote-download
field refers only to absence of a separate connector download; its explicit
[transport correction](native-hu100-diagnosis-artifacts/retrieval-transport-correction.json)
prevents a no-download claim. All originals and partial/failure evidence are retained.
