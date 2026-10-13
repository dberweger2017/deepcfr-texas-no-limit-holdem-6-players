# HU100 pot-pressure diagnosis

v0.5.2 loses to 1.5× pot pressure by **-26.37 [-41.78, -10.96] BB/100**. The 1× size control gains **+72.36 [+58.58, +86.13]** relative to baseline. A check-else-call intervention at unresolved keys also improves return; the folding intervention and descriptive K50−v1 contrast are inconclusive. No release or policy default changes.

## Fixed experiment

Owner-authorized Stage A on the M4 follows [Stage C](hu200-k50-first.md) and precedes roots-per-seat Stage B. Six exposures implement the five requested arms. Each has 16,384 seat-swapped duplicate blocks /32,768 hands. All arms share the exact deals, opponent seeds and candidate sampling streams from #236’s scripted harness. Blinds are 50/100 chips, stacks 100 BB, no rake, reset each hand.

The baseline style is `Style(0.40, 0.60, 0.95, 0.20, 3)` (1.5× pot); the size control changes only its numerator to 2 (1× pot). Baseline, size, passive and folding use the downloaded v0.5.2 average with released translation (512 states /128 events). Descriptive K50 and v0.5.1 use their 2B averages, both with translation off. “Missing” means unresolved after translation; successful translation is a separate category. Overrides consume the original sampler draw and alter only missing-key choices.

Bundled verifiers passed `--require-publication` for v0.5.2 source `3a3bc4218bab96637d6a27a69b73383ee8bb196c` and v0.5.1 source `dca7547e6eca82ce77f3e42230b82fabc1c29a4c`. PR246’s currently merged status was checked before read-only restoration from its averages archive; its root/uploads were never changed. The actual small-archive member is `averages/2000000000.average.gz`. All full model and K50 table SHA256s passed, retained in [inputs](hu100-pot-pressure-artifacts/inputs.json).

Fresh roots `20261013100101` (timing pilot) and `20261013100102` (final) were checked against current main and every other current open PR head’s tracked configs, reports, plans and scripts before outcomes. The 128-block pilot replayed and reproduced all arms without payoff inspection. It projected 1,153.56 seconds for the requested final count, so no reduction was made. The exact [canonical plan](hu100-pot-pressure-artifacts/canonical-plan.json) was [posted and read back on PR251](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/251#issuecomment-6136765568) before admitted final hands. Scientific source stayed fixed at `ff2e0521098a8b4a32b29da903c02dd7c84e077b`.

All **196,608 final hands** passed independent rules/observation/menu/key/action/opponent/settlement replay and deterministic reproduction before readout. There were no failed-hand reruns or outcome-driven extensions. Nominal Student-t 95% intervals use paired duplicate blocks; they condition on these fixed models and one training seed, with no multiple-comparison adjustment.

| Exposure | BB/100 [nominal 95%] |
|---|---:|
| 1. Baseline v0.5.2, 1.5× pot | -26.37 [-41.78, -10.96] |
| 2. Size control v0.5.2, 1× pot | +45.99 [+34.77, +57.21] |
| 3. Missing: check, else call | -8.42 [-25.17, +8.33] |
| 4. Missing: check, else fold | -15.26 [-24.07, -6.45] |
| 5a. K50 2B, translation off | -112.06 [-131.28, -92.83] |
| 5b. v1 2B /v0.5.1, translation off | -110.77 [-129.36, -92.17] |

| Paired contrast | Gain BB/100 [nominal 95%] |
|---|---:|
| folding-baseline | +11.11 [-2.74, +24.95] |
| k50-v1 | -1.29 [-18.24, +15.66] |
| passive-baseline | +17.95 [+4.58, +31.32] |
| size1x-baseline | +72.36 [+58.58, +86.13] |

## Baseline decomposition

These disjoint classes sum to the total −26.37 BB/100. Contribution divides class net chips by **all** hands; conditional return divides by hands in the class. With a 100-chip BB, average net chips per hand equals BB/100. Class membership follows the realized path and is descriptive.

| Hand class | Hands | Share | Contribution BB/100 [95%] | Conditional BB/100 |
|---|---:|---:|---:|---:|
| No missing or translated decisions | 27,449 | 83.77% | +44.74 [+41.88, +47.60] | +53.41 |
| At least one unresolved missing key | 1,310 | 4.00% | -77.56 [-91.78, -63.33] | -1940.00 |
| Translated, none missing | 4,009 | 12.23% | +6.45 [+0.16, +12.73] | +52.68 |

All 27,449 hands in the first class used positive-mass exact decisions exclusively. Three known zero-mass decisions occurred in other classes; they were not changed by the diagnostic wrappers.

| Decision class | Check | Call | Fold | Raise | Total |
|---|---:|---:|---:|---:|---:|
| known_zero_mass | 0 (0.00%) | 1 (33.33%) | 2 (66.67%) | 0 (0.00%) | 3 |
| missing | 29 (1.73%) | 479 (28.61%) | 520 (31.06%) | 646 (38.59%) | 1,674 |
| positive_mass_exact | 6,602 (22.07%) | 6,756 (22.58%) | 4,176 (13.96%) | 12,385 (41.40%) | 29,919 |
| translated | 1,512 (17.73%) | 2,846 (33.38%) | 2,260 (26.51%) | 1,908 (22.38%) | 8,526 |

Street accounting below is over all 32,768 baseline hands. “Terminal” is the last decision street. Gross losses are absolute net losses on losing hands; commitments count chips invested on each street, including posted blinds preflop. Commitments do not identify the street that caused a loss.

| Street | Net by terminal street, chips /BB/100 | Gross lost by terminal street, chips /BB/100 | Committed chips | Committed in losing hands |
|---|---:|---:|---:|---:|
| preflop | +387,100 /+11.81 | 1,598,750 /48.79 | 7,770,950 | 2,780,550 |
| flop | -20,600 /-0.63 | 1,200,700 /36.64 | 2,962,700 | 1,091,700 |
| turn | -355,100 /-10.84 | 1,532,500 /46.77 | 3,350,700 | 1,520,600 |
| river | -875,500 /-26.72 | 2,497,400 /76.21 | 2,981,500 | 1,436,500 |

## What the data supports

**Bet size.** Changing only the scripted raise size to 1× pot turns the baseline loss into +45.99 [34.77, 57.21] BB/100, a paired +72.36 [58.58, 86.13]. The larger size is a substantial cause of this opponent-specific weakness. This intervention changes later public histories, coverage and pot sizes as well as the immediate price, so it does not isolate pure price sensitivity from abstraction/translation effects.

**Missing keys.** Only 4.00% of hands encounter an unresolved key, but their −77.56 contribution outweighs +51.19 from the other two classes. Missing decisions raise 38.59%, call 28.61%, fold 31.06% and check 1.73%. Replacing these choices by check-else-call improves return +17.95 [4.58, 31.32]; check-else-fold gives +11.11 [−2.74, 24.95]. This supports harmful unresolved fallback as one contributor, with evidence for the passive intervention; it does not establish a production remedy. Neither intervention separates all coverage/history effects or proves that all missing hands can be repaired.

**Stored decisions.** Positive-mass exact-only hands contribute +44.74 [41.88, 47.60] and translated-only hands +6.45 [0.16, 12.73]. The observed loss therefore does not come from the aggregate of covered-only hands in this schedule. These are selected paths: stored actions may lead into missing histories, and this split cannot certify the quality of stored play against the 1.5× size. Translation-off K50 and v1 both lose about 111–112 BB/100, with K50−v1 −1.29 [−18.24, 15.66]; the K50 direct-match advantage in #246 does not show a pot-pressure improvement here.

## Safety and preserved setup records

Before archival: 448 five-second readings, peak family RSS 3.315 GiB, minimum free memory 86%, maximum swap growth 0.000 GB from A’s baseline, minimum free disk 27.61 GiB, pressure levels [1]. No guard line was crossed. Final play/replay/reproduction took 1,111.61 seconds and peaked at 2.887 GiB. Every reading and operation receipt is preserved; final closeout resource summary includes archive work.

Two setup/admission failures are retained. Initial input preparation used a full-archive member namespace for PR246’s small ZIP; exact small-member mapping was corrected after checking its manifest. The subsequent comment-post command lacked `gh` in PATH; a prematurely launched final command was rejected by the missing publication receipt before any hand. The unchanged plan was then posted and exact readback verified before the admitted final run. Both failures played **zero hands**, had no resource breach and were recorded as setup/admission failures rather than sustained resource stops. No failed scientific hand was retried.

After C was merged and its unchanged six local checkpoint/current/gate originals were matched against current accepted archival provenance and dependency checks, 8,983,554,470 bytes were reclaimed. The [cleanup receipt](hu100-pot-pressure-artifacts/c-cleanup-receipt.json) records each restoration locator; C’s average/tables and PR246’s entire root/uploads remain intact.

## Evidence

The full evidence ZIP includes releases, the restored K50 average, all final/pilot traces and reproductions, audits, frozen plan/publication proof, failures and all resource readings. PR163’s tables remain an explicitly indexed dependency. Archive/member verification and native plus independent cloud acceptance receipts are recorded in [RESULTS_INDEX](../../RESULTS_INDEX.md). Compact results and checks are [here](hu100-pot-pressure-artifacts/summary.json). No binaries or raw hand traces enter Git.
