# HU20 exact turn check: owner-stopped campaign

The owner stopped this work to prioritize Ollama on the M4. **2 of 288 frozen
spot-policy jobs completed; zero roots have all six exports.** No pooled values,
bootstrap intervals, R decision, H0 decision or training recommendation are
admitted. The diagnostic and its new monitoring sidecar exited. Existing
TensorBoard servers and Ollama are left untouched. No automatic restart,
training, promotion, rental or merge is scheduled.

The separate history-alias audit is completed in draft
[PR #146](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/146).
This turn extension amends draft
[PR #145](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/145).
Turn/river values cannot answer the original flop question. Within-root alias
cost also cannot measure aliases inherited across different preflop/flop roots.

## Frozen design and qualification

The [protocol](../hu20-exact-turn-check-protocol.md),
[config](../../configs/diagnostics/hu20-exact-turn-check.json), source inventory
and sampled public roots were pushed in commit
`31030f4270b92f6780937edd90282e058b9a05b4` before main-root values. Set A selects
16 stratified weighted roots from 67 LBR-selected turn roots behind 74 observed
decisions; the selected sample contains 17 decisions. Set B selects 32 roots
from 1,088 fresh live turn roots in 3,000 current-policy self-play deals, before
sampling turn actions. All six B500M current/stored-average exports were frozen.
The conservative node-scaled full-pipeline cost estimate is 21.06 hours with
twofold contingency; cumulative main ceiling is 24 hours. No selection was
changed after values. All 48 roots are excluded from pooled inference because
their common six-export intersection is empty; 46 roots never started.

Qualification passed before admission:

- Gate K: 100,000 real information-key samples, zero mismatches.
- V1: all native line/action amounts match in three real turn fixtures and both
  completed main jobs.
- V2: 300 native terminal samples per real fixture, zero integer payout
  mismatches; floating representation error ≤0.000101 chips.
- V3: current-binary river BR values agree with `profile_quality` within
  6e-7 BB. Secondary-range river current/BR values agree within 7.53e-7 BB.
- V4: three real B500M exports locked end to end; exact native-policy EV lies
  inside each independent 20,000-deal native 95% interval. The current solver
  binary reproduces each qualified lock EV.
- Independent full-native compact fixture: ten projection/BR comparisons across
  412 aliased public nodes agree within 1.53e-7 BB. Artificial four-holding
  fixture ranges are validation evidence, not playing-strength results.
- Secondary identity: all ten reweighted comparisons per cost fixture agree
  with their primary values within 1.53e-7 BB.

| Real fixture | Exact BP EV, seat 0 BB | Native MC BB [95%] | Equilibrium residual, % pot | Full pipeline seconds | Plain / compressed GiB | Peak owned RSS GiB |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Limped | +0.211196 | +0.1751 [0.130884, 0.219316] | 0.186956 | 360.58 | 2.100 / 1.063 | 2.600 |
| Min-raised | +1.525193 | +1.5359 [1.433996, 1.637804] | 0.160866 | 128.78 | 0.936 / 0.477 | 1.280 |
| 3-bet | −0.748324 | −0.6987 [−0.822558, −0.574842] | 0.172093 | 89.52 | 0.650 / 0.331 | 0.938 |

All used native compressed menus, with no removed lines or raise-cap fallback.
The three cost fixtures were declared before their pilot values and excluded
from the main population. Every completed equilibrium reaches ≤0.2% of pot.
No admitted native turn root was oversize. The original flop resource failure
remains in the [flop report](hu20-exact-flop-check.md).

## Stopped main execution

Admission used 4 GiB owned RSS, one external solver, two Rayon threads and nice
priority 10. The first job completed in 44.58 solver seconds, peak owned RSS
0.557 GiB; the second in 17.45 seconds, peak 0.327 GiB. Both V1 and V5 pass.
Neither root completed the six required policy profiles. At 80.70 main elapsed
seconds the between-job guard refused a third job:
`MemoryError('Measured job headroom fell below budget')`.

A newly loaded Ollama model process was observed during this interval. The
subsequent system snapshot showed swap at 2,314.88 MiB, versus the frozen
745.38-MiB baseline, exceeding the declared 1-GiB growth limit. The retained
later snapshot shows 2,082.88 MiB swap and 1.579 GiB reclaimable memory. The
diagnostic did not stop Ollama or alter other jobs. The owner then explicitly
requested cleanup and a stop for now. The first guard failure was memory
admission; the later swap observation independently prevents re-admission.

The following are retained **individual incomplete-profile evidence**, averaged
over the two target seats. They are not pooled campaign estimates or a basis
for choosing an intervention. Full per-seat BB/%-pot values, both-blueprint EV,
secondary support and turn-fold/key/decile tables remain in
[main-results.jsonl.gz](hu20-exact-turn-check-artifacts/main-results.jsonl.gz).

| Set / export | Root pot BB | BP loss BB | Full v1 BB | Per-line v1 BB | Equity 50 BB | Equity 200 BB | Signed alias cost | Residual % pot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A / seed 2026093001 current | 8 | 4.625971 | 0.778463 | 0.778463 | 0.362597 | 0.221207 | 0 | 0.134033 |
| B / seed 2026093002 current | 14 | 4.534742 | 0.437007 | 0.437007 | 0.326375 | 0.228936 | 0 | 0.190257 |

Both roots are 3-bet pots. Zero within-root alias difference on these two roots
cannot clear the global history alias issue. The frozen descriptive LBR
accounting would use `(134/768) × mean e_bp / 0.65`; its required mean is
unavailable. Likewise, equity headroom and fold-frequency decision summaries
are unavailable. No hypothesis is classified.

## Retained failures and resource attempts

All unsuccessful qualification attempts remain in the local archive:

| Attempt | Failure / disposition |
| --- | --- |
| compact-validation-alias-01 | JSON action-field ordering differed in the Python comparison; canonical action identity fixed and regression-tested. |
| compact-validation-alias-02 | A 1-GiB fixture watchdog limit was exceeded; retained. |
| compact-validation-alias-03 | Wrong input-plan path; no solver allocation. |
| compact-validation-alias-04 | Simultaneous Python profile and locked solver exceeded 4-GiB aggregate RSS. Separate workers release the profile before solving. |
| compact-validation-alias-05 | Synthetic smaller-stack fixture failed table-roster validation; no solve. |
| compact-validation-alias-06 | Compilation correctly refused a non-HU20 initial stack; no solve. That synthetic path was removed. |
| compact-validation-alias-07 | Full native HU20 fixture passes after worker separation, peak 2.815 GiB. |
| cost-preflight-limp-01 | Typo in worker executable path; exit 127, no allocation. Corrected limp-02 passes. |
| main-01 | Two completed atomic jobs, then memory admission refused the third. Subsequent swap exceeds baseline +1 GiB; owner stops and requests cleanup. |

No failed attempt disappears from the record. No capped loss values were
admitted without removed-reach auditing. The native→3→2 cap ladder remains a
declared option requiring that audit; it was not needed in admitted fixtures.

## Provenance, cleanup and limitations

The AGPL solver and custom Rust integration stayed outside the MIT repository,
upstream commit `9d1509fe5077d019825f833eed04b16d342dfda1`, binary SHA-256
`cdc46b10d985d64747982ed1e3d40a1697d533cfbc444c0d16ed284cc4148952`.
[inventory-main.json](hu20-exact-turn-check-artifacts/inventory-main.json)
records verified source files, all six policies, all three stored hand archives,
the native engine and Python/toolchain provenance. Main request/response/file
hashes are in
[main-file-inventory.json](hu20-exact-turn-check-artifacts/main-file-inventory.json).
Atomic main records, admission, failure, process ownership and post-stop machine
state are committed beside them. The machine-readable report contains all
frozen-root exclusions, with zero eligible roots and no ratio intervals.

The M4 staging workspace, copied inputs, external tool/build files and new
turn TensorBoard run are archived to
`/Users/dberweger/Local/hu20-m4-archive-20261002` on M1 before cleanup. Per-file
SHA-256 verification covers 3,103 records / 3,152,458,858 bytes with zero
mismatches. [Cleanup evidence](hu20-exact-turn-check-artifacts/m4-cleanup.json)
records the four deleted temporary roots; shared source/venv, both existing TensorBoard
servers, original unrelated runs and Ollama are preserved. No automatic resume
is scheduled. The owner can decide later whether to continue under a new
machine admission; the stopped clock and original evidence remain retained.

Ranges condition on preflop and flop policy likelihoods, so earlier errors are
excluded. Projections are feasible strategies rather than abstraction
equilibria; high projection loss is an upper bound on minimum abstraction loss,
not proof of H1. Per-turn buckets are more favourable than global blueprint
buckets. Signed alias differences are not a causal loss decomposition.
Secondary values reweight the primary equilibrium strategy instead of solving
a new-range equilibrium, with unsupported empirical mass disclosed. Incomplete
six-export coverage prevents every frozen campaign decision. Turn/river results
do not answer the flop question.

Validation at the frozen admission commit: 107 local diagnostic tests and full
[Linux CI](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/36979892958)
pass. The stopped-run reporter additionally counts every unstarted frozen root;
its regression brings the local diagnostic suite to 108 passing tests. The
report-source inventory distinguishes that reporting correction from the frozen
solver source. Draft PRs only.

## Owner-requested TensorBoard shutdown

After the initial verified workspace cleanup, the owner also requested stopping
TensorBoard and unnecessary background work to prioritize Ollama. Both servers
(ports 6006 and 16008) and an orphaned thermal-monitor shell/pmset/tail family
were terminated; their exit and closed listener ports were verified. No poker
training, solver or monitor process remains. Ollama and remote-access/system
services are preserved. TensorBoard logs stay in their original directories;
[cleanup/restart records](hu20-exact-turn-check-artifacts/tensorboard-cleanup.json)
also remain in the M1 archive. There is no automatic restart. Earlier cleanup
records showing the servers alive describe the preceding cleanup stage.
