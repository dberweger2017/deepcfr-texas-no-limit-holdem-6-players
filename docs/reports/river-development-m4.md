# River CFR development curves on M4

The frozen [development plan](../river-development.md) ran at source
`9ead777d77fbc76e26f898f08703732b9197f797` with the unchanged 12M
checkpoint. All 12 small heads-up cases and five full-range roots completed.
The [raw rows, manifest, profiles, result and checksums](river-development-m4/)
are retained and verified after transfer from the M4. The run took 525.66
seconds, below its 900-second limit; peak process RSS was 6.59 GiB, below
10.5 GiB. System swap used 769.38 MiB before and after the run.

## Small-game average-strategy curve

| Complete sweeps | Median exploitability / root pot | Worst case | Cases at or below 0.1% |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.1491% | 0.4251% | 5/12 |
| 16,384 | 0.0953% | 0.3543% | 6/12 |
| 32,768 | 0.0700% | 0.2260% | 9/12 |

The three cases still above the proposed `1e-3` root-pot threshold at
32,768 sweeps are `hu-paired-kicker` (0.1960%), `hu-low-draw` (0.1792%)
and `hu-mixed` (0.2260%). The median final-played strategy at that point
is much worse, at 7.55% of the pot. All 36 new average-profile snapshots
agree with the independent enumerated values and best-response gains within
`4.44e-16` BB. This development curve supports average extraction, but the
proposed all-case numerical target remains unmet.

## Full-range decision-time curve

All five roots have 1,081 board-compatible holdings per active seat. The
table reports own-reach-weighted average-profile exploitability in the
restricted declared game, as a percentage of the fixed root pot. Setup is
included in each decision clock.

| Root | Pot / remaining stack (BB) | Declared range shape | 5 s | 15 s | 30 s | 60 s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Dry check | 2 / 9 | Blueprint | 2.733% | 1.499% | 0.844% | 0.476% |
| Paired flop bet | 4 / 3 | Uniform | 1.336% | 0.513% | 0.270% | 0.164% |
| Four-flush turn bet | 8 / 6 | Blueprint | 0.984% | 0.433% | 0.254% | 0.148% |
| Low-draw double bet | 12 / 4 | Squared blueprint | 0.380% | 0.126% | 0.083% | 0.043% |
| Deep board straight | 32 / 84 | Blueprint | 0.0046% | 0.00041% | 0.00010% | 0.00002% |

The public trees have 29–53 nodes. At 60 seconds, 1,370–2,546 full sweeps
completed, except the deep board-straight case at 1,814. The measured
segment wall cost is about 0.023–0.046 seconds per completed sweep; it
depends on the public tree. Final-played strategies are retained in the
rows and remain much more exploitable in the nontrivial cases.

**Range limitation discovered after the run:** the effective support was
1,081 of 1,081 holdings for both seats in four of five roots. Their
blueprint-based action likelihoods were effectively uniform at these
histories; squaring one such marginal did not create a different range.
The dry-check root had effective supports 345 and 206. Thus this first
run varied boards, pots and stacks, but supplied little actual range-shape
variation. A separately frozen, explicitly labeled development amendment
will exercise nonuniform stipulated ranges and a nontrivial deep board.
These findings do not change or overwrite the completed run.

These exact best-response values are conditional on the stated board,
public ranges and restricted action menu. No corrected-rollout comparison,
whole-hand BB/100 result, or player promotion follows from them.
