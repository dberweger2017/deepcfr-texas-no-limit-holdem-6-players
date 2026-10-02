# Complete turn cost preflight declaration

Declared before the complete-pipeline cost pilot on 2026-10-02. This is not the
main protocol or authorization for a main campaign. Only the three fixed native
HU20 turn fixtures from deal seed 202610010901 are used: limped, min-raised, and
min/min/call three-bet, with a checked flop. Board: 6s 3d Jc 6c. These roots are
excluded from both main sets. The source is current B500M lineage 2026093001,
SHA-256 e0dfb7c0a0ebde1e8a904fce89481919f4e9b32a633b912e88d241d039e69628.

The initial memory/real-export gates passed. The full cost pilot now times exact
equilibrium, four projections, ten target-only BR comparisons, secondary
reweighting and both-policy EV, with all raw outputs retained. Pilot strategy
values do not enter a hypothesis decision or select the main corpus. Only time,
memory and validation determine admission and the outcome-blind corpus size.

Use one nice-10 external process, two Rayon threads, 16-bit compression, a 4 GiB
aggregate RSS budget subject to measured headroom, the initial run swap baseline
781587578 bytes, and stop at >1 GiB swap growth, nonfinite values, or any gate
failure. Solve to 0.2% of root pot, at most 10,000 iterations/900 solve seconds;
full pilot wall ceiling 1,200 seconds per root. No cap is needed for these roots.
The existing server/jobs stay untouched. No paid compute is authorized.

For this secondary-cost fixture, responder empirical weights equal primary
weights, so every secondary value must reproduce its primary value within
3e-5 BB. A separate river-native gate biases only responder weights and checks
both current EV and BR against the independent `profile_quality`; target ranges
and locks stay fixed. The secondary baseline is the reweighted primary
equilibrium strategy's value, not an equilibrium under a different opponent law.
The retained 20,000-deal native V4 sample is reused only for the identical frozen
source, root and ranges; its confidence interval remains fixed.

The main protocol, corpus, decision rule, cost estimate and final binary/source
inventory will be committed before any main-root losses are produced.
