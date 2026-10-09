# Trained global K50 bench: prospective protocol

Authorized October 9, 2026 on the free Apple M4 (10 cores, 16 GiB). This is
abstraction step 3, independent of the HU100 ladder. No trainer source changes.

Use #162's same 40 #149 turn roots, two frozen halves and lineage B500M
seed 2026093001 ranges. Native `bench-train`, seed 202610050001, unchanged
linear CFR, ordered histories and action menus. One v1 and one pinned global
K50 training trajectory per fold. Both averages and current are exported;
only opponent-sampled averages decide. Uniform fallback on missing/zero mass.

Checkpoints are 1M, 3M, 10M plus a separately frozen equity matched iteration
per fold. Before any scoring, measure the key ratio and mean traverser visits
at 3M. Freeze `ceil(3M * v1 mean visits/key / equity mean visits/key)`;
this accounts for any different traverser work as well as the key ratio.
Report actual key counts, total/mean visits, quantiles and visit bands at every
checkpoint, including any residual matching error. No outcome-based adjustment.
The 3M timing runs are deterministic prefixes; final runs must reproduce their
1M/3M files byte for byte, with one final trajectory per variant/fold.

Score all exports in one #149 lock-only pass per held-out root. Use #190's
pinned K50 labels and card-removal sentinel, with the trained equity schema
and scalar bucket payload when hashing the unchanged history templates.
Transport group metric `equity-k50` through native slot `eq50-fit0`; this is
an alias only. Never project an equity policy through v1 or the fitted witness
keys. Retain all 40 boards, both seats, B/L/P references and native outputs.

Before any scoring, primary rule: equity opponent-sampled E at matched visits
minus v1 E at 3M passes if the paired 95% bootstrap upper bound is below
−0.10 BB and equity E at 10M is below v1 E at 10M (point estimate).
Fail if the matched contrast's lower bound is above −0.10 BB. Otherwise
inconclusive. A clear matched gain with a nonnegative 10M contrast is also
inconclusive. No extra samples or favorable checkpoint selection.

E is BB per spot; Q=(E−P)/(B−P). Freeze 2,000 paired board bootstrap draws,
Python Random seed 202610050002, sorted spot IDs, percentile indices 49/1949.
Traverser-reach, current, learning curves and the gap to #190's aggregate
K50 witness E=0.3917 are descriptive. That witness includes three lineages;
it is not a new trained baseline or an additional decision opportunity.

Measure training speed with the 3M timing prefixes and scoring cost with one
full-policy root. Pilot outcomes do not choose the budget or samples. Post
the pilot quote on the PR before final scoring. One independent source review
before final runs; one evidence review at closeout. A then B; if combined
quotes exceed about ten hours, complete A, run B training/direct primary and
omit B scripted secondary. Quote storage as originals plus ZIPs and 16 GiB floor.

Whole-family soft/hard ceilings 7/9 GiB, normal pressure, at least 15% system
free, **3,000,000,000 bytes total system swap**, AC and >=16 GiB disk.
Any exactness mismatch, information leak, invalid action, accounting error
or guard breach stops this stage without scientific retry. Trainer defects
stop this stage and are reported on its PR; `native/hu20-trainer/` is immutable.
Models/tables/raw files remain ignored. Hash and manifest research ZIPs in
`~/Local/Research-Cloud/PR-<n>-hu20-equity-bench/`, index restoration, preserve
failures and existing hash-verified inputs without duplicate model copies.
No release, publication, tag or cleanup of other PR evidence.
