# HU20 history compression: Phase 0 M1 result

**Status: the frozen 2M density gate fails. C/D paid training remains blocked.**

Source `da42ac5285edf30a732cad930b970580640be387`; Python 3.11.14 and engine5db20e3. All six from-zero workers completed; no paid jobs or poker-strength outcomes. The original schemas/checkpoints, M4 #136 and New Guy's #143 remain separate.

## Common river decisions

The model-independent corpus has 16,681 decisions from 2,048 deals × two button rotations; 545 are river decisions. The same 545 decisions are probed for each lineage; 1,635 pooled encounters are descriptive repeats across three lineages, not 1,635 independent deals.

| Metric at 2M | Full | Compressed |
| --- | ---: | ---: |
| Median visits | 1 | 4 |
| Missing / zero visits | 34.01% | 16.82% |
| Below 10 visits | 88.50% | 73.64% |
| Below 100 visits | 100.00% | 100.00% |

All three seed medians rise 1→4; every seed improves missing/<10 coverage. However, every observed river key in both tables remains below 100 visits. None of the three seeds meets the complete predeclared gate (≥2× median, ≥5 percentage-point reductions in both low-visit fractions). The candidate has a real measured density benefit but **does not satisfy admission**, and this is not evidence about poker strength.

## Resource results

| Cell | Seed suffix | Completed nodes | Entries | Step nodes/sec | Wall sec | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full | 3001 | 2,000,143 | 193,295 | 16,265 | 130.54 | 338.6 |
| compressed | 3001 | 2,000,099 | 129,939 | 15,256 | 136.41 | 236.5 |
| full | 3002 | 2,000,104 | 190,826 | 16,132 | 130.90 | 335.2 |
| compressed | 3002 | 2,000,539 | 128,634 | 15,529 | 134.07 | 240.0 |
| full | 3003 | 2,000,207 | 191,071 | 15,945 | 132.83 | 338.6 |
| compressed | 3003 | 2,000,319 | 129,275 | 15,347 | 135.75 | 236.6 |

These measurements include the same key/street observer in both workers; wall time also includes common-corpus probing and four checkpoint saves. They are not mature-100M host forecasts. CSV and per-worker JSON retain all 250k/500k/1M/2M prefix counts, overshoots, per-street stored/unique/common-encounter histograms and save/export overhead. No guard failed; no lineage was dropped.

## Correctness and patch-version comparison

22 focused tests pass on the pinned interpreter, including observation-only summaries, preflop key equality, current-street token preservation, public chip/line boundaries, hidden-information/rotation isolation, artifact rejection and fresh-process resume. Earlier fixture/dependency failures are retained in [initial validation](dr2x2-history-artifacts/initial-validation.json). Original full-history 32-iteration work, checkpoint and export bytes match merged main571bcb2.

A separate short comparison on #143's unchanged abstraction/solver/artifact/descriptor files tests Python3.11.14 versus3.11.15: v1-full and v2-full, all three seeds, 100k complete nodes followed by a fresh-process next iteration. All six pairs have identical complete checkpoint/export bytes, payloads, keys/regrets/averages/visits (covered by exact checkpoint rows), deterministic work and derived RNG identities. This is a small cross-patch check, not mature cross-platform proof or a compressed-v2 test. B retains its own frozen3.11.15 runtime; C/D still need exact Linux/M1 parity before main training.

GitHub's Python3.11.16 exposed a unit-test fixture issue: the observer comparison called the campaign worker's strict3.11.14 admission check. Runtime validation is now a separate function; the portable observer unit test substitutes only that check, while explicit worker tests verify rejection of a wrong Python version or engine before training. All25 focused tests pass on3.11.14 and3.11.15. The CLI worker still enforces the original runtime pin. No scientific worker was rerun, and the frozen source/results and compact evidence above remain unchanged.

## Proposed next measurement — not executed

Recommend a **separate prospective 10M-node density continuation** of these same six retained 2M prefixes, preserving the schema, seeds, recipe and common public corpus. Apply the existing numerical median/<10/<100 requirements at the new declared boundary; retain the failed 2M result as a separate result, not a retroactive pass. At this prefix every sampled river decision lies below100 visits, so 2M cannot show a reduction at that cutoff. More unpaid sampling is a narrower next step than changing the schema or launching100M paid jobs.

This revises the initial5M proposal after independently checking the [ad hoc review](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/144#issuecomment-5934446228) against the committed histograms. The maximum encounter visits are94 compressed and79 full. Assuming each existing key's visits scale linearly with completed nodes, the projected compressed-minus-full fraction reaching100 visits is1.65 percentage points at5M,3.43 at7M and6.85 at10M. This assumption ignores strategy drift and is not evidence that10M will pass; it suggests5M is a poor boundary for testing the unchanged5-point requirement.

The uniform-menu corpus also differs from the target policy's reached decisions. Any approved continuation should prospectively add a separately reported, fixed policy-reach diagnostic before inspecting strength outcomes; it must not replace the common corpus or change its gate. Sparse-key associations in retained LBR hands can be useful descriptive evidence, but assigning a whole-hand loss to a visited key does not establish that the key caused it. No #136 analysis or evaluation is added here, and its frozen work retains priority. Neither the density result nor projected key growth predicts a unique causal mechanism or a detectable strength improvement.

This requires a committed amendment before execution and an explicit owner decision. A successful larger preflight would still leave D's exact-descriptor integration/resource prefix, C/D Linux recovery parity, live-priced all-in cost cap and owner budget approval pending. No dollar cap or paid launch has been approved for #144.

## Retrieval and ownership

Large prefix checkpoints, iteration logs, current exports and native-corpus probe records remain on M1 at:

```text
/Users/dberweger/.codex/worktrees/hu20-history-2x2/deepcfr-texas-no-limit-holdem-6-players/results/dr2x2-history-preflight-m1-20261001
```

Use the committed [artifact verification](dr2x2-history-artifacts/phase0-20261001/artifact-verification.json) for exact checkpoint/export paths, sizes and SHA256. Thirty checkpoint/export hashes were independently rechecked after all workers closed. Compact evidence is indexed by [manifest](dr2x2-history-artifacts/phase0-20261001/manifest.json). Preserve original copies until an independently verified archive destination exists.

M4 remains #136-only with its ACTIVE90-minute monitor. New Guy owns #143/B and its independent rental ledger. All future C/D pods, volumes and artifact roots use `dr2x2-`; no C/D rental exists. A/B/C/D strength comparison will require one fresh common schedule, restricted-river range law and all five preregistered factorial contrasts. No promotion or300M extension.
