# v0.4.1 confirmation for O

The exact #165 opponent-sampled averages meet every predeclared statistical release check against the three v0.4.0 R lineages, including a fresh direct match against shipped R1. No release is published. [Protocol](../hu20-v041-o-confirmation.md), [predeclaration](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/176#issuecomment-6013898242), [frozen counts and hash](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/176#issuecomment-6014180194).

**All four statistical release checks pass.** The full 40-board turn/river diagnostic and its independent audit also pass. No publication or default change.

| Release check | Paired result, BB/100 (95% CI) | Rule | Result |
| --- | --- | --- | --- |
| Direct O vs shipped R1 | +10.50 [+7.90, +13.10] | lower > 0 | PASS |
| Bounded LBR O−R | +37.56 [+27.75, +47.37] | lower > 0 | PASS |
| Native pressure O−R | +6.92 [+0.39, +13.45] | lower > −10 | PASS |
| Other eleven panels | All upper bounds above −20 | no upper < −20 | PASS |

All 592,896 final poker hands replayed with every action and settlement checked; independent statistics match. Fixed arena root 202610062101 and direct root 202610062301. No sample extension.

| Panel | Blocks | O absolute | R absolute | Paired O−R | Gate |
| --- | ---: | --- | --- | --- | --- |
| lbr | 5,632 | -28.17 [-38.42, -17.93] | -65.73 [-74.62, -56.84] | +37.56 [+27.75, +47.37] | PASS |
| loose_aggressive | 256 | +48.34 [+7.51, +89.17] | +26.89 [-7.00, +60.78] | +21.45 [-9.70, +52.60] | PASS |
| loose_passive | 256 | +14.23 [-21.04, +49.49] | +25.20 [-3.70, +54.09] | -10.97 [-33.09, +11.15] | PASS |
| minraise-cap2 | 256 | +82.03 [+24.21, +139.86] | +64.55 [+8.03, +121.07] | +17.48 [-25.49, +60.45] | PASS |
| native-pressure | 16,384 | +130.73 [+123.53, +137.92] | +123.81 [+117.23, +130.38] | +6.92 [+0.39, +13.45] | PASS |
| passive | 256 | +103.65 [+61.74, +145.55] | +120.80 [+81.99, +159.61] | -17.15 [-51.37, +17.06] | PASS |
| pot_pressure | 256 | +32.39 [-12.86, +77.64] | +37.70 [-4.17, +79.56] | -5.31 [-28.71, +18.10] | PASS |
| pressure-cap2 | 256 | +31.12 [-17.24, +79.48] | -1.20 [-45.00, +42.59] | +32.32 [-5.27, +69.91] | PASS |
| selective-stackoff | 256 | +49.25 [+29.39, +69.11] | +36.59 [+18.38, +54.79] | +12.66 [-1.52, +26.85] | PASS |
| tight_aggressive | 256 | +54.20 [+31.31, +77.09] | +56.51 [+34.22, +78.80] | -2.31 [-20.48, +15.86] | PASS |
| tight_passive | 256 | +60.22 [+45.93, +74.51] | +53.03 [+41.46, +64.59] | +7.19 [-1.74, +16.13] | PASS |
| train_pressure | 256 | +29.69 [-7.03, +66.41] | +30.57 [-1.29, +62.42] | -0.88 [-26.60, +24.85] | PASS |
| uniform | 256 | +124.80 [+71.62, +177.99] | +121.71 [+70.91, +172.52] | +3.09 [-45.21, +51.39] | PASS |

| Direct O seed vs shipped R1 | Duplicate blocks | O−R1, BB/100 (95% CI) | Label |
| --- | ---: | --- | --- |
| 2026100601 | 49,152 | +11.12 [+7.85, +14.39] | better |
| 2026100602 | 49,152 | +10.80 [+7.53, +14.07] | better |
| 2026100603 | 49,152 | +9.59 [+6.34, +12.84] | better |

The fresh head-to-head confirms the earlier +11.45 result: the three O policies average +10.50 BB/100 against shipped v0.4.0, with each lineage individually better. The bounded LBR attacker earns **28.17 [17.93, 38.42] BB/100 against O**, versus **65.73 [56.84, 74.62] against R**; lower is better, and this bounded probe supplies a lower bound on exploitability rather than a certificate.

Achieved paired half-widths: native pressure **6.53 ≤8**, LBR **9.81 ≤10**, direct aggregate **2.60 ≤4** and every direct lineage ≤3.28. The eleven small panels remain imprecise; passing the declared severe-regression rule does not establish improvement on each.

O retains the no-floor direct gain seen for T in #175, while avoiding the traverser-average native-pressure drop seen for T/shield in earlier runs. There is no new O-vs-T or O-vs-shield direct comparison. Conditional unpublished release preparation uses the prospectively fixed O seed 2026100601; v0.4.0 remains stable pending owner review and explicit chat approval.

## Design and validation

Exact manifest-pinned O seeds 2026100601/02/03 use 1B-node opponent-sampled averages without the regret floor. R seeds 2026093001/02/03 retain 100M current extraction; R1 is shipped. No training or re-extraction occurred. The #173/#175 direct runner, reporter, auditor and duplicate format are unchanged. The #165 arena play/loader/opponents/panels are unchanged, with report/audit arm enumeration reduced to R and O. All source identity and installed engine receipts are retained. Execution source is `523e4a347d012beb8cb523497d2e9a4cbb90035f`; release integration uses a separate worktree.

Paired nominal Student-t 95% intervals average both seats and three lineages inside each independent deal block; they are conditional on these retained training seeds, without retrospective multiplicity adjustment. Root searches covered tracked configurations/reports and readable retained plans on both Macs; broken historical symlinks are recorded rather than treated as searched. Arena pilot 202610062001 and direct pilot 202610062201 are excluded from all final estimates. Only SDs, costs, memory and completeness were inspected before freezing. No outcome-driven extension occurred.

The pilot contains 48,384 independently replayed hands / 276,624 actions. Final arena contains 297,984 hands / 1,738,648 actions; direct contains 294,912 hands / 1,501,906 actions. Every action, actor, observation, own cards, legal bound, public event, settlement and chip total passes independent replay. Raw chips reproduce all aggregates, lineage/position splits and confidence intervals. All 145,109 bounded-LBR decisions complete, with zero incomplete decisions and zero soft-budget overruns. Complete coverage/LBR street telemetry is in [arena audit](hu20-v041-o-confirmation-artifacts/arena-audit.json); retained zero-mass and missing keys are reported separately.

All play, resource sizing and audits use free M4 compute, no RunPod. Frozen guards are at most three workers, 6 GiB RSS per worker, at least 15 GiB free disk and a three-hour aggregate final cap. The admitted quote is 167.1 minutes including 25% headroom. The earlier completed poker audit uses spare capacity while one lock-only worker runs; this changes reporting order only, and the final audit rehashes every previously verified raw trace before reusing its proof.

## Historical comparison

#165 O−R native pressure +0.44 [−18.47, 19.34] failed only the width-based safeguard, without measuring a regression. #175 then measured O−R1 +11.45 [8.83, 14.07]. [Claude’s accepted #175 analysis](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/175#issuecomment-6013778852) separates the floor-related direct loss from traverser-reach averaging’s native-pressure drop: earlier T ≈104 and shield ≈103 versus O ≈128 and R ≈128 BB/100. Proximity across different-root runs is descriptive, not an equivalence test. #175’s direct matched-seed shield−T −14.26 [−16.90, −11.62] isolates the floor recipe, while T−R1 +8.12 [5.50, 10.73] shows the common budget/averaging changes alone are insufficient to explain shield’s loss. No new O−T or O−shield direct claim follows.

## Progress: turn/river distance from equilibrium

O seed 2026100601 and shipped R1 use the identical #149 lock-only pipeline used for shield: all 40 fixed boards, both seats, the same original B500M ranges, compact trees and approximate reference. There is no new solve. E is conditional best-response gain above that retained reference, in BB per spot; Q=(E−P)/(B−P) places E against common B/P anchors. Lower E/Q is better; Q is not bounded to [0,1]. The reference has residual error (the original 0.2%-pot stopping criterion), and these measurements do not establish full-game Nash distance.

| Policy / anchor | E, BB per conditional spot [95% bootstrap interval] | Q [95% bootstrap interval] |
| --- | --- | --- |
| O · seed 2026100601 | 1.0665 [1.0023, 1.1315] | 0.5226 [0.4807, 0.5768] |
| Shipped R1 | 2.7147 [2.4922, 2.9323] | 2.6792 [2.3247, 3.0578] |
| Common B anchor | 1.4313 [1.3212, 1.5364] | 1 by definition |
| Common P anchor | 0.6670 [0.6390, 0.6973] | 0 by definition |

Paired O−R1 E is **-1.6483 [-1.8563, -1.4479] BB per spot**. The independently audited 2,000 paired board-bootstrap draws use seed 202610050002 and the identical percentile indices 49/1949. Every one of 160 raw metrics and all bootstrap arithmetic verify. The historical R1 E/Q and common B/P anchors reproduce exactly; legacy metric key `cfrplus` labels O in this run.

For context, shield's earlier identical-board E is 0.9160 [0.8628, 0.9704], Q 0.3258 [0.2871, 0.3626]. Shield has lower point E/Q on this conditional diagnostic, while its earlier direct loss and native-pressure regression remain. These adjacent policy values are descriptive; no new O−shield direct match or full-game exploitability comparison is claimed. T has no comparable full-export 40-board measurement in this task.

Final evaluation and all independent scientific audits finish **113.35 minutes** after the frozen final start, below the three-hour cap. Largest retained command/native job RSS observation is **4.66 GiB**, below 6 GiB. Resource guards remain active throughout; no scientific play, action, settlement, native metric or audit failure occurs. Auxiliary harness/path/CI-invocation mistakes and their corrections are retained separately. The two integration hands are additional engineering checks, excluded from every playing-strength estimate.

## Unpublished candidate and archive

All four gates pass, so the prospectively fixed first-seed O export is prepared with a verifier, model card, release notes, manifest and SHA256SUMS. [Readiness](../releases/v0.4.1/READINESS.md) records 34 focused checks, green full CI on candidate runtime source, two real-model HTTP hands with replay/restart, and the publication hold. The release package contains inference only, no engine binary or private journal. `--o-candidate` is opt-in. The owner must review these numbers and approve publication in chat first; no tag, GitHub release, Latest change or stable-model replacement occurs.

The complete research ZIP is retained under `~/Local/Research-Cloud/PR-176-HU20-v041-O-confirmation/`, with an embedded member size/SHA256 manifest and full ZIP-member readback. [Research index](../../RESULTS_INDEX.md) gives retrieval and archive hashes. Exact source, six policy exports, manifests, pilot/final plans, every raw action/settlement, lock-only requests/compact/reference/response files, native binary, resource logs, auxiliary failures, reports/audits, candidate preparation and restoration instructions are included. Every M4 snapshot member and independently audited raw trace is hash-verified after retrieval. Originals remain; nothing in the synced folder is deleted or evicted. Cloud upload acceptance is recorded separately from local byte verification.

Verified main ZIP: **792 members / 8,038,176,698 logical bytes**, ZIP **3,389,594,093 bytes**. SHA256 `9f473aa22b16037549716e8fd938b4b47670ca89e29b398bf241b1448fe597a3`; embedded manifest SHA256 `6cd6cca672569b42c062837de823043c08d7fd8dfc1d3fb4587fe5628ec875fe`. Every member passes readback. Final documentation/PR/CI/upload receipts are retained in a separate member-hashed closeout ZIP in the same folder; original files remain.
