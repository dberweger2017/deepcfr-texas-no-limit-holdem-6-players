# Card-diversity admission report

## Scope

This report covers admission only for the frozen nested-board protocol in
[`holdem-card-diversity.md`](../holdem-card-diversity.md). It does not contain
learning results: no main reference generation, 1,024-step campaign fit, sealed
test, model comparison, or promotion ran. The only fits executed were the
declared eight-step calibration fits on two training contexts.

## Calibration history

An earlier standalone calibration, before the final verification hardening,
projected 3,036.3 seconds of reference work and 759.7 seconds of fitting with
the 50% allowance. Its temporary output was not retained; it was an admission
estimate, not campaign evidence.

The launch command was then run twice with the unchanged committed plan, source,
seeds, board generator, and 3,600/1,200-second stage caps:

| Run | Source | Reference projection | Fit projection | Result |
| --- | --- | ---: | ---: | --- |
| initial launch | `b86edc7` | 4,598.9 s | 1,341.3 s | blocked at calibration |
| unchanged retry | `b86edc7` | 3,984.8 s | 1,177.1 s | blocked at calibration |

Both attempts stopped before main references and campaign fits. The differing
estimates show local timing variance; they do not justify increasing the frozen
budget or drawing a learning conclusion.

## Provenance and artifacts

The full prelaunch focused suite at `b86edc7` passed 82 tests. The final
post-admission check reran the 18 card-diversity/feature/representation tests;
it is a subset smoke check, not a claim that only 18 checks were run overall.

Both launch reports record source commit `b86edc77427f1543b90e41a6ae915777d07ad2bd`
and source fingerprint
`4b464ab13602470f372678bb973a7db7d3bb1ad02e64919df072ee157f1ee9f4`.
The materialized context hash is
`760cf732bd23d51e00d4d7bf5a3da0f04cb53b2b2137683470b340d55904ef91` for both.

Retained local artifact hashes are:

| Run | `calibration.json` | `report.json` |
| --- | --- | --- |
| initial launch | `e3f3faee09b76ef5deefd85f5ae8a7a615506925da3b4dc3be0d00d9551059e2` | `22e9317fca4af2c41dd5d86f35f7df8f5f4168d298b5020f69e5a6280471426e` |
| unchanged retry | `63d97271254f330ba2c5b04a9bb323392a562bd19d527c2e027e6032f1cd3a7d` | `9efac948b887a8a6e656a45ef23bc4fcd5bccebbf7aed51f3868d42030f91e52` |

## Decision

The admission gate failed and the task stops here. The next task is to revise
the resource plan before rerunning this same frozen scientific comparison. That
revision must be reviewed separately; this report makes no claim about board
diversity, visible-card features, or learning quality.
