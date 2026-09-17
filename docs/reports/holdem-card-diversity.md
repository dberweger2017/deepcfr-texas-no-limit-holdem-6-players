# Nested-board card-diversity results

## Outcome

The frozen diagnostic completed locally after the runtime-only amendment in
[`holdem-card-diversity-runtime-amendment.md`](holdem-card-diversity-runtime-amendment.md).
It provides finite regression evidence, not a full-game strength result. No
candidate is promoted and production defaults remain unchanged.

The 768 contexts comprise 48 training boards, 8 validation boards, and 8 test
boards, with six hero holdings and two betting situations per board. The 16 and
48 training arms received the same 1,024 updates and 32,768 sampled examples;
their effective exposures were 170.67 and 56.89 context passes respectively.
All 24 fits and all three predeclared test arms completed.

## Validation

Validation decision cost is the mean BB loss under the fixed reference values.
The primary comparisons were fixed before fitting: scaled48 versus scaled16 for
board diversity, and features48 versus scaled48 for the visible-card control.

| Training arm | Model | Mean validation cost (BB) | Mean relative RMSE |
| --- | --- | ---: | ---: |
| 16 boards | original | 0.8052 | 0.6859 |
| 16 boards | scaled | 0.8096 | 0.6730 |
| 16 boards | cards | 0.3741 | 0.6380 |
| 16 boards | features | 0.2921 | 0.5492 |
| 48 boards | original | 0.5274 | 0.6594 |
| 48 boards | scaled | 0.6813 | 0.6752 |
| 48 boards | cards | 0.3491 | 0.5270 |
| 48 boards | features | 0.3639 | 0.5189 |

The board-diversity comparison passed zero of three seed checks. The
visible-card control passed two of three validation seed checks; seed 911 did
not satisfy the complete declared rule. The lower aggregate costs are
descriptive and do not override the per-seed gate.

## Sealed test

The three predeclared arms were evaluated once regardless of validation:

| Arm | Mean test cost (BB) | Mean relative RMSE |
| --- | ---: | ---: |
| scaled16 | 0.5359 | 0.7000 |
| scaled48 | 0.3967 | 0.7629 |
| features48 | 0.2168 | 0.6363 |

The board-diversity test comparison failed all three seed checks. The
visible-card test comparison passed all three test seed checks, but its overall
comparison is **not confirmed** because validation passed only two of three
seeds. The report stores validation and test checks separately and marks both
primary comparisons unconfirmed.

## Runtime, provenance, and verification

The run took 2,308.60 seconds (38.48 minutes): 1,790.11 seconds in exact
reference enumeration over 10,556,040 nodes and 427.15 seconds in fitting.
Calibration passed at 3,197.25 seconds projected reference work and 838.04
seconds projected fitting work with the 1.5 allowance. No rental or adaptive
sweep was used. Every fit used the declared clipping rule; all 24 fits recorded
1,024 clipped steps under this fixed recipe.

Source commit: `eb67b2d86906ffcc5c9a7a622fd0f39bf01a5f33`.
Source fingerprint: `4b464ab13602470f372678bb973a7db7d3bb1ad02e64919df072ee157f1ee9f4`.
Plan hash: `9967134f877e09856690f544c23a314f2fe1df8287e21098ce212a87d1f94b73`.
Materialized-plan hash: `23a38532c5c72fc16725dbb13f77ac27b10e45701d9d4396eb7636828b009c68`.
Context hash: `97b1af5da4ea54374f57cc6048ce8b930e7e892b4d2137683470b340d55904ef91`.

The fresh-process verifier reloaded all 24 fit weights and 9 test evaluations,
reproduced stored metrics, checked all 28 artifact hashes, rejected test data
in fitting curves, and recomputed validation/test comparisons. The external
source archive is `logs/card-diversity-campaign-source-eb67b2d.tar.gz` with
SHA-256 `fe6de11e91878e5336408c480fdadb8770934f3b013bba4d576d52e7d2f098ea`.

## Decision and limits

The board-coverage hypothesis did not pass its declared per-seed criterion.
The visible-card control shows a promising but unconfirmed test result because
one validation seed failed. These results do not establish a preferred
representation, full-game learning, or professional playing strength. Keep
production defaults unchanged and retain the complete local artifacts for
review.
