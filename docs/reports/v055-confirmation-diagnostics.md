# v0.5.5 confirmation: card luck and public-action diagnosis

Research in progress. Both analyses must finish before this report is complete.

The inputs are the 30,000 confirmation hands from [#318](https://github.com/dberweger2017/CFR-NLHE-6max/pull/318), the three seed runs from #304, the blend exploratory run and #316’s receipts. Archive SHA256s and every journal hash chain are checked before analysis.

The first analysis runs card-only AIVAT with #316’s per-hand wrapper, reports plain and cross-fitted estimates, intervals, variance reduction, removed luck, street weights and seats, and produces the six charts. The second runs the existing profile tool and derives flop responses by size and position, terminal flop-fold contributions, button opens and big-blind response outcomes from public actions. The same tables cover each earlier run. Single-run patterns remain provisional; observed losses do not establish a fix.

Protocol version 2 uses raw BB/100. Its qualification verdict stands; AIVAT is informational. There is no new play, training, model loading or own-action correction.

## Reproduction and receipts

The [public-action appendix](v055-confirmation-diagnostics/public-tables.md) and [CSV tables](v055-confirmation-diagnostics/tables/) have run on all 70,000 distinct hands (the 30,000-hand confirmation pool is also reported). They reproduce #318’s **−12.925 BB/100** from flop folds and **−32.793 BB/100** in the big blind.

Card-only AIVAT is still running. Its results, six charts, comparison, integrity receipts and final research archive are pending. The corrected launcher passed a two-hand smoke check; the initial launcher failure and traceback are preserved in the research evidence and recorded in the PR.
