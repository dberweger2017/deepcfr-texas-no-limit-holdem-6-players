# Streamed checkpoint density audit

The [streaming audit](../../scripts/audit_blueprint_density.py) verified the
full SHA-256 of each immutable checkpoint, then read every gzip JSONL row
without loading a policy dictionary. All three scans used clean source
`5c86107` on the M4. The copied [machine-readable results and checksums](blueprint-seat-density-m4/)
match the M4 files. The 5.83M and 12M files were accessible for later play;
the 58.02M file was accessible for this whole-table stream but would exceed
the 10.5-GiB in-memory play limit.

| Checkpoint | Iteration | Entries | Total visits | Mean | Median | p99 | Exactly 1 | At most 2 | At most 5 | At least 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.83M | 8,733 | 5,834,622 | 6,704,960 | 1.149 | 1 | 4 | 93.37% | 97.66% | 99.48% | 3,081 (0.053%) |
| 12M | 18,455 | 12,000,551 | 14,714,005 | 1.226 | 1 | 5 | 90.98% | 96.50% | 99.12% | 10,541 (0.088%) |
| 58.02M | 99,646 | 58,015,659 | 92,086,645 | 1.587 | 1 | 11 | 83.17% | 92.05% | 97.40% | 224,425 (0.387%) |

| Checkpoint | Entries offering fold and check | Near-pure current policy (max probability ≥ 0.99) | p99.9 visits |
| --- | ---: | ---: | ---: |
| 5.83M | 32.85% | 58.65% | 13 |
| 12M | 32.42% | 58.86% | 18 |
| 58.02M | 30.41% | 59.85% | 43 |

The near-pure count describes regret-matched **current** policies, not the
stored average policy. All-nonpositive regrets are treated as a uniform menu
distribution. Near-purity is not a quality diagnosis. These are whole-table
counts; hashed keys do not contain
recoverable street labels. The reached-decision audit will separately measure
which entries actual held-out play queries and how many updates those entries
have. In particular, the 58.02M total of 92,086,645 recorded visits is
measured from rows, not inferred from traversal-node totals.
