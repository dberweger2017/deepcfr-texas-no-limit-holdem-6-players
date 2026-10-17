# Reproduce the confirmation diagnosis

Use the source revision and input pins recorded in the integrity and storage receipts. Work in a fresh ignored `results/` directory, with Python 3.11, NumPy and the pinned engine installed. Matplotlib is needed for the six charts. No model, policy cache or card-abstraction tables are needed.

## Restore the inputs

The [research index](../../../RESULTS_INDEX.md) gives the archive IDs, SHA256s and retrieval commands. This campaign's final bundle also retains the three source ZIPs and the exploratory journals. Place these files before running the preparation helper:

- `results/inputs/v055-blend-confirmation-20261017.zip` (#318).
- `results/inputs/hu200-v055-slumbot-confirmation-20261015.zip` (#304).
- `results/inputs/slumbot-aivat-real-runs-perhand-20261017.zip` (#316).
- `results/inputs/exploratory-prefix.jsonl`: the verified 8,300-hand prefix.
- `results/inputs/exploratory-part2.jsonl`: its 1,700-hand continuation.
- `results/inputs/exploratory/prefix/PREFIX.json`: the prefix preservation receipt.

For the exploratory files, copy the final bundle's `inputs/exploratory/prefix/journal.jsonl` and `inputs/exploratory/part2/journal.jsonl` to the two names above. The helper verifies the indexed ZIP/manifest/member SHA256s, the preserved prefix hash and every journal's complete hash chain. It then writes a run specification and integrity receipt.

```sh
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p results/outputs
python docs/reports/v055-confirmation-diagnostics/prepare_inputs.py
```

## Compute card corrections

The guarded entry point invokes #316's unchanged `dump_aivat.py`, preserving its per-hand output and analyzer summaries. This example runs each account sequentially with at most six workers. Independent accounts can run on separate hosts with identical sources and verified inputs; collect complete outputs with SHA256 verification before pooling. On macOS, run the commands under `caffeinate -i`.

```sh
for letter in a b c; do
  set -- results/inputs/pr318/account-${letter}-*
  AIVAT_DUMP=results/outputs/perhand-confirmation-${letter}.jsonl \
    python docs/reports/v055-confirmation-diagnostics/run_aivat.py "$1" \
    --aivat --workers 6 --hands 10000 \
    --out results/outputs/aivat-confirmation-${letter}.json
done
python docs/reports/v055-confirmation-diagnostics/summarize_aivat.py
```

The pool reuses the saved deterministic corrections and **refits** the cross-fitted weights on all 30,000 hands, in A/B/C order. It does not average account fits or recompute card equities. The same helper matches every saved hand to its journal, reconstructs the earlier card-only estimates from #316's first four columns, and reproduces its seed receipts to numerical precision. Own-action columns are excluded.

## Profile and diagnose public actions

```sh
python docs/reports/v055-confirmation-diagnostics/run_profiles.py
python docs/reports/v055-confirmation-diagnostics/public_tables.py \
  --runs results/outputs/runs.json \
  --integrity results/outputs/input-integrity.json \
  --out-dir results/outputs/public-final
python docs/reports/v055-confirmation-diagnostics/check_public_tables.py
python docs/reports/v055-confirmation-diagnostics/check_profiles.py
python docs/reports/v055-confirmation-diagnostics/write_public_report.py
```

The profile runner calls `scripts.profile_slumbot_runs` on all seven runs and the confirmation pool. The detailed helper reuses a completed journal verification only while the journal's exact bytes still match the integrity receipt. Reconciliation checks cover terminal-fold chips, all BB first responses, flop opportunities for both sides, raw rates and analyzer decision counts. The older analyzer JSON includes abandoned-hand prefixes, which the completed-hand tables consistently exclude.

## Draw the six charts

```sh
python -m scripts.plot_slumbot_aivat --cards-only \
  --run 'Account A=results/outputs/perhand-confirmation-a.jsonl,results/outputs/cards-confirmation-a.json' \
  --run 'Account B=results/outputs/perhand-confirmation-b.jsonl,results/outputs/cards-confirmation-b.json' \
  --run 'Account C=results/outputs/perhand-confirmation-c.jsonl,results/outputs/cards-confirmation-c.json' \
  --run 'Pooled 30,000=results/outputs/perhand-confirmation-pooled.jsonl,results/outputs/cards-confirmation-pooled.json' \
  --out-dir docs/reports/v055-confirmation-diagnostics/charts
python docs/reports/v055-confirmation-diagnostics/write_aivat_report.py
```

`--cards-only` shows raw, plain-card and weighted-card estimates and omits uncomputed own-action series. The pooled data set is a real fourth input, so all six charts include it. Protocol v2 still judges raw BB/100; none of these commands changes its verdict.
