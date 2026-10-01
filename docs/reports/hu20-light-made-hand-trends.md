# Completed 100M–500M light-panel made-hand trends

I analyzed every completed selective-stackoff light task from #136: three original
lineages × 25 checkpoints × 256 paired blocks × two positions = **38,400 hands**.
The transferred files contain all seven panels (268,800 hands); their full coordinate,
identity, replay-evidence and zero-sum checks passed before selecting this panel.
This is descriptive, post-hoc simulator analysis against the frozen post-Luna stress
opponent. It does not change #136’s evaluation or supply independent confirmation.

## Findings

- All-hand profit goes from **+748.50 BB at 100M to +840.50 BB at 500M** over
  1,536 hands per checkpoint (+48.73 → +54.72 BB/100, descriptive). The curve
  fluctuates: +957.00 BB at 480M. These are the light schedule, not the pending
  broader or held-out evaluation; do not substitute one schedule’s estimate for another.
- First large target raises increase **12 → 18 hands**. After a same-street rival
  raise: **5 → 8**, all continued. Without a rival raise: **7 → 10**, with **5 → 6**
  continuations. Large means the rival owes ≥800 chips after its existing commitment.
- After a rival raise, continued made-hand order is **4 ahead / 1 behind / 0 tied**
  at 100M versus **5 / 3 / 0** at 500M. One-pair hands are **1/5 → 2/8 postflop
  continuations**. Without a prior rival raise, the corresponding order is **1/3/1 →
  3/3/0**, with one pair **2/5 → 1/6**. This sparse evidence does not demonstrate
  a monotone correction of weak-hand aggression.
- All-hand full-stack wins/losses are **4/3 → 6/4** (each denominator 1,536).
  The selected-event totals happen to include all these endpoint stack outcomes;
  intermediate checkpoints need their own denominators. At 380M losses reach 8.
- Lineages differ: whole-hand profit at 100M→500M is **+303.00→+247.50 BB** for
  seed 2026093001, **+224.50→+278.00 BB** for seed 2026093002 and
  **+221.00→+315.00 BB** for seed 2026093003 (512 hands each). At 500M,
  after-raise one-pair continuations are **0/2, 2/3, 0/3**, respectively. Pooling
  would obscure that the two observed one-pair re-raises come from seed 2.

The [complete dashboard](hu20-light-made-hand-trends-artifacts/dashboard.md) covers
all 25 checkpoints, both situations and each lineage. [Summary JSON](hu20-light-made-hand-trends-artifacts/summary.json)
retains folds, continuations, streets, categories, trained/fallback and all-hand
counters. [457 exact selected events](hu20-light-made-hand-trends-artifacts/events.jsonl)
retain current cards/board, exact wager, response and whole-hand payoff.

## Interpretation and unresolved questions

Whole-hand profit includes earlier/later bets and is **not individual-bet EV**.
Ahead/behind/tied compares best made hands at that raise; draws and future cards
are excluded. Both cards are joined offline, never supplied to a playing opponent.
The same blocks recur across checkpoints and lineages, and selecting large raises
changes the sample at each checkpoint. I make no significance or strength claim.

The useful lead is to inspect the concrete seed-2 500M one-pair re-raises and
continued behind-hand events. Are they justified draws/blockers, weak abstraction
collisions, or poor policies within otherwise reasonable buckets? These records
cannot distinguish those causes. No policy intervention follows from this report.

## Provenance and reproduction

The [protocol](../hu20-light-made-hand-trend-protocol.md) was committed before
measurement. Original scientific source: `17b4c9a08ed0765d0fb8f05240c0409b21e43977`;
reporting implementation: `da64fd3`. Canonical campaign plan SHA-256:
`74f0c18024a4a409d223276d4781c2464b27fc8d5d32a51ccb22a224332f2219`.
Doctor Research authorized one read-only transfer through the existing ownership
note. All 75 raw/result pairs and the plan were verified locally; compressed hand
bytes total 157,629,578. Originals remained unchanged and the transfer window was
released explicitly. [Input manifest](hu20-light-made-hand-trends-artifacts/input-manifest.json)
SHA-256: `37080f74fda0b9c6f4ef580869f12f3d02c4a35c0f783ad00644c1e1e72a7338`.
The first 74 identities are full published subsets; the final identity omits
nonessential fields, which are retained from its separately hash-verified result.
No input/model bytes were rewritten.

M1, one worker, Python 3.11.15: **21.87 seconds**, peak RSS **30.81 MiB**.
No models, new hands, M4 analysis or paid compute. Raw inputs remain under ignored
`results/`; compact outputs and their [checksums](hu20-light-made-hand-trends-artifacts/manifest.json)
are published here. Tests use generated records: **43 focused tests pass**, including
existing first-event/current-board comparisons and new transfer, arithmetic,
duplicate, incomplete-lineage and pairing checks.

```sh
python -m pytest tests/diagnostics/test_light_made_hands.py tests/diagnostics/test_stackoff.py -q
python -m scripts.report_hu20_light_made_hands \
  --inputs results/hu20-light-trend-inputs-20261001 \
  --out results/hu20-light-made-hand-trends-reproduction
```

The reproduction requires the unchanged raw files named by the input manifest;
it does not download private paths or load their referenced models.
