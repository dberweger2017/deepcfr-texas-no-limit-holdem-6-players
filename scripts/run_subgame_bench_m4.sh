#!/bin/bash
# Overnight M4 run of the subgame bench: train both held-out folds in parallel,
# then score every checkpoint on all 40 held-out roots, then report.
# Launch detached from the repo checkout:
#   nohup scripts/run_subgame_bench_m4.sh > ../bench.log 2>&1 < /dev/null &
# Rerunning resumes: training continues from its last checkpoint state and
# evaluation skips roots that already have a result.
set -euo pipefail
POOL=${POOL:-/Users/dberweger/Local/hu20-board-pooling-20261004}
WORK=${WORK:-/Users/dberweger/Local/hu20-trainer-bench-20261005}
PY=${PY:-/Users/dberweger/Local/deepcfr-training/.venv/bin/python}
ITERATIONS=${ITERATIONS:-3000000}
CHECKPOINTS=${CHECKPOINTS:-100000,300000,1000000}
export PYTHONPATH=.
common=(--prepared "$POOL/prepared-03" --lineage-index 0)
for fold in 0 1; do
  if ! grep -q '"complete": true' "$WORK/runs/fold-$fold/status.json" 2>/dev/null; then
    nice -n 10 "$PY" -m scripts.subgame_bench train "${common[@]}" --out "$WORK/runs" \
      --evaluation-fold "$fold" --iterations "$ITERATIONS" --checkpoints "$CHECKPOINTS" &
  fi
done
wait
"$PY" -m scripts.subgame_bench evaluate "${common[@]}" --out "$WORK/eval" --runs "$WORK/runs" \
  --main "$POOL/main-06" --binary "$POOL/pooling-engineering-05-mac"
"$PY" -m scripts.subgame_bench report "${common[@]}" --out "$WORK/eval" --main "$POOL/main-06"
echo "BENCH COMPLETE"
