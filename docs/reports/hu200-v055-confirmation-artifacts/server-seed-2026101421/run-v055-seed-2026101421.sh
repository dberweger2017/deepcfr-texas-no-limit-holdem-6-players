#!/bin/sh
# PR278 confirmation seed 2026101421. Prepared only; the owner logs in and starts it once.
set -eu
cd "$HOME/deepcfr-slumbot/repo-completed-hands"
test "$(git rev-parse HEAD)" = ff6cc77aa9510854eef49114355f272d6c465570
OUT="$HOME/deepcfr-slumbot/v055-runs-pr278/seed-2026101421-$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$OUT"
exec "$HOME/deepcfr-slumbot/venv/bin/python" -m scripts.play_slumbot --hands 10000 \
  --timeout 30 --max-abandoned 200 --max-rejected 50 --login --account deepcfr-v055-s21 \
  --model "$HOME/deepcfr-slumbot/inputs/model-seed-2026101421/final.average.gz" \
  --model-sha256 c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444 \
  --card-buckets "$HOME/deepcfr-slumbot/inputs/tables" \
  --policy-cache "$HOME/deepcfr-slumbot/cache-seed-2026101421" --cache-read-only \
  --action-translation --translation-max-states 512 --translation-max-events 128 \
  --seed 20261014055201 --out "$OUT"
