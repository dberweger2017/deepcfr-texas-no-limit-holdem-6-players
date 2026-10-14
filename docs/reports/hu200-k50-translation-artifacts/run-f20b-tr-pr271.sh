#!/bin/sh
# Prepared only. The owner supplies the dedicated account and starts this launcher.
set -eu
cd "$HOME/deepcfr-slumbot/repo-translation-pr271"
test "$(git rev-parse HEAD)" = fd44d53f22c4c795d2d725f94babc065bde14f90
exec "$HOME/deepcfr-slumbot/venv/bin/python" -m scripts.play_slumbot --hands 5000 \
  --max-abandoned 200 --max-rejected 50 --login --account deepcfr-f20b-tr \
  --model "$HOME/deepcfr-slumbot/inputs/model-frozen/hu200-k50-frozen-20b.average.gz" \
  --model-sha256 9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6 \
  --card-buckets "$HOME/deepcfr-slumbot/inputs/tables" \
  --policy-cache "$HOME/deepcfr-slumbot/cache-frozen" --cache-read-only \
  --action-translation --translation-max-states 512 --translation-max-events 128 \
  --seed 20261014027101 \
  --out "$HOME/deepcfr-slumbot/translation-runs-pr271/5000-hands-seed-20261014027101"
