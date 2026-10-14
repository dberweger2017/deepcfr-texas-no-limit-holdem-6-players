#!/bin/sh
# Owner starts this after creating the deepcfr-k50-frozen-20b account.
# Prepared for PR268; no account creation, credentials or live play were used.
set -eu
cd "$HOME/deepcfr-slumbot/repo"
"$HOME/deepcfr-slumbot/venv/bin/python" -m scripts.play_slumbot --hands 5000 \
  --max-abandoned 200 --max-rejected 50 --login --account deepcfr-k50-frozen-20b \
  --model "$HOME/deepcfr-slumbot/inputs/model-frozen/hu200-k50-frozen-20b.average.gz" \
  --model-sha256 9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6 \
  --card-buckets "$HOME/deepcfr-slumbot/inputs/tables" \
  --policy-cache "$HOME/deepcfr-slumbot/cache-frozen" \
  --seed 20261014026801 \
  --out "$HOME/deepcfr-slumbot/runs/hu200-k50-frozen-20b-$(date +%Y%m%d-%H%M%S)"
