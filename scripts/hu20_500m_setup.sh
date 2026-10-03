#!/bin/bash
set -euo pipefail
REVISION="$1"
export UV_CACHE_DIR=/workspace/uv-cache UV_PYTHON_INSTALL_DIR=/workspace/python
export CARGO_HOME=/workspace/cargo RUSTUP_HOME=/workspace/rustup CARGO_BUILD_JOBS=1
export PATH="$CARGO_HOME/bin:/root/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p /workspace/results
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/source
git -C /workspace/source checkout "$REVISION"
cd /workspace/source
uv python install 3.11.14
uv venv --python 3.11.14 .venv
uv pip install --python .venv/bin/python pip pytest 'pokers @ git+https://github.com/dberweger2017/pokers.git@5db20e3d5d6862b32a7402035c1340b622d3b005'
.venv/bin/python -m pytest -q tests/test_hu20_500m_campaign.py > /workspace/results/focused-tests.log
.venv/bin/python -m scripts.hu20_500m_pod \
  --plan configs/blueprint/hu20-500m-campaign.json --parent /workspace/parent.json \
  --reference /workspace/reference.json --control /workspace/control.json --root /workspace/results/worker
