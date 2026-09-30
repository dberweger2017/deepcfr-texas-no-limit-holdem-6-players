#!/bin/bash
# One compiler/trainer at a time on this pod; source and work are frozen.
set -euo pipefail
DRIVER_REVISION="$1"
DEADLINE="$2"
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/python
export CARGO_HOME=/workspace/cargo
export RUSTUP_HOME=/workspace/rustup
export CARGO_BUILD_JOBS=1
export PATH="$CARGO_HOME/bin:/root/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p /workspace/results
lscpu > /workspace/results/setup-lscpu.txt
uname -a > /workspace/results/setup-kernel.txt
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
git clone --branch feature/runpod-mature-cpu --single-branch https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/driver
git -C /workspace/driver checkout "$DRIVER_REVISION"
git clone --branch feature/trainer-observation-reuse --single-branch https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/runtime
git -C /workspace/runtime checkout 50326afcd4776308054e0c9efce8681de1e877eb
cd /workspace/runtime
uv python install 3.11.14
uv venv --python 3.11.14 .venv
uv pip install --python .venv/bin/python pytest 'pokers @ git+https://github.com/dberweger2017/pokers.git@5db20e3d5d6862b32a7402035c1340b622d3b005'
uv pip freeze --python .venv/bin/python > /workspace/results/packages.txt
PYTHONPATH=/workspace/driver .venv/bin/python -m pytest -q /workspace/driver/tests/test_mature_cpu_worker.py > /workspace/results/worker-tests.log
cd /workspace/driver
/workspace/runtime/.venv/bin/python -m scripts.mature_cpu_linux_worker \
  --plan /workspace/driver/configs/blueprint/runpod-mature-cpu-pilot.json \
  --runtime /workspace/runtime --parent /workspace/parent.json.gz \
  --out /workspace/results/work --deadline "$DEADLINE"
