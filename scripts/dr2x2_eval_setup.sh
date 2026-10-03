#!/bin/bash
# An isolated container, pinned source/engine/runtime; one compiler/worker.
set -euo pipefail
V2_SOURCE="$1"
EVAL_SEED="$2"
PLAN_SHA="$3"
export UV_CACHE_DIR=/workspace/uv-cache UV_PYTHON_INSTALL_DIR=/workspace/python
export UV_PYTHON_INSTALL_MIRROR=https://github.com/astral-sh/python-build-standalone/releases/download
export CARGO_HOME=/workspace/cargo RUSTUP_HOME=/workspace/rustup CARGO_BUILD_JOBS=1
export PATH="$CARGO_HOME/bin:/workspace/uv:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p /workspace/results /workspace/uv
curl -fLsS --retry 2 https://github.com/astral-sh/uv/releases/download/0.12.21/uv-x86_64-unknown-linux-gnu.tar.gz -o /workspace/uv.tar.gz
printf '%s  %s\n' '23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0' /workspace/uv.tar.gz | sha256sum -c -
tar -xzf /workspace/uv.tar.gz --strip-components=1 -C /workspace/uv
curl -fLsS --retry 2 https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init -o /workspace/rustup-init
printf '%s  %s\n' '20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c' /workspace/rustup-init | sha256sum -c -
chmod +x /workspace/rustup-init
/workspace/rustup-init -y --profile minimal --default-toolchain 1.90.0
sha256sum /workspace/uv.tar.gz /workspace/rustup-init > /workspace/results/setup-downloads.sha256
# Git is public; no provider or GitHub credential is copied into this container.
git clone --branch feature/hu20-history-2x2 --single-branch https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/source
git -C /workspace/source checkout "$V2_SOURCE"
cd /workspace/source
uv python install 3.11.14
uv venv --python 3.11.14 .venv
uv pip install --python .venv/bin/python pip pytest==9.0.3 numpy==1.26.4 scipy==1.17.1 'pokers @ git+https://github.com/dberweger2017/pokers.git@5db20e3d5d6862b32a7402035c1340b622d3b005'
uv pip freeze --python .venv/bin/python > /workspace/results/packages.txt
 .venv/bin/python -m pytest -q tests/test_dr2x2_ac_executor.py tests/diagnostics/test_history_mechanisms.py tests/test_dr2x2_evaluation_preflight.py tests/diagnostics/test_cfr_average.py > /workspace/results/focused-tests.log
PYTHONPATH=/workspace/source .venv/bin/python /workspace/eval-pod.py --source "$V2_SOURCE" --seed "$EVAL_SEED" --plan-sha "$PLAN_SHA" \
 --plan /workspace/plan.json --inputs /workspace/inputs --reference /workspace/reference.json \
 --control /workspace/control.json --root /workspace/results/dr2x2-eval
