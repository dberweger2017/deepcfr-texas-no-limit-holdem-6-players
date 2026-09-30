#!/bin/bash
set -euo pipefail
mkdir -p /workspace/hu20-setup
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/python
export CARGO_HOME=/workspace/cargo
export RUSTUP_HOME=/workspace/rustup
export CARGO_BUILD_JOBS=1
export PATH="$CARGO_HOME/bin:/root/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
git clone --branch feature/runpod-hu20-parity --single-branch https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/hu20-parity
cd /workspace/hu20-parity
git checkout e18f007
uv python install 3.11.14
uv venv --python 3.11.14 .venv
uv pip install --python .venv/bin/python pytest 'pokers @ git+https://github.com/dberweger2017/pokers.git@5db20e3d5d6862b32a7402035c1340b622d3b005'
mkdir -p results/platform-pilot-hardware
lscpu > results/platform-pilot-hardware/lscpu.txt
free -b > results/platform-pilot-hardware/memory.txt
df -B1 > results/platform-pilot-hardware/filesystems.txt
uname -a > results/platform-pilot-hardware/uname.txt
.venv/bin/python -m pip freeze > results/platform-pilot-hardware/packages.txt 2>/dev/null || uv pip freeze --python .venv/bin/python > results/platform-pilot-hardware/packages.txt
for name in cpu.max memory.max memory.current memory.peak memory.swap.current; do
    if test -r /sys/fs/cgroup/$name; then cat /sys/fs/cgroup/$name > results/platform-pilot-hardware/$name.txt; fi
done
.venv/bin/python -m pytest -q tests/test_hu20_platform_pilot.py > results/platform-pilot-hardware/focused-tests.log
.venv/bin/python -m scripts.hu20_platform_pilot run --plan configs/blueprint/runpod-hu20-parity.json --out results/platform-pilot/direct
.venv/bin/python -m scripts.hu20_platform_pilot run --plan configs/blueprint/runpod-hu20-parity.json --out results/platform-pilot/resumed --resume results/platform-pilot/direct
.venv/bin/python -m scripts.hu20_platform_pilot compare --left results/platform-pilot/direct --right results/platform-pilot/resumed --out results/platform-pilot/resume-comparison.json
for path in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.swap.current /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.max_usage_in_bytes /sys/fs/cgroup/memory/memory.memsw.max_usage_in_bytes /sys/fs/cgroup/memory/memory.failcnt; do
 if test -r "$path"; then name=$(basename "$path"); cat "$path" > "results/platform-pilot-hardware/cgroup-$name.txt"; fi
done
printf 'complete\n' > results/platform-pilot-hardware/setup-finished.txt
