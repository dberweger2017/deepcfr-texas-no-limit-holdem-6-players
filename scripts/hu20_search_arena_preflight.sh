#!/usr/bin/env bash
# Provisioning is separate. This script admits and checks only its actual host.
set -euo pipefail
commit=6a5ff44ec3e5d313be594d177244c3579081ff00
B=/workspace/bundle; E=/workspace/evidence
mkdir -p "$E"
exec > >(tee -a "$E/preflight.log") 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CARGO_HOME=/workspace/cargo RUSTUP_HOME=/workspace/rustup PATH=/workspace/cargo/bin:/workspace/uv:$PATH
(cd "$B" && sha256sum -c SHA256SUMS) > "$E/bundle-verify.txt"
test "$(uname -m)" = x86_64
uname -a > "$E/uname.txt"; lscpu > "$E/lscpu.txt"; free -b > "$E/free.txt"
df -B1 / /workspace > "$E/df.txt"
cat /sys/fs/cgroup/cpu.max > "$E/cgroup-cpu.max"
cat /sys/fs/cgroup/memory.max > "$E/cgroup-memory.max"
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
quota=len(os.sched_getaffinity(0)) if cpu[0]=='max' else min(len(os.sched_getaffinity(0)),int(cpu[0])/int(cpu[1]))
mem=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))*1024
cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
if cap!='max':mem=min(mem,int(cap))
record={'quota_cpus':quota,'admitted_ram_bytes':mem,'affinity_cpus':len(os.sched_getaffinity(0)),
        'volume_free_bytes':shutil.disk_usage('/workspace').free,'minimum_cpu':23.8,'minimum_ram_bytes':60*10**9}
Path('/workspace/evidence/host-admission.json').write_text(json.dumps(record,sort_keys=True)+'\n')
assert quota>=23.8 and mem>=60*10**9,record
assert record['volume_free_bytes']>=35*10**9,record
PY
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0 > "$E/rustup.log" 2>&1
rustc -Vv > "$E/rustc.txt"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/uv UV_NO_MODIFY_PATH=1 sh > "$E/uv.log" 2>&1
uv venv -q -p 3.11 /workspace/venv
. /workspace/venv/bin/activate
uv pip install -q numpy==1.26.4 scipy==1.17.1 pytest==9.0.3
uv pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -q "pokers @ git+https://github.com/dberweger2017/pokers.git@5db20e3d5d6862b32a7402035c1340b622d3b005"
uv pip freeze > "$E/freeze.txt"
git clone -q https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/repo
cd /workspace/repo
git checkout -q "$commit"
git rev-parse HEAD > "$E/source.txt"
# Operational scripts are separately hash-bound; approved policy/evaluator source stays at 6a5ff44.
cp "$B"/ops/scripts/*.py scripts/
cp "$B"/ops/tests/test_hu20_search_arena_ops.py tests/
cp "$B"/ops/SHA256SUMS "$E/operations.sha256"
export PYTHONPATH=/workspace/repo
tar -xzf "$B/external.tar.gz" -C /workspace
git config --global --add safe.directory /workspace/hu20-turn-search-tool/upstream
find /workspace/hu20-turn-search-tool -name '._*' -delete
test -z "$(git -C /workspace/hu20-turn-search-tool/upstream status --porcelain)"
bash scripts/build_hu20_search_solver.sh /workspace/hu20-turn-search-tool native > "$E/build.log" 2>&1
BIN=/workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool
sha256sum "$BIN" > "$E/binary.sha256"
python -m pytest -q tests/test_hu20_search_arena_ops.py tests/test_hu20_turn_search.py tests/test_hu20_search_campaign.py tests/test_hu20_search_protocol.py > "$E/tests.log" 2>&1
python -m scripts.check_hu20_turn_search_solver --binary "$BIN" --config "$B/config.json" \
    --reference "$B/macos-reference/reference.json" --out "$E/checker" > "$E/checker.log" 2>&1
python -m scripts.replay_hu20_search_requests --binary "$BIN" --solves "$B/requests" --out "$E/replay" > "$E/replay.log" 2>&1
python - <<'PY'
import json
from pathlib import Path
x=json.loads(Path('/workspace/evidence/replay/summary.json').read_text())
assert x['requests']==96 and x['passed']==96 and x['max_profile_difference']==0,x
PY
# Reference and fresh profile bodies are eligible only after the full comparison.
python -m scripts.hu20_search_evidence finalize "$B/requests" --owned-pod-root /workspace > "$E/reference-retention.json"
python -m scripts.hu20_search_evidence finalize "$E" --owned-pod-root /workspace > /workspace/parity-retention.json
mv /workspace/parity-retention.json "$E/parity-retention.json"
python -m scripts.hu20_search_evidence retention-check "$B/requests" > "$E/reference-retention-check.json"
python -m scripts.hu20_search_evidence retention-check "$E" > "$E/parity-retention-check.json"
touch "$E/PREFLIGHT_PASSED"
