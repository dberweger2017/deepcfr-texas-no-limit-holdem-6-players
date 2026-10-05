#!/usr/bin/env bash
# Owner-approved x86_64 Linux pilot for HU20 turn search: native build, parity against the M4,
# then an outcome-blind timing run for the arena quote. Never provisions or terminates hardware.
#
#   bash hu20_search_pod_pilot.sh SOURCE_COMMIT   (bundle at /workspace/bundle)
set -euo pipefail
commit="${1:?source commit required}"
B=/workspace/bundle; E=/workspace/evidence; mkdir -p "$E"
exec > >(tee -a "$E/pilot.log") 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }

step host
test "$(uname -m)" = x86_64
uname -a > "$E/uname.txt"; lscpu > "$E/lscpu.txt"; nproc > "$E/nproc.txt"
cat /sys/fs/cgroup/cpu.max > "$E/cgroup-cpu.max" 2>/dev/null || true
free -b > "$E/free.txt"; df -B1 /workspace > "$E/df.txt"
(cd "$B" && sha256sum -c SHA256SUMS) > "$E/bundle-verify.txt"

step toolchains
export CARGO_HOME=/workspace/cargo RUSTUP_HOME=/workspace/rustup PATH=/workspace/cargo/bin:/workspace/uv:$PATH
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0 > "$E/rustup.log" 2>&1
rustc -Vv > "$E/rustc.txt"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/uv UV_NO_MODIFY_PATH=1 sh > "$E/uv.log" 2>&1
uv venv -q -p 3.11 /workspace/venv
. /workspace/venv/bin/activate
uv pip install -q numpy==1.26.4 scipy==1.17.1 pytest==9.0.3
uv pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -q "pokers @ git+https://github.com/dberweger2017/pokers.git@5db20e3d5d6862b32a7402035c1340b622d3b005"
python -VV > "$E/python.txt"; uv pip freeze > "$E/freeze.txt"

step source
git clone -q https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git /workspace/repo
cd /workspace/repo && git checkout -q "$commit" && git rev-parse HEAD > "$E/source.txt"
export PYTHONPATH=/workspace/repo

step solver
tar -xzf "$B/external.tar.gz" -C /workspace
# The archive keeps the Mac owner's uid; git refuses such a checkout for root without this.
git config --global --add safe.directory /workspace/hu20-turn-search-tool/upstream
bash scripts/build_hu20_search_solver.sh /workspace/hu20-turn-search-tool native > "$E/build.log" 2>&1
BIN=/workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool
sha256sum "$BIN" > "$E/binary.sha256"

step tests
python -m pytest -q tests/test_hu20_turn_search.py tests/test_hu20_search_campaign.py > "$E/tests.log" 2>&1 \
    || echo "focused tests failed; see tests.log"

step parity
python -m scripts.check_hu20_turn_search_solver --binary "$BIN" --config "$B/config.json" \
    --reference "$B/macos-reference/reference.json" --out "$E/checker" > "$E/checker.log" 2>&1
python -m scripts.replay_hu20_search_requests --binary "$BIN" --solves "$B/requests" --out "$E/replay" > "$E/replay.log" 2>&1

step approval
# Timing runs only after both parity checks pass; the document binds the owner approval, plan and settings.
python - <<PY
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from src.arena.schedule import digest
from src.blueprint.hu20_turn_search import TurnSearchConfig
B, E = Path("$B"), Path("$E")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((B/"timing-plan.json").read_text())
config = TurnSearchConfig(**json.loads((B/"config.json").read_text()))
approval = {"owner_approved": True, "kind": "timing-pilot",
    "owner_decision_url": "https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5994292377",
    "arena_plan_sha256": digest(plan), "search_config_sha256": digest(asdict(config)),
    "selected_settings_parity": "passed", "quote_sha256": sha(B/"estimate.md"),
    "parity_sha256": digest({"checker": sha(E/"checker.log"), "replay": sha(E/"replay/summary.json")}),
    "worker_seconds": 2700, "rss_limit_bytes": 12*1024**3}
(E/"pilot-approval.json").write_text(json.dumps(approval, indent=1))
PY

step timing
mkdir -p "$E/timing"
for i in 0 1 2 3; do
    python -m scripts.evaluate_hu20_turn_search --phase timing --plan "$B/timing-plan.json" --inputs "$B/inputs" \
        --out "$E/timing/worker-$i" --binary "$BIN" --search-config "$B/config.json" \
        --paid-approval "$E/pilot-approval.json" --worker-index "$i" --worker-count 4 > "$E/timing/worker-$i.log" 2>&1 &
done
wait || true

step done
touch "$E/DONE"
