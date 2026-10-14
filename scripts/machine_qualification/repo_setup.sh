#!/bin/bash
# Runs inside the detached on-host guard, never as root.
set -euo pipefail
export PATH="$QUALIFICATION_ROOT/node/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$QUALIFICATION_ROOT"
python3 install_node.py
node --version
test ! -e repo-main
git clone https://github.com/dberweger2017/CFR-NLHE-6max.git repo-main
cd repo-main
git rev-parse HEAD > ../source-main.txt
git status --porcelain > ../source-status.txt
uv venv --python 3.11.17 .venv
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python - <<'PY'
import json, platform, sys, torch
from pathlib import Path
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.startswith('2.5.1+cpu'), torch.__version__
assert torch.version.cuda is None
Path('../environment.json').write_text(json.dumps({
    'python': sys.version, 'platform': platform.platform(),
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
}, indent=2) + '\n')
PY
uv pip freeze --python .venv/bin/python > ../requirements-installed.txt
.venv/bin/python -m scripts.build_native_trainer --receipt ../build-first.json
cp native/hu20-trainer/target/release/hu20-trainer ../trainer-first
