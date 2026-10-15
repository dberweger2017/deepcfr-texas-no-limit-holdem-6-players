#!/bin/bash
# Serial Python suite and bounded Cargo compilation avoid CPU oversubscription.
set -euo pipefail
export PATH="$QUALIFICATION_ROOT/node/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$QUALIFICATION_ROOT/repo-main"
cargo test --locked -j 4 --manifest-path native/hu20-trainer/Cargo.toml --lib
.venv/bin/python -m pytest -q -ra --maxfail=1 --durations=25
.venv/bin/python -m scripts.check_game --players 6 --hands 20
.venv/bin/python -m scripts.check_session --hands 30
sha256sum native/hu20-trainer/target/release/hu20-trainer
