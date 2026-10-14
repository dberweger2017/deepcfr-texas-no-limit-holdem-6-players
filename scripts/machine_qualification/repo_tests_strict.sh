#!/bin/bash
# Keep the strict assertions; select oneMKL's reproducible CPU arithmetic mode.
set -euo pipefail
export PATH="$QUALIFICATION_ROOT/node/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export MKL_CBWR=AVX2,STRICT
export K50_REAL_TABLES="$QUALIFICATION_ROOT/tables"
cd "$QUALIFICATION_ROOT"
python3 verify_tables.py
cd repo-main
cargo build --release --locked -j 4 --manifest-path native/hu20-buckets/Cargo.toml
exec bash "$QUALIFICATION_ROOT/repo_tests.sh"
