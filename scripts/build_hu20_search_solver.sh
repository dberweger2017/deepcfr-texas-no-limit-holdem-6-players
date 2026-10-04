#!/usr/bin/env bash
# Build only the separately installed external AGPL project.
set -euo pipefail
task_tool_root="${1:?external harness root required}"
task_target="${2:-native}"
task_pin=9d1509fe5077d019825f833eed04b16d342dfda1
test "$(git -C "$task_tool_root/upstream" rev-parse HEAD)" = "$task_pin"
test -z "$(git -C "$task_tool_root/upstream" status --porcelain)"
export CARGO_BUILD_JOBS=2
# The pinned upstream predates this compiler lint; its source remains unchanged.
export RUSTFLAGS='-A dangerous_implicit_autorefs'
case "$task_target" in
  native)
    cargo build --release --locked --manifest-path "$task_tool_root/harness/Cargo.toml"
    ;;
  x86_64-linux)
    command -v cargo-zigbuild >/dev/null || { echo 'cargo-zigbuild unavailable; build natively on approved RunPod hardware.' >&2; exit 2; }
    cargo zigbuild --release --locked --target x86_64-unknown-linux-gnu --manifest-path "$task_tool_root/harness/Cargo.toml"
    ;;
  *) echo 'Use native or x86_64-linux. Emulation is prohibited.' >&2; exit 2 ;;
esac
