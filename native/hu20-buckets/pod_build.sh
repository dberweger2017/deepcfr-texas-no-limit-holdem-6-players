#!/bin/bash
# Build the full-deck HU20 bucket tables on a disposable Linux pod.
# Usage on the pod:  nohup bash pod_build.sh COMMIT > /workspace/build.log 2>&1 &
# Outputs land in /workspace/hu20-buckets-out with SHA256SUMS; retrieve them, then terminate the pod.
set -euo pipefail
COMMIT=${1:?reviewed commit}
WORK=/workspace
OUT=$WORK/hu20-buckets-out
export CARGO_HOME=$WORK/cargo RUSTUP_HOME=$WORK/rustup PATH=$WORK/cargo/bin:$PATH
mkdir -p "$OUT"
stage() { echo "[$(date -u +%FT%TZ)] $*"; echo "{\"stage\": \"$*\", \"time\": \"$(date -u +%FT%TZ)\"}" > "$OUT/pod-status.json"; }

stage "host"
{ lscpu; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null; free -g; df -h $WORK; } > "$OUT/host.txt" 2>&1 || true
# Size the thread pool from the cgroup quota, not the host thread count.
THREADS=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${THREADS:-0}" -ge 1 ] || THREADS=$(nproc)
echo "threads=$THREADS" >> "$OUT/host.txt"

stage "toolchain"
command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
stage "checkout $COMMIT"
[ -d $WORK/poker ] || git clone -q https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git $WORK/poker
cd $WORK/poker && git fetch -q origin && git checkout -q "$COMMIT"
cd native/hu20-buckets

stage "tests (including the full seven-card category gate)"
RAYON_NUM_THREADS=$THREADS cargo test --release -- --include-ignored 2>&1 | tee "$OUT/tests.log"

stage "compile release binary"
cargo build --release 2>&1 | tail -3
test -x ./target/release/hu20-buckets

stage "build tables on $THREADS threads"
RAYON_NUM_THREADS=$THREADS ./target/release/hu20-buckets build --out "$OUT" \
  --ranks 13 --ks 50,200 --bins 50 --seed 202610050003 --iterations 100 2> >(tee "$OUT/build-stderr.log" >&2) &
BUILD=$!
# The base image has no GNU time; sample the builder's resident memory instead.
PEAK=0
while kill -0 $BUILD 2>/dev/null; do
  RSS=$(ps -o rss= -p $BUILD 2>/dev/null | tr -d ' ')
  [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS && echo "peak_rss_kib=$PEAK" > "$OUT/peak-rss.txt"
  sleep 5
done
wait $BUILD

stage "checksums"
cd "$OUT" && sha256sum *.bin summary.json host.txt tests.log peak-rss.txt > SHA256SUMS
git -C $WORK/poker rev-parse HEAD > "$OUT/commit.txt"
stage "BUILD COMPLETE"
