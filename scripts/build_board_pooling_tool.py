"""Verify and fingerprint an externally retained AGPL tool; never vendor it."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tool-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); root = a.tool_root.resolve()
    if a.out.exists():
        raise FileExistsError("Preserve each build attempt's fingerprint")
    if sys.platform == "linux" and platform.machine() != "x86_64":
        raise ValueError("Production build requires native x86_64 Linux, no emulation")
    expected = json.loads(Path("docs/reports/hu20-board-pooling-artifacts/external-tool.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(root / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root / "upstream"), "status", "--porcelain"], text=True).strip()
    if commit != expected["upstream_commit"] or status:
        raise ValueError("Pinned external upstream differs")
    for row in expected["sources"]:
        if file_hash(root / row["path"]) != row["sha256"]:
            raise ValueError("External harness source/lockfile fingerprint differs")
    for row in expected["reference_sources"]:
        if file_hash(root / row["path"]) != row["sha256"]:
            raise ValueError("Original reference harness source differs")
    rust = subprocess.check_output(["rustc", "-Vv"], text=True)
    if "release: 1.96.0\n" not in rust:
        raise ValueError("Use the prospective Rust 1.96.0 toolchain")
    env = dict(os.environ, RUSTFLAGS=expected["build_flags"])
    subprocess.run(["cargo", "build", "--locked", "--release", "--manifest-path", str(root / "harness/Cargo.toml")], check=True, env=env)
    subprocess.run(["cargo", "build", "--locked", "--release", "--manifest-path", str(root / "reference-harness/Cargo.toml")], check=True, env=env)
    binary = root / "harness/target/release/hu20-exact-flop-tool"
    result = dict(expected, platform=sys.platform, machine=platform.machine(), rustc=rust,
                  binary_path=str(binary), binary_sha256=file_hash(binary),
                  reference_binary_sha256=file_hash(root / "reference-harness/target/release/hu20-exact-flop-tool"))
    if sys.platform == "linux":
        result["linux_binary"] = result["binary_sha256"]
        result["glibc"] = subprocess.check_output(["ldd", "--version"], text=True).splitlines()[0]
        result["cpu"] = subprocess.check_output(["lscpu"], text=True)
    atomic_json(a.out, result)


if __name__ == "__main__":
    main()
