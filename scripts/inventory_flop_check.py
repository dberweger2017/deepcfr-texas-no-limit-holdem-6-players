"""Verify retained input bytes and the separately installed pinned solver source."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pokers

from src.diagnostics.flop_check import SOLVER_COMMIT, atomic_json
from src.diagnostics.saved_hu20 import file_hash


def inventory(plan, repo, inputs, tool, expected):
    rows = []
    for spec in plan["policies"]:
        path = inputs / spec["path"]; actual = file_hash(path)
        rows.append({"path": str(path), "sha256": actual, "expected_sha256": spec["sha256"],
                     "bytes": path.stat().st_size, "verified": actual == spec["sha256"]})
    for relative, wanted in sorted(plan["stored_hands"].items()):
        path = repo / "docs/reports/hu20-card-v2-artifacts/production" / relative
        actual = file_hash(path)
        rows.append({"path": str(path), "sha256": actual, "expected_sha256": wanted,
                     "bytes": path.stat().st_size, "verified": actual == wanted})
    if expected["commit"] != SOLVER_COMMIT:
        raise ValueError("Solver source inventory is for another commit")
    upstream = []
    for relative, wanted in expected["files"].items():
        path = tool / "upstream" / relative; actual = file_hash(path)
        upstream.append({"path": relative, "sha256": actual, "verified": actual == wanted})
    if not all(r["verified"] for r in rows + upstream):
        raise ValueError("Input or pinned source hash mismatch")
    external = {str(path.relative_to(tool)): file_hash(path) for path in (
        tool / "harness/Cargo.toml", tool / "harness/Cargo.lock", *sorted((tool / "harness/src").glob("*.rs")),
        tool / "harness/target/release/hu20-exact-flop-tool")}
    source_paths = [*sorted((repo / "src/diagnostics").glob("flop_check*.py")),
                    *sorted((repo / "scripts").glob("*flop_check*.py")),
                    *sorted((repo / "scripts").glob("*turn*check*.py")),
                    repo / "src/diagnostics/turn_check.py",
                    repo / "src/diagnostics/exact_ranker.py", repo / "src/arena/endgame_quality.py",
                    repo / "src/game/observation.py", repo / "src/game/types.py",
                    repo / "src/blueprint/abstraction.py", repo / "src/blueprint/hu20_river.py",
                    repo / "src/blueprint/river_cfr.py", repo / "src/game/hand.py"]
    native = {module.__file__: file_hash(module.__file__)
              for name, module in list(sys.modules.items()) if name.startswith("pokers")
              and getattr(module, "__file__", "").endswith((".so", ".dylib", ".pyd"))}
    return {"inputs": rows, "all_inputs_verified": True, "upstream_commit": SOLVER_COMMIT,
            "upstream_files": upstream, "external_tool_files_sha256": external,
            "repository_source_sha256": {str(p.relative_to(repo)): file_hash(p) for p in source_paths},
            "python": sys.version, "pokers_version": importlib.metadata.version("pokers"),
            "pokers_package_path": pokers.__file__,
            "pokers_native_sha256": native,
            "rustc": subprocess.check_output([str(Path.home() / ".cargo/bin/rustc"), "--version"], text=True).strip(),
            "build_flags": {"RUSTFLAGS": "-A dangerous_implicit_autorefs", "cargo": "build --release --locked"},
            "license_boundary": "external AGPL tool; no solver or harness source vendored"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "inputs", "tool", "upstream-inventory", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = inventory(json.loads(args.plan.read_text()), args.repo, args.inputs, args.tool,
                       json.loads(args.upstream_inventory.read_text()))
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()
