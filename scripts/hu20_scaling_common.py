"""Identity, host and resource checks for continued HU20 lineages."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import time

from scripts.run_tp20_campaign import swap_bytes
from scripts.train_hu20 import rss, system, write_json
from src.arena.schedule import digest
from src.blueprint.artifact import load_training
from src.blueprint.solver import HU20_UNCAPPED_GAME
from src.blueprint.windowed import _hash


def identity():
    import pokers
    package = Path(pokers.__file__).parent
    return {"host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "pid": os.getpid(),
            "ram": system(["sysctl", "-n", "hw.memsize"]),
            "processor": system(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "native_binaries": {p.name: _hash(p) for p in package.glob("*.so")},
            "dependencies": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}


def check(plan, out, deadline, swap_before):
    if time() >= deadline:
        raise TimeoutError("Single absolute campaign/phase deadline")
    if rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
        raise MemoryError("Per-host process RSS guard")
    if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
        raise RuntimeError("Free disk guard")
    current = swap_bytes(system(["sysctl", "vm.swapusage"]))
    before = swap_bytes(swap_before)
    if current is not None and before is not None and current - before > .5 * 1024**3:
        raise MemoryError("Per-host swap growth guard")


def parent_trainer(spec, maximum_entries=None):
    path = Path(spec["checkpoint_path"])
    if _hash(path) != spec["checkpoint_sha256"]:
        raise ValueError("Transferred training parent hash")
    trainer = load_training(path)
    c = trainer.config
    if (c.game != HU20_UNCAPPED_GAME or c.raise_cap is not None
            or c.seed != spec["seed"] or trainer.iteration != spec["iteration"]
            or c.roots_per_seat != 1 or c.postflop_replicates != 1
            or len(trainer.nodes) != spec["entries"]):
        raise ValueError("Continued lineage/configuration mismatch")
    if maximum_entries is not None and maximum_entries != c.max_entries:
        # This is a frozen safety bound, not a change to the regret update recipe.
        from dataclasses import replace
        trainer.config = replace(c, max_entries=maximum_entries)
    return trainer


def inventory(root):
    return {str(p.relative_to(root)): {"sha256": _hash(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file()}


def acquire(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "owner.json", {"pid": os.getpid(), "host": platform.node(), "started": time()})


def specification(seed, nodes, iteration, checkpoint, policy, cp_hash, policy_hash):
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
    from src.blueprint.artifact import HU20_UNCAPPED_FORMAT
    return {"name": f"B-{seed}-{nodes}", "seed": seed, "nodes": nodes,
            "iteration": iteration, "players": 2, "arm": "B",
            "milestone": nodes, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": cp_hash, "path": str(policy), "sha256": policy_hash,
            "abstraction": HU20_UNCAPPED_SCHEMA, "format": HU20_UNCAPPED_FORMAT,
            "dual_menu_telemetry": True}
