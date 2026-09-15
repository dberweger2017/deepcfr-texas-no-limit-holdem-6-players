"""Auditable run inputs and deterministic artifact comparison."""

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path

from src.arena.registry import PolicyRegistry
from src.arena.report import CONFIDENCE, MINIMUM_BLOCKS
from src.arena.schedule import Plan, digest, schedule_document
from src.game.observation import RULES_PROFILE, SCHEMA_VERSION
from src.game.session import SESSION_PROFILE

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_VERSION = 2


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def environment() -> dict:
    engine = metadata.distribution("pokers")
    direct = json.loads(engine.read_text("direct_url.json") or "{}")
    binaries = {
        str(path): sha256(Path(engine.locate_file(path)).read_bytes()).hexdigest()
        for path in engine.files or ()
        if str(path).endswith((".so", ".pyd", ".dylib"))
    }
    if not binaries:
        raise ValueError("Cannot fingerprint the installed poker engine")
    return {
        "engine": {
            "commit": direct.get("vcs_info", {}).get("commit_id"),
            "binaries": binaries,
        },
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            dist.metadata["Name"]: dist.version for dist in metadata.distributions()
        },
    }


def source_fingerprint() -> str:
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    relevant = sorted(
        {p for p in paths if p.endswith(".py") or p.startswith("requirements")}
    )
    return digest(
        {
            p: sha256((ROOT / p).read_bytes()).hexdigest()
            for p in relevant
            if (ROOT / p).is_file()
        }
    )


def manifest(plan: Plan, registry: PolicyRegistry | None = None) -> dict:
    registry = registry or PolicyRegistry(plan)
    if registry.plan != plan:
        raise ValueError("Policy registry belongs to a different plan")
    return {
        "version": ARTIFACT_VERSION,
        "plan": asdict(plan),
        "schedule_sha256": digest(schedule_document(plan)),
        "rules": {
            "hand": RULES_PROFILE,
            "session": SESSION_PROFILE,
            "observation_schema": SCHEMA_VERSION,
        },
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_fingerprint(),
        "policies": registry.fingerprints(),
        "environment": environment(),
        "protocol": {
            "inference": registry.inference_config,
            "confidence": CONFIDENCE,
            "minimum_blocks": MINIMUM_BLOCKS,
            "interval": "student-t-over-independent-block-means",
            "session_reloads": "restore busted seats to their initial stack before each hand",
            "rate_denominator": "scheduled table hands per arm; includes dealt-out session hands",
        },
    }


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_manifest(value: dict, registry: PolicyRegistry | None = None) -> Plan:
    if value.get("version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported arena manifest version")
    plan = Plan.from_dict(value["plan"])
    current = manifest(plan, registry)
    for field in (
        "schedule_sha256",
        "source_sha256",
        "policies",
        "rules",
        "environment",
        "protocol",
    ):
        if value[field] != current[field]:
            raise ValueError(f"Reproduction mismatch: {field}")
    return plan
