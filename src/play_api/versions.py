"""Version selection between pinned policies; a session never changes its model."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.policies import v040, v041
from src.play_api.service import PlayError, PlayService

DEFAULT_VERSION = "v0.4.1"


@dataclass(frozen=True)
class Release:
    version: str
    asset_name: str
    verify: Callable[[Path], str]
    load: Callable[[Path], object]


# Only reviewed release pins belong here; research candidates use explicit CLI paths.
RELEASES = (
    Release("v0.4.1", v041.ASSET_NAME, v041.verify, v041.load_policy),
    Release("v0.4.0", v040.EXPECTED_NAME, v040.verify, v040.load_policy),
)


class VersionedTables:
    def __init__(self, services: dict[str, PlayService], *, default_version=DEFAULT_VERSION):
        if default_version not in services or any(not isinstance(v, str) or not v for v in services):
            raise ValueError("The default release model and valid version names are required")
        self.services = services
        self.default_version = default_version
        self.lock = threading.RLock()

    def model_info(self):
        return self.services[self.default_version].model_info()

    def model_catalog(self):
        return {"default": self.default_version,
                "models": [{"version": version, **service.model_info()}
                           for version, service in self.services.items()]}

    def _for_session(self, session_id):
        found = []
        for service in self.services.values():
            with service.lock:
                if service.db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                    found.append(service)
        if not found:
            raise PlayError("Unknown session", 404)
        if len(found) != 1:
            raise RuntimeError("Ambiguous persisted session identity")
        return found[0]

    def _check_key(self, key, selected):
        # Preserve the single-table server's global idempotency boundary when
        # switching models: a retry cannot create a second session or wager.
        if not isinstance(key, str):
            raise PlayError("Invalid idempotency key")
        for service in self.services.values():
            if service is selected:
                continue
            with service.lock:
                if service.db.execute("SELECT 1 FROM operations WHERE key=?", (key,)).fetchone():
                    raise PlayError("Idempotency key belongs to another model", 409)

    def create(self, key, body):
        body = dict(body)
        version = body.pop("modelVersion", self.default_version)
        if not isinstance(version, str) or version not in self.services:
            raise PlayError("Choose an available model version")
        with self.lock:
            service = self.services[version]
            self._check_key(key, service)
            return service.create(key, body)

    def _read(self, name, session_id, *args):
        with self.lock:
            return getattr(self._for_session(session_id), name)(session_id, *args)

    def _write(self, name, session_id, key, body):
        with self.lock:
            service = self._for_session(session_id)
            self._check_key(key, service)
            return getattr(service, name)(session_id, key, body)

    def state(self, session_id):
        return self._read("state", session_id)

    def history(self, session_id):
        return self._read("history", session_id)

    def diagnostics(self, session_id, hand_id):
        return self._read("diagnostics", session_id, hand_id)

    def benchmark_result(self, session_id):
        return self._read("benchmark_result", session_id)

    def verify_replay(self, session_id):
        return self._read("verify_replay", session_id)

    def new_hand(self, session_id, key, body):
        return self._write("new_hand", session_id, key, body)

    def act(self, session_id, key, body):
        return self._write("act", session_id, key, body)

    def advance(self, session_id, key, body):
        return self._write("advance", session_id, key, body)

    def end_benchmark(self, session_id, key, body):
        return self._write("end_benchmark", session_id, key, body)

    def close(self):
        for service in self.services.values():
            service.close()


def load_tables(models: Path, data: Path, source_version: str):
    # Validate all artifacts before allocating any reader. Never silently
    # fall back to an older model if the intended default is absent or corrupt.
    for release in RELEASES:
        release.verify(models / release.asset_name)
    services = {}
    try:
        for release in RELEASES:
            version = release.version
            policy = release.load(models / release.asset_name)
            if version == "v0.4.0":
                policy.name = "v0.4.0 · B100M · seed 2026093001"
            services[version] = PlayService(data / version / "private.sqlite", policy,
                                            source_version=source_version)
        return VersionedTables(services)
    except Exception:
        for service in services.values():
            service.close()
        raise
