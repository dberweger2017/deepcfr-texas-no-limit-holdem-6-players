"""Version selection between pinned policies; a session never changes its model."""

import threading
from pathlib import Path

from scripts.verify_v04_model import EXPECTED_NAME, verify as verify_v040
from src.play_api.o_candidate import ASSET_NAME, load_o_candidate, verify as verify_v041
from src.play_api.service import PlayError, PlayService, load_b100m

DEFAULT_VERSION = "v0.4.1"


class VersionedTables:
    def __init__(self, services: dict[str, PlayService]):
        if set(services) != {"v0.4.1", "v0.4.0"}:
            raise ValueError("Both pinned release models are required")
        self.services = services
        self.lock = threading.RLock()

    def model_info(self):
        return self.services[DEFAULT_VERSION].model_info()

    def model_catalog(self):
        return {"default": DEFAULT_VERSION,
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
        version = body.pop("modelVersion", DEFAULT_VERSION)
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
    # Validate both artifacts before allocating either reader. Never silently
    # fall back to an older model if the intended default is absent or corrupt.
    verify_v041(models / ASSET_NAME)
    verify_v040(models / EXPECTED_NAME)
    services = {}
    try:
        for version, loader, name in (("v0.4.1", load_o_candidate, ASSET_NAME),
                                      ("v0.4.0", load_b100m, EXPECTED_NAME)):
            policy = loader(models / name)
            if version == "v0.4.0":
                policy.name = "v0.4.0 · B100M · seed 2026093001"
            services[version] = PlayService(data / version / "private.sqlite", policy,
                                            source_version=source_version)
        return VersionedTables(services)
    except Exception:
        for service in services.values():
            service.close()
        raise
