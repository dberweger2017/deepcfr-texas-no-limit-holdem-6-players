"""Version selection between pinned policies; a session never changes its model."""

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.policies import v040, v041
from src.play_api.service import PlayError, PlayService, _model_info

DEFAULT_VERSION = "v0.4.1"


@dataclass(frozen=True)
class Release:
    version: str
    asset_name: str
    verify: Callable[[Path], str]
    load: Callable[[Path], object]
    manifest_sha256: str

    def identity(self, policy) -> dict:
        path = Path(__file__).resolve().parents[2] / 'configs/play/release-manifests' / f'{self.version}.json'
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.manifest_sha256:
            raise ValueError('Pinned release manifest hash differs')
        manifest = json.loads(data)
        model = manifest['model']
        info = _model_info(policy)
        if (manifest.get('release', manifest.get('candidate')) != self.version
                or any(model[key] != info[key] for key in ('sha256', 'game', 'schema', 'format'))
                or model['file'] != self.asset_name or model['players'] != 2 or model['raise_cap'] is not None):
            raise ValueError('Release manifest differs from loaded model')
        if not info['name'].startswith(self.version + ' · '):
            info['name'] = f"{self.version} · {info['name']}"
        base = 'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/download'
        return {'version': self.version, **info, 'manifestSha256': self.manifest_sha256,
                'manifestUrl': f'{base}/{self.version}/release-manifest.json'}


# Only reviewed release pins belong here; research candidates use explicit CLI paths.
RELEASES = (
    Release("v0.4.1", v041.ASSET_NAME, v041.verify, v041.load_policy,
            "8d1a85bea7fd2bad3d4a8526ad95e858239fa162d14e19369a47c739097659f5"),
    Release("v0.4.0", v040.EXPECTED_NAME, v040.verify, v040.load_policy,
            "1383de5fe5f60ef829ad0410e75bd871a936e5297c8030f534f69eb69c2362af"),
)


class VersionedTables:
    def __init__(self, services: dict[str, PlayService], *, default_version=DEFAULT_VERSION,
                 spectator=None, identities=None):
        if default_version not in services or any(not isinstance(v, str) or not v for v in services):
            raise ValueError("The default release model and valid version names are required")
        self.services = services
        self.default_version = default_version
        self.spectator = spectator
        self.identities = identities
        self.stores = [*services.values(), *([spectator] if spectator is not None else [])]
        self.lock = threading.RLock()

    def model_info(self):
        return self.services[self.default_version].model_info()

    def model_catalog(self):
        return {"default": self.default_version,
                "models": [self.identities[version] if self.identities else
                           {"version": version, **service.model_info()}
                           for version, service in self.services.items()],
                "spectatorAvailable": self.spectator is not None}

    def _for_session(self, session_id):
        found = []
        for service in self.stores:
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
        for service in self.stores:
            if service is selected:
                continue
            with service.lock:
                if service.db.execute("SELECT 1 FROM operations WHERE key=?", (key,)).fetchone():
                    raise PlayError("Idempotency key belongs to another model", 409)

    def create(self, key, body):
        body = dict(body)
        if body.get('sessionType') == 'spectator':
            if self.spectator is None:
                raise PlayError('Spectator mode requires pinned release models')
            with self.lock:
                self._check_key(key, self.spectator)
                return self.spectator.create(key, body)
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
        for service in self.stores:
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
        from src.play_api.spectator import SpectatorService
        identities = {release.version: release.identity(services[release.version].policy)
                      for release in RELEASES}
        spectator = SpectatorService(data / 'spectator' / 'private.sqlite',
                                     {v: s.policy for v, s in services.items()}, identities,
                                     source_version=source_version)
        return VersionedTables(services, spectator=spectator, identities=identities)
    except Exception:
        for service in services.values():
            service.close()
        raise
