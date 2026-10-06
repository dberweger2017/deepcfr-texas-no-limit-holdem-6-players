"""Loopback-only HTTP entrypoint for the local HU20 table."""

import argparse
import hmac
import json
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from src.play_api.service import PlayError, PlayService, load_b100m

ASSETS = Path(__file__).resolve().parents[2] / "apps" / "poker-web"
ASSET_TYPES = {"/": ("index.html", "text/html; charset=utf-8"),
               "/app.js": ("app.js", "text/javascript; charset=utf-8"),
               "/style.css": ("style.css", "text/css; charset=utf-8")}
SESSION = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})$")
HISTORY = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/history$")
HAND = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/hands$")
ACTION = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/actions$")
ADVANCE = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/advance$")
DIAGNOSTICS = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/hands/([A-Za-z0-9_-]{24})/diagnostics$")
BENCHMARK_RESULT = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/benchmark/result$")
BENCHMARK_EXPORT = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/benchmark/export$")
BENCHMARK_END = re.compile(r"^/api/sessions/([A-Za-z0-9_-]{24})/benchmark/end$")


def token_file(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.is_symlink():
        raise ValueError("Access token path must be a regular file")
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write(secrets.token_urlsafe(32) + "\n")
    if not path.is_file():
        raise ValueError("Access token path must be a regular file")
    os.chmod(path, 0o600)
    token = path.read_text().strip()
    if not token:
        raise ValueError("Access token file is empty")
    return token


def handler_for(service, token, port):
    allowed_host = f"127.0.0.1:{port}"
    allowed_origin = f"http://{allowed_host}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, status, value, *, download_name=None):
            data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'")
            if download_name is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            self.wfile.write(data)

        def _error(self, error):
            self._send(error.status, {"error": str(error)})

        def _guard(self, mutation=False):
            if self.headers.get("Host") != allowed_host:
                raise PlayError("Host not allowed", 403)
            origin = self.headers.get("Origin")
            if origin is not None and origin != allowed_origin:
                raise PlayError("Origin not allowed", 403)
            if mutation and origin is None and self.headers.get("X-Play-Token") is None:
                raise PlayError("Access denied", 403)

        def _auth(self):
            supplied = self.headers.get("X-Play-Token", "")
            if not hmac.compare_digest(supplied, token):
                raise PlayError("Access denied", 403)

        def _body(self):
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise PlayError("Expected application/json", 415)
            value = self.headers.get("Content-Length", "")
            if not value.isdecimal() or not 0 < int(value) <= 8192:
                raise PlayError("Invalid request length", 413)
            try:
                body = json.loads(self.rfile.read(int(value)))
            except (ValueError, UnicodeDecodeError):
                raise PlayError("Invalid JSON") from None
            if not isinstance(body, dict):
                raise PlayError("Expected a JSON object")
            return body

        def do_GET(self):
            try:
                self._guard()
                path = urlsplit(self.path).path
                if path in ASSET_TYPES and self.path == path:
                    name, kind = ASSET_TYPES[path]
                    asset = ASSETS / name
                    if asset.is_symlink() or not asset.is_file():
                        raise PlayError("Asset unavailable", 404)
                    data = asset.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._auth()
                if self.path != path:
                    raise PlayError("Unknown endpoint", 404)
                if path == "/api/model":
                    self._send(200, service.model_info())
                elif match := SESSION.fullmatch(path):
                    self._send(200, service.state(match[1]))
                elif match := HISTORY.fullmatch(path):
                    self._send(200, service.history(match[1]))
                elif match := BENCHMARK_RESULT.fullmatch(path):
                    self._send(200, service.benchmark_result(match[1]))
                elif match := BENCHMARK_EXPORT.fullmatch(path):
                    report = service.benchmark_result(match[1])
                    self._send(200, report, download_name=f'hu20-benchmark-{report["benchmarkId"]}.json')
                elif match := DIAGNOSTICS.fullmatch(path):
                    self._send(200, service.diagnostics(match[1], match[2]))
                else:
                    raise PlayError("Unknown endpoint", 404)
            except PlayError as error:
                self._error(error)
            except Exception:
                self._send(500, {"error": "Internal service error"})

        def do_POST(self):
            try:
                self._guard(mutation=True)
                self._auth()
                path = urlsplit(self.path).path
                if self.path != path:
                    raise PlayError("Unknown endpoint", 404)
                body = self._body()
                key = self.headers.get("Idempotency-Key")
                if path == "/api/sessions":
                    response = service.create(key, body)
                elif match := HAND.fullmatch(path):
                    response = service.new_hand(match[1], key, body)
                elif match := ACTION.fullmatch(path):
                    response = service.act(match[1], key, body)
                elif match := ADVANCE.fullmatch(path):
                    response = service.advance(match[1], key, body)
                elif match := BENCHMARK_END.fullmatch(path):
                    response = service.end_benchmark(match[1], key, body)
                else:
                    raise PlayError("Unknown endpoint", 404)
                self._send(200, response)
            except PlayError as error:
                self._error(error)
            except Exception:
                self._send(500, {"error": "Internal service error"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--policy", type=Path)
    opponent.add_argument("--o-candidate", type=Path,
                          help="Opt-in pinned O1B average candidate; does not change v0.4.0")
    opponent.add_argument("--uniform-random", action="store_true",
                          help="Benchmark-only control over the same restricted HU20 menu; loads no model")
    parser.add_argument("--data-dir", type=Path, default=Path("results/play-web"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--source-version", default="unknown")
    parser.add_argument("--verify-session")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1–65535")
    os.umask(0o077)
    if args.uniform_random:
        from src.play_api.uniform_random import UniformRestrictedPolicy
        policy = UniformRestrictedPolicy()
    elif args.o_candidate:
        from src.play_api.o_candidate import load_o_candidate
        policy = load_o_candidate(args.o_candidate)
    else:
        policy = load_b100m(args.policy)
    service = PlayService(args.data_dir / "private.sqlite", policy, source_version=args.source_version)
    try:
        if args.verify_session:
            print(f"Verified {service.verify_replay(args.verify_session)} completed hands")
            return 0
        token_path = args.data_dir / "access.token"
        token = token_file(token_path)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service, token, args.port))
        print(f"Local table: http://127.0.0.1:{args.port}/", flush=True)
        print(f"Enter the access token stored at {token_path}; it is not put in a URL or log.", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
