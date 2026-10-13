"""A bounded HTTPS JSON transport for external services.

One request at a time, a fixed timeout, a minimum interval between requests and a request
cap. Nothing is retried: a POST that times out may or may not have acted, so it raises
`AmbiguousRequest` and the caller ends that hand. Errors name the endpoint and status, plus a
JSON error body's short `error_msg` with anything token-like removed; request and response
bodies can carry tokens or credentials and never enter messages otherwise.
"""

import json
import re
import socket
import time
import urllib.error
import urllib.request


class AmbiguousRequest(Exception):
    """The request may have reached the service, but no complete response came back."""


class ServiceError(Exception):
    """The service answered with an error status or an unreadable body."""
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def service_reason(raw: bytes) -> str | None:
    """A JSON error body's `error_msg`, shortened, with runs that could be tokens removed."""
    try:
        message = json.loads(raw).get('error_msg')
    except (ValueError, AttributeError):
        return None
    if not isinstance(message, str):
        return None
    return re.sub(r'[A-Za-z0-9_-]{16,}', '...', message)[:120] or None


class HttpTransport:
    def __init__(self, base: str, *, timeout: float = 10.0, min_interval: float = 0.5, max_requests: int = 1000):
        self.base, self.timeout, self.min_interval, self.max_requests = base.rstrip('/'), timeout, min_interval, max_requests
        self.requests = 0
        self.latencies = []
        self._last = None

    def post(self, endpoint: str, body: dict) -> dict:
        if self.requests >= self.max_requests:
            raise ServiceError('Request cap reached')
        if self._last is not None:
            time.sleep(max(0.0, self._last + self.min_interval - time.monotonic()))
        self.requests += 1
        request = urllib.request.Request(self.base + endpoint, json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                reason = service_reason(error.read())
            except OSError:
                reason = None
            raise ServiceError(f'{endpoint} returned HTTP {error.code}' + (f': {reason}' if reason else ''),
                               status=error.code) from None
        except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError):
            raise AmbiguousRequest(f'{endpoint} gave no complete response') from None
        finally:
            self._last = time.monotonic()
        self.latencies.append(self._last - started)
        try:
            return json.loads(raw)
        except ValueError:
            raise ServiceError(f'{endpoint} returned a body that is not JSON') from None
