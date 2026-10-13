"""A bounded HTTPS JSON transport for external services.

One request at a time, a fixed timeout, a minimum interval between requests and a request
cap. Nothing is retried: a POST that times out may or may not have acted, so it raises
`AmbiguousRequest` and the caller ends that hand. Errors name the endpoint and status only;
request and response bodies can carry tokens or credentials and never enter messages.
"""

import json
import socket
import time
import urllib.error
import urllib.request


class AmbiguousRequest(Exception):
    """The request may have reached the service, but no complete response came back."""


class ServiceError(Exception):
    """The service answered with an error status or an unreadable body."""


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
            raise ServiceError(f'{endpoint} returned HTTP {error.code}') from None
        except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError):
            raise AmbiguousRequest(f'{endpoint} gave no complete response') from None
        finally:
            self._last = time.monotonic()
        self.latencies.append(self._last - started)
        try:
            return json.loads(raw)
        except ValueError:
            raise ServiceError(f'{endpoint} returned a body that is not JSON') from None
