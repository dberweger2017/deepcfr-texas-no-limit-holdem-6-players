"""Keep the isolated user test session alive only until its fixed cutoff."""
import datetime
import time

deadline = datetime.datetime.fromisoformat('2026-10-15T04:50:00+00:00').timestamp()
while time.time() < deadline:
    time.sleep(min(30, deadline - time.time()))
