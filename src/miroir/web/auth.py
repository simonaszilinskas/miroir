"""Access behind one shared password (functional spec §8), signed cookies, and rate limits.

A session is a random id with an expiry, signed with HMAC-SHA256. Nothing is stored on the
server, so a container restart keeps people logged in. Rate limits are kept in memory: the
web container runs alone (max one container), so one counter sees every request.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

COOKIE = "miroir_session"
#: A login lasts a week.
SESSION_SECONDS = 7 * 24 * 3600
#: Searches per session per minute (each one may wake a GPU).
SEARCHES_PER_MINUTE = 30
#: Login attempts per client address per minute.
LOGINS_PER_MINUTE = 5
#: Letters used for generated passwords: no 0/o, 1/l/i, easy to type on a phone.
_PASSWORD_LETTERS = "abcdefghjkmnpqrstuvwxyz23456789"  # noqa: S105 (an alphabet)


def new_password(groups: int = 3, size: int = 4) -> str:
    """A random password like `k7rq-m2xp-9fha` (about 60 bits with the defaults)."""
    return "-".join(
        "".join(secrets.choice(_PASSWORD_LETTERS) for _ in range(size)) for _ in range(groups)
    )


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def password_ok(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.strip().encode(), expected.encode())


@dataclass(frozen=True, slots=True)
class Sessions:
    """Signs and checks session tokens."""

    key: bytes
    lifetime_s: int = SESSION_SECONDS

    def issue(self, now: float | None = None) -> str:
        expires = int((time.time() if now is None else now) + self.lifetime_s)
        body = f"{secrets.token_urlsafe(12)}.{expires}"
        return f"{body}.{self._sign(body)}"

    def check(self, token: str | None, now: float | None = None) -> str | None:
        """The session id when the token is genuine and current, else None."""
        if not token or token.count(".") != 2:
            return None
        session_id, expires, signature = token.split(".")
        body = f"{session_id}.{expires}"
        if not hmac.compare_digest(signature, self._sign(body)):
            return None
        if not expires.isdigit() or int(expires) < (time.time() if now is None else now):
            return None
        return session_id

    def _sign(self, body: str) -> str:
        digest = hmac.new(self.key, body.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")


class RateLimit:
    """At most `limit` events per key in any `window_s` seconds (sliding window)."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        at = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= at - self.window_s:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(at)
            if len(self._events) > 10_000:  # forget idle keys now and then
                self._events = {k: v for k, v in self._events.items() if v}
            return True
