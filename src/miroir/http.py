"""HTTP with retries: back off when a server is busy or limits us, give up after a few tries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

#: Answers that mean "try again later": timeout, too many requests, server trouble.
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class Retry:
    attempts: int = 6
    backoff_s: float = 1.0
    """First wait; it doubles after each failed try, up to `max_wait_s`."""
    max_wait_s: float = 60.0
    sleep: Callable[[float], None] = time.sleep


def send(client: httpx.Client, request: httpx.Request, retry: Retry) -> httpx.Response:
    """Send `request`, retrying busy and limit answers and dropped connections.

    Returns the last response, whatever its status, so callers decide what a 404 means.
    Raises the transport error if the connection fails on every try.
    """
    for attempt in range(1, retry.attempts + 1):
        last = attempt == retry.attempts
        try:
            response = client.send(request)
        except httpx.TransportError as err:
            if last:
                raise
            wait = _wait(retry, attempt, None)
            logger.debug("%s %s: %s; retry in %.1f s", request.method, request.url, err, wait)
        else:
            if response.status_code not in RETRY_STATUS or last:
                return response
            wait = _wait(retry, attempt, response.headers.get("retry-after"))
            logger.debug(
                "%s %s: HTTP %d; retry in %.1f s",
                request.method,
                request.url,
                response.status_code,
                wait,
            )
            response.close()
        retry.sleep(wait)
    msg = "unreachable: the last attempt always returns or raises"
    raise AssertionError(msg)


def _wait(retry: Retry, attempt: int, retry_after: str | None) -> float:
    backoff = min(retry.backoff_s * 2 ** (attempt - 1), retry.max_wait_s)
    if retry_after is not None:
        try:
            return min(max(float(retry_after), backoff), retry.max_wait_s)
        except ValueError:
            pass
    return backoff
