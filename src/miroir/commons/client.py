"""A polite, cached client for Wikidata's SPARQL endpoint, the Commons API and Commons files.

Wikimedia's rules for bots: name the client and give a contact (`WIKIMEDIA_USER_AGENT`), send
`maxlag` with every API call and wait when the servers say they lag, and keep the pace low.
Here every request goes out one at a time, at most `MAX_PER_SECOND` a second. Every answer is
kept on disk (`Settings.commons_cache_dir`), so a rebuild sends nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
from PIL import UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from miroir.config import WIKIMEDIA_USER_AGENT
from miroir.http import Retry, send
from miroir.images.fetch import to_jpeg

logger = logging.getLogger(__name__)

#: Wikimedia asks for no more than a few requests a second from one client.
MAX_PER_SECOND = 4
#: Seconds of replication lag past which the Commons API refuses us (Wikimedia's advice: 5).
MAXLAG_S = 5
#: Times to wait and ask again when the API says it lags.
MAXLAG_TRIES = 10


class WikimediaError(RuntimeError):
    """A Wikimedia service answered with an error we cannot wait out."""


def open_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": WIKIMEDIA_USER_AGENT},
        timeout=httpx.Timeout(120.0, connect=20.0),
        follow_redirects=True,
    )


class WikimediaClient:
    """GET JSON or images, one request at a time, cached on disk forever."""

    def __init__(
        self,
        http: httpx.Client,
        cache_dir: Path,
        *,
        retry: Retry | None = None,
        max_per_second: float = MAX_PER_SECOND,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._cache = cache_dir
        self._retry = retry or Retry(attempts=6, backoff_s=2.0)
        self._interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._last = -1e9
        self.requests = 0
        """Requests sent (cache misses), for the log."""

    def get_json(self, url: str, params: Mapping[str, str]) -> object:
        """The decoded JSON answer to a GET, from the cache when we asked before."""
        path = self._cache / "api" / f"{_key(url, params)}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as cached:
                return json.load(cached)
        payload = self._fetch_json(url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False)
        tmp.replace(path)
        return payload

    def get_jpeg(self, url: str) -> bytes | None:
        """An image resized to 512 px on its long side, or None when it is gone or unreadable."""
        path = self._cache / "thumbs" / f"{_key(url, {})}.jpg"
        missing = path.with_suffix(".missing")
        if path.exists():
            return path.read_bytes()
        if missing.exists():
            return None
        response = self._get(url, {})
        path.parent.mkdir(parents=True, exist_ok=True)
        if response.status_code != 200:
            logger.warning("HTTP %d for %s", response.status_code, url)
            if response.status_code == 404:
                missing.touch()
            return None
        try:
            jpeg = to_jpeg(response.content)
        except (UnidentifiedImageError, OSError) as err:
            logger.warning("cannot decode %s: %s", url, err)
            missing.touch()
            return None
        tmp = path.with_suffix(".part")
        tmp.write_bytes(jpeg)
        tmp.replace(path)
        return jpeg

    def _fetch_json(self, url: str, params: Mapping[str, str]) -> object:
        for _ in range(MAXLAG_TRIES):
            response = self._get(url, params)
            if response.status_code != 200:
                msg = f"GET {url}: HTTP {response.status_code}: {response.text[:300]}"
                raise WikimediaError(msg)
            try:
                payload: object = response.json()
            except ValueError as err:
                msg = f"GET {url}: answer is not JSON"
                raise WikimediaError(msg) from err
            code = _error_code(payload)
            if code is None:
                return payload
            if code != "maxlag":
                msg = f"GET {url} {dict(params)}: error {code}: {str(payload)[:300]}"
                raise WikimediaError(msg)
            wait = float(response.headers.get("retry-after", MAXLAG_S))
            logger.info("Commons servers lag; waiting %.0f s", wait)
            self._sleep(wait)
        msg = f"GET {url}: still lagging after {MAXLAG_TRIES} tries"
        raise WikimediaError(msg)

    def _get(self, url: str, params: Mapping[str, str]) -> httpx.Response:
        wait = self._last + self._interval - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._last = self._clock()
        self.requests += 1
        request = self._http.build_request("GET", url, params=dict(params))
        response = send(self._http, request, self._retry)
        response.read()
        return response


def _key(url: str, params: Mapping[str, str]) -> str:
    text = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    digest = hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()
    return f"{digest[:2]}/{digest}"


class _Error(BaseModel):
    code: str = "unknown"


class _ErrorBody(BaseModel):
    error: _Error | None = None


def _error_code(payload: object) -> str | None:
    """The API's error code when it answered 200 with an error (as it does for maxlag)."""
    if not isinstance(payload, dict):
        return None
    try:
        body = _ErrorBody.model_validate(payload)
    except ValidationError:
        return "unknown"
    return None if body.error is None else body.error.code
