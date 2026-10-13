"""How fast the POP asset bucket serves images (technical spec §12, question 2).

`fetch_timed` runs inside a Modal CPU container: it downloads a list of image URLs with a
given number of parallel requests, once each and without retries, so every failure counts.
It times each download and the resize the image volume fill does (`to_jpeg`, 512 px), and
hashes the bytes so the Mac can find the same file under two names. The Mac picks the
images, sends the batches and reads the results (`miroir.jobs.fetch_probe`).
"""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from miroir.images.fetch import iiif_url, to_jpeg
from miroir.images.phash import phash

#: Seconds before one download gives up. A file that takes longer counts as a timeout.
TIMEOUT_S = 60.0
USER_AGENT = "miroir-research/0.1 (French heritage visual search prototype; one-off speed test)"


@dataclass(frozen=True, slots=True)
class TimedFetch:
    image_id: str
    status: int
    """HTTP status of the bucket; 0 when no answer came (timeout, connection error)."""
    fallback_status: int
    """HTTP status of the IIIF service, tried after a bucket 404; 0 when not tried."""
    size_bytes: int
    """Bytes of the file served (bucket, else IIIF)."""
    fetch_s: float
    """From the first request to the last byte, fallback included."""
    resize_s: float
    """Decode and shrink to 512 px JPEG, as the image volume fill does."""
    width: int
    height: int
    sha1: str
    """Of the bytes served; empty when nothing was."""
    phash: int
    """Perceptual hash of the 512 px JPEG (0 when none): finds one picture saved twice."""
    error: str
    """Why no image came out: "", "timeout", "connection", "http", "unreadable"."""


@dataclass(frozen=True, slots=True)
class TimedBatch:
    concurrency: int
    wall_s: float
    started: float
    """Container clock (Unix seconds) when the batch began, to line up parallel batches."""
    fetches: tuple[TimedFetch, ...]


def fetch_timed(
    items: Sequence[tuple[str, str]], concurrency: int, not_before: float = 0.0
) -> TimedBatch:
    """Download `(image_id, url)` pairs, `concurrency` at a time. Runs in a Modal container.

    Waits until Unix time `not_before`, so batches sent to several containers start together.
    """
    time.sleep(max(0.0, not_before - time.time()))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    headers = {"User-Agent": USER_AGENT}
    started = time.time()
    start = time.monotonic()
    with (
        httpx.Client(timeout=TIMEOUT_S, limits=limits, headers=headers) as http,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):

        def one(item: tuple[str, str]) -> TimedFetch:
            return _fetch(http, item[0], item[1])

        fetches = tuple(pool.map(one, items))
    return TimedBatch(concurrency, time.monotonic() - start, started, fetches)


def _fetch(http: httpx.Client, image_id: str, url: str) -> TimedFetch:
    start = time.monotonic()
    status, fallback, data, error = _get(http, url)
    if status == 404:
        fallback, _, data, error = _get(http, iiif_url(url))
    fetch_s = time.monotonic() - start
    if data is None:
        return TimedFetch(image_id, status, fallback, 0, fetch_s, 0.0, 0, 0, "", 0, error)
    start = time.monotonic()
    width = height = fingerprint = 0
    small = b""
    try:
        with PILImage.open(io.BytesIO(data)) as opened:
            width, height = opened.size
        small = to_jpeg(data)
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError):
        error = "unreadable"
    resize_s = time.monotonic() - start
    if small:
        fingerprint = phash(small)
    digest = hashlib.sha1(data, usedforsecurity=False).hexdigest()
    return TimedFetch(
        image_id,
        status,
        fallback,
        len(data),
        fetch_s,
        resize_s,
        width,
        height,
        digest,
        fingerprint,
        error,
    )


def _get(http: httpx.Client, url: str) -> tuple[int, int, bytes | None, str]:
    """(status, 0, body or None, error)."""
    try:
        response = http.get(url)
    except httpx.TimeoutException:
        return 0, 0, None, "timeout"
    except httpx.TransportError:
        return 0, 0, None, "connection"
    if response.status_code != 200:
        return response.status_code, 0, None, "http"
    return 200, 0, response.content, ""
