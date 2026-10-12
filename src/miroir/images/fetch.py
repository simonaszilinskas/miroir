"""Fetch collection images by URL, resized to a small JPEG, with a bounded on-disk cache.

Used when the Mac sends images to an outside model service. The asset bucket is tried
first; when it answers 404 (some file names with accents), the IIIF image service for the
same file is tried, and it resizes on the server. Nothing here decides who may receive an
image: callers that send images out go through `miroir.jobs.embed`, which checks
`publishable`.
"""

from __future__ import annotations

import io
import logging
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote, unquote

import httpx
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from miroir.config import POP_ASSET_HOST, POP_IIIF_IMAGE_HOST
from miroir.http import Retry, send
from miroir.parallel import bounded_map
from miroir.records.model import Image

logger = logging.getLogger(__name__)

#: Longest side, in pixels, of every image sent to an outside model.
LONG_SIDE = 512
JPEG_QUALITY = 90
#: Parallel downloads from POP. Kept low: the bucket and IIIF server are public services,
#: and the Mac should stay responsive.
CONCURRENCY = 4


class FetchSkip(StrEnum):
    """Why an image could not be fetched."""

    MISSING = "missing"
    """404 on the bucket and on the IIIF service."""
    HTTP_ERROR = "http_error"
    """Another error status after retries."""
    UNREADABLE = "unreadable"
    """The bytes are not an image Pillow can open."""


@dataclass(frozen=True, slots=True)
class Fetched:
    image_id: str
    jpeg: bytes | None
    """JPEG bytes, longest side at most LONG_SIDE. None when skipped."""
    skip: FetchSkip | None


def iiif_url(asset_url: str, long_side: int = LONG_SIDE) -> str:
    """IIIF URL of the same file as `asset_url`, scaled on the server to fit `long_side`."""
    prefix = f"{POP_ASSET_HOST}/"
    if not asset_url.startswith(prefix):
        msg = f"not a POP asset URL: {asset_url}"
        raise ValueError(msg)
    path = unquote(asset_url.removeprefix(prefix))
    return (
        f"{POP_IIIF_IMAGE_HOST}/{quote(path, safe='')}/full/!{long_side},{long_side}/0/default.jpg"
    )


def to_jpeg(data: bytes, long_side: int = LONG_SIDE) -> bytes:
    """Decode any image, shrink it to fit `long_side` (never enlarge), encode as RGB JPEG."""
    with PILImage.open(io.BytesIO(data)) as opened:
        opened.thumbnail((long_side, long_side), PILImage.Resampling.LANCZOS)
        rgb = opened.convert("RGB")
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


class ImageCache:
    """Resized JPEGs keyed by `image_id`, capped at `max_bytes`; oldest files are removed first.

    `max_bytes=None` means no cap: no eviction, and no scan of the folder at start (a Modal
    volume holding every screening image, where listing 19,000 files would be slow).
    """

    def __init__(self, directory: Path, max_bytes: int | None) -> None:
        self._dir = directory
        self._max = max_bytes
        self._lock = threading.Lock()
        self._size = 0 if max_bytes is None else sum(p.stat().st_size for p in self._files())

    def get(self, image_id: str) -> bytes | None:
        path = self._path(image_id)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def put(self, image_id: str, jpeg: bytes) -> None:
        path = self._path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(jpeg)
        tmp.replace(path)
        with self._lock:
            self._size += len(jpeg)
            if self._max is not None and self._size > self._max:
                self._evict()

    @property
    def size_bytes(self) -> int:
        return self._size

    def _evict(self) -> None:
        """Remove oldest files until the cache is at 90% of its cap. Caller holds the lock."""
        target = int((self._max or 0) * 0.9)
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime)
        removed = 0
        for path in files:
            if self._size <= target:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            self._size -= size
            removed += 1
        logger.debug("image cache: removed %d old file(s)", removed)

    def _files(self) -> Iterator[Path]:
        return self._dir.glob("*/*.jpg") if self._dir.exists() else iter(())

    def _path(self, image_id: str) -> Path:
        return self._dir / image_id[:2] / f"{image_id}.jpg"


class ImageFetcher:
    """Download, resize and cache collection images, several at a time."""

    def __init__(
        self,
        client: httpx.Client,
        cache: ImageCache,
        *,
        retry: Retry | None = None,
        concurrency: int = CONCURRENCY,
    ) -> None:
        self._client = client
        self._cache = cache
        self._retry = retry or Retry()
        self._concurrency = concurrency

    def fetch(self, image: Image) -> Fetched:
        cached = self._cache.get(image.image_id)
        if cached is not None:
            return Fetched(image.image_id, cached, None)
        result = self._download(image)
        if result.jpeg is not None:
            self._cache.put(image.image_id, result.jpeg)
        return result

    def fetch_all(self, images: Iterable[Image]) -> Iterator[Fetched]:
        """Fetch in parallel; yields results in the order of `images`, a few at a time."""
        yield from bounded_map(self.fetch, images, workers=self._concurrency)

    def _download(self, image: Image) -> Fetched:
        direct = self._get(image.url)
        if direct.status_code == 200:
            return self._convert(image.image_id, direct.content)
        if direct.status_code == 404:
            fallback = self._get(iiif_url(image.url))
            if fallback.status_code == 200:
                logger.debug("%s: bucket 404, IIIF served it", image.image_id)
                return self._convert(image.image_id, fallback.content)
            if fallback.status_code == 404:
                logger.debug("%s: missing on bucket and IIIF: %s", image.image_id, image.url)
                return Fetched(image.image_id, None, FetchSkip.MISSING)
            direct = fallback
        logger.warning("%s: HTTP %d for %s", image.image_id, direct.status_code, direct.url)
        return Fetched(image.image_id, None, FetchSkip.HTTP_ERROR)

    def _get(self, url: str) -> httpx.Response:
        response = send(self._client, self._client.build_request("GET", url), self._retry)
        response.read()
        return response

    def _convert(self, image_id: str, data: bytes) -> Fetched:
        try:
            return Fetched(image_id, to_jpeg(data), None)
        except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as err:
            # A scan over Pillow's pixel limit (about 179 million) is skipped like a broken file.
            logger.warning("%s: cannot decode image: %s", image_id, err)
            return Fetched(image_id, None, FetchSkip.UNREADABLE)


def count_skips(results: Iterable[Fetched]) -> Counter[FetchSkip]:
    return Counter(r.skip for r in results if r.skip is not None)
