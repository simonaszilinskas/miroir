"""Download every POP image to Object Storage: one shard, run on a Scaleway CPU machine.

`miroir scw download --shard k --of n` (started by `miroir.jobs.scw_bulk`) reads the image
list from the bucket, skips the images already stored, and for each other one fetches the
POP file (the IIIF service when the asset bucket answers 404), shrinks it to at most
`FULL_LONG_SIDE` px on its long side (never enlarges it) and stores it at `full_key`. POP
serves one file per image: the Mémoire `_p` file is the largest there is (checked 2026-11-02:
the asset folders hold that file alone, and the IIIF service gives it as its full size).

Politeness: a busy answer (429, 503) or a dropped connection pauses every thread of the
process (`Pacer`), longer each time it happens again soon, before the request is tried
again. Every image that fails is written with its collection and why (`Outcome`).
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pyarrow.parquet as pq
from PIL import Image as PILImage
from PIL import ImageFile, UnidentifiedImageError

from miroir.config import Settings
from miroir.images.fetch import iiif_url, to_jpeg
from miroir.jobs.bulk import (
    FULL_LONG_SIDE,
    IMAGE_LIST_KEY,
    JobState,
    JobStatus,
    failures_key,
    full_key,
    shard_of,
    shard_prefixes,
    status_key,
)
from miroir.jobs.objects import Bucket, bucket_name, listing, s3_client
from miroir.parallel import unordered_map

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

JOB = "download"
#: Tries per URL; busy answers and dropped connections are tried again after the pause.
ATTEMPTS = 5
#: Answers that mean the server wants us to slow down.
BUSY_STATUS = frozenset({429, 502, 503, 504})
#: Seconds between two status files.
STATUS_EVERY_S = 30.0


@dataclass(frozen=True, slots=True)
class Item:
    image_id: str
    url: str
    collection: str


@dataclass(frozen=True, slots=True)
class Outcome:
    image_id: str
    collection: str
    reason: str
    """"ok", "ok_truncated" (stored, decoded from a file cut short), or why the image is
    not stored: "missing" (404 on the bucket and on IIIF),
    "http_<status>", "not_an_image" (a web page instead), "unreadable", "connection" or
    "store"."""
    detail: str = ""
    via: str = ""
    """"asset" or "iiif" when fetched."""
    width: int = 0
    """The fetched file's size, before shrinking."""
    height: int = 0
    stored_bytes: int = 0


class Pacer:
    """One pause shared by every thread: a busy answer stops all requests for a while.

    The pause starts at `first_s` and doubles, up to `max_s`, while busy answers keep coming
    within `calm_s` of the last one; after a calm spell it starts small again."""

    def __init__(
        self,
        first_s: float = 5.0,
        max_s: float = 300.0,
        calm_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._first = first_s
        self._max = max_s
        self._calm = calm_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._until = 0.0
        self._last = -1e18
        self._pause = 0.0
        self.trips = 0

    def wait(self) -> None:
        while (left := self._until - self._clock()) > 0:
            self._sleep(min(left, 5.0))

    def trip(self, why: str) -> None:
        with self._lock:
            now = self._clock()
            if now < self._until:  # already pausing for this burst
                return
            self._pause = (
                self._first if now - self._last > self._calm else min(self._pause * 2, self._max)
            )
            self._last = now
            self._until = now + self._pause
            self.trips += 1
        logger.warning("server busy (%s): every thread pauses %.0f s", why, self._pause)


class Downloader:
    def __init__(self, http: httpx.Client, s3: S3Client, bucket: str, pacer: Pacer) -> None:
        self._http = http
        self._s3 = s3
        self._bucket = bucket
        self._pacer = pacer

    def get(self, url: str) -> httpx.Response | str:
        """The final answer for `url`, or "connection" when every try failed to connect."""
        for attempt in range(ATTEMPTS):
            self._pacer.wait()
            try:
                response = self._http.get(url)
            except httpx.TransportError as err:
                self._pacer.trip(type(err).__name__)
                if attempt == ATTEMPTS - 1:
                    return "connection"
                continue
            if response.status_code in BUSY_STATUS and attempt < ATTEMPTS - 1:
                self._pacer.trip(f"HTTP {response.status_code}")
                continue
            return response
        return "connection"

    def fetch(self, item: Item) -> tuple[bytes | None, str, Outcome | None]:
        """The file's bytes and where they came from, or the failure."""
        first = self.get(item.url)
        if isinstance(first, httpx.Response) and first.status_code == 200:
            return first.content, "asset", None
        if isinstance(first, httpx.Response) and first.status_code == 404:
            second = self.get(iiif_url(item.url, FULL_LONG_SIDE))
            if isinstance(second, httpx.Response) and second.status_code == 200:
                return second.content, "iiif", None
            if isinstance(second, httpx.Response) and second.status_code == 404:
                return None, "", Outcome(item.image_id, item.collection, "missing")
            first = second
        if isinstance(first, str):
            return None, "", Outcome(item.image_id, item.collection, first)
        reason = f"http_{first.status_code}"
        return None, "", Outcome(item.image_id, item.collection, reason, str(first.url)[:200])

    def one(self, item: Item) -> Outcome:
        data, via, failed = self.fetch(item)
        if failed is not None:
            return failed
        assert data is not None
        try:
            jpeg, width, height, cut = shrink(data)
        except (
            UnidentifiedImageError,
            OSError,
            PILImage.DecompressionBombError,
            ValueError,
        ) as err:
            reason = "not_an_image" if data.lstrip()[:1] == b"<" else "unreadable"
            return Outcome(item.image_id, item.collection, reason, str(err)[:200], via)
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=full_key(item.image_id),
                Body=jpeg,
                ContentType="image/jpeg",
                Metadata={"source-width": str(width), "source-height": str(height), "via": via},
            )
        except Exception as err:
            return Outcome(item.image_id, item.collection, "store", str(err)[:200], via)
        reason = "ok_truncated" if cut else "ok"
        return Outcome(item.image_id, item.collection, reason, "", via, width, height, len(jpeg))


def shrink(data: bytes) -> tuple[bytes, int, int, bool]:
    """The file as stored, its size, and whether its end was missing.

    Kept as it is when it is a whole JPEG Pillow reads as RGB or grey within
    `FULL_LONG_SIDE`; else shrunk (never enlarged) and saved as RGB JPEG. A file cut short
    by a few bytes (its end marker, often) is decoded as far as it goes, as browsers do."""
    with PILImage.open(io.BytesIO(data)) as opened:
        width, height = opened.size
        small = max(width, height) <= FULL_LONG_SIDE
        whole = opened.format == "JPEG" and opened.mode in {"RGB", "L"} and small
        try:
            opened.load()
        except OSError as err:
            if "truncated" not in str(err):
                raise
            return _truncated(data), width, height, True
    return (data if whole else to_jpeg(data, FULL_LONG_SIDE)), width, height, False


def _truncated(data: bytes) -> bytes:
    """`to_jpeg` of a file cut short. Pillow's switch for it is global: it is on only while
    one such file decodes."""
    with _TRUNCATED:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return to_jpeg(data, FULL_LONG_SIDE)
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False


_TRUNCATED = threading.Lock()


def read_image_list(s3: S3Client, bucket: str, shard: int, of: int) -> list[Item]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "images.parquet"
        s3.download_file(bucket, IMAGE_LIST_KEY, str(path))
        table = pq.read_table(path, columns=["image_id", "url", "collection"]).to_pydict()  # pyright: ignore[reportUnknownMemberType]
    return [
        Item(i, u, c)
        for i, u, c in zip(table["image_id"], table["url"], table["collection"], strict=True)
        if shard_of(str(i), of) == shard
    ]


def stored_ids(s3: S3Client, bucket: str, shard: int, of: int) -> set[str]:
    return {Path(s.key).stem for p in shard_prefixes(shard, of) for s in listing(s3, bucket, p)}


def run_download(settings: Settings, *, shard: int, of: int, workers: int) -> JobStatus:
    """Fetch and store this shard's missing images. Returns the final status."""
    s3 = s3_client(settings)
    bucket = bucket_name(settings, Bucket.IMAGES)
    items = read_image_list(s3, bucket, shard, of)
    have = stored_ids(s3, bucket, shard, of)
    todo = [i for i in items if i.image_id not in have]
    status = JobStatus(JOB, shard, of, total=len(todo), note=f"{len(have):,} already stored")
    logger.info(
        "shard %d/%d: %d images, %d stored, %d to do", shard, of, len(items), len(have), len(todo)
    )
    failures = Path(tempfile.gettempdir()) / f"failures-{JOB}-{shard}.jsonl"
    failures.write_text("")
    pacer = Pacer()
    counts: Counter[str] = Counter()
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    headers = {"User-Agent": "miroir (ministere de la Culture, recherche visuelle POP)"}
    last = 0.0
    with (
        httpx.Client(timeout=60.0, limits=limits, follow_redirects=True, headers=headers) as http,
        failures.open("a") as out,
    ):
        downloader = Downloader(http, s3, bucket, pacer)
        for outcome in unordered_map(downloader.one, todo, workers=workers):
            counts[outcome.reason] += 1
            status = replace(status, done=status.done + 1)
            if outcome.reason not in {"ok", "ok_truncated"}:
                out.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
            if time.time() - last > STATUS_EVERY_S:
                out.flush()
                status = _report(s3, bucket, status, counts, pacer, failures)
                last = time.time()
        out.flush()
    status = _report(s3, bucket, replace(status, state=JobState.DONE), counts, pacer, failures)
    logger.info("%s", status.line())
    return status


def _report(  # noqa: PLR0917 (one call site)
    s3: S3Client,
    bucket: str,
    status: JobStatus,
    counts: Counter[str],
    pacer: Pacer,
    failures: Path,
) -> JobStatus:
    status = status.now(counts={**dict(counts), "busy_pauses": pacer.trips})
    s3.put_object(Bucket=bucket, Key=status_key(JOB, status.shard), Body=status.to_json().encode())
    s3.upload_file(str(failures), bucket, failures_key(JOB, status.shard))
    logger.info("%s", status.line())
    return status
