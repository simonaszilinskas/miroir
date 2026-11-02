"""Scaleway Object Storage (S3): the buckets that replace the Modal volumes.

| Bucket | Modal volume | What it holds |
|---|---|---|
| `<prefix>-weights` | `miroir-weights` | Hugging Face and torch hub caches (`hf/`, `torch/`) |
| `<prefix>-benchmarks` | `miroir-benchmarks` | AmsterTime |
| `<prefix>-images` | `miroir-images` | POP images at 512 px, `<id[:2]>/<id>.jpg` |
| `<prefix>-index` | `miroir-web-index` | The website's index bundles, `<name>/<file>` |

The GPU machine mirrors a bucket into a local folder at the old mount path (`/weights`, ...),
so the model code reads files as it did on Modal, and sends new files back when a call ends.
Used on the Mac (bucket creation, index upload) and on the machine (`miroir.remote.worker`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from miroir.config import Settings
from miroir.parallel import bounded_map

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

#: Parallel transfers between a folder and a bucket.
TRANSFERS = 32
UPLOAD_ATTEMPTS = 5


class Bucket(StrEnum):
    WEIGHTS = "weights"
    BENCHMARKS = "benchmarks"
    IMAGES = "images"
    INDEX = "index"


def bucket_name(settings: Settings, bucket: Bucket) -> str:
    return f"{settings.scw_bucket_prefix}-{bucket.value}"


def endpoint(settings: Settings) -> str:
    return f"https://s3.{settings.scw_region}.scw.cloud"


def s3_client(settings: Settings) -> S3Client:
    """A boto3 client for the project's region. Keys come from the settings, never the shell."""
    import boto3  # noqa: PLC0415 (the scaleway extra)
    from botocore.config import Config  # noqa: PLC0415

    if settings.scw_access_key is None or settings.scw_secret_key is None:
        msg = "Scaleway is not set up: SCW_ACCESS_KEY and SCW_SECRET_KEY go in .env"
        raise RuntimeError(msg)
    # Scaleway does not take the checksums newer boto3 adds to every upload.
    config = Config(
        max_pool_connections=TRANSFERS,
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"max_attempts": 5, "mode": "standard"},
    )
    return boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "s3",
        region_name=settings.scw_region,
        endpoint_url=endpoint(settings),
        aws_access_key_id=settings.scw_access_key.get_secret_value(),
        aws_secret_access_key=settings.scw_secret_key.get_secret_value(),
        config=config,
    )


def ensure_buckets(settings: Settings, client: S3Client) -> dict[str, str]:
    """Create every bucket that does not exist yet. Returns each name with what happened."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    owned = {b.get("Name") for b in client.list_buckets().get("Buckets", [])}
    done: dict[str, str] = {}
    for bucket in Bucket:
        name = bucket_name(settings, bucket)
        if name in owned:
            done[name] = "exists"
            continue
        try:
            client.create_bucket(
                Bucket=name,
                CreateBucketConfiguration={"LocationConstraint": settings.scw_region},  # pyright: ignore[reportArgumentType]
            )
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code")
            if code == "BucketAlreadyOwnedByYou":  # a retried request had made it
                done[name] = "exists"
                continue
            if code == "BucketAlreadyExists":
                msg = (
                    f"bucket {name!r} belongs to another Scaleway customer: pick another "
                    "prefix (`Settings.scw_bucket_prefix`, in .env)"
                )
                raise RuntimeError(msg) from err
            raise
        done[name] = "created"
    return done


@dataclass(frozen=True, slots=True)
class Stored:
    key: str
    size: int


def listing(client: S3Client, bucket: str, prefix: str = "") -> Iterator[Stored]:
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for item in page.get("Contents", []):
            yield Stored(item.get("Key", ""), item.get("Size", 0))


def download_tree(client: S3Client, bucket: str, root: Path, prefix: str = "") -> int:
    """Copy every object under `prefix` missing from `root` (or of another size). Returns count."""
    todo = [
        s
        for s in listing(client, bucket, prefix)
        if not _same(root / s.key, s.size) and not s.key.endswith("/")
    ]

    def get(item: Stored) -> None:
        path = root / item.key
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        client.download_file(bucket, item.key, str(part))
        part.replace(path)

    list(bounded_map(get, todo, workers=TRANSFERS))
    logger.info("%s/%s: %d files down to %s", bucket, prefix, len(todo), root)
    return len(todo)


def upload_tree(client: S3Client, bucket: str, root: Path, prefix: str = "") -> int:
    """Copy every file under `root / prefix` the bucket lacks (or holds at another size)."""
    stored = {s.key: s.size for s in listing(client, bucket, prefix)}
    base = root / prefix if prefix else root
    files = [p for p in base.rglob("*") if p.is_file() and not p.name.endswith(".part")]
    todo = [p for p in files if stored.get(_key(root, p)) != p.stat().st_size]
    upload_files(client, bucket, root, todo)
    logger.info("%s: %d files up from %s", bucket, len(todo), base)
    return len(todo)


def upload_files(client: S3Client, bucket: str, root: Path, files: Iterable[Path]) -> None:
    """Upload these files, keyed by their path under `root`."""

    def put(path: Path) -> None:
        _again(lambda: client.upload_file(str(path), bucket, _key(root, path)))

    list(bounded_map(put, list(files), workers=TRANSFERS))


def _again(call: Callable[[], None]) -> None:
    """A new bucket answers NoSuchBucket now and then for a while; boto3 does not retry that."""
    from boto3.exceptions import S3UploadFailedError  # noqa: PLC0415

    for attempt in range(UPLOAD_ATTEMPTS):
        try:
            call()
            return
        except S3UploadFailedError as err:
            if attempt == UPLOAD_ATTEMPTS - 1:
                raise
            logger.warning("upload failed, trying again: %s", err)
            time.sleep(2**attempt)


def download_keys(client: S3Client, bucket: str, root: Path, keys: Iterable[str]) -> set[str]:
    """Fetch these keys into `root` when the bucket has them. Returns the keys it did not have."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    def get(key: str) -> str | None:
        path = root / key
        if path.exists():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        try:
            client.download_file(bucket, key, str(part))
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return key
            raise
        part.replace(path)
        return None

    return {k for k in bounded_map(get, list(keys), workers=TRANSFERS) if k is not None}


def _same(path: Path, size: int) -> bool:
    return path.exists() and path.stat().st_size == size


def _key(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
