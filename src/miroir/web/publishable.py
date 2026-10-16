"""Which of a bundle's images may go to an outside model service (technical spec §9).

The conversation page (step H5) shows the model thumbnails of results. Only images of
publishable records and Commons photos may leave, as for every paid run (CLAUDE.md rule 8).
The rule itself is `miroir.jobs.privacy.PublishableImages`, which needs the record store: so
it runs on the Mac when the bundle is built (`miroir web publishable`), and the bundle keeps
the ids it lets through in `publishable.txt.gz`. A bundle without the file lets none through.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

PUBLISHABLE_FILE = "publishable.txt.gz"


class NotPublishableError(PermissionError):
    """Asked to send an image that may not leave to an outside service."""


class Publishable:
    """The image ids that may go to an outside service."""

    def __init__(self, image_ids: Iterable[str]) -> None:
        self._ids = frozenset(image_ids)

    def __len__(self) -> int:
        return len(self._ids)

    def allows(self, image_id: str) -> bool:
        return image_id in self._ids

    def check(self, image_id: str, service: str) -> None:
        if image_id not in self._ids:
            msg = f"image {image_id} is not publishable; it may not go to {service}"
            raise NotPublishableError(msg)


def write_publishable(path: Path, image_ids: Iterable[str]) -> int:
    ids = sorted(set(image_ids))
    path.write_bytes(gzip.compress("".join(f"{i}\n" for i in ids).encode(), compresslevel=6))
    return len(ids)


def read_publishable(path: Path) -> Publishable:
    """The ids in the bundle's file; none when the bundle has no file."""
    if not path.exists():
        logger.warning("no %s in the bundle: no image goes to the model", path.name)
        return Publishable(())
    text = gzip.decompress(path.read_bytes()).decode()
    found = Publishable(line for line in text.splitlines() if line)
    logger.info("%d images may go to the model", len(found))
    return found
