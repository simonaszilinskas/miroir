"""Commons query photos as 512 px JPEGs, for the model services that embed them.

Commons photos are queries only (T2, T2b, T8c), never gallery images. They are openly
licensed, so they may go to any service (technical spec §7). The Mac fetches each one through
`WikimediaClient`: one request at a time, at most four a second, with a User-Agent naming the
project, and every copy kept in `Settings.commons_cache_dir`, so a second run sends nothing.
Modal containers never fetch from Wikimedia: the Mac sends them the bytes.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager

from miroir.commons.client import WikimediaClient, open_client
from miroir.commons.model import CommonsPhoto, is_commons_image
from miroir.config import Settings
from miroir.images.fetch import Fetched, FetchSkip
from miroir.records.store import read_commons

logger = logging.getLogger(__name__)


class CommonsImages:
    """The Commons photos of `commons.parquet`, fetched on demand at 512 px."""

    def __init__(self, photos: Mapping[str, CommonsPhoto], client: WikimediaClient) -> None:
        self._photos = photos
        self._client = client

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._photos

    def fetch(self, image_id: str) -> Fetched:
        """The photo's 960 px thumbnail, shrunk to 512 px on its long side."""
        photo = self._photos.get(image_id)
        if photo is None:
            msg = f"{image_id} is not a photo of commons.parquet"
            raise KeyError(msg)
        jpeg = self._client.get_jpeg(photo.thumb_url)
        if jpeg is None:
            logger.warning("%s: could not fetch %s", image_id, photo.thumb_url)
            return Fetched(image_id, None, FetchSkip.MISSING)
        return Fetched(image_id, jpeg, None)

    def fetch_all(self, image_ids: Iterable[str]) -> Iterator[Fetched]:
        """One after another: the client keeps Wikimedia's pace."""
        for image_id in image_ids:
            yield self.fetch(image_id)


def commons_photos(settings: Settings) -> dict[str, CommonsPhoto]:
    """Every Commons photo by image id; empty before `miroir testsets commons` has run."""
    if not settings.commons_parquet.exists():
        return {}
    return {p.image_id: p for p in read_commons(settings.commons_parquet)}


@contextmanager
def open_commons_images(settings: Settings) -> Generator[CommonsImages]:
    with open_client() as http:
        yield CommonsImages(
            commons_photos(settings), WikimediaClient(http, settings.commons_cache_dir)
        )


def split_commons(image_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """POP image ids and Commons photo ids, each in the given order."""
    pop: list[str] = []
    commons: list[str] = []
    for image_id in image_ids:
        (commons if is_commons_image(image_id) else pop).append(image_id)
    return pop, commons
