"""Images the Mac sends to the GPU machine as bytes (`GpuRunner.embed_supplied`).

Commons photos added to a gallery (step F2), fetched politely from Wikimedia, and the images
of the records copied into extra/ (`miroir records add`).
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager

import httpx

from miroir.commons.client import WikimediaClient, open_client
from miroir.commons.images import CommonsImages
from miroir.config import Settings
from miroir.images.fetch import Fetched, ImageFetcher, local_cache
from miroir.records.extra import read_extra_images
from miroir.records.model import Image
from miroir.records.store import read_commons

logger = logging.getLogger(__name__)


class PopImages:
    """POP images fetched on the Mac (image cache, else the bucket), for records the images
    bucket does not hold: those copied from the whole download (`records/extra`)."""

    def __init__(self, pop: ImageFetcher, images: Mapping[str, Image]) -> None:
        self._pop = pop
        self._images = images

    def fetch_all(self, image_ids: Iterable[str]) -> Iterator[Fetched]:
        yield from self._pop.fetch_all(self._images[i] for i in image_ids)


@contextmanager
def open_extra_images(settings: Settings) -> Generator[PopImages]:
    cache = local_cache(settings)
    images = {i.image_id: i for i in read_extra_images(settings)}
    with httpx.Client(timeout=60.0, follow_redirects=True) as pop:
        yield PopImages(ImageFetcher(pop, cache), images)


@contextmanager
def open_gallery_photos(settings: Settings, gallery: str) -> Generator[CommonsImages]:
    """The F2 photos of a gallery, fetched from Wikimedia at its pace (cached)."""
    photos = {p.image_id: p for p in read_commons(settings.commons_gallery_parquet(gallery))}
    with open_client() as wikimedia:
        yield CommonsImages(photos, WikimediaClient(wikimedia, settings.commons_cache_dir))
