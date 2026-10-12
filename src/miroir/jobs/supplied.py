"""Images the Mac sends to Modal as bytes (`ModalRunner.embed_supplied`).

Two sources: Commons photos added to a gallery (step F2), fetched politely from Wikimedia,
and grey copies of POP images and Commons query photos (step F1), made here with Pillow
(a pixel operation, no model).
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager

import httpx

from miroir.commons.client import WikimediaClient, open_client
from miroir.commons.images import CommonsImages, commons_photos
from miroir.commons.model import is_commons_image
from miroir.config import Settings
from miroir.eval.sets import load_queries, read_gallery
from miroir.images.colour import already_grey, greyscale
from miroir.images.fetch import Fetched, ImageCache, ImageFetcher
from miroir.records.model import Image
from miroir.records.store import read_commons, read_images

logger = logging.getLogger(__name__)


class GreyCopies:
    """Grey copies of POP images (image cache, else POP) and of Commons query photos."""

    def __init__(
        self, pop: ImageFetcher, images: Mapping[str, Image], commons: CommonsImages
    ) -> None:
        self._pop = pop
        self._images = images
        self._commons = commons

    def original(self, image_id: str) -> Fetched:
        if is_commons_image(image_id):
            return self._commons.fetch(image_id)
        return self._pop.fetch(self._images[image_id])

    def fetch_all(self, image_ids: Iterable[str]) -> Iterator[Fetched]:
        for image_id in image_ids:
            fetched = self.original(image_id)
            if fetched.jpeg is None:
                yield fetched
            else:
                yield Fetched(image_id, greyscale(fetched.jpeg), None)

    def not_grey(self, image_ids: Sequence[str]) -> list[str]:
        """The ids whose image has colour: greying the others changes no pixel worth naming."""
        out: list[str] = []
        unread = 0
        for image_id in image_ids:
            fetched = self.original(image_id)
            if fetched.jpeg is None:
                unread += 1
            elif not already_grey(fetched.jpeg):
                out.append(image_id)
        logger.info(
            "%d of %d images have colour (%d unreadable, left out)",
            len(out),
            len(image_ids),
            unread,
        )
        return out


@contextmanager
def open_grey_copies(settings: Settings) -> Generator[GreyCopies]:
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1_000_000)
    images = {i.image_id: i for i in read_images(settings.images_parquet)}
    with httpx.Client(timeout=60.0) as pop, open_client() as wikimedia:
        commons = CommonsImages(
            commons_photos(settings), WikimediaClient(wikimedia, settings.commons_cache_dir)
        )
        yield GreyCopies(ImageFetcher(pop, cache), images, commons)


def grey_candidates(settings: Settings, gallery: str, set_ids: Sequence[str]) -> list[str]:
    """Every image of `gallery` and every tune query image of `set_ids`, once each."""
    ids = list(read_gallery(settings, gallery))
    ids.extend(q.query_image_id for s in set_ids for q in load_queries(settings, s, "tune"))
    return list(dict.fromkeys(ids))


@contextmanager
def open_gallery_photos(settings: Settings, gallery: str) -> Generator[CommonsImages]:
    """The F2 photos of a gallery, fetched from Wikimedia at its pace (cached)."""
    photos = {p.image_id: p for p in read_commons(settings.commons_gallery_parquet(gallery))}
    with open_client() as wikimedia:
        yield CommonsImages(photos, WikimediaClient(wikimedia, settings.commons_cache_dir))
