"""Which images may leave the Mac (technical spec §9, CLAUDE.md rule 8).

Only images of `publishable` records go to an outside service (OpenRouter, any other) or to
the GPU machine; the full-sample index is the one exception (`--all-images`).
Commons photos (test queries T2, T2b, T8c, F2 gallery photos) are openly licensed, so they may
go to any service. The OpenRouter runners (`miroir.jobs.embed`, `miroir.jobs.describe`) check
every id through `PublishableImages` before sending. Every call to the GPU machine goes through
`ImageGate`, inside `miroir.jobs.scw_run.ScalewayCalls`, whoever makes it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

from miroir.commons.images import commons_photos
from miroir.config import Settings
from miroir.records.extra import read_site_records
from miroir.records.model import Image
from miroir.records.store import read_commons

logger = logging.getLogger(__name__)


class PrivacyError(PermissionError):
    """Asked to send an image from a record that is not publishable to an outside service."""


class PublishableImages:
    """Every image of the record store, and which of them may go to an outside service."""

    def __init__(
        self,
        images: dict[str, Image],
        publishable: frozenset[str],
        commons: frozenset[str] = frozenset(),
    ) -> None:
        self.images = images
        self._publishable = publishable
        self._commons = commons
        """Commons photo ids: openly licensed, so any service may receive them."""

    @classmethod
    def load(cls, settings: Settings) -> PublishableImages:
        records = read_site_records(settings)
        images = {i.image_id: i for r in records for i in r.images}
        publishable = frozenset(i.image_id for r in records if r.publishable for i in r.images)
        commons = set(commons_photos(settings))
        for path in settings.commons_gallery_parquets():  # F2 gallery photos, also Commons
            commons.update(p.image_id for p in read_commons(path))
        return cls(images, publishable, frozenset(commons))

    def check_known(self, image_ids: Sequence[str]) -> None:
        """Raise unless every id is in images.parquet or commons.parquet (publishable or not)."""
        unknown = [i for i in image_ids if i not in self.images and i not in self._commons]
        if unknown:
            msg = (
                f"{len(unknown)} image id(s) in neither images.parquet nor commons.parquet, "
                f"e.g. {unknown[0]}"
            )
            raise KeyError(msg)

    def check(self, image_ids: Sequence[str], service: str) -> None:
        """Raise unless every id is a known Commons photo, or a known and publishable image."""
        self.check_known(image_ids)
        private = [i for i in image_ids if i not in self._publishable and i not in self._commons]
        if private:
            msg = (
                f"{len(private)} image(s) come from records that are not publishable "
                f"(e.g. {private[0]}); they may not go to {service}"
            )
            raise PrivacyError(msg)

    def only_publishable(self, image_ids: Sequence[str]) -> list[str]:
        """Drop ids from records that are not publishable (logged), keeping order."""
        self.check([i for i in image_ids if i not in self.images], "any service")
        kept = [i for i in image_ids if i in self._publishable or i in self._commons]
        if len(kept) < len(image_ids):
            logger.info("left out %d non-publishable image(s)", len(image_ids) - len(kept))
        return kept


#: Splits a derived id ("<image id>#<anything>": a cut or a copy of that image) from its image.
DERIVED = "#"


class ImageGate:
    """The one check every image bound for the GPU machine passes (`miroir.jobs.scw_run`).

    `ScalewayCalls` runs it inside each call that sends an image or names one, so no caller
    can send an image without it. An id may name a copy made on the Mac (a cut, a grey copy)
    as `<image id>#<anything>`: the image it comes from is checked. With `all_images`, the
    full-sample index exception (technical spec §9), any known image passes.
    """

    def __init__(
        self,
        load: Callable[[], PublishableImages],
        *,
        all_images: bool = False,
        service: str = "the GPU machine",
    ) -> None:
        self._load = load
        self._loaded: PublishableImages | None = None
        self.all_images = all_images
        self._service = service

    @property
    def store(self) -> PublishableImages:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def check(self, image_ids: Iterable[str]) -> None:
        """Raise unless every image may go to the machine."""
        sources = [i.split(DERIVED, 1)[0] for i in image_ids]
        if self.all_images:
            self.store.check_known(sources)
        else:
            self.store.check(sources, self._service)
