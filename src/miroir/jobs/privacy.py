"""Which images may leave the Mac (technical spec §9, CLAUDE.md hard rule 8).

Only images of `publishable` records go to an outside service: OpenRouter, Modal, any other.
Commons photos (test queries T2, T2b, T8c) are openly licensed, so they may go to any service.
Runners load the record store through `PublishableImages` and check every id before sending.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from miroir.commons.images import commons_photos
from miroir.config import Settings
from miroir.records.model import Image
from miroir.records.store import read_records

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
        records = read_records(settings.records_parquet, settings.images_parquet)
        images = {i.image_id: i for r in records for i in r.images}
        publishable = frozenset(i.image_id for r in records if r.publishable for i in r.images)
        return cls(images, publishable, frozenset(commons_photos(settings)))

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
