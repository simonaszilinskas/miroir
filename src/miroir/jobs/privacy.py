"""Which images may leave the Mac (technical spec §9, CLAUDE.md hard rule 8).

Only images of `publishable` records go to an outside service: OpenRouter, Modal, any other.
Runners load the record store through `PublishableImages` and check every id before sending.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from miroir.config import Settings
from miroir.records.model import Image
from miroir.records.store import read_records

logger = logging.getLogger(__name__)


class PrivacyError(PermissionError):
    """Asked to send an image from a record that is not publishable to an outside service."""


class PublishableImages:
    """Every image of the record store, and which of them may go to an outside service."""

    def __init__(self, images: dict[str, Image], publishable: frozenset[str]) -> None:
        self.images = images
        self._publishable = publishable

    @classmethod
    def load(cls, settings: Settings) -> PublishableImages:
        records = read_records(settings.records_parquet, settings.images_parquet)
        images = {i.image_id: i for r in records for i in r.images}
        publishable = frozenset(i.image_id for r in records if r.publishable for i in r.images)
        return cls(images, publishable)

    def check(self, image_ids: Sequence[str], service: str) -> None:
        """Raise unless every id is known and publishable."""
        unknown = [i for i in image_ids if i not in self.images]
        if unknown:
            msg = f"{len(unknown)} image id(s) not in images.parquet, e.g. {unknown[0]}"
            raise KeyError(msg)
        private = [i for i in image_ids if i not in self._publishable]
        if private:
            msg = (
                f"{len(private)} image(s) come from records that are not publishable "
                f"(e.g. {private[0]}); they may not go to {service}"
            )
            raise PrivacyError(msg)

    def only_publishable(self, image_ids: Sequence[str]) -> list[str]:
        """Drop ids from records that are not publishable (logged), keeping order."""
        self.check([i for i in image_ids if i not in self.images], "any service")
        kept = [i for i in image_ids if i in self._publishable]
        if len(kept) < len(image_ids):
            logger.info("left out %d non-publishable image(s)", len(image_ids) - len(kept))
        return kept
