"""Archive prints on the site: search the picture, not its mount, and use what the caption says.

A photo of a Mémoire print glued on a card with a typed caption (results.md, T1 archive):

1. `miroir.images.mount.find_picture` finds the print on the card (no model, a few ms); the
   photo search then embeds the print alone.
2. Only then does the GPU container's small open vision model read the caption
   (`miroir.remote.caption`); a photo without a mount costs nothing more.
3. The commune the caption names (`miroir.caption.Gazetteer`) acts as the photo's position:
   step H2's soft prior around it. The caption's words refine the search as typed words do.

The visitor sees what was read and can correct or remove it; the page then sends the caption
back (`SearchRequest.caption`) and the model is not asked again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from miroir.caption import Gazetteer
from miroir.web.catalog import Catalog
from miroir.web.nearby import EntityPlaces


class CaptionReader(Protocol):
    def read(self, jpeg: bytes) -> str:
        """The caption line of an archive photo, or "" when there is none."""
        ...


@dataclass(frozen=True, slots=True)
class ArchiveRule:
    """What the site does with a mounted print (results.md, T1 archive)."""

    crop: bool = True
    """Search the print alone."""
    place_bonus: float = 0.0
    """Bonus of the soft prior around the caption's commune; 0 leaves the place out."""
    place_scale_km: float = 1.0
    words: float = 0.0
    """Weight of the caption's words beside the photo; 0 leaves them out."""


SITE_ARCHIVE = ArchiveRule(crop=False, words=1.0)


def gazetteer(catalog: Catalog, places: EntityPlaces) -> Gazetteer:
    """The index's communes, each at the median position of its placed entities."""
    commune = [""] * catalog.entity_count
    departement = [""] * catalog.entity_count
    for record, entity in zip(catalog.records, catalog.record_entity, strict=True):
        row = int(entity)
        if record.commune and not commune[row]:
            commune[row] = record.commune
            departement[row] = record.departement or ""
    rows = (
        (commune[e], departement[e], float(places.lat[e]), float(places.lon[e]))
        for e in range(catalog.entity_count)
        if commune[e] and np.isfinite(places.lat[e])
    )
    return Gazetteer.build(rows)
