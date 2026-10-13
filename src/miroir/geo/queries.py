"""Where each Commons query photo was taken (step H2): EXIF GPS, else the file page's camera
location. A phone gives the first; the second is what Commons users typed or a bot copied
from EXIF, so it is a camera position too. The page's "object location" is never used."""

from __future__ import annotations

import logging
from collections import Counter

from miroir.commons.api import camera_positions
from miroir.commons.client import WikimediaClient, open_client
from miroir.config import Settings
from miroir.geo.model import PositionSource, QueryPosition
from miroir.records.store import read_commons, write_positions

logger = logging.getLogger(__name__)


def build_positions(settings: Settings) -> Counter[tuple[str, str]]:
    """Write `query-positions.parquet` for every photo of `commons.parquet`.

    Returns counts by (set, source or "none").
    """
    photos = read_commons(settings.commons_parquet)
    with open_client() as http:
        client = WikimediaClient(http, settings.commons_cache_dir)
        camera = camera_positions(client, [p.file_title for p in photos])
        logger.info("Commons: %d requests sent for camera locations", client.requests)
    positions: list[QueryPosition] = []
    counts: Counter[tuple[str, str]] = Counter()
    for photo in photos:
        position: QueryPosition | None = None
        if photo.latitude is not None and photo.longitude is not None:
            position = QueryPosition(
                photo.image_id, photo.latitude, photo.longitude, PositionSource.EXIF
            )
        elif photo.file_title in camera:
            lat, lon = camera[photo.file_title]
            position = QueryPosition(photo.image_id, lat, lon, PositionSource.CAMERA)
        counts[photo.set_id.value, "none" if position is None else position.source.value] += 1
        if position is not None:
            positions.append(position)
    write_positions(positions, settings.query_positions_parquet)
    return counts
