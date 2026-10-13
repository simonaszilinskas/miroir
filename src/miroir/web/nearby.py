"""Step H2 on the site: when the photo or the person gives a position, favour what is near.

The bundle holds one position per entity (`places.npz`, from `miroir geo places`: Mérimée's
point for a building, Muséofile's for a museum object, else the commune centre). A search
with a position adds `PRIOR_BONUS * exp(-d / PRIOR_SCALE_KM)` to each placed entity's fused
score (d in km; the fused score runs from 0 to 1 for a photo): the soft prior that scored best
on T2 (results.md, H2). Nothing is removed, so with nothing known nearby the ranking is the
usual one. Entities without a place (Rose Valland works, a few Mémoire photos) get no bonus.

The position comes with the request and goes nowhere else: it is not logged or stored.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.geo.model import distance_km

logger = logging.getLogger(__name__)

PLACES_FILE = "places.npz"
#: The soft prior picked on T2 tune (results.md, H2): bonus at distance 0, and the distance in
#: km over which it falls by a factor e.
PRIOR_BONUS = 1.0
PRIOR_SCALE_KM = 1.0
#: The page reports how many known entities lie this close, in km.
NEAR_KM = 2.0

Float64s = NDArray[np.float64]
Bools = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Position:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            msg = f"not a position: {self.latitude}, {self.longitude}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EntityPlaces:
    """Latitude and longitude of each catalog entity, in `Catalog.entity_ids` order; NaN where
    the entity has no place."""

    lat: Float64s
    lon: Float64s

    def distances(self, position: Position) -> Float64s:
        """Km from the position to each entity; NaN for an entity without a place."""
        with np.errstate(invalid="ignore"):
            return distance_km(position.latitude, position.longitude, self.lat, self.lon)

    def prior(
        self,
        position: Position,
        *,
        bonus: float = PRIOR_BONUS,
        scale_km: float = PRIOR_SCALE_KM,
    ) -> NDArray[np.float32]:
        """The bonus each entity gets: highest where the photo was taken, 0 without a place."""
        d = np.nan_to_num(self.distances(position), nan=np.inf)
        return (bonus * np.exp(-d / scale_km)).astype(np.float32)

    def near(self, position: Position, km: float = NEAR_KM) -> Bools:
        """Entities placed within `km` of the position."""
        return np.asarray(np.nan_to_num(self.distances(position), nan=np.inf) <= km)


def write_places(
    path: Path, entity_ids: Sequence[str], places: Mapping[str, tuple[float, float]]
) -> int:
    """Store the places of `entity_ids` (the catalog's order). Returns how many have one."""
    lat = np.asarray([places[e][0] if e in places else np.nan for e in entity_ids])
    lon = np.asarray([places[e][1] if e in places else np.nan for e in entity_ids])
    np.savez_compressed(path, entity_ids=np.asarray(entity_ids, dtype=str), lat=lat, lon=lon)
    return int(np.isfinite(lat).sum())


def read_places(path: Path, entity_ids: Sequence[str]) -> EntityPlaces | None:
    """The places stored for the catalog's entities, or None when the bundle has none."""
    if not path.exists():
        logger.info("no %s in the bundle: searches ignore positions", path.name)
        return None
    with np.load(path, allow_pickle=False) as data:
        stored = [str(e) for e in data["entity_ids"]]
        row = {e: i for i, e in enumerate(stored)}
        lat_all = np.asarray(data["lat"], dtype=np.float64)
        lon_all = np.asarray(data["lon"], dtype=np.float64)
    rows = np.asarray([row.get(e, -1) for e in entity_ids], dtype=np.int64)
    lat = np.where(rows >= 0, lat_all[np.maximum(rows, 0)], np.nan)
    lon = np.where(rows >= 0, lon_all[np.maximum(rows, 0)], np.nan)
    logger.info("places: %d of %d entities placed", int(np.isfinite(lat).sum()), len(entity_ids))
    return EntityPlaces(lat, lon)
