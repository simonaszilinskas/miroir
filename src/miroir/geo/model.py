"""Positions of entities and of query photos, and the distance between two positions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

#: Mean Earth radius, in kilometres.
EARTH_KM = 6371.0088

Float64s = NDArray[np.float64]


class PlaceSource(StrEnum):
    """Where an entity's position comes from, most precise first."""

    MERIMEE = "merimee"
    """The building's own point in Mérimée (POP)."""
    MUSEUM = "museum"
    """The museum holding the object, from Muséofile (POP)."""
    COMMUNE = "commune"
    """The centre of the commune the record names (geo.api.gouv.fr)."""


@dataclass(frozen=True, slots=True)
class Place:
    entity_id: str
    latitude: float
    longitude: float
    source: PlaceSource


class PositionSource(StrEnum):
    """Where a query photo's position comes from."""

    EXIF = "exif"
    """The camera's GPS, in the file's EXIF."""
    CAMERA = "camera"
    """The camera location the Commons file page gives ({{Location}}), often from EXIF."""


@dataclass(frozen=True, slots=True)
class QueryPosition:
    image_id: str
    latitude: float
    longitude: float
    source: PositionSource


def distance_km(lat: float, lon: float, lats: Float64s, lons: Float64s) -> Float64s:
    """Great-circle distance from one point to many, in kilometres (haversine)."""
    p1, p2 = np.radians(lat), np.radians(lats)
    dp = p2 - p1
    dl = np.radians(lons) - np.radians(lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def moved(lat: float, lon: float, km: float, bearing_rad: float) -> tuple[float, float]:
    """The point `km` away from (lat, lon) in the direction `bearing_rad` (0 = north)."""
    d = km / EARTH_KM
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing_rad))
    l2 = l1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2)
    )
    return float(np.degrees(p2)), float((np.degrees(l2) + 540) % 360 - 180)
