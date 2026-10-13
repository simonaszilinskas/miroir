"""Commune centres by INSEE code, from geo.api.gouv.fr (the State's open geography API).

Two calls: today's communes (with Paris, Lyon and Marseille's arrondissements) and the
former communes merged into another (associated or delegated), whose codes old records still
carry. Each answer is kept in `Settings.geo_cache_dir`, so a rebuild sends nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel

from miroir.config import GEO_API_URL, WIKIMEDIA_USER_AGENT
from miroir.http import Retry, send

logger = logging.getLogger(__name__)

#: (cache file, path, query) of each call.
_CALLS = (
    (
        "communes.json",
        "/communes",
        {"fields": "nom,code,centre", "type": "commune-actuelle,arrondissement-municipal"},
    ),
    ("communes-deleguees.json", "/communes_associees_deleguees", {"fields": "nom,code,centre"}),
)


class _Centre(BaseModel):
    coordinates: tuple[float, float]
    """GeoJSON order: longitude, latitude."""


class _Commune(BaseModel):
    code: str
    centre: _Centre | None = None


def commune_centres(
    cache_dir: Path, http: httpx.Client | None = None
) -> dict[str, tuple[float, float]]:
    """(latitude, longitude) of every commune centre, by INSEE code. Today's communes win
    over a former commune that shares a code."""
    centres: dict[str, tuple[float, float]] = {}
    for name, path, params in reversed(_CALLS):
        for commune in _read(cache_dir / name, path, params, http):
            if commune.centre is not None:
                lon, lat = commune.centre.coordinates
                centres[commune.code] = (lat, lon)
    logger.info("commune centres: %d INSEE codes", len(centres))
    return centres


def _read(
    cached: Path, path: str, params: dict[str, str], http: httpx.Client | None
) -> list[_Commune]:
    if not cached.exists():
        if http is None:
            with _client() as own:
                payload = _fetch(own, path, params)
        else:
            payload = _fetch(http, path, params)
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cached)
    data: object = json.loads(cached.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"{cached}: expected a list of communes"
        raise TypeError(msg)
    return [_Commune.model_validate(row) for row in data]  # pyright: ignore[reportUnknownVariableType]


def _fetch(http: httpx.Client, path: str, params: dict[str, str]) -> object:
    request = http.build_request("GET", GEO_API_URL + path, params={**params, "format": "json"})
    response = send(http, request, Retry(attempts=4, backoff_s=2.0))
    if response.status_code != 200:
        msg = f"GET {path}: HTTP {response.status_code}"
        raise RuntimeError(msg)
    logger.info("geo.api.gouv.fr %s: %d bytes", path, len(response.content))
    payload: object = response.json()
    return payload


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": WIKIMEDIA_USER_AGENT}, timeout=httpx.Timeout(120.0, connect=20.0)
    )
