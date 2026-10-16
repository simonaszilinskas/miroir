from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from miroir.commons.api import camera_positions
from miroir.commons.client import WikimediaClient
from miroir.geo.communes import commune_centres
from miroir.geo.model import Place, PlaceSource, distance_km, moved
from miroir.geo.places import NO_HINT, UNKNOWN_COMMUNE, place_entities
from miroir.records.mappers import (
    PlaceHint,
    PopPoint,
    merimee_point,
    museum_point,
    place_hint,
)
from miroir.records.model import Collection, Record
from tests.fakes import NO_WAIT


def test_distance_and_moved_agree() -> None:
    # Notre-Dame de Paris to the Louvre pyramid: about 1.4 km.
    d = distance_km(48.85296, 2.34991, np.asarray([48.86100]), np.asarray([2.33584]))
    assert d[0] == pytest.approx(1.36, abs=0.05)
    lat, lon = moved(48.85, 2.35, 0.3, 1.0)
    back = distance_km(48.85, 2.35, np.asarray([lat]), np.asarray([lon]))
    assert back[0] == pytest.approx(0.3, abs=1e-6)


def test_points_from_raw_records() -> None:
    merimee = {"REF": "PA00078066", "coordonnees_au_format_WGS84": "48.39165,4.52479"}
    assert merimee_point(merimee) == PopPoint("PA00078066", 48.39165, 4.52479)
    assert merimee_point({"REF": "PA1", "coordonnees_au_format_WGS84": ""}) is None
    assert merimee_point({"REF": "PA1", "coordonnees_au_format_WGS84": "0,0"}) is None
    assert merimee_point({"REF": "PA1", "coordonnees_au_format_WGS84": "91,2"}) is None
    museum = {"REF": "M1128", "Coordonnees": "45.790491, 4.797411"}
    assert museum_point(museum) == PopPoint("M1128", 45.790491, 4.797411)


def test_place_hint_reads_insee_and_museum() -> None:
    assert place_hint({"REF": "AP1", "INSEE": ["05151"]}) == PlaceHint("AP1", "05151", None)
    assert place_hint({"REF": "AP2", "INSEE": ["2A004"]}).insee == "2A004"
    assert place_hint({"REF": "AP3", "INSEE": ["n.c."]}).insee is None
    assert place_hint({"REF": "J1", "MUSEO": "M0569"}) == PlaceHint("J1", None, "M0569")


def _record(ref: str, entity: str, collection: Collection = Collection.MEMOIRE) -> Record:
    return Record(
        ref=ref,
        collection=collection,
        entity_id=entity,
        title="",
        description="",
        denomination=(),
        domain=(),
        authors=(),
        period=(),
        region=None,
        departement=None,
        commune=None,
        publishable=True,
        pop_url="",
        linked_refs=(),
        images=(),
    )


def test_place_entities_takes_the_most_precise_source() -> None:
    records = [
        _record("AP1", "PA1"),  # Mérimée point
        _record("AP2", "PA2"),  # no Mérimée point: its commune
        _record("J1", "J1", Collection.JOCONDE),  # its museum
        _record("AP3", "AP3"),  # a commune code the API does not know
        _record("RV1", "RV1", Collection.ROSE_VALLAND),  # nothing
    ]
    hints = {
        "AP1": PlaceHint("AP1", "75056", None),
        "AP2": PlaceHint("AP2", "75056", None),
        "J1": PlaceHint("J1", None, "M1"),
        "AP3": PlaceHint("AP3", "99999", None),
    }
    report = place_entities(
        records,
        hints,
        buildings={"PA1": PopPoint("PA1", 48.0, 2.0)},
        museums={"M1": PopPoint("M1", 45.0, 4.0)},
        centres={"75056": (48.86, 2.34)},
    )
    assert report.places == (
        Place("J1", 45.0, 4.0, PlaceSource.MUSEUM),
        Place("PA1", 48.0, 2.0, PlaceSource.MERIMEE),
        Place("PA2", 48.86, 2.34, PlaceSource.COMMUNE),
    )
    assert report.counts[("memoire", UNKNOWN_COMMUNE)] == 1
    assert report.counts[("rose-valland", NO_HINT)] == 1
    assert report.coverage("memoire") == (2, 3)


def test_commune_centres_read_from_the_cache(tmp_path: Path) -> None:
    today = [
        {"code": "75056", "nom": "Paris", "centre": {"coordinates": [2.34, 48.86]}},
        {"code": "01001", "nom": "X"},
    ]
    former = [
        {"code": "75056", "nom": "old", "centre": {"coordinates": [0.0, 0.0]}},
        {"code": "01120", "nom": "Cordieux", "centre": {"coordinates": [5.03, 45.92]}},
    ]
    (tmp_path / "communes.json").write_text(json.dumps(today))
    (tmp_path / "communes-deleguees.json").write_text(json.dumps(former))
    centres = commune_centres(tmp_path)
    assert centres == {"75056": (48.86, 2.34), "01120": (45.92, 5.03)}


def test_commune_centres_fetched_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert "MIROIR" in request.headers["user-agent"]
        return httpx.Response(200, json=[])

    http = httpx.Client(
        transport=httpx.MockTransport(handler), headers={"User-Agent": "MIROIR test"}
    )
    commune_centres(tmp_path, http)
    commune_centres(tmp_path, http)
    assert sorted(calls) == ["/communes", "/communes_associees_deleguees"]


def test_camera_positions_keep_the_camera_only(tmp_path: Path) -> None:
    answer = {
        "query": {
            "pages": [
                {
                    "title": "File:a.jpg",
                    "coordinates": [
                        {"lat": 43.95, "lon": 4.81, "primary": True, "type": "camera"},
                    ],
                },
                {
                    "title": "File:b.jpg",
                    "coordinates": [
                        {"lat": 42.5, "lon": 2.9, "primary": False, "type": "landmark"},
                    ],
                },
                {"title": "File:c.jpg"},
            ]
        }
    }
    client = WikimediaClient(
        httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=answer))),
        tmp_path,
        retry=NO_WAIT,
        max_per_second=1e9,
        sleep=lambda _s: None,
    )
    found = camera_positions(client, ["File:a.jpg", "File:b.jpg", "File:c.jpg"])
    assert found == {"File:a.jpg": (43.95, 4.81)}


def test_place_hint_keeps_the_museum_name() -> None:
    hint = place_hint({"REF": "J1", "MUSEO": "M0569", "NOMOFF": "musée des Amériques - Auch"})
    assert hint.museum_name == "musée des Amériques - Auch"
    assert place_hint({"REF": "AP1", "NOMOFF": "x"}).museum_name is None
