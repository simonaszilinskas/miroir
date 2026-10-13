"""One position per building or object (step H2): POP's own point, else its commune's centre.

In order of precision:

1. a Mémoire building (`PA` id) with a point in Mérimée (`coordonnees_au_format_WGS84`);
2. a Joconde object: its museum's point in Muséofile (`MUSEO`, `Coordonnees`);
3. the centre of the commune a record names (Mémoire `INSEE`), from geo.api.gouv.fr.

Rose Valland works have no place: they were looted and moved. Mérimée and Muséofile come from
the whole POP download (`Settings.full_exports_dir`); without it, only commune centres remain.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from miroir.config import Settings
from miroir.geo.communes import commune_centres
from miroir.geo.model import Place, PlaceSource
from miroir.records.build import iter_raw
from miroir.records.mappers import PlaceHint, PopPoint, merimee_point, museum_point, place_hint
from miroir.records.model import Collection, Record
from miroir.records.store import read_records, write_places

logger = logging.getLogger(__name__)

#: Why an entity got no position.
NO_HINT = "no commune or museum named"
UNKNOWN_COMMUNE = "commune code not in geo.api.gouv.fr"


@dataclass(frozen=True, slots=True)
class PlacesReport:
    places: tuple[Place, ...]
    counts: Mapping[tuple[str, str], int]
    """(collection, source or reason missing): entities."""

    def coverage(self, collection: str) -> tuple[int, int]:
        """(entities placed, entities) of one collection."""
        placed = sum(
            n for (c, s), n in self.counts.items() if c == collection and s in set(PlaceSource)
        )
        total = sum(n for (c, _), n in self.counts.items() if c == collection)
        return placed, total


def place_entities(
    records: Sequence[Record],
    hints: Mapping[str, PlaceHint],
    buildings: Mapping[str, PopPoint],
    museums: Mapping[str, PopPoint],
    centres: Mapping[str, tuple[float, float]],
) -> PlacesReport:
    """The best position of every entity of `records`, and counts by source."""
    by_entity: defaultdict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_entity[record.entity_id].append(record)
    places: list[Place] = []
    counts: Counter[tuple[str, str]] = Counter()
    for entity_id in sorted(by_entity):
        group = by_entity[entity_id]
        found = _place(entity_id, group, hints, buildings, museums, centres)
        if isinstance(found, Place):
            places.append(found)
            counts[group[0].collection.value, found.source.value] += 1
        else:
            counts[group[0].collection.value, found] += 1
    return PlacesReport(tuple(places), dict(counts))


def _place(  # noqa: PLR0917 (private; the four sources in order of precision)
    entity_id: str,
    group: Sequence[Record],
    hints: Mapping[str, PlaceHint],
    buildings: Mapping[str, PopPoint],
    museums: Mapping[str, PopPoint],
    centres: Mapping[str, tuple[float, float]],
) -> Place | str:
    building = buildings.get(entity_id)
    if building is not None:
        return Place(entity_id, building.latitude, building.longitude, PlaceSource.MERIMEE)
    record_hints = [h for r in group if (h := hints.get(r.ref)) is not None]
    for hint in record_hints:
        museum = museums.get(hint.museum or "")
        if museum is not None:
            return Place(entity_id, museum.latitude, museum.longitude, PlaceSource.MUSEUM)
    codes = [h.insee for h in record_hints if h.insee]
    if not codes:
        return NO_HINT
    for code in codes:
        if code in centres:
            lat, lon = centres[code]
            return Place(entity_id, lat, lon, PlaceSource.COMMUNE)
    return UNKNOWN_COMMUNE


def read_hints(settings: Settings) -> dict[str, PlaceHint]:
    """Commune and museum of every July export record, by ref."""
    hints: dict[str, PlaceHint] = {}
    for collection in Collection:
        for raw in iter_raw(settings.export_jsonl(collection)):
            hint = place_hint(raw)
            hints[hint.ref] = hint
    return hints


def read_points(settings: Settings, base: str) -> dict[str, PopPoint]:
    """Positions of one base of the whole POP download ("merimee", "museo"), by ref."""
    path = settings.full_export_jsonl(base)
    if not path.exists():
        logger.warning("%s missing: no %s positions", path, base)
        return {}
    parse = merimee_point if base == "merimee" else museum_point
    points = {p.ref: p for raw in iter_raw(path) if (p := parse(raw)) is not None}
    logger.info("%s: %d records with a position", base, len(points))
    return points


def build_places(settings: Settings) -> PlacesReport:
    """Write `places.parquet` for every entity of `records.parquet`."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    hints = read_hints(settings)
    buildings = read_points(settings, "merimee")
    centres = commune_centres(settings.geo_cache_dir)
    report = place_entities(records, hints, buildings, read_points(settings, "museo"), centres)
    write_places(report.places, settings.places_parquet)
    logger.info("places: %d entities placed", len(report.places))
    return report
