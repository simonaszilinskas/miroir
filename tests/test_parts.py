from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from miroir.config import Settings
from miroir.eval.parts import (
    VET_ENTITY,
    Kind,
    Rule,
    build_vet,
    detail_subject,
    entity_pair,
    is_blind,
    is_detail,
    whole_subject,
)
from miroir.eval.sets import Query, load_queries, write_set
from miroir.records.images import make_image_id
from miroir.records.model import Collection, Image, ImageType, Record
from miroir.records.store import read_records, write_records


def _record(
    collection: Collection,
    ref: str,
    entity: str,
    description: str = "",
    *,
    publishable: bool = True,
    types: tuple[ImageType, ...] = (ImageType.MAIN,),
) -> Record:
    images = tuple(
        Image(
            image_id=make_image_id(ref, f"https://x/{ref}/{n}.jpg"),
            ref=ref,
            entity_id=entity,
            collection=collection,
            url=f"https://x/{ref}/{n}.jpg",
            image_type=t,
            position=n,
            is_cover=n == 0,
        )
        for n, t in enumerate(types)
    )
    return Record(
        ref=ref,
        collection=collection,
        entity_id=entity,
        title="",
        description=description,
        denomination=(),
        domain=(),
        authors=(),
        period=(),
        region=None,
        departement=None,
        commune=None,
        publishable=publishable,
        pop_url="",
        linked_refs=(),
        images=images,
    )


def _photo(ref: str, entity: str, description: str, *, publishable: bool = True) -> Record:
    return _record(Collection.MEMOIRE, ref, entity, description, publishable=publishable)


def test_captions_name_the_detail_and_what_it_is_of() -> None:
    assert is_detail("verrière de la vierge, détail : saint jean")
    assert is_detail("retable, vue partielle")
    assert not is_detail("retable, vue générale")
    assert detail_subject("verrière de la vierge, détail : saint jean") == "verrière de la vierge"
    assert detail_subject("détail de l'aile droite") == ""
    assert whole_subject("verrière de la vierge, vue générale") == "verrière de la vierge"
    assert is_blind("calice, détail du poinçon")
    assert is_blind("retable, revers, détail")
    assert not is_blind("retable, détail : la mise au tombeau")


def test_an_object_detail_is_answered_by_the_photo_of_the_same_subject() -> None:
    records = [
        _photo("A1", "PM1", "Retable, haut-relief : Mise au tombeau, détail"),
        _photo("A2", "PM1", "Retable, haut-relief : Mise au tombeau, vue générale"),
        _photo("A3", "PM1", "Retable, vue générale"),
        _photo("A4", "PM1", "Retable, détail du poinçon"),  # blind: never the query
        _photo("A5", "PM1", "Retable, détail", publishable=False),
        _record(Collection.PALISSY, "PM1", "PM1", "Retable de la Passion"),
    ]
    pair = entity_pair("PM1", records, Kind.ALTARPIECE, random.Random(0))
    assert pair is not None
    assert pair.rule == Rule.SAME_SUBJECT
    q = pair.query
    assert q.query_image_id == records[0].images[0].image_id
    assert q.expected_entity_ids == ("PM1",)
    assert "A2" in q.why
    assert q.kind == "altarpiece"
    # Every detail photo leaves the gallery; the whole views stay.
    details = {r.images[0].image_id for r in (records[0], records[3], records[4])}
    assert set(q.exclude_image_ids) == details


def test_an_object_record_image_is_not_a_whole() -> None:
    records = [
        _photo("A1", "PM1", "statue, détail"),
        _record(Collection.PALISSY, "PM1", "PM1", "Statue : Vierge à l'Enfant"),
        _photo("A2", "PM1", "retable, vue générale"),  # another part of the same entity
    ]
    assert entity_pair("PM1", records, Kind.STATUE, random.Random(0)) is None


def test_a_detail_of_a_part_is_answered_by_the_whole_object() -> None:
    records = [
        _photo("A1", "PM1", "Retable, tableau : saint Maurice, détail"),
        _photo("A2", "PM1", "Retable"),
    ]
    pair = entity_pair("PM1", records, Kind.ALTARPIECE, random.Random(0))
    assert pair is not None
    assert pair.rule == Rule.CONTAINS


def test_a_building_detail_needs_a_photo_of_the_same_part() -> None:
    records = [
        _photo("B1", "PA1", "Portail ouest, détail"),
        _photo("B2", "PA1", "Vue générale depuis le sud"),
    ]
    assert entity_pair("PA1", records, Kind.BUILDING, random.Random(0)) is None
    records.append(_photo("B3", "PA1", "Portail ouest"))
    records.append(_photo("B6", "PA1", "Portail ouest, vue partielle"))
    records.append(_photo("B7", "PA1", "Portail ouest, détail des voussures"))
    records.append(_photo("B4", "PA1", "Tableau : chien et faisan, détail"))
    records.append(_photo("B5", "PA1", "Tableau : chien et faisan"))
    pair = entity_pair("PA1", records, Kind.BUILDING, random.Random(0))
    assert pair is not None
    assert pair.rule == Rule.SAME_SUBJECT
    assert "B3" in pair.query.why  # never the painting: the set's buildings are building parts


def test_vet_copies_its_photos_and_has_a_tune_half_only(settings: Settings, tmp_path: Path) -> None:
    sources = []
    for n in range(3):
        path = tmp_path / f"{n}.jpg"
        path.write_bytes(b"jpeg %d" % n)
        sources.append(path)
    queries = build_vet(settings, sources)
    assert [q.query_id for q in load_queries(settings, "VET", "tune")] == [
        "VET-1",
        "VET-2",
        "VET-3",
    ]
    assert all(q.expected_entity_ids == (VET_ENTITY,) for q in queries)
    copied = settings.outside_image_path(queries[1].query_image_id)
    assert copied.read_bytes() == b"jpeg 1"
    with pytest.raises(ValueError, match="takes 3 images"):
        build_vet(settings, sources[:2])


def test_sets_without_kind_or_why_are_written_as_before(settings: Settings) -> None:
    plain = Query("T1-x", "T1", "PA1", "x", ("PA1",), ())
    write_set(settings, "T1", [plain], [])
    line = json.loads(settings.testset_path("T1", "tune").read_text())
    assert "kind" not in line
    assert "why" not in line


def test_read_records_of_some_entities_only(settings: Settings) -> None:
    records = [_photo("A1", "PM1", "a"), _photo("A2", "PM2", "b"), _photo("A3", "PM1", "c")]
    settings.data_root.mkdir(parents=True)
    write_records(records, settings.records_parquet, settings.images_parquet)
    some = read_records(settings.records_parquet, settings.images_parquet, {"PM1"})
    assert [r.ref for r in some] == ["A1", "A3"]
    assert all(len(r.images) == 1 for r in some)
