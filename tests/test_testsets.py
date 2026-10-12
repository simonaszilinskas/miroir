from __future__ import annotations

import pytest

from miroir.config import Settings
from miroir.eval import build_sets
from miroir.eval.build_sets import (
    RecordIndex,
    Sizes,
    SplitSet,
    check_halves_apart,
    needed_images,
    same_shoot,
    screening_gallery,
    split_by_entity,
    t1_queries,
    t3_queries,
    t5_queries,
)
from miroir.eval.sets import Query, load_queries, read_gallery, write_set
from miroir.records.build import build as build_records
from miroir.records.images import make_image_id
from miroir.records.model import Collection, Image, ImageType, Record
from miroir.records.store import read_records


def _record(
    collection: Collection,
    ref: str,
    n_images: int,
    *,
    entity_id: str | None = None,
    publishable: bool = True,
    photographers: tuple[str, ...] = (),
    photo_date: str = "",
) -> Record:
    entity = entity_id or ref
    images = tuple(
        Image(
            image_id=make_image_id(ref, f"https://x/{ref}/{n}.jpg"),
            ref=ref,
            entity_id=entity,
            collection=collection,
            url=f"https://x/{ref}/{n}.jpg",
            image_type=ImageType.MAIN,
            position=n,
            is_cover=n == 0,
        )
        for n in range(n_images)
    )
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
        publishable=publishable,
        pop_url="",
        linked_refs=(),
        images=images,
        photographers=photographers,
        photo_date=photo_date,
    )


def _corpus() -> list[Record]:
    """40 buildings with 1-4 one-photo records each, 40 Joconde records, 30 loose records.

    A building's photos are refs 100 apart, so each is its own shoot.
    """
    records: list[Record] = []
    for b in range(40):
        building = f"PA{b:08d}"
        records.extend(
            _record(Collection.MEMOIRE, f"APMH{b * 1000 + p * 100:08d}", 1, entity_id=building)
            for p in range(1 + b % 4)
        )
    records.extend(_record(Collection.JOCONDE, f"J{n:05d}", 1 + n % 3) for n in range(40))
    records.extend(_record(Collection.MEMOIRE, f"LOOSE{n:03d}", 1) for n in range(30))
    return records


SMALL = Sizes(t1=20, t5=20, t8_per_part=4, screening=150)


def test_t1_one_query_per_building_with_two_or_more_photos() -> None:
    records = _corpus()
    queries = t1_queries(records, size=100)
    eligible = {f"PA{b:08d}" for b in range(40) if 1 + b % 4 >= 2}
    assert {q.entity_id for q in queries} == eligible  # fewer than asked: all of them
    by_image = {i.image_id: r for r in records for i in r.images}
    for q in queries:
        assert q.expected_entity_ids == (q.entity_id,)
        record = by_image[q.query_image_id]
        assert q.exclude_image_ids == tuple(i.image_id for i in record.images)


def _photo(ref: str, building: str, who: str = "", when: str = "") -> Record:
    photographers = (who,) if who else ()
    return _record(
        Collection.MEMOIRE,
        ref,
        1,
        entity_id=building,
        photographers=photographers,
        photo_date=when,
    )


def test_same_shoot_is_photographer_and_date_or_near_refs() -> None:
    a = _photo("APMH00001000", "PA1", "Hurault", "1944")
    assert same_shoot(a, _photo("APMH00009000", "PA1", "Hurault", "1944"))
    assert not same_shoot(a, _photo("APMH00009000", "PA1", "Hurault", "1945"))
    assert not same_shoot(a, _photo("APMH00009000", "PA1", "Bonnel", "1944"))
    assert same_shoot(a, _photo("APMH00001050", "PA1"))  # no metadata, ref within the window
    assert not same_shoot(a, _photo("APMH00001051", "PA1"))
    assert not same_shoot(a, _photo("APLP00001001", "PA1"))  # another series
    assert same_shoot(_photo("AP2322T000470", "PA1"), _photo("AP2322T000471", "PA1"))
    assert same_shoot(a, a)


def test_t1_hides_the_query_shoot_and_drops_buildings_with_one_shoot() -> None:
    two_shoots = [
        _photo("APMH00001000", "PA1", "Hurault", "1944"),
        _photo("APMH00001001", "PA1", "Hurault", "1944"),
        _photo("APMH00007000", "PA1", "Bonnel", "1970"),
    ]
    one_shoot = [_photo(f"APMH0000200{n}", "PA2", "Bonnel", "1968") for n in range(3)]
    records = [*two_shoots, *one_shoot]
    ref_of = {r.images[0].image_id: r.ref for r in records}
    queries = {q.entity_id: q for q in t1_queries(records, size=2)}

    q = queries["PA1"]
    hidden = {ref_of[i] for i in q.exclude_image_ids}
    if ref_of[q.query_image_id] == "APMH00007000":
        assert hidden == {"APMH00007000"}
    else:
        assert hidden == {"APMH00001000", "APMH00001001"}
    assert build_sets.answerable(q, RecordIndex(records))
    assert not build_sets.answerable(queries["PA2"], RecordIndex(records))


def test_dropped_queries_keep_their_images_in_the_galleries() -> None:
    one_shoot = [_photo(f"APMH0900000{n}", "PA99", "Bonnel", "1968") for n in range(3)]
    records = [*_corpus(), *one_shoot]
    built = build_sets.build_sets(records, Sizes(t1=100, t5=20, t8_per_part=0, screening=150))
    kept = {q.entity_id for s in built.splits for q in s.queries if s.set_id == "T1"}
    drawn = {q.entity_id for s in built.drawn for q in s.queries if s.set_id == "T1"}
    assert drawn - kept == {"PA99"}
    assert built.dropped["T1"] == 1
    needed = needed_images((q for s in built.drawn for q in s.queries), RecordIndex(records))
    assert {r.images[0].image_id for r in one_shoot} <= needed


def test_t5_leaves_the_other_views_in_the_gallery() -> None:
    queries = t5_queries(_corpus(), size=5)
    assert len(queries) == 5
    for q in queries:
        assert q.exclude_image_ids == (q.query_image_id,)
        assert q.expected_entity_ids == (q.entity_id,)


def test_t3_queries_an_artwork_crop_and_names_its_room_photo(settings: Settings) -> None:
    build_records(settings)
    records = read_records(settings.records_parquet, settings.images_parquet)
    queries = {q.entity_id: q for q in t3_queries(records)}
    # Fixture: room photo JDP-0241 and two artworks hung in it, each with two crops.
    assert set(queries) == {"JDP-0228", "JDP-0230"}
    room = next(r for r in records if r.ref == "JDP-0241")
    artwork = next(r for r in records if r.ref == "JDP-0230")
    q = queries["JDP-0230"]
    assert q.scene_image_ids == (room.cover.image_id,)
    assert q.query_image_id in {i.image_id for i in artwork.images}
    assert q.exclude_image_ids == (q.query_image_id,)


def test_split_puts_each_entity_in_one_half_only() -> None:
    splits = build_sets.build_sets(_corpus(), SMALL).splits
    check_halves_apart(splits)
    tune = {q.entity_id for s in splits for q in s.tune}
    sealed = {q.entity_id for s in splits for q in s.sealed}
    assert tune
    assert sealed
    assert not tune & sealed
    for split in splits:
        assert abs(len(split.tune) - len(split.sealed)) <= 1


def test_t8_removes_the_whole_entity_and_expects_nothing() -> None:
    records = _corpus()
    index = RecordIndex(records)
    t8 = {s.set_id: s for s in build_sets.build_sets(records, SMALL).splits}["T8"]
    assert len(t8.tune) == len(t8.sealed) == 2 * SMALL.t8_per_part
    for q in t8.queries:
        assert q.expected_entity_ids == ()
        assert set(q.exclude_image_ids) == set(index.entity_images[q.entity_id])


def test_overlapping_halves_are_refused() -> None:
    q = Query("T1-a", "T1", "PA1", "a", ("PA1",), ("a",))
    other = Query("T8-b", "T8", "PA1", "b", (), ("b",))
    with pytest.raises(ValueError, match="both halves"):
        check_halves_apart([SplitSet("T1", (q,), ()), SplitSet("T8", (), (other,))])


def test_same_seed_same_sets_other_seed_other_sets() -> None:
    records = _corpus()
    first = build_sets.build_sets(records, SMALL, seed=1)
    assert build_sets.build_sets(list(reversed(records)), SMALL, seed=1) == first
    assert build_sets.build_sets(records, SMALL, seed=2) != first


def test_split_by_entity_rounds_tune_up() -> None:
    queries = [Query(f"T3-{n}", "T3", f"E{n}", str(n), (f"E{n}",), (str(n),)) for n in range(5)]
    split = split_by_entity("T3", queries)
    assert (len(split.tune), len(split.sealed)) == (3, 2)


def test_screening_gallery_keeps_needed_images_and_fills_with_distractors() -> None:
    records = _corpus()
    index = RecordIndex(records)
    splits = build_sets.build_sets(records, SMALL).splits
    needed = needed_images((q for s in splits for q in s.queries), index)
    gallery = screening_gallery(index.image_publishable, needed, size=150)
    assert len(gallery) == 150
    assert needed <= gallery
    with pytest.raises(ValueError, match="screening"):
        screening_gallery(index.image_publishable, needed, size=len(needed) - 1)


def test_sealed_half_loads_only_with_allow_sealed(settings: Settings) -> None:
    q = Query("T1-a", "T1", "PA1", "a", ("PA1",), ("a",))
    s = Query("T1-b", "T1", "PA2", "b", ("PA2",), ("b",))
    write_set(settings, "T1", [q], [s])
    assert load_queries(settings, "T1", "tune") == [q]
    with pytest.raises(PermissionError):
        load_queries(settings, "T1", "sealed")
    assert load_queries(settings, "T1", "sealed", allow_sealed=True) == [s]


def test_build_writes_sets_and_galleries(settings: Settings) -> None:
    build_records(settings)
    report = build_sets.build(settings, Sizes(t1=5, t5=5, t8_per_part=1, screening=25))
    full = read_gallery(settings, build_sets.FULL)
    screening = set(read_gallery(settings, build_sets.SCREENING))
    publishable = set(read_gallery(settings, build_sets.SCREENING_PUBLISHABLE))
    assert len(screening) == report.screening == 25
    assert publishable <= screening <= set(full)
    assert len(publishable) == report.screening_publishable
    t3 = load_queries(settings, "T3", "tune") + load_queries(
        settings, "T3", "sealed", allow_sealed=True
    )
    assert len(t3) == 2
