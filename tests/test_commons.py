from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from miroir.commons import api
from miroir.commons.api import CommonsFile, WikidataItem
from miroir.commons.build import Targets, drop_shared, find_commons_photos, unambiguous
from miroir.commons.check import check_sample
from miroir.commons.client import WikimediaClient
from miroir.commons.model import (
    CommonsPhoto,
    CommonsSet,
    Source,
    View,
    commons_image_id,
    is_commons_image,
)
from miroir.commons.select import Candidate, Kind, Reject, photo_problem, pick, rank, view_of
from miroir.config import COMMONS_API_URL, WIKIDATA_SPARQL_URL, Half, Settings
from miroir.eval import build_sets
from miroir.eval.build_sets import Sizes, split_following
from miroir.eval.sets import Query, load_queries, read_gallery
from miroir.images.phash import distance, phash
from miroir.records.model import Collection, Image, ImageType, Record
from miroir.records.store import read_commons, write_commons, write_records
from tests.fakes import NO_WAIT

FIXTURES = Path(__file__).parent / "fixtures" / "commons"


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- pictures ---------------------------------------------------------------------


def _picture(seed: int, size: tuple[int, int] = (640, 480)) -> bytes:
    """A JPEG with shapes placed by `seed`, drawn at 640 x 480 then resized to `size`."""
    img = PILImage.new("RGB", (640, 480), (200, 190, 170))
    draw = ImageDraw.Draw(img)
    for n in range(6):
        x = (seed * 97 + n * 131) % 640
        y = (seed * 53 + n * 71) % 480
        shade = (seed * 40 + n * 30) % 255
        draw.rectangle((x, y, x + 200, y + 120), fill=(shade, 255 - shade, 90))
    out = io.BytesIO()
    img.resize(size).save(out, format="JPEG", quality=90)
    return out.getvalue()


def test_phash_ignores_size_and_tells_pictures_apart() -> None:
    a = phash(_picture(1))
    assert distance(a, phash(_picture(1, (320, 240)))) <= 4
    assert distance(a, phash(_picture(7))) > 16


# --- parsing real answers ---------------------------------------------------------


def test_sparql_rows_become_one_item_each(tmp_path: Path) -> None:
    client = _client(_FakeWikimedia(), tmp_path)
    items = {i.qid: i for i in api.items_with(client, api.MERIMEE)}
    item = items["Q427340"]
    assert item.ids == ("PA00086257",)
    assert item.images == ("File:Église Saint-Gervais.jpg",)
    assert item.categories == ("Category:Église Saint-Gervais-Saint-Protais de Paris",)


def test_category_listing_keeps_licence_author_exif_and_pop_links(tmp_path: Path) -> None:
    client = _client(_FakeWikimedia(), tmp_path)
    listing = api.category_files(client, "Category:Église Saint-Gervais-Saint-Protais de Paris")
    assert "Category:Interior of Église Saint-Gervais-Saint-Protais" in listing.subcategories
    files = {f.title.split(" ")[0]: f for f in listing.files}
    window = files["File:F0799"]
    assert window.licence == "CC BY-SA 4.0"
    assert window.author == "Mbzt"
    assert window.camera == "FUJIFILM X-Pro2"
    assert window.year == 2016
    assert window.latitude is None
    assert "?" not in window.thumb_url
    assert "/960px-F0799" in window.thumb_url
    plaque = next(f for f in listing.files if "plaque" in f.title)
    assert plaque.latitude == pytest.approx(48.8559, abs=1e-4)
    assert plaque.longitude == pytest.approx(2.3545, abs=1e-4)
    perugino = api.category_files(client, "Category:Marriage of the Virgin (Perugino)")
    by_title = {f.title: f for f in perugino.files}
    assert by_title["File:Pietro Perugino cat66.jpg"].pop_refs == ("06570007477",)


# --- which files pass -------------------------------------------------------------


def _file(**changes: object) -> CommonsFile:
    base = CommonsFile(
        title="File:Eglise Saint-Pierre.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Eglise_Saint-Pierre.jpg",
        url="https://upload.wikimedia.org/x/Eglise_Saint-Pierre.jpg",
        thumb_url="https://thumb.wikimedia.org/x/960px-Eglise_Saint-Pierre.jpg",
        width=4000,
        height=3000,
        mime="image/jpeg",
        licence="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0",
        author="A. Visitor",
        credit="Own work",
        taken="2019:06:01 10:00:00",
        latitude=None,
        longitude=None,
        camera="Apple iPhone 11",
        categories=("Église Saint-Pierre",),
        description="",
    )
    return replace(base, **changes)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({}, None),
        ({"mime": "image/png"}, Reject.NOT_JPEG),
        ({"width": 400}, Reject.TOO_SMALL),
        ({"pop_refs": ("PA00000002",)}, Reject.OTHER_RECORD),
        ({"title": "File:Plan de l'église Saint-Pierre.jpg"}, Reject.NOT_A_PHOTO),
        ({"categories": ("Postcards of Paris",)}, Reject.NOT_A_PHOTO),
        ({"categories": ("Médiathèque de l'architecture et du patrimoine",)}, Reject.ARCHIVE),
        ({"camera": ""}, Reject.NO_CAMERA),
        ({"camera": "EPSON Perfection V600"}, Reject.SCANNER),
        ({"camera": "Phase One P65+"}, Reject.SCANNER),
        ({"author": "Atget, Eugène (1857-1927), photographe"}, Reject.ARCHIVE),
        ({"categories": ("Images from Paris Musées",)}, Reject.ARCHIVE),
        ({"taken": ""}, Reject.NO_DATE),
        ({"taken": "1998:05:01 12:00:00"}, Reject.OLD),
    ],
)
def test_photo_problem(changes: dict[str, object], reason: Reject | None) -> None:
    assert photo_problem(_file(**changes), "PA00000001", Kind.BUILDING) == reason


def test_a_painting_is_a_photo_of_an_artwork_but_not_of_a_building() -> None:
    painting = _file(categories=("Paintings in the Musée des Augustins",))
    assert photo_problem(painting, "000PE000001", Kind.ARTWORK) is None
    assert photo_problem(painting, "PA00000001", Kind.BUILDING) == Reject.NOT_A_PHOTO


def test_a_file_linked_to_the_answer_passes() -> None:
    assert photo_problem(_file(pop_refs=("PA00000001",)), "PA00000001", Kind.BUILDING) is None


def test_view_from_subcategory_then_words() -> None:
    assert view_of(_file(), "Category:Interior of Église Saint-Pierre") == View.INTERIOR
    assert view_of(_file(title="File:Saint-Pierre, portail ouest.jpg")) == View.DETAIL
    assert view_of(_file(categories=("Naves in France",))) == View.INTERIOR
    assert view_of(_file()) == View.EXTERIOR


def _candidate(
    title: str, view: View, author: str = "A", source: Source = Source.CATEGORY
) -> Candidate:
    return Candidate(_file(title=title, author=author), source, "Category:X", view)


def test_pick_takes_one_per_view_then_other_visits_up_to_the_cap() -> None:
    ranked = [
        _candidate("File:a.jpg", View.EXTERIOR, "A"),
        _candidate("File:b.jpg", View.EXTERIOR, "A"),
        _candidate("File:c.jpg", View.EXTERIOR, "B"),
        _candidate("File:d.jpg", View.INTERIOR, "A"),
        _candidate("File:e.jpg", View.DETAIL, "C"),
    ]
    assert [c.file.title for c in pick(ranked, 3)] == ["File:a.jpg", "File:e.jpg", "File:c.jpg"]
    assert len(pick(ranked, 5)) == 5
    assert pick(ranked, 1) == [ranked[0]]


def test_rank_puts_p18_first_and_counts_a_file_once() -> None:
    import random  # noqa: PLC0415

    same = _candidate("File:a.jpg", View.EXTERIOR, source=Source.P18)
    ranked = rank(
        [_candidate("File:a.jpg", View.EXTERIOR), _candidate("File:b.jpg", View.EXTERIOR), same],
        random.Random(1),
    )
    assert [c.file.title for c in ranked] == ["File:a.jpg", "File:b.jpg"]
    assert ranked[0].source == Source.P18


def test_unambiguous_drops_ids_shared_between_items() -> None:
    items = [
        WikidataItem("Q1", ("PA1",), ("File:a.jpg",), ()),
        WikidataItem("Q2", ("PA2",), (), ("Category:B",)),
        WikidataItem("Q3", ("PA2",), (), ("Category:C",)),
        WikidataItem("Q4", ("PA3", "PA4"), (), ("Category:D",)),
        WikidataItem("Q5", ("PA5",), (), ()),
    ]
    found, dropped = unambiguous(items, lambda i: i != "PA1")
    assert set(found) == set()  # PA2 shared, PA3/PA4 on one item, PA5 has nowhere to look
    assert dropped == 3
    found, _ = unambiguous(items, lambda _i: True)
    assert set(found) == {"PA1"}


# --- the client -----------------------------------------------------------------------


def test_client_waits_out_maxlag_then_serves_from_cache(tmp_path: Path) -> None:
    calls: list[str] = []
    answers = iter(
        [
            httpx.Response(200, json={"error": {"code": "maxlag"}}, headers={"retry-after": "1"}),
            httpx.Response(200, json={"query": {"pages": []}}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return next(answers)

    slept: list[float] = []
    client = WikimediaClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        tmp_path,
        retry=NO_WAIT,
        clock=lambda: 0.0,
        sleep=slept.append,
    )
    params = {"action": "query", "titles": "File:a.jpg"}
    assert client.get_json(COMMONS_API_URL, params) == {"query": {"pages": []}}
    assert client.get_json(COMMONS_API_URL, params) == {"query": {"pages": []}}
    assert len(calls) == 2
    assert 1.0 in slept  # the maxlag wait
    assert client.requests == 2


def test_client_sends_at_most_four_a_second(tmp_path: Path) -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    client = WikimediaClient(
        httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={}))),
        tmp_path,
        retry=NO_WAIT,
        clock=lambda: now[0],
        sleep=sleep,
    )
    for n in range(5):
        client.get_json(COMMONS_API_URL, {"n": str(n)})
    assert now[0] == pytest.approx(1.0)


# --- the whole search, against a fake Wikimedia -------------------------------------


class _FakeWikimedia:
    """Serves the fixture answers: SPARQL, category listings, file info and thumbnails."""

    def __init__(self, thumbs: dict[str, bytes] | None = None) -> None:
        self.thumbs = thumbs or {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        if url.startswith(WIKIDATA_SPARQL_URL):
            name = "sparql-p380.json" if "P380" in query["query"] else "sparql-p347.json"
            return httpx.Response(200, json=_fixture(name))
        if url.startswith(COMMONS_API_URL):
            if "gcmtitle" in query:
                name = _CATEGORY_FIXTURES.get(query["gcmtitle"])
                body = _fixture(name) if name else {"batchcomplete": True}
                return httpx.Response(200, json=body)
            return httpx.Response(200, json=_fixture("titles.json"))
        path = urlsplit(url).path
        for name, data in self.thumbs.items():
            if name in path:
                return httpx.Response(200, content=data)
        return httpx.Response(404)


_CATEGORY_FIXTURES = {
    "Category:Église Saint-Gervais-Saint-Protais de Paris": "category-saint-gervais.json",
    "Category:Interior of Église Saint-Gervais-Saint-Protais": (
        "category-saint-gervais-interior.json"
    ),
    "Category:Marriage of the Virgin (Perugino)": "category-perugino.json",
}


def _client(fake: _FakeWikimedia, cache: Path) -> WikimediaClient:
    http = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return WikimediaClient(
        http,
        cache,
        retry=NO_WAIT,
        max_per_second=1e9,
        sleep=lambda _s: None,
    )


def _record(collection: Collection, ref: str, entity: str, images: int = 1) -> Record:
    imgs = tuple(
        Image(
            image_id=f"{ref.lower()[-8:]}{n:08d}"[-16:].rjust(16, "0"),
            ref=ref,
            entity_id=entity,
            collection=collection,
            url=f"https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/{ref}/{n}.jpg",
            image_type=ImageType.MAIN,
            position=n,
            is_cover=n == 0,
        )
        for n in range(images)
    )
    return Record(
        ref=ref,
        collection=collection,
        entity_id=entity,
        title=f"Title of {ref}",
        description="",
        denomination=(),
        domain=(),
        authors=(),
        period=(),
        region=None,
        departement=None,
        commune=None,
        publishable=True,
        pop_url=f"https://pop.culture.gouv.fr/notice/x/{ref}",
        linked_refs=(),
        images=imgs,
    )


def test_find_photos_for_a_building_and_drops_the_museum_copy(tmp_path: Path) -> None:
    rose = _picture(3)
    fake = _FakeWikimedia(
        thumbs={
            # One file is cropped to the painting: the Joconde image again. The other is not.
            "Cropped": _picture(3, (500, 600)),
            "500px-Caen": _picture(9, (500, 375)),
        }
    )
    records = [
        _record(Collection.MEMOIRE, "APMH00000001", "PA00086257", images=2),
        _record(Collection.JOCONDE, "06570007477", "06570007477"),
    ]
    photos, report = find_commons_photos(
        records,
        _client(fake, tmp_path),
        lambda _image: rose,
        Targets(t2=10, t8_buildings=0, t8_artworks=0),
    )
    by_set = {s: [p for p in photos if p.set_id == s] for s in CommonsSet}
    assert 1 <= len(by_set[CommonsSet.T2]) <= 3
    assert {p.entity_id for p in by_set[CommonsSet.T2]} == {"PA00086257"}
    assert {p.view for p in by_set[CommonsSet.T2]} >= {View.INTERIOR}
    assert all(is_commons_image(p.image_id) for p in photos)
    t2b = by_set[CommonsSet.T2B]
    assert [p.file_title.split(" ")[0] for p in t2b] == ["File:Caen"]
    assert t2b[0].phash_distance is not None
    assert t2b[0].phash_distance > Targets().near_copy_bits
    assert report.sets[1].rejects.get(Reject.NEAR_COPY) == 1
    assert report.requests > 0
    # A second run reads everything from the cache.
    again, report2 = find_commons_photos(
        records,
        _client(_FakeWikimedia(), tmp_path),
        lambda _image: rose,
        Targets(t2=10, t8_buildings=0, t8_artworks=0),
    )
    assert again == photos
    assert report2.requests == 0


def test_a_file_chosen_for_two_entities_is_dropped() -> None:
    photo = _photo("File:a.jpg", CommonsSet.T2, "PA1")
    kept, shared = drop_shared([photo, replace(photo, entity_id="PA2"), _photo("File:b.jpg")])
    assert [p.file_title for p in kept] == ["File:b.jpg"]
    assert shared == 1


# --- sets and galleries -----------------------------------------------------------------


def _photo(title: str, set_id: CommonsSet = CommonsSet.T2, entity: str = "PA1") -> CommonsPhoto:
    return CommonsPhoto(
        image_id=commons_image_id(title),
        set_id=set_id,
        entity_id=entity,
        wikidata_id="Q1",
        file_title=title,
        page_url="https://commons.wikimedia.org/wiki/" + title,
        url="https://upload.wikimedia.org/" + title,
        thumb_url="https://thumb.wikimedia.org/" + title,
        width=4000,
        height=3000,
        licence="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0",
        author="A. Visitor",
        credit="Own work",
        taken="2019:06:01",
        latitude=48.8,
        longitude=2.3,
        camera="Apple iPhone 11",
        source=Source.CATEGORY,
        category="Category:X",
        view=View.EXTERIOR,
    )


def test_commons_table_round_trips(tmp_path: Path) -> None:
    photos = [_photo("File:a.jpg"), replace(_photo("File:b.jpg"), latitude=None, phash_distance=17)]
    write_commons(photos, tmp_path / "commons.parquet")
    assert read_commons(tmp_path / "commons.parquet") == photos


def test_split_following_keeps_known_halves_and_balances_the_rest() -> None:
    queries = [Query(f"T2-{n}", "T2", f"E{n}", f"c{n}", (f"E{n}",), ()) for n in range(10)]
    fixed: dict[str, Half] = {"E0": "tune", "E1": "tune", "E2": "sealed"}
    split = split_following("T2", queries, fixed)
    tune = {q.entity_id for q in split.tune}
    sealed = {q.entity_id for q in split.sealed}
    assert {"E0", "E1"} <= tune
    assert "E2" in sealed
    assert len(tune) == len(sealed) == 5
    plain = build_sets.split_by_entity("T2", queries)
    assert split_following("T2", queries, {}) == plain


def _corpus_with_commons() -> tuple[list[Record], list[CommonsPhoto]]:
    records = [
        *(
            _record(Collection.MEMOIRE, f"APMH{b:08d}", f"PA{b:08d}", images=1 + b % 3)
            for b in range(30)
        ),
        *(
            _record(Collection.JOCONDE, f"J{n:05d}", f"J{n:05d}", images=1 + n % 2)
            for n in range(30)
        ),
    ]
    photos: list[CommonsPhoto] = [
        *(
            _photo(f"File:b{b}-{k}.jpg", CommonsSet.T2, f"PA{b:08d}")
            for b in range(0, 30, 2)
            for k in range(2)
        ),
        *(_photo(f"File:w{n}.jpg", CommonsSet.T2B, f"J{n:05d}") for n in range(0, 30, 3)),
        *(_photo(f"File:x{n}.jpg", CommonsSet.T8C, f"PA9{n:07d}") for n in range(6)),
        _photo("File:stale.jpg", CommonsSet.T8C, "PA00000001"),  # now in the table: dropped
    ]
    return records, photos


def test_build_adds_commons_sets_and_a_v2_gallery_with_their_answers(settings: Settings) -> None:
    records, photos = _corpus_with_commons()
    write_records(records, settings.records_parquet, settings.images_parquet)
    write_commons(photos, settings.commons_parquet)
    sizes = Sizes(t1=10, t5=10, t8_per_part=2, screening=60)
    report = build_sets.build(settings, sizes)

    sets = {s.set_id: s for s in report.sets}
    assert sets["T2"].tune + sets["T2"].sealed == 30
    assert sets["T2b"].tune + sets["T2b"].sealed == 10
    assert sets["T8"].tune + sets["T8"].sealed == 8
    assert sets["T8c"].tune + sets["T8c"].sealed == 6
    assert sets["T8c"].dropped == 1

    def both(set_id: str) -> list[Query]:
        return load_queries(settings, set_id, "tune") + load_queries(
            settings, set_id, "sealed", allow_sealed=True
        )

    halves: dict[str, str] = {}
    for set_id in ("T1", "T3", "T5", "T8", "T2", "T2b", "T8c"):
        for half in ("tune", "sealed"):
            queries = load_queries(settings, set_id, half, allow_sealed=True)
            for q in queries:
                assert halves.setdefault(q.entity_id, half) == half
    assert not any(is_commons_image(q.query_image_id) for q in both("T8"))
    t8_commons = both("T8c")
    assert len(t8_commons) == 6
    assert all(q.expected_entity_ids == () and q.exclude_image_ids == () for q in t8_commons)

    screening = set(read_gallery(settings, build_sets.SCREENING))
    v2 = set(read_gallery(settings, build_sets.SCREENING_V2))
    delta = set(read_gallery(settings, build_sets.SCREENING_V2 + build_sets.DELTA))
    assert screening <= v2
    assert delta == v2 - screening
    assert len(delta) == report.delta
    entity_images = build_sets.RecordIndex(records).entity_images
    for q in both("T2") + both("T2b"):
        assert set(entity_images[q.entity_id]) <= v2
        assert q.exclude_image_ids == ()
    for name in (
        build_sets.FULL,
        build_sets.SCREENING,
        build_sets.SCREENING_PUBLISHABLE,
        build_sets.SCREENING_V2,
        build_sets.SCREENING_V2_PUBLISHABLE,
    ):
        assert not any(is_commons_image(i) for i in read_gallery(settings, name))


def test_build_without_commons_writes_no_v2_gallery(settings: Settings) -> None:
    records, _ = _corpus_with_commons()
    write_records(records, settings.records_parquet, settings.images_parquet)
    report = build_sets.build(settings, Sizes(t1=10, t5=10, t8_per_part=2, screening=60))
    assert report.screening_v2 == 0
    assert not settings.gallery_path(build_sets.SCREENING_V2).exists()


def test_closed_models_may_see_commons_photos() -> None:
    records, _ = _corpus_with_commons()
    index = build_sets.RecordIndex(records)
    photo = commons_image_id("File:a.jpg")
    q = Query(f"T2-{photo}", "T2", "PA00000000", photo, ("PA00000000",), ())
    gallery = set(index.entity_images["PA00000000"])
    assert build_sets.closed_ok(q, gallery, index)
    assert not build_sets.closed_ok(q, set(), index)


def test_check_sample_pairs_the_photo_with_an_answer_image(settings: Settings) -> None:
    records, photos = _corpus_with_commons()
    write_records(records, settings.records_parquet, settings.images_parquet)
    write_commons(photos, settings.commons_parquet)
    build_sets.build(settings, Sizes(t1=10, t5=10, t8_per_part=2, screening=60))
    sample = check_sample(settings, size=8)
    assert len(sample) == 8
    tune = {q.query_id for s in ("T2", "T2b") for q in load_queries(settings, s, "tune")}
    for row in sample:
        assert row["query_id"] in tune
        assert row["commons_thumb_url"].startswith("https://thumb.wikimedia.org/")
        assert row["answer_image_url"].startswith("https://popcorn-prd-perf-assets")
        assert row["pop_title"].startswith("Title of")
