"""The website's search: grouping by entity, filters, G4 routing and fusion, the cut-off."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miroir.config import Settings
from miroir.geo.model import Place, PlaceSource
from miroir.records.store import write_places as write_places_parquet
from miroir.web import facets, ranking
from miroir.web.bundle import add_places, cutoff_for
from miroir.web.catalog import CATALOG_FILE, CatalogData, Filters, write_catalog
from miroir.web.engine import Engine, SearchRequest, prepare_photo
from miroir.web.levels import WORDS_ROUTE, Level, Levels, RouteLevels
from miroir.web.nearby import NEAR_KM, PLACES_FILE, Position, read_places, write_places
from miroir.web.routing import Bucket, PhotoType, Routing
from tests import web_helpers as h

# --- grouping (functional spec §6) ---------------------------------------------------------


def test_an_entity_scores_its_best_image() -> None:
    catalog = h.catalog()
    scores = np.asarray([0.1, 0.7, 0.2, 0.9, 0.3, 0.4], dtype=np.float32)
    per_entity = ranking.entity_max(scores, catalog)
    assert dict(zip(catalog.entity_ids, per_entity.tolist(), strict=True)) == pytest.approx(
        {"OBJ1": 0.9, "OBJ2": 0.3, "PA1": 0.7}
    )


def test_hidden_images_drop_out_and_an_emptied_entity_is_removed() -> None:
    catalog = h.catalog()
    scores = np.asarray([0.1, -np.inf, -np.inf, -np.inf, 0.3, 0.4], dtype=np.float32)
    per_entity = dict(zip(catalog.entity_ids, ranking.entity_max(scores, catalog), strict=True))
    assert per_entity["PA1"] == pytest.approx(0.4)  # the Commons photo now leads
    assert per_entity["OBJ1"] == -np.inf


def test_one_card_per_entity_with_its_best_image_marked_when_not_the_cover() -> None:
    engine, _ = h.engine(building=False)
    response = engine.search(SearchRequest(image=h.photo(130)))  # row 3: the back of OBJ1
    ids = [c.entity_id for c in response.cards]
    assert len(ids) == len(set(ids)) == 3
    first = response.cards[0]
    assert first.entity_id == "OBJ1"
    assert first.matched == "Revers"
    assert first.image_url is not None
    assert first.image_url.endswith("j2.jpg")
    assert first.title == "Vase"  # empty title: the denomination stands in
    assert first.photo_count == 2


def test_a_commons_match_shows_the_entity_pop_image() -> None:
    engine, _ = h.engine(building=True)
    response = engine.search(SearchRequest(image=h.photo(210)))  # row 5: the Commons photo
    first = response.cards[0]
    assert first.entity_id == "PA1"
    assert first.image_url is not None
    assert first.image_url.split("/")[-1] in {"m1.jpg", "m2.jpg"}  # a POP photo, not Commons
    assert first.matched is None  # both are covers
    assert first.photo_count == 2  # the Commons photo is not counted
    assert first.title.startswith("Église Saint-Pierre")


def test_an_entity_with_only_a_commons_photo_has_a_card_without_image() -> None:
    engine, _ = h.engine(building=True)
    engine.catalog = h.catalog(pop_images=False)
    first = engine.search(SearchRequest(image=h.photo(210))).cards[0]
    assert first.entity_id == "PA1"
    assert first.image_url is None
    assert first.photo_count == 0


# --- filters (F3) --------------------------------------------------------------------------


def test_filters_apply_before_ranking() -> None:
    engine, _ = h.engine(building=False)
    request = SearchRequest(image=h.photo(130), filters=Filters(collections=("memoire",)))
    response = engine.search(request)
    assert [c.entity_id for c in response.cards] == ["PA1"]


def test_a_commons_photo_follows_its_entity_records_through_filters() -> None:
    catalog = h.catalog()
    mask = catalog.image_mask(Filters(region="Occitanie"))
    assert mask.tolist() == [True, True, False, False, False, True]


def test_old_region_names_count_as_today_s_region() -> None:
    catalog = h.catalog()
    assert catalog.record_mask(Filters(region="Occitanie")).tolist() == [True, True, False, False]
    assert facets.region("Ile-de-France") == "Île-de-France"
    assert facets.region(None) is None


def test_domain_and_period_filters_use_tidied_values() -> None:
    catalog = h.catalog()
    assert catalog.record_mask(Filters(domain="Architecture")).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert catalog.record_mask(Filters(period="18e siècle")).tolist() == [
        False,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        (["4e quart 19e siècle"], ("19e siècle",)),
        (["années 1960"], ("20e siècle",)),
        (["romain impérial"], ("Antiquité",)),
        (["1724"], ("18e siècle",)),
        (["XIIe siècle"], ("Moyen Âge",)),
        (["paléolithique supérieur"], ("Préhistoire",)),
        (["3e quart 3e siècle"], ("Antiquité",)),
        ([], ()),
    ],
)
def test_periods_become_broad_periods(written: list[str], expected: tuple[str, ...]) -> None:
    assert facets.periods(written) == expected


def test_facet_options_count_records() -> None:
    options = h.catalog().facet_options()
    assert [(v.value, v.count) for v in options.collections] == [
        ("joconde", 1),
        ("memoire", 2),
        ("rose-valland", 1),
    ]
    assert [v.value for v in options.departements[""]] == ["Haute-Garonne", "Seine-Maritime"]
    assert ("Architecture", 2) in [(v.value, v.count) for v in options.domains]


# --- routing and fusion (G4) ---------------------------------------------------------------


def test_fusion_is_a_weighted_sum_of_min_max_scores() -> None:
    a = np.asarray([0.2, 0.6, 1.0], dtype=np.float32)
    b = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    fused = ranking.fuse([ranking.Signal(a, 0.2), ranking.Signal(b, 0.8)])
    assert fused.tolist() == pytest.approx([0.8, 0.1, 0.6])


def test_an_entity_removed_from_one_signal_stays_out() -> None:
    a = np.asarray([0.2, -np.inf, 1.0], dtype=np.float32)
    b = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    fused = ranking.fuse([ranking.Signal(a, 0.5), ranking.Signal(b, 0.5)])
    assert fused[1] == -np.inf
    assert ranking.top_entities(fused, 5).tolist() == [2, 0]


def test_routing_picks_the_first_bucket_that_fits() -> None:
    routing = Routing(
        buckets=(
            Bucket("archive", 1.0, 0.9, PhotoType.BUILDING, greyscale=True),
            Bucket("colour-building", 0.3, 0.8, PhotoType.BUILDING),
            Bucket("object", 0.2, 0.7, PhotoType.OBJECT),
        ),
        grey_below=2.0,
    )
    assert routing.choose(PhotoType.BUILDING, 0.5).name == "archive"
    assert routing.choose(PhotoType.BUILDING, 40.0).name == "colour-building"
    assert routing.choose(PhotoType.BUILDING, None).name == "archive"
    assert routing.choose(PhotoType.OBJECT, 0.5).name == "object"


def test_a_building_guess_routes_to_embeddinggemma_alone() -> None:
    engine, _ = h.engine(building=True)
    response = engine.search(SearchRequest(image=h.photo(10)))
    assert response.guessed_type == "building"
    assert response.bucket == "building"
    assert response.cards[0].entity_id == "PA1"


def test_a_grey_photo_reports_no_chroma() -> None:
    engine, _ = h.engine(building=True)
    response = engine.search(SearchRequest(image=h.photo(120)))
    assert response.chroma is not None
    assert response.chroma < 1


# --- "not found" (T8 cut-off) --------------------------------------------------------------


def test_cutoff_keeps_at_most_the_target_share_confident() -> None:
    wrong = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    cutoff = cutoff_for(wrong, 0.10)
    assert 0.8 < cutoff < 0.8 + 1e-9  # one of ten may stay confident: just above the second
    assert (wrong >= cutoff).mean() == pytest.approx(0.1)
    assert (wrong >= cutoff_for(wrong, 0.05)).mean() == 0.0


def test_every_card_has_a_level_and_only_the_first_can_be_more_than_a_likeness() -> None:
    engine, _ = h.engine(building=True)
    engine.levels = Levels(routes=(RouteLevels("building", 0.5, 0.0, 0.0, 0.0),))
    cards = engine.search(SearchRequest(image=h.photo(10))).cards
    assert cards[0].level is Level.VERY_CLOSE
    assert cards[0].level_label == "Très proche"
    assert {c.level for c in cards[1:]} == {Level.LIKENESS}
    engine.levels = Levels(routes=(RouteLevels("building", 1.5, 0.0, 0.5, 0.0),))
    cards = engine.search(SearchRequest(image=h.photo(10))).cards
    assert [c.level for c in cards][:2] == [Level.CLOSE, Level.LIKENESS]


def test_without_levels_below_the_cutoff_every_card_is_a_likeness() -> None:
    sure, _ = h.engine(building=True, cutoff=0.5)
    unsure, _ = h.engine(building=True, cutoff=1.5)
    request = SearchRequest(image=h.photo(10))
    assert sure.search(request).cards[0].level is Level.VERY_CLOSE
    found = unsure.search(request)
    assert {c.level for c in found.cards} == {Level.LIKENESS}
    assert found.cards  # the closest cards are still listed


def test_words_alone_read_the_words_route() -> None:
    engine, _ = h.engine()
    engine.levels = Levels(routes=(RouteLevels(WORDS_ROUTE, None, None, 0.0, 0.0),))
    response = engine.search(SearchRequest(text="lièvre"))
    assert response.mode == "text"
    assert response.confidence is None
    assert response.cards[0].level is Level.CLOSE
    assert response.words_found is True
    assert response.cards[0].entity_id == "OBJ2"


def test_words_no_record_holds_are_flagged_and_still_answered() -> None:
    engine, _ = h.engine()
    response = engine.search(SearchRequest(text="qsdfghjk"))
    assert response.words_found is False
    assert response.cards  # the text vector still ranks images
    assert engine.search(SearchRequest(image=h.photo(130))).words_found is None


def test_words_refine_a_photo_search() -> None:
    engine, _ = h.engine(building=False)
    plain = engine.search(SearchRequest(image=h.photo(130)))
    refined = engine.search(SearchRequest(image=h.photo(130), text="Toulouse église"))
    assert plain.cards[0].entity_id == "OBJ1"
    assert refined.mode == "photo+text"
    assert refined.cards[0].entity_id == "PA1"


def test_a_search_needs_a_photo_or_words() -> None:
    engine, _ = h.engine()
    with pytest.raises(ValueError, match="photo or words"):
        engine.search(SearchRequest())


def test_query_photos_are_shrunk_like_gallery_images() -> None:
    import io  # noqa: PLC0415

    from PIL import Image as PILImage  # noqa: PLC0415

    jpeg = prepare_photo(h.photo(100, size=(2000, 1000)))
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        assert opened.format == "JPEG"
        assert opened.size == (512, 256)


def test_a_guess_saved_by_index_guess_reads_too(tmp_path: Path) -> None:
    from miroir.web.routing import read_guess  # noqa: PLC0415

    path = tmp_path / "guess.npz"
    np.savez(path, mean=np.zeros(3), scale=np.ones(3), coef=np.ones(3), bias=np.float32(-2.0))
    guess = read_guess(path)
    assert guess.bias == pytest.approx(-2.0)
    assert guess.photo_type(np.asarray([3.0, 0.0, 0.0], dtype=np.float32))[0] is PhotoType.BUILDING


# --- position (H2) ---------------------------------------------------------------------------


def _placed_engine(tmp_path: Path) -> Engine:
    engine, _ = h.engine(building=False)
    path = tmp_path / PLACES_FILE
    write_places(path, engine.catalog.entity_ids, {"PA1": (43.6, 1.44), "OBJ1": (49.44, 1.09)})
    places = read_places(path, engine.catalog.entity_ids)
    assert places is not None
    return Engine(
        catalog=engine.catalog,
        routing=engine.routing,
        guess=engine.guess,
        encoders=engine.encoders,
        gallery=engine.gallery,
        places=places,
    )


def test_a_position_lifts_what_is_near(tmp_path: Path) -> None:
    engine = _placed_engine(tmp_path)
    toulouse = Position(43.601, 1.441)  # about 100 m from PA1
    plain = engine.search(SearchRequest(image=h.photo(130)))
    assert plain.cards[0].entity_id == "OBJ1"  # the photo looks like OBJ1, in Rouen
    near = engine.search(SearchRequest(image=h.photo(130), position=toulouse))
    assert near.cards[0].entity_id == "PA1"
    assert len(near.cards) == 3  # nothing is removed; OBJ2 (no place) gets no bonus
    assert near.nearby == 1
    assert near.near_km == NEAR_KM


def test_with_nothing_near_the_ranking_is_unchanged(tmp_path: Path) -> None:
    engine = _placed_engine(tmp_path)
    plain = engine.search(SearchRequest(image=h.photo(130)))
    far = engine.search(SearchRequest(image=h.photo(130), position=Position(-33.9, 151.2)))
    assert far.nearby == 0
    assert [c.entity_id for c in far.cards] == [c.entity_id for c in plain.cards]


def test_without_a_position_nothing_changes(tmp_path: Path) -> None:
    engine = _placed_engine(tmp_path)
    response = engine.search(SearchRequest(image=h.photo(130)))
    assert response.nearby is None
    assert response.cards[0].entity_id == "OBJ1"


def test_a_bundle_without_places_ignores_positions(tmp_path: Path) -> None:
    assert read_places(tmp_path / PLACES_FILE, ("PA1",)) is None
    engine, _ = h.engine(building=False)
    response = engine.search(SearchRequest(image=h.photo(130), position=Position(43.6, 1.44)))
    assert response.nearby is None
    assert len(response.cards) == 3


def test_places_go_into_a_built_bundle(tmp_path: Path, settings: Settings) -> None:
    folder = tmp_path / "bundle"
    folder.mkdir()
    catalog = h.catalog()
    write_catalog(folder / CATALOG_FILE, CatalogData(catalog.images, catalog.records))
    write_places_parquet(
        [Place("PA1", 43.6, 1.44, PlaceSource.MERIMEE), Place("XX", 0.1, 0.1, PlaceSource.COMMUNE)],
        settings.places_parquet,
    )
    assert add_places(settings, folder) == 1
    places = read_places(folder / PLACES_FILE, catalog.entity_ids)
    assert places is not None
    assert places.near(Position(43.6, 1.44)).tolist() == [False, False, True]  # OBJ1, OBJ2, PA1
