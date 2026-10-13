"""Archive prints: the print found on its mount, the caption's commune, and the search."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from miroir.caption import Gazetteer, clean_caption, name_forms, normalise
from miroir.images.mount import crop_picture, find_picture
from miroir.web.archive import ArchiveRule, gazetteer
from miroir.web.engine import Engine, SearchRequest
from miroir.web.nearby import PLACES_FILE, read_places, write_places
from tests import web_helpers as h

PAPER = 210


def card(*, colour: bool = False, caption: bool = True) -> bytes:
    """A 384 x 288 card: grey paper, a dark textured print on the right, typed lines left."""
    rng = np.random.default_rng(3)
    grey = np.full((288, 384), PAPER, dtype=np.uint8)
    grey[30:250, 200:360] = rng.integers(0, 120, (220, 160), dtype=np.uint8)
    if caption:
        for y in range(200, 260, 6):  # typed lines: short dark dashes
            for x in range(20, 150, 7):
                grey[y, x : x + 4] = 40
    rgb = np.stack([grey] * 3, axis=-1)
    if colour:
        rgb[30:250, 200:360, 0] = 200  # a red object
    out = io.BytesIO()
    PILImage.fromarray(rgb).save(out, format="PNG")
    return out.getvalue()


# --- the print on its mount --------------------------------------------------------------------


def test_the_print_is_found_on_its_card() -> None:
    picture = find_picture(card())
    assert picture is not None
    assert abs(picture.left - 200) <= 2
    assert abs(picture.right - 360) <= 2
    assert abs(picture.top - 30) <= 2
    assert abs(picture.bottom - 250) <= 2
    with PILImage.open(io.BytesIO(crop_picture(card(), picture))) as cut:
        assert abs(cut.width - 160) <= 4
        assert abs(cut.height - 220) <= 4


def test_no_mount_without_writing_or_in_colour() -> None:
    assert find_picture(card(caption=False)) is None  # an object on a plain background
    assert find_picture(card(colour=True)) is None
    assert find_picture(h.photo(130, (400, 300))) is None  # one flat tone: no picture


def test_the_box_follows_the_photo_size() -> None:
    small = find_picture(card())
    with PILImage.open(io.BytesIO(card())) as opened:
        big = opened.resize((768, 576))
    data = io.BytesIO()
    big.save(data, format="PNG")
    large = find_picture(data.getvalue())
    assert small is not None
    assert large is not None
    assert abs(large.left - 2 * small.left) <= 4


# --- the caption ---------------------------------------------------------------------------


def test_a_caption_is_one_clean_line() -> None:
    assert clean_caption('  "BEAUNE (Côte d\'Or)\n Eglise"  ') == "BEAUNE (Côte d'Or) Eglise"
    assert clean_caption("AUCUNE") == ""
    assert clean_caption("") == ""


def test_names_are_compared_without_accents_or_case() -> None:
    assert normalise("ST-QUENTIN (Aisne)") == "saint quentin aisne"
    assert name_forms("Arques (Les)") == {"arques les", "arques", "les arques"}


def places() -> Gazetteer:
    return Gazetteer.build(
        [
            ("Beaune", "Côte-d'Or", 47.02, 4.84),
            ("Beaune", "Côte-d'Or", 47.04, 4.84),
            ("Saint-Quentin", "Aisne", 49.85, 3.29),
            ("Plan", "Isère", 45.3, 5.4),
            ("Arques (Les)", "Lot", 44.6, 1.2),
        ]
    )


def test_the_commune_a_caption_names() -> None:
    found = places().find("BEAUNE (Côte d'Or) Eglise Notre-Dame. Façade ouest")
    assert found is not None
    assert found.label == "Beaune (Côte-d'Or)"
    assert found.with_departement
    assert found.latitude == 47.03  # the median of its entities


def test_a_commune_in_capitals_counts_without_its_departement() -> None:
    found = places().find("ST-QUENTIN. Collégiale, bas-côté nord")
    assert found is not None
    assert found.commune == "Saint-Quentin"
    assert places().find("LES ARQUES, église") is not None


def test_a_common_word_is_not_a_commune() -> None:
    assert places().find("Plan de l'église") is None  # not in capitals, no département
    assert places().find("") is None


# --- the search -----------------------------------------------------------------------------


class Reader:
    def __init__(self, caption: str) -> None:
        self.caption = caption
        self.calls = 0

    def read(self, jpeg: bytes) -> str:
        del jpeg
        self.calls += 1
        return self.caption


def archive_engine(tmp_path: Path, reader: Reader, rule: ArchiveRule) -> Engine:
    base, encoders = h.engine(building=False)
    path = tmp_path / PLACES_FILE
    write_places(path, base.catalog.entity_ids, {"PA1": (43.6, 1.44), "OBJ1": (49.44, 1.09)})
    placed = read_places(path, base.catalog.entity_ids)
    assert placed is not None
    return Engine(
        catalog=base.catalog,
        routing=base.routing,
        guess=base.guess,
        encoders=encoders,
        gallery=base.gallery,
        places=placed,
        captions=reader,
        archive=rule,
    )


def test_a_mounted_print_is_searched_alone_and_its_caption_read(tmp_path: Path) -> None:
    reader = Reader("TOULOUSE (Haute-Garonne) Eglise Saint-Pierre")
    engine = archive_engine(tmp_path, reader, ArchiveRule(place_bonus=1.0))
    assert isinstance(engine.encoders, h.RowEncoders)
    response = engine.search(SearchRequest(image=card()))
    assert response.mount
    assert response.picture is not None
    assert response.picture[0] > 0.5  # the print sits on the right of the card
    assert reader.calls == 1
    assert response.caption_read
    assert response.caption == "TOULOUSE (Haute-Garonne) Eglise Saint-Pierre"
    assert response.caption_place == "Toulouse (Haute-Garonne)"
    assert response.cards[0].entity_id == "PA1"  # the building in Toulouse comes first
    assert engine.encoders.last_photo is not None
    with PILImage.open(io.BytesIO(engine.encoders.last_photo)) as embedded:
        assert embedded.width < embedded.height  # the print (portrait), not the card


def test_a_photo_without_a_mount_is_not_read(tmp_path: Path) -> None:
    reader = Reader("TOULOUSE")
    engine = archive_engine(tmp_path, reader, ArchiveRule(place_bonus=1.0))
    response = engine.search(SearchRequest(image=h.photo(130)))
    assert reader.calls == 0
    assert not response.mount
    assert response.picture is None
    assert response.caption is None


def test_a_caption_the_person_gives_or_removes_is_not_read_again(tmp_path: Path) -> None:
    reader = Reader("ROUEN")
    engine = archive_engine(tmp_path, reader, ArchiveRule(place_bonus=1.0))
    given = engine.search(SearchRequest(image=card(), caption="Toulouse (Haute-Garonne)"))
    removed = engine.search(SearchRequest(image=card(), caption=""))
    whole = engine.search(SearchRequest(image=card(), mount=False))
    assert reader.calls == 0
    assert given.caption_place == "Toulouse (Haute-Garonne)"
    assert not given.caption_read
    assert removed.caption is None
    assert removed.caption_place is None
    assert whole.picture is None


def test_the_index_communes_sit_where_their_entities_are(tmp_path: Path) -> None:
    engine = archive_engine(tmp_path, Reader(""), ArchiveRule())
    assert engine.places is not None
    found = gazetteer(engine.catalog, engine.places).find("ROUEN (Seine-Maritime)")
    assert found is not None
    assert (found.latitude, found.longitude) == (49.44, 1.09)


def test_shoot_images_can_be_hidden_for_measuring() -> None:
    engine, _ = h.engine(building=False)
    response = engine.search(SearchRequest(image=h.photo(130), exclude_images=("j1", "j2")))
    assert "OBJ1" not in [c.entity_id for c in response.cards]


def test_without_cropping_the_whole_card_is_searched_and_its_caption_still_read(
    tmp_path: Path,
) -> None:
    reader = Reader("TOULOUSE (Haute-Garonne)")
    engine = archive_engine(tmp_path, reader, ArchiveRule(crop=False, words=1.0))
    assert isinstance(engine.encoders, h.RowEncoders)
    response = engine.search(SearchRequest(image=card()))
    assert response.mount
    assert response.picture is None
    assert reader.calls == 1
    assert engine.encoders.last_photo is not None
    with PILImage.open(io.BytesIO(engine.encoders.last_photo)) as embedded:
        assert embedded.width > embedded.height  # the whole card (landscape)
