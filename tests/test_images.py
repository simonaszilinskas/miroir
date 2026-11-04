from __future__ import annotations

import unicodedata

import pytest

from miroir.records.images import (
    ImageSkip,
    asset_url,
    bucket_key,
    image_type_from_path,
    make_image_id,
    skip_reason,
)
from miroir.records.mappers import map_record
from miroir.records.model import Collection, ImageType
from tests.helpers import raw_record


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("mnr/OAR00119/oar-119-c-musee-du-louvre-1.jpg", ImageType.MAIN),
        ("mnr/MNR10/MNR10_c_MuseeduLouvre_detail_8.JPG", ImageType.DETAIL),
        ("mnr/OAR620/OAR620_CopyrightMuséeduLouvre2021_Détail.JPG", ImageType.DETAIL),
        ("mnr/MNR963/MNR963_revers_ThB-AP_1.jpg", ImageType.BACK),
        ("mnr/MNR528/MNR528dos-copyright_Musee_d_Orsay.jpg", ImageType.BACK),
        ("mnr/REC103/REC103-verso_CopyrightRMNGP.jpg", ImageType.BACK),
        ("mnr/MNR836/MNR836_detail-inscription-3.jpg", ImageType.INSCRIPTION),
        ("mnr/X/X_signature.jpg", ImageType.INSCRIPTION),
        ("mnr/MNR709/MNR709_detail-etiquette-au-dos-du-cadre.jpg", ImageType.LABEL),
        ("mnr/OAR437/OAR437_étiquette verso.jpg", ImageType.LABEL),
        ("mnr/OAR766/mnr766_cachetcire.jpg", ImageType.LABEL),
        ("joconde/X/eveque constitutionnel du calvados 2010.0.10.jpg", ImageType.MAIN),
        ("joconde/X/dossier.jpg", ImageType.MAIN),
        ("mnr/OAR74/OAR74_CopyrightMuseeDuLouvre_F-ENS.jpg", ImageType.MAIN),
        ("mnr/OAR74/OAR74_CopyrightMuseeDuLouvre_F-DET-6-6.jpg", ImageType.DETAIL),
        ("mnr/OAR74/OAR74_CopyrightMuseeDuLouvre_R-ENS.jpg", ImageType.BACK),
        ("mnr/OAR74/OAR74_CopyrightMuseeDuLouvre_R-DET-1-6.jpg", ImageType.BACK),
    ],
)
def test_image_type_from_file_name(path: str, expected: ImageType) -> None:
    assert image_type_from_path(path) == expected


def test_only_the_file_name_sets_the_type() -> None:
    assert image_type_from_path("mnr/detail-folder/face.jpg") == ImageType.MAIN


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA=", ImageSkip.PLACEHOLDER),
        ("", ImageSkip.PLACEHOLDER),
        ("memoire/MHR11_20257500080/SUR_DEMANDE.jpg", ImageSkip.ON_REQUEST),
        ("memoire/APMH00125797/sap91_mh00125797_p.jpg", None),
    ],
)
def test_skip_reason(path: str, expected: ImageSkip | None) -> None:
    assert skip_reason(path) == expected


def test_asset_url_percent_encodes_the_path() -> None:
    url = asset_url("mnr/rfr21/L'Enfant prodigue")
    assert (
        url
        == "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/mnr/rfr21/L%27Enfant%20prodigue"
    )


def test_image_id_depends_on_ref_and_key_only() -> None:
    a = make_image_id("REF1", "mnr/REF1/a.jpg")
    assert a == make_image_id("REF1", "mnr/REF1/a.jpg")
    assert len(a) == 16
    assert a != make_image_id("REF2", "mnr/REF1/a.jpg")
    assert a != make_image_id("REF1", "mnr/REF1/b.jpg")


# Golden ids: a change here changes every image id, so every vector run and test set made
# before it (technical spec §3.3). Change them only with a full re-index.
@pytest.mark.parametrize(
    ("collection", "ref", "position", "image_id"),
    [
        (Collection.JOCONDE, "M1222019757", 0, "31e508966c676ef6"),
        (Collection.MEMOIRE, "APMH00125797", 0, "8b9a425a80ae10cc"),
        # "ER 2_photo_Georges_Poncet.JPG": the URL encodes the space, the id does not.
        (Collection.ROSE_VALLAND, "AEG00007", 0, "7d6c1cf4f5df7836"),
    ],
)
def test_image_ids_of_real_export_lines_are_pinned(
    collection: Collection, ref: str, position: int, image_id: str
) -> None:
    record = map_record(collection, raw_record(collection, ref)).record
    assert record is not None
    assert record.images[position].image_id == image_id


def test_image_id_ignores_how_an_accented_name_is_spelt() -> None:
    path = "mnr/OAR620/OAR620_CopyrightMuséeduLouvre2021_Détail.JPG"
    decomposed = unicodedata.normalize("NFD", path)
    assert decomposed != path
    assert bucket_key(decomposed) == bucket_key("/" + path) == path
    assert make_image_id("OAR620", bucket_key(decomposed)) == "b339f33b2cd9e85d"
    assert make_image_id("OAR620", bucket_key(path)) == "b339f33b2cd9e85d"


def test_one_file_spelt_two_ways_is_one_image() -> None:
    raw = raw_record(Collection.ROSE_VALLAND, "MNR00564")
    path = "mnr/MNR00564/MNR564_détail.jpg"
    twice = {**raw, "VIDEO": [path, unicodedata.normalize("NFD", path)]}
    result = map_record(Collection.ROSE_VALLAND, twice)
    assert result.record is not None
    assert len(result.record.images) == 1
    assert result.image_skips == (ImageSkip.DUPLICATE,)
