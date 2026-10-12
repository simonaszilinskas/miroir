from __future__ import annotations

import pytest

from miroir.records.images import (
    ImageSkip,
    asset_url,
    image_type_from_path,
    make_image_id,
    skip_reason,
)
from miroir.records.model import ImageType


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


def test_image_id_depends_on_ref_and_url_only() -> None:
    a = make_image_id("REF1", "https://x/a.jpg")
    assert a == make_image_id("REF1", "https://x/a.jpg")
    assert len(a) == 16
    assert a != make_image_id("REF2", "https://x/a.jpg")
    assert a != make_image_id("REF1", "https://x/b.jpg")
