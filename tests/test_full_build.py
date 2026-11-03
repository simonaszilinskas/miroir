from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from miroir.config import Settings
from miroir.records.full import build_full, count_full
from miroir.records.images import ImageSkip, copy_key, skip_reason
from miroir.records.model import Collection, Record
from miroir.records.store import read_records

FIXTURE_FULL = Path(__file__).parent / "fixtures" / "exports-full"


@pytest.fixture
def full(settings: Settings) -> Settings:
    """Settings whose data dir holds the full-export fixtures (real lines, one per case)."""
    shutil.copytree(FIXTURE_FULL, settings.full_exports_dir)
    return settings


def _records(settings: Settings) -> dict[str, Record]:
    records = read_records(settings.full_records_parquet, settings.full_images_parquet)
    return {r.ref: r for r in records}


def test_copy_key_joins_a_bucket_copy_to_its_image() -> None:
    assert copy_key("joconde/1/M500145_0001074_P.JPG") == copy_key("joconde/1/0001074.jpg")
    assert copy_key("joconde/1/M040000_1429-uenpgb_1.JPG") == "1429-uenpgb"
    assert copy_key("joconde/1/0000225A.jpg") == copy_key("joconde/1/0000225a.jpg")
    assert copy_key("joconde/1/0001075.jpg") != copy_key("joconde/1/0001074.jpg")


def test_links_and_documents_are_not_images() -> None:
    assert skip_reason("http://www.culture.gouv.fr/Wave/image/merimee/PDF/x.pdf") == (
        ImageSkip.NOT_AN_IMAGE
    )
    assert skip_reason("palissy/PM32000416/19880902_CMH_31-03.pdf") == ImageSkip.NOT_AN_IMAGE
    assert skip_reason("[object Object]") == ImageSkip.NOT_AN_IMAGE
    assert skip_reason("memoire/AP55P00431/sap01_55p00431_p.jpg") is None


def test_full_build_maps_every_base(full: Settings) -> None:
    build = build_full(full)
    records = _records(full)

    assert {r.collection for r in records.values()} == set(Collection)
    # Joconde: the bucket's renamed copies go; publishable only when the July export says so.
    copies = records["50010004377"]
    assert len(copies.images) == 7
    assert all(not i.url.split("/")[-1].startswith("M500145") for i in copies.images)
    assert build.counts[Collection.JOCONDE]["skip_image:renamed_copy"] >= 7
    assert not copies.publishable  # not in the July export: unknown, so not publishable
    assert records["M1222019757"].publishable
    assert not records["M0809006594"].publishable  # July DIFFU "non"
    # Mérimée lists Mémoire photos.
    building = records["PA00078066"]
    assert building.title == "Halles (vieilles)"
    assert building.commune == "Brienne-le-Château"
    assert building.publishable
    assert [i.url.split("/")[-2] for i in building.images] == ["AP55P00431", "AP55P00432"]
    # Palissy: a photo outside the Mémoire file leaves rights unknown; "[object Object]" is
    # no image, so that record goes.
    assert not records["PM09000059"].publishable
    assert "PM01001224" not in records
    assert build.counts[Collection.PALISSY]["skip_image:not_an_image"] == 1
    assert records["M1128"].pop_url == "https://pop.culture.gouv.fr/notice/museo/M1128"
    assert len(records["AEG00001"].images) > 1


def test_full_counts_each_file_once(full: Settings) -> None:
    report = count_full(full, build_full(full))

    # The seven Mémoire photos that Mérimée records list are counted once.
    assert report["unique_images_in_several_bases"] == 7
    rows, unique = report["image_rows"], report["unique_images"]
    assert isinstance(rows, int)
    assert isinstance(unique, int)
    assert rows == unique + 7
    by_kind = report["unique_images_by_kind"]
    assert isinstance(by_kind, dict)
    assert by_kind["building"] == 7 + 2  # Mérimée photos and the museum's two
    unique = full.full_image_list("unique").read_text(encoding="utf-8").split()
    assert len(unique) == report["unique_images"]
    assert full.full_report_json.exists()
