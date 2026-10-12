from __future__ import annotations

from miroir.records.images import ImageSkip
from miroir.records.mappers import DESCRIPTION_MAX_CHARS, MapResult, RecordSkip, map_record
from miroir.records.model import Collection, ImageType, Record
from tests.helpers import raw_record

J, M, RV = Collection.JOCONDE, Collection.MEMOIRE, Collection.ROSE_VALLAND


def _map(collection: Collection, ref: str) -> MapResult:
    return map_record(collection, raw_record(collection, ref))


def _record(collection: Collection, ref: str) -> Record:
    record = _map(collection, ref).record
    assert record is not None
    return record


# --- entity_id ----------------------------------------------------------------


def test_memoire_entity_is_the_pm_object_when_lbase_also_holds_a_pa_building() -> None:
    # LBASE: PA00080616, PM05000292. The photo shows the object.
    assert _record(M, "APMH00125797").entity_id == "PM05000292"


def test_memoire_entity_is_pa_when_lbase_holds_no_pm() -> None:
    assert _record(M, "AP2322T000470").entity_id == "PA00087673"


def test_memoire_entity_is_pm_when_pm_comes_first() -> None:
    assert _record(M, "APMH00020693").entity_id == "PM21001451"


def test_memoire_entity_skips_lbase_ids_that_are_not_pa_or_pm() -> None:
    # LBASE holds an IM id, then a PM id.
    assert _record(M, "IVR26_19852101771ZA").entity_id == "PM21001704"


def test_memoire_entity_falls_back_to_ref_without_lbase() -> None:
    assert _record(M, "AP2322T003524").entity_id == "AP2322T003524"


def test_other_bases_use_ref_as_entity() -> None:
    assert _record(J, "M1222019757").entity_id == "M1222019757"
    assert _record(RV, "OAR00119").entity_id == "OAR00119"


# --- skips ----------------------------------------------------------------------


def test_placeholder_gif_record_is_skipped() -> None:
    for collection, ref in [(J, "08580516465"), (M, "APMH00132533")]:
        result = _map(collection, ref)
        assert result.record is None
        assert result.record_skip == RecordSkip.NO_IMAGE
        assert result.image_skips == (ImageSkip.PLACEHOLDER,)


def test_image_on_request_is_skipped() -> None:
    result = _map(M, "MHR11_20257500080")
    assert result.record is None
    assert result.image_skips == (ImageSkip.ON_REQUEST,)


def test_duplicate_paths_in_one_record_are_dropped() -> None:
    result = _map(RV, "OAR00074")
    assert result.record is not None
    assert result.image_skips == (ImageSkip.DUPLICATE,)
    urls = [image.url for image in result.record.images]
    assert len(urls) == len(set(urls))


# --- images ---------------------------------------------------------------------


def test_rose_valland_multi_image_record_with_revers() -> None:
    record = _record(RV, "MNR00564")
    assert [i.image_type for i in record.images] == [ImageType.MAIN, ImageType.BACK, ImageType.MAIN]
    assert [i.position for i in record.images] == [0, 1, 2]
    assert record.cover.position == 0
    assert all(i.ref == "MNR00564" and i.entity_id == "MNR00564" for i in record.images)


def test_cover_is_first_main_image_even_when_back_comes_first() -> None:
    record = _record(RV, "REC0122V")
    assert record.images[0].image_type == ImageType.BACK
    assert record.cover.position == 1
    assert sum(i.is_cover for i in record.images) == 1


def test_label_beats_detail() -> None:
    types = [i.image_type for i in _record(RV, "AEG00007").images]
    assert types == [ImageType.MAIN, ImageType.LABEL, ImageType.MAIN, ImageType.LABEL]


def test_memoire_keeps_one_image_the_largest_variant() -> None:
    record = _record(M, "APMH00125797")
    assert len(record.images) == 1
    assert record.images[0].url.endswith("/memoire/APMH00125797/sap91_mh00125797_p.jpg")


# --- text and filter fields -----------------------------------------------------


def test_memoire_fields() -> None:
    record = _record(M, "APMH00125797")
    assert record.title == "Église paroissiale"
    assert record.description == "Vantail de porte de la sacristie"
    assert record.period == ("années 1940",)
    assert record.departement == "Hautes-Alpes"
    assert record.commune == "Saint-Martin-de-Queyrières"
    assert record.pop_url == "https://pop.culture.gouv.fr/notice/memoire/APMH00125797"


def test_memoire_denomination_string_is_split() -> None:
    assert _record(M, "IVR26_19967100284X").denomination == ("cloche", "mouton")


def test_publishable() -> None:
    assert _record(J, "M0569000124").publishable
    assert not _record(J, "000SC026869").publishable  # DIFFU = "non"
    assert not _record(M, "MHR93_00054348").publishable  # DIFF asks for permission
    assert _record(RV, "OAR00119").publishable


def test_rose_valland_uses_mongo_collection_in_pop_url() -> None:
    record = _record(RV, "OAR00119")
    assert record.collection == Collection.ROSE_VALLAND
    assert record.pop_url == "https://pop.culture.gouv.fr/notice/mnr/OAR00119"


def test_description_is_capped() -> None:
    record = _record(RV, "OAR00119")
    assert record.description.startswith("Chaise en acajou")
    assert len(record.description) <= DESCRIPTION_MAX_CHARS
