from __future__ import annotations

from miroir.config import Settings
from miroir.records.build import build
from miroir.records.model import Collection
from miroir.records.store import read_images, read_records


def test_build_writes_parquet_that_reads_back(settings: Settings) -> None:
    report = build(settings)

    records = read_records(settings.records_parquet, settings.images_parquet)
    assert len(records) == 17  # 20 fixture records, 3 without a usable image
    assert {r.collection for r in records} == set(Collection)
    assert report.counts[Collection.MEMOIRE]["skip_record:no_image"] == 2
    assert report.counts[Collection.JOCONDE]["skip_image:duplicate"] == 1

    images = read_images(settings.images_parquet)
    assert len(images) == sum(len(r.images) for r in records)
    assert len({i.image_id for i in images}) == len(images)
    assert {r.ref: r.images for r in records}["MNR00564"][1].image_type == "back"


def test_image_ids_are_stable_across_runs(settings: Settings) -> None:
    build(settings)
    first = [i.image_id for i in read_images(settings.images_parquet)]
    build(settings, collections=(Collection.ROSE_VALLAND, Collection.JOCONDE, Collection.MEMOIRE))
    second = [i.image_id for i in read_images(settings.images_parquet)]
    assert sorted(first) == sorted(second)
