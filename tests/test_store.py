from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from miroir.config import Settings
from miroir.records.build import build
from miroir.records.full import build_full
from miroir.records.mappers import map_record
from miroir.records.model import Collection, Record
from miroir.records.store import (
    SCHEMA_VERSION_KEY,
    RecordsWriter,
    read_images,
    read_records,
    write_records,
)
from tests.helpers import FIXTURE_FULL, raw_record


def _record() -> Record:
    record = map_record(Collection.JOCONDE, raw_record(Collection.JOCONDE, "M0569000124")).record
    assert record is not None
    return record


def _paths(folder: Path) -> tuple[Path, Path]:
    return folder / "records.parquet", folder / "images.parquet"


def test_a_failed_batch_build_leaves_no_table(tmp_path: Path) -> None:
    records, images = _paths(tmp_path)

    def fail() -> None:
        with RecordsWriter(records, images) as writer:
            writer.write([_record()])
            raise RuntimeError

    with pytest.raises(RuntimeError):
        fail()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_full_build_keeps_the_last_good_tables(settings: Settings) -> None:
    shutil.copytree(FIXTURE_FULL, settings.full_exports_dir)
    build_full(settings)
    before = settings.full_records_parquet.read_bytes()
    settings.full_export_jsonl("palissy").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expecting"):
        build_full(settings)
    assert settings.full_records_parquet.read_bytes() == before
    names = sorted(p.name for p in settings.full_pop_dir.iterdir())
    assert names == ["images.parquet", "records.parquet"]


def test_write_records_leaves_only_the_two_files(tmp_path: Path) -> None:
    records, images = _paths(tmp_path)
    write_records([_record()], records, images)
    write_records([_record()], records, images)
    assert [r.ref for r in read_records(records, images)] == ["M0569000124"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["images.parquet", "records.parquet"]


def test_tables_carry_their_schema_version(settings: Settings) -> None:
    build(settings)
    for path in (settings.records_parquet, settings.images_parquet):
        metadata = pq.read_schema(path).metadata
        assert metadata is not None
        assert SCHEMA_VERSION_KEY in metadata


def test_a_table_missing_a_column_is_refused(tmp_path: Path) -> None:
    records, images = _paths(tmp_path)
    write_records([_record()], records, images)
    pq.write_table(pq.read_table(records).drop_columns(["photo_date"]), records)
    with pytest.raises(ValueError, match="rebuild"):
        read_records(records, images)


def test_a_table_of_another_schema_version_is_refused(tmp_path: Path) -> None:
    records, images = _paths(tmp_path)
    write_records([_record()], records, images)
    old = pq.read_table(images).replace_schema_metadata({SCHEMA_VERSION_KEY: b"0"})
    pq.write_table(old, images)
    with pytest.raises(ValueError, match="rebuild"):
        read_images(images)


def test_a_table_from_before_versions_with_the_same_columns_still_reads(tmp_path: Path) -> None:
    records, images = _paths(tmp_path)
    write_records([_record()], records, images)
    for path in (records, images):
        pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)
    assert len(read_records(records, images)) == 1
