from __future__ import annotations

from pathlib import Path

from miroir.config import Settings
from miroir.records.build import iter_raw
from miroir.records.mappers import RawRecord
from miroir.records.model import Collection

FIXTURE_EXPORTS = Path(__file__).parent / "fixtures" / "exports"


def raw_record(collection: Collection, ref: str) -> RawRecord:
    """One real export line from tests/fixtures/exports, by REF."""
    path = Settings(exports_dir=FIXTURE_EXPORTS).export_jsonl(collection)
    return next(raw for raw in iter_raw(path) if raw["REF"] == ref)
