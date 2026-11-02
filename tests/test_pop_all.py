"""All of POP: the download's rules, the file list, the index rows and the coverage counts."""

from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image as PILImage

from miroir.config import Settings
from miroir.eval.vectors import Route, RunInfo, VectorSet
from miroir.index.coverage import coverage
from miroir.index.pop_all import Gathered, Sources, index_rows
from miroir.jobs.bulk import (
    FULL_LONG_SIDE,
    JobState,
    JobStatus,
    piece_ranges,
    shard_of,
    shard_prefixes,
)
from miroir.jobs.pop_download import Pacer, shrink
from miroir.jobs.scw_bulk import image_files, image_list
from miroir.parallel import unordered_map


def jpeg(width: int, height: int, mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    PILImage.new(mode, (width, height), 128).save(out, format="JPEG", quality=90)
    return out.getvalue()


def test_a_shard_is_the_listing_of_its_prefixes() -> None:
    ids = [f"{p:02x}{'0' * 14}" for p in range(256)]
    for shard in range(8):
        mine = {f"full/{i[:2]}/" for i in ids if shard_of(i, 8) == shard}
        assert mine == set(shard_prefixes(shard, 8))


def test_piece_ranges_cover_every_piece_once() -> None:
    ranges = piece_ranges(40, 4095, 4)
    covered = [p for a, b in ranges for p in range(a, b + 1)]
    assert covered == list(range(40, 4096))


def test_a_status_round_trips_and_stamps_its_changes() -> None:
    status = JobStatus("download", 2, 8, total=10)
    later = status.now(done=4, state=JobState.DONE)
    assert later.done == 4
    assert later.updated >= status.updated
    assert JobStatus.from_json(later.to_json()) == later


def test_a_small_jpeg_is_stored_as_it_is() -> None:
    data = jpeg(600, 400)
    stored, width, height, cut = shrink(data)
    assert stored == data
    assert (width, height, cut) == (600, 400, False)


def test_a_large_image_is_shrunk_to_the_long_side() -> None:
    stored, width, height, _ = shrink(jpeg(3000, 1500))
    assert (width, height) == (3000, 1500)
    with PILImage.open(io.BytesIO(stored)) as opened:
        assert max(opened.size) == FULL_LONG_SIDE


def test_a_png_is_saved_as_jpeg() -> None:
    out = io.BytesIO()
    PILImage.new("RGBA", (200, 100)).save(out, format="PNG")
    stored, *_ = shrink(out.getvalue())
    with PILImage.open(io.BytesIO(stored)) as opened:
        assert opened.format == "JPEG"


def test_a_file_cut_short_is_decoded_as_far_as_it_goes() -> None:
    data = jpeg(800, 600)
    stored, width, height, cut = shrink(data[: len(data) - 40])
    assert cut
    assert (width, height) == (800, 600)
    with PILImage.open(io.BytesIO(stored)) as opened:
        assert opened.size == (800, 600)


def test_the_pacer_waits_longer_when_busy_answers_keep_coming() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    pacer = Pacer(first_s=5, max_s=20, calm_s=60, clock=lambda: now[0], sleep=sleep)
    pacer.trip("429")
    pacer.trip("429")  # same burst: no longer pause
    pacer.wait()
    assert now[0] == pytest.approx(5)
    pacer.trip("429")
    pacer.wait()
    assert now[0] == pytest.approx(15)
    now[0] += 120  # calm again
    pacer.trip("429")
    pacer.wait()
    assert now[0] == pytest.approx(140)
    assert pacer.trips == 3


def test_unordered_map_yields_every_result_slow_ones_last() -> None:
    def work(n: int) -> int:
        time.sleep(0.05 if n == 0 else 0.0)
        return n

    out = list(unordered_map(work, range(20), workers=4))
    assert sorted(out) == list(range(20))
    assert out[-1] == 0 or out.index(0) > 0


def write_images(path: Path, rows: list[tuple[str, str, str]]) -> None:
    ids, urls, collections = zip(*rows, strict=True)
    table = pa.table({"image_id": ids, "url": urls, "collection": collections})
    pq.write_table(table, path)


def test_one_file_per_url_named_by_an_embedded_row(settings: Settings) -> None:
    settings.data_root.mkdir(parents=True)
    settings.full_images_parquet.parent.mkdir(parents=True)
    write_images(
        settings.full_images_parquet,
        [("a1", "u/a", "memoire"), ("a2", "u/a", "merimee"), ("b1", "u/b", "joconde")],
    )
    write_images(settings.images_parquet, [("a2", "u/a", "memoire"), ("c1", "u/c", "memoire")])
    files = image_files(settings, frozenset({"a2"})).to_pydict()
    assert dict(zip(files["image_id"], files["file_id"], strict=True)) == {
        "a1": "a2",
        "a2": "a2",
        "b1": "b1",
        "c1": "c1",
    }
    listed = image_list(image_files(settings, frozenset({"a2"}))).to_pydict()
    assert listed["image_id"] == ["a2", "b1", "c1"]
    assert listed["collection"] == ["memoire", "joconde", "memoire"]


def test_index_rows_one_per_file_and_entity() -> None:
    rows = index_rows(
        image_ids=["m1", "p1", "m2", "x1"],
        entity_ids=["E", "E", "F", "G"],
        file_of={"m1": "f1", "p1": "f1", "m2": "f1", "x1": "f9"},
        has_vector={"f1"},
        july=frozenset({"p1"}),
    )
    assert rows.ids == ("p1", "m2")  # the July row wins; f9 has no vector
    assert rows.files == ("f1", "f1")


def run(ids: tuple[str, ...], rows: list[list[float]]) -> VectorSet:
    info = RunInfo("A7", "m", 512, "head", "g", date(2026, 11, 2), 0.0, 0.0, Route.HOSTED)
    return VectorSet(ids, np.array(rows, dtype=np.float16), info)


def test_gathered_rows_come_from_july_or_the_pieces() -> None:
    july = run(("j1", "j2"), [[1.0, 0.0], [0.0, 1.0]])
    pieces = np.array([[0.6, 0.8]], dtype=np.float16)
    sources = Sources(july, july.index(), pieces, {"n1": 0})
    gathered = Gathered(("n1", "j2", "j1"), sources)
    assert gathered.shape == (3, 2)
    np.testing.assert_allclose(gathered[0:3], [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]], atol=1e-3)


def test_coverage_counts_records_and_takes_their_first_image_s_reason() -> None:
    table = coverage(
        records={"r1": ("memoire", "E"), "r2": ("memoire", "E"), "r3": ("joconde", "F")},
        images={"i1": ("r1", "E"), "i2": ("r2", "E"), "i3": ("r3", "F")},
        file_of={"i1": "f1", "i2": "f2", "i3": "f3"},
        with_vectors={"f1"},
        indexed_entities={"E"},
        reasons={"f2": "download: missing", "f3": "download: not_an_image"},
    )
    assert table["memoire"].records == 2
    assert table["memoire"].with_vectors == 1
    assert table["memoire"].in_index == 2
    assert dict(table["memoire"].reasons) == {"download: missing": 1}
    assert table["joconde"].in_index == 0
