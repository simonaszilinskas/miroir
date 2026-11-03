"""Crop vectors kept and shortened: the job's spec, the projected form, the slice measure's
tables (no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miroir.eval.crop_short import (
    Projection,
    aligned,
    best_per_file,
    other_rows,
    projected,
    short_form,
    shortlisted,
    storage_gb,
    whole_form,
)
from miroir.eval.crop_short import SliceCrops as Crops
from miroir.eval.sets import Query
from miroir.index.crop_bundle import piece_paths, write_crop_files
from miroir.jobs.bulk import PIECES
from miroir.jobs.crop_embed import (
    GRID,
    SLICE_PIECE_FILES,
    CropSpec,
    Form,
    Precision,
    approx_cosines,
    full_spec,
    in_stock,
    project_crops,
    query_terms,
    read_crop_spec,
    spec_json,
    write_piece,
)
from miroir.jobs.crop_measure import MeasureOut, QueryRanks, measure_queries, summarize
from miroir.jobs.objects import link_siblings
from miroir.jobs.scaleway import Offer
from miroir.web.crops import (
    COORDS_FILE,
    MEAN_DOT_FILE,
    PROJECTION_FILE,
    ROW_FILE,
    SCALE_FILE,
    CropQuery,
    CropRule,
    NumpyCrops,
    read_crops,
    write_rule,
)
from miroir.web.store import ExactStore, quantize_rows
from tests import web_helpers as h

RNG = np.random.default_rng(7)


def _unit(rows: int, dim: int) -> np.ndarray:
    x = RNG.normal(size=(rows, dim)).astype(np.float32) + 0.5
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _pca(vectors: np.ndarray, dims: int) -> Projection:
    mean = vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    return Projection(mean.astype(np.float32), vt[:dims].astype(np.float32))


# --- the spec ----------------------------------------------------------------------------------


def test_a_listed_spec_cuts_its_files_into_pieces_and_reads_back() -> None:
    files = tuple(f"f{n:05d}" for n in range(SLICE_PIECE_FILES * 2 + 3))
    spec = CropSpec(
        name="s", grid=GRID, precisions=(Precision.FP32, Precision.BF16), form=Form.WHOLE,
        files=files,
    )  # fmt: skip
    assert spec.pieces() == ["0000", "0001", "0002"]
    assert spec.piece_files("0002") == list(files[-3:])
    assert sum(len(spec.piece_files(p)) for p in spec.pieces()) == len(files)
    assert read_crop_spec(spec_json(spec)) == spec


def test_the_full_run_works_by_file_id_piece() -> None:
    spec = full_spec(Precision.BF16, "crops/pop-3x3/projection.npz", 256)
    assert len(spec.pieces()) == PIECES
    assert spec.pieces()[:2] == ["000", "001"]
    assert spec.form is Form.PROJECTED


class _Api:
    def __init__(self, states: dict[str, dict[str, str]]) -> None:
        self.states = states
        self.asked: list[str] = []

    def availability(self, zone: str) -> dict[str, str]:
        self.asked.append(zone)
        return self.states.get(zone, {})


def test_offers_in_stock_keep_their_order_and_drop_shortages() -> None:
    offers = [
        Offer("a", "L40S", 1.47),
        Offer("b", "L40S", 1.47),
        Offer("a", "L4", 0.79),
        Offer("a", "RENDER-S", 1.25),
    ]
    api = _Api({"a": {"L40S": "scarce", "L4": "shortage", "RENDER-S": "available"}})
    found = in_stock(api, offers)  # pyright: ignore[reportArgumentType]
    assert [(o.zone, o.commercial_type) for o in found] == [("a", "L40S"), ("a", "RENDER-S")]
    assert api.asked == ["a", "b"]  # one question per zone


# --- the projected form --------------------------------------------------------------------------


def test_projected_crops_put_back_the_whole_cosine() -> None:
    # Most of the spread on 8 axes, as image vectors have it on their first axes.
    basis = RNG.normal(size=(8, 32)).astype(np.float32)
    gallery = RNG.normal(size=(400, 8)).astype(np.float32) @ basis + 2.0
    gallery += 0.05 * RNG.normal(size=gallery.shape).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    pca = _pca(gallery, 32)
    crops, query = gallery[:50], gallery[-1]
    coords, mean_dot = project_crops(crops, pca.mean, pca.components)
    q_coords, offset = query_terms(query, pca.mean, pca.components)
    # Every axis kept: exact.
    assert np.allclose(approx_cosines(coords, mean_dot, q_coords, offset), crops @ query, atol=1e-5)
    # Fewer axes: close, and on the same scale (not centred, not renormalised).
    short = approx_cosines(coords, mean_dot, q_coords[:16], offset)
    assert np.abs(short - crops @ query).max() < 0.01


def test_short_forms_score_like_the_jobs_projection() -> None:
    gallery = _unit(300, 24)
    pca = _pca(gallery, 24)
    queries = _unit(3, 24)
    coords, mean_dot = projected(gallery.astype(np.float16), pca, 12)
    approx = short_form("a", coords, mean_dot, queries, pca, scale="approx", int8=False)
    job_coords, job_dot = project_crops(gallery.astype(np.float16), pca.mean, pca.components[:12])
    q_coords, offset = query_terms(queries[1], pca.mean, pca.components[:12])
    want = approx_cosines(job_coords, job_dot, q_coords, offset)
    assert np.allclose(approx.score(slice(0, 300))[:, 1], want, atol=1e-4)
    quantized = short_form("q", coords, mean_dot, queries, pca, scale="approx", int8=True)
    assert np.abs(quantized.score(slice(0, 300)) - approx.score(slice(0, 300))).max() < 0.02
    norm = short_form("n", coords, mean_dot, queries, pca, scale="norm", int8=False)
    assert np.abs(norm.score(slice(0, 300))).max() <= 1 + 1e-5


# --- tables --------------------------------------------------------------------------------------


def test_each_file_takes_its_best_crop() -> None:
    crops = np.eye(6, 4, dtype=np.float16)  # 3 files of 2 crops
    queries = np.eye(2, 4, dtype=np.float32)  # query 0 likes crop 0, query 1 crop 1
    table = best_per_file(whole_form("w", crops, queries), 2)
    assert table.shape == (2, 3)
    assert table[0].tolist() == [1, 0, 0]
    assert table[1].tolist() == [1, 0, 0]


def test_tables_follow_the_slice_tests_columns() -> None:
    table = np.asarray([[0.1, 0.2]], dtype=np.float32)
    out = aligned(table, ["b", "a"], ["a", "c", "b"])
    assert out[0, 0] == np.float16(0.2)
    assert np.isneginf(out[0, 1])
    assert out[0, 2] == np.float16(0.1)


def test_a_shortlist_keeps_each_querys_best_files_but_not_its_hidden_ones() -> None:
    pick = np.asarray([[0.9, 0.8, 0.1, 0.7]], dtype=np.float16)
    value = np.asarray([[0.5, 0.4, 0.3, 0.2]], dtype=np.float16)
    out = shortlisted(pick, value, 2, [np.asarray([0], dtype=np.int64)])
    assert np.isneginf(out[0, 0])  # hidden
    assert out[0, 1] == np.float16(0.4)
    assert out[0, 3] == np.float16(0.2)
    assert np.isneginf(out[0, 2])


def test_a_crop_pca_never_sees_an_answers_crops() -> None:
    crops = Crops(("a", "b", "c"), 2, {})
    assert other_rows(crops, {"b"}).tolist() == [0, 1, 4, 5]


def test_storage_counts_the_two_numbers_beside_each_crop() -> None:
    assert storage_gb(1_000_000_000, 256, int8=True) == 264.0


# --- crops at the site -------------------------------------------------------------------------


def _crop_bundle(folder: Path, crops: np.ndarray, rows_file: list[int], rule: CropRule) -> None:
    """A bundle's crop files with an identity PCA (every axis kept, mean 0): exact cosines."""
    dim = crops.shape[1]
    coords, scale = quantize_rows(crops)
    np.save(folder / COORDS_FILE, coords)
    np.save(folder / SCALE_FILE, scale)
    np.save(folder / MEAN_DOT_FILE, np.zeros(len(crops), dtype=np.float32))
    np.save(folder / ROW_FILE, np.asarray(rows_file, dtype=np.int32))
    np.savez(
        folder / PROJECTION_FILE,
        mean=np.zeros(dim, dtype=np.float32),
        components=np.eye(dim, dtype=np.float32),
    )
    write_rule(folder, rule)


def test_each_image_takes_its_files_best_crop(tmp_path: Path) -> None:
    matrix = h.unit_rows(6, 8)
    crops = h.unit_rows(12, 8, seed=3)  # 6 files of 2 crops
    crops[11] = matrix[0]
    _crop_bundle(tmp_path, crops, [0, 1, 2, 3, 4, 5], CropRule(per_file=2, shortlist=2))
    index = read_crops(tmp_path)
    assert index is not None
    mask = np.ones(6, dtype=bool)
    mask[1] = False
    scores = index.row_scores(matrix[0], mask)
    assert scores[5] == pytest.approx(1.0, abs=0.01)
    assert np.isneginf(scores[1])
    want = (crops @ matrix[0]).reshape(6, 2).max(axis=1)
    assert np.allclose(scores[mask], want[mask], atol=0.02)
    assert read_crops(tmp_path / "none") is None


def test_the_cpu_scores_crops_in_blocks_as_in_one(monkeypatch: pytest.MonkeyPatch) -> None:
    crops = h.unit_rows(90, 8, seed=4)
    coords, scale = quantize_rows(crops)
    store = NumpyCrops(coords, scale, np.zeros(90, dtype=np.float32), per_file=9)
    query = CropQuery(h.unit_rows(1, 8)[0], 0.25)
    whole = store.best_per_file(query)
    monkeypatch.setattr("miroir.web.crops.BLOCK_ROWS", 20)
    assert np.allclose(store.best_per_file(query), whole, atol=1e-6)
    assert whole.shape == (10,)


def test_a_crop_of_an_image_lifts_its_megaloc_score_and_brings_it_on_the_shortlist(
    tmp_path: Path,
) -> None:
    engine, _ = h.engine(building=False)
    matrix = h.unit_rows(6, 8)
    crops = h.unit_rows(12, 8, seed=3)
    crops[11] = matrix[0]  # a window of image 5 shows what photo 0 shows
    _crop_bundle(tmp_path, crops, [0, 1, 2, 3, 4, 5], CropRule(per_file=2, shortlist=1))
    engine.crops = read_crops(tmp_path)
    rows, scale = quantize_rows(matrix)
    engine.exact = ExactStore(rows=rows, scale=scale)
    mask = np.ones(6, dtype=bool)
    mask[0] = False  # the photo's own image is hidden
    timings: dict[str, float] = {}
    on = engine.vector_cosines(matrix[0], matrix[0], 10.0, mask, timings)
    assert on.a3[5] == pytest.approx(1.0, abs=0.02)
    assert "crops" in timings
    engine.use_crops = False
    off = engine.vector_cosines(matrix[0], matrix[0], 10.0, mask, {})
    assert off.a3[5] < 0.9
    assert np.isfinite(on.a3).sum() >= np.isfinite(off.a3).sum()


class _Pages:
    def __init__(self, objects: dict[str, int]) -> None:
        self.objects = objects

    def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, object]]:  # noqa: N803 (boto3's names)
        del Bucket
        found = [{"Key": k, "Size": s} for k, s in self.objects.items() if k.startswith(Prefix)]
        return [{"Contents": found}]


class _S3:
    def __init__(self, objects: dict[str, int]) -> None:
        self.objects = objects

    def get_paginator(self, name: str) -> _Pages:
        assert name == "list_objects_v2"
        return _Pages(self.objects)


def test_a_copied_bundle_links_the_files_it_shares_and_fetches_only_the_rest(
    tmp_path: Path,
) -> None:
    old = tmp_path / "pop-all"
    old.mkdir()
    (old / "a7.npy").write_bytes(b"x" * 10)
    (old / "manifest.json").write_bytes(b"{}")
    s3 = _S3(
        {
            "pop-all-crops/a7.npy": 10,
            "pop-all-crops/manifest.json": 5,
            "pop-all-crops/crops.json": 3,
        }
    )
    linked = link_siblings(s3, "b", tmp_path, "pop-all-crops/")  # pyright: ignore[reportArgumentType]
    assert linked == 1
    new = tmp_path / "pop-all-crops" / "a7.npy"
    assert new.stat().st_ino == (old / "a7.npy").stat().st_ino
    assert not (tmp_path / "pop-all-crops" / "manifest.json").exists()  # another size


def test_crop_files_follow_the_index_rows_and_keep_the_first_axes(tmp_path: Path) -> None:
    pieces = tmp_path / "pieces"
    pieces.mkdir()
    crops = h.unit_rows(4 * 9, 8, seed=5)
    for name, files, rows in (("000", ["fa", "fb"], crops[:18]), ("001", ["fc", "fd"], crops[18:])):
        (pieces / f"{name}.npz").write_bytes(
            write_piece(files, 9, {
                "coords": rows.astype(np.float16), "mean_dot": np.zeros(18, np.float32),
                "failed": np.array(["fx"]), "reasons": np.array(["read: X"]),
            })
        )  # fmt: skip
    projection = tmp_path / "projection.npz"
    np.savez(projection, mean=np.zeros(8, np.float32), components=np.eye(8, dtype=np.float32))
    out = tmp_path / "bundle"
    facts = write_crop_files(
        piece_paths(tmp_path), out, index_ids=["i1", "i2", "c1"],
        file_of={"i1": "fc", "i2": "fa"}, projection=projection, dims=4, rule=CropRule(),
    )  # fmt: skip
    assert facts["crops"] == 36
    assert facts["failed_files"] == 2
    assert np.load(out / ROW_FILE).tolist() == [2, 0, -1]
    index = read_crops(out)
    assert index is not None
    assert index.components.shape == (4, 8)
    query = np.zeros(8, dtype=np.float32)
    query[:4] = crops[20, :4] / np.linalg.norm(crops[20, :4])
    scores = index.row_scores(query, np.ones(3, dtype=bool))
    assert int(np.argmax(scores)) == 0  # row i1, file fc, holds crop 20


def test_measure_queries_say_where_their_vectors_come_from() -> None:
    queries = [
        Query(query_id="q1", set_id="T5", entity_id="E", query_image_id="pop1",
              expected_entity_ids=("E",), exclude_image_ids=("pop2",)),
        Query(query_id="q2", set_id="T2", entity_id="F", query_image_id="commons-1",
              expected_entity_ids=("F",), exclude_image_ids=()),
        Query(query_id="q3", set_id="T2", entity_id="F", query_image_id="commons-2",
              expected_entity_ids=("F",), exclude_image_ids=()),
    ]  # fmt: skip
    spec = measure_queries(queries, {"pop1", "pop2"}, {"pop1": "file1"}, {"commons-1"}, "x")
    assert [q.query_id for q in spec.queries] == ["q1", "q2"]
    assert spec.skipped == ("q3",)
    assert spec.queries[0].file_id == "file1"
    assert spec.queries[0].hidden == ("pop1", "pop2")


def test_the_summary_counts_hits_and_first_places_per_set() -> None:
    def ranks(query_id: str, off: int, on: int) -> QueryRanks:
        return QueryRanks(query_id=query_id, set_id="PARTS", bucket="object", off=off, on=on,
                          off_ms=10.0, on_ms=30.0, crops_ms=15.0)  # fmt: skip

    found = (ranks("a", 1, 2), ranks("b", 30, 1), ranks("c", 0, 0))
    out = MeasureOut(index="x", gpu="g", ranks=found, photo_search_ms={})
    row = summarize(out)["PARTS"]
    assert row["queries"] == 2
    assert row["hit@1 off"] == 50.0
    assert row["hit@10 on"] == 100.0
    assert row["first lost"] == 1
    assert row["first gained"] == 1
    assert row["median crops ms"] == 15.0
