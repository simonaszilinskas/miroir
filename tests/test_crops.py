"""The crop slice test: crop windows, gallery picking, and ranking with crops (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from miroir.eval.crops import (
    VARIANTS,
    FullScores,
    Groups,
    QueryCase,
    Scores,
    full_rank,
    hits,
    look_alikes,
    rank_slice,
    summarize_full,
    usable_rows,
)
from miroir.eval.sets import Query
from miroir.jobs.crop_slice import (
    GRIDS,
    Grid,
    QuerySource,
    SliceSpec,
    build_spec,
    crop_boxes,
    pick_a7_files,
    pick_gallery,
    read_spec,
    spec_json,
)

NEG = -np.inf


# --- windows ---------------------------------------------------------------------------------


def test_grid_windows_cover_the_photo_and_overlap() -> None:
    two, three = GRIDS
    boxes = crop_boxes(1000, 500, two)
    assert boxes == [
        (0, 0, 600, 300),
        (400, 0, 1000, 300),
        (0, 200, 600, 500),
        (400, 200, 1000, 500),
    ]
    boxes = crop_boxes(1000, 1000, three)
    assert len(boxes) == 9
    assert boxes[0] == (0, 0, 400, 400)
    assert boxes[-1] == (600, 600, 1000, 1000)
    assert boxes[1][0] < boxes[0][2]  # neighbours share a strip


def test_grid_refuses_windows_that_leave_gaps() -> None:
    with pytest.raises(ValueError, match="side"):
        Grid(3, 0.3)


# --- the gallery -------------------------------------------------------------------------------


def _query(
    query_id: str, set_id: str, image: str, entity: str, hidden: tuple[str, ...] = ()
) -> Query:
    return Query(
        query_id=query_id,
        set_id=set_id,
        entity_id=entity,
        query_image_id=image,
        expected_entity_ids=(entity,),
        exclude_image_ids=hidden,
    )


def test_gallery_keeps_every_answer_samples_others_and_leaves_queries_out() -> None:
    ids = [f"i{n}" for n in range(20)]
    entity_of = {i: ("T" if n < 3 else f"E{n}") for n, i in enumerate(ids)}
    answers, others = pick_gallery(ids, entity_of, {"T"}, {"i0", "i10"}, size=8, seed=1)
    assert answers == ["i1", "i2"]
    assert len(others) == 6
    assert "i10" not in others
    assert pick_gallery(ids, entity_of, {"T"}, {"i0", "i10"}, size=8, seed=1)[1] == others


def test_a7_sample_takes_answers_then_seeded_others() -> None:
    picked = pick_a7_files(["f1", "f1", ""], ["f2", "f3", "f4", "f1"], 2, seed=3)
    assert picked[0] == "f1"
    assert len(picked) == 3
    assert set(picked[1:]) <= {"f2", "f3", "f4"}


def test_spec_names_each_query_source_and_round_trips() -> None:
    queries = [
        _query("VET-1", "VET", "outside-VET-1", "V"),
        _query("P-1", "PARTS", "i0", "P", hidden=("i0", "i1")),
        _query("T2-1", "T2", "commons-aa", "B"),
    ]
    ids = ["i0", "i1", "i2", "i3", "commons-bb"]
    entity_of = {"i0": "P", "i1": "P", "i2": "V", "i3": "X", "commons-bb": "B"}
    file_of = {"i0": "f0", "i1": "f1", "i2": "f2", "i3": "f3"}
    spec = build_spec(queries, ids, entity_of, file_of, size=10, a7_distractors=5)
    assert "i0" not in spec.gallery
    assert dict(zip(spec.gallery, spec.files, strict=True))["commons-bb"] == ""
    sources = {q.query_id: q.source for q in spec.queries}
    assert sources == {
        "VET-1": QuerySource.OUTSIDE,
        "P-1": QuerySource.CATALOG,
        "T2-1": QuerySource.SUPPLIED,
    }
    assert spec.queries[1].file_id == "f0"
    assert set(spec.a7_files) == {"f1", "f2", "f3"}  # VET and PARTS answers, then others
    again = read_spec(spec_json(spec))
    assert again == spec
    assert sorted(again.crop_files()) == ["f1", "f2", "f3"]


# --- ranking with crops ------------------------------------------------------------------------


def _scores(
    whole3: list[list[float]],
    crops3: list[list[float]],
    entities: tuple[str, ...],
    *,
    weight: float = 0.0,
) -> Scores:
    n_q, n_rows = len(whole3), len(entities)
    w3 = np.asarray(whole3, dtype=np.float16)
    c3 = np.asarray(crops3, dtype=np.float16)
    return Scores(
        query_ids=tuple(f"q{i}" for i in range(n_q)),
        a7_weight=np.full(n_q, weight, dtype=np.float32),
        buckets=("colour-building",) * n_q,
        row_entities=entities,
        row_file=np.arange(n_rows, dtype=np.int64),
        whole7=np.zeros((n_q, n_rows), dtype=np.float16),
        whole3=w3,
        crops3={"g2": c3, "g3": np.full_like(c3, NEG)},
        crops7={"g2": np.full_like(c3, NEG), "g3": np.full_like(c3, NEG)},
        a7_files=np.zeros(n_rows, dtype=bool),
        done=np.ones(n_rows, dtype=bool),
    )


def _case(query_id: str, right: str, hidden: list[int] | None = None) -> QueryCase:
    return QueryCase(query_id, "S", "", np.asarray(hidden or [], dtype=np.int64), (right,))


def test_a_crop_lifts_the_whole_its_part_belongs_to() -> None:
    # Row 0 is the whole (entity A); its whole vector matches the detail badly, a crop well.
    scores = _scores([[0.2, 0.5, 0.1]], [[0.9, 0.3, 0.1]], ("A", "B", "C"))
    ranked = rank_slice(scores, [_case("q0", "A")], VARIANTS)
    assert ranked.ranks["a-whole"] == [2]
    assert ranked.ranks["b-whole+2x2"] == [1]
    assert ranked.first["b-whole+2x2"] == ["A"]


def test_hidden_rows_lose_their_crops_too() -> None:
    scores = _scores([[0.2, 0.5, 0.1]], [[0.9, 0.3, 0.1]], ("A", "A", "C"))
    ranked = rank_slice(scores, [_case("q0", "A", hidden=[0])], VARIANTS)
    assert ranked.ranks["b-whole+2x2"] == [1]  # row 1 (A, 0.5) still beats C
    gone = rank_slice(scores, [_case("q0", "C", hidden=[2])], VARIANTS)
    assert gone.ranks["a-whole"] == [0]  # nothing of C left to find


def test_rows_without_crops_done_are_left_out() -> None:
    scores = _scores([[0.2, 0.5]], [[0.9, 0.3]], ("A", "B"))
    cut = Scores(
        **{**{f: getattr(scores, f) for f in Scores.__slots__}, "done": np.array([True, False])}
    )
    assert usable_rows(cut).tolist() == [0]
    assert usable_rows(cut, a7=True).tolist() == []


def test_groups_take_the_best_row_per_entity() -> None:
    groups = Groups.of(["B", "A", "B"])
    assert groups.entity_ids == ("A", "B")
    out = groups.entity_max(np.asarray([0.1, 0.2, 0.7], dtype=np.float32))
    assert out.tolist() == pytest.approx([0.2, 0.7])


def test_full_rank_puts_the_crop_score_against_every_other_entity() -> None:
    e7 = np.zeros(5, dtype=np.float32)
    e3 = np.asarray([0.3, 0.9, 0.8, 0.1, NEG], dtype=np.float32)
    full = summarize_full(e7, e3, 0.0, np.asarray([0, 3], dtype=np.int64), top=3)
    assert full.top.tolist() == [1, 2, 0]
    assert (full.lo3, full.hi3) == pytest.approx((0.1, 0.9))
    top_in_gallery = np.asarray([-1, -1, 0], dtype=np.int64)
    no_crops = np.full(2, NEG, dtype=np.float32)
    kwargs = {"top_in_gallery": top_in_gallery, "crop7": no_crops, "a7_weight": 0.0, "right": [0]}
    assert full_rank(full, crop3=no_crops, **kwargs) == 3
    lifted = np.asarray([0.95, NEG], dtype=np.float32)
    assert full_rank(full, crop3=lifted, **kwargs) == 1


def test_full_rank_counts_an_answer_off_the_shortlist_as_missed() -> None:
    full = FullScores(
        0.0, 1.0, 0.0, 1.0,
        top=np.asarray([0], dtype=np.int64),
        top_e7=np.asarray([0.5], dtype=np.float32),
        top_e3=np.asarray([0.5], dtype=np.float32),
        gallery_e7=np.asarray([0.4], dtype=np.float32),
        gallery_e3=np.asarray([NEG], dtype=np.float32),
    )  # fmt: skip
    none = np.full(1, NEG, dtype=np.float32)
    args = {"top_in_gallery": np.asarray([-1]), "crop7": none, "a7_weight": 0.2, "right": [0]}
    assert full_rank(full, crop3=none, **args) == 2
    assert full_rank(full, crop3=np.asarray([0.9], dtype=np.float32), **args) == 1


def test_hits_and_look_alikes() -> None:
    assert hits([1, 3, 12, 0]) == {"queries": 3.0, "hit@1": 33.3, "hit@10": 66.7}
    moved = look_alikes([1, 2, 1, 0], [2, 1, 1, 5])
    assert moved == {
        "lost_first": 1,
        "gained_first": 1,
        "wrong_first_before": 1,
        "wrong_first_after": 1,
    }


def test_spec_rejects_unknown_fields() -> None:
    spec = SliceSpec(gallery=(), files=(), a7_files=(), queries=())
    with pytest.raises(ValueError, match="other"):
        read_spec(spec_json(spec).replace(b'"note"', b'"other": 1, "note"'))


def _write_outputs(folder, spec: SliceSpec) -> None:
    """What `miroir.remote.crop_slice` saves, with the same keys, for a tiny gallery."""
    n_q = len(spec.queries)
    files = spec.crop_files()
    entities = np.array(["A", "B", "C"])
    np.savez(
        folder / "queries.npz",
        query_ids=np.array([q.query_id for q in spec.queries]),
        a7=np.zeros((n_q, 4), dtype=np.float32),
        chroma=np.zeros(n_q, dtype=np.float32),
        a7_weight=np.zeros(n_q, dtype=np.float32),
        buckets=np.array(["colour-building"] * n_q),
    )
    np.savez(
        folder / "whole.npz",
        image_ids=np.array(spec.gallery),
        files=np.array(spec.files),
        entities=entities,
        whole7=np.asarray([[0.3, 0.6, 0.1]] * n_q, dtype=np.float16),
        whole3=np.asarray([[0.3, 0.6, 0.1]] * n_q, dtype=np.float16),
    )
    col = {f: i for i, f in enumerate(files)}
    crop = np.full((n_q, len(files)), NEG, dtype=np.float16)
    crop[:, col["fa"]] = 0.95
    np.savez(
        folder / "crops.npz",
        files=np.array(files),
        done=np.ones(len(files), dtype=bool),
        a7_files=np.array([f in spec.a7_files for f in files]),
        a3_g2=crop,
        a3_g3=np.full_like(crop, NEG),
        a7_g2=crop,
        a7_g3=np.full_like(crop, NEG),
        a3_g2_best=np.zeros((n_q, len(files)), dtype=np.int8),
        a3_g3_best=np.zeros((n_q, len(files)), dtype=np.int8),
    )
    # Full catalog: entities A, B, C at rows 0, 1, 2, and D (row 7), never in the gallery.
    np.savez(
        folder / "full.npz",
        top=np.asarray([[7, 1, 0]] * n_q, dtype=np.int64),
        top_e7=np.asarray([[0.9, 0.6, 0.3]] * n_q, dtype=np.float32),
        top_e3=np.asarray([[0.9, 0.6, 0.3]] * n_q, dtype=np.float32),
        gallery_e7=np.asarray([[0.3, 0.6, 0.1]] * n_q, dtype=np.float32),
        gallery_e3=np.asarray([[0.3, 0.6, 0.1]] * n_q, dtype=np.float32),
        range=np.asarray([[0.0, 0.9, 0.0, 0.9]] * n_q),
        named_rows=np.asarray([0, 1, 7], dtype=np.int64),
        named_entities=np.array(["A", "B", "D"]),
        gallery_entities=np.array(["A", "B", "C"]),
    )


def test_report_reads_what_the_machine_writes(tmp_path) -> None:
    from miroir.eval.crops import cases_of, load_scores, report  # noqa: PLC0415

    queries = [_query("VET-1", "VET", "outside-VET-1", "A"), _query("T2-1", "T2", "commons-q", "B")]
    ids = ["ia", "ib", "ic"]
    spec = build_spec(
        queries, ids, {"ia": "A", "ib": "B", "ic": "C"}, {"ia": "fa", "ib": "fb", "ic": "fc"}
    )
    _write_outputs(tmp_path, spec)
    scores = load_scores(tmp_path, spec)
    out = report(scores, cases_of(spec, queries))
    vet = out["vet_ranks"]["VET-1"]  # type: ignore[index]
    assert vet == {"a-whole": 2, "b-whole+2x2": 1, "c-whole+2x2+3x3": 1}
    full = out["full_ranks"]["VET-1"]  # type: ignore[index]
    assert full["a-whole"] == 3  # D and B above A
    assert full["b-whole+2x2"] == 1  # the crop (0.95) passes D (0.9)
    assert out["look_alikes"]["T2"]["b-whole+2x2"]["lost_first"] == 1  # type: ignore[index]
