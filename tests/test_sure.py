"""Sure object answers: the margin rule (`miroir.web.sure`) and how it is set (`eval.sure`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from miroir.eval.site_rerank import Searched
from miroir.eval.sure import choose, sure_rows
from miroir.web.bundle import SiteHit
from miroir.web.engine import SearchRequest
from miroir.web.routing import PhotoType
from miroir.web.sure import SureRule, margin_of, read_sure, write_sure
from tests import web_helpers as h


def test_margin_is_the_gap_to_the_best_other_answer() -> None:
    assert margin_of([0.9, 0.7, 0.85]) == pytest.approx(0.05)
    assert margin_of([0.9]) == pytest.approx(0.9)
    # Two records sharing one image tie: never sure, however high the cosine.
    assert margin_of([1.0, 1.0, 0.6]) == 0.0


def test_the_rule_reads_back(tmp_path: Path) -> None:
    write_sure(tmp_path / "sure.json", SureRule(margin=0.09, note="T3, T5, T2b"))
    assert read_sure(tmp_path / "sure.json") == SureRule(margin=0.09, note="T3, T5, T2b")
    assert read_sure(tmp_path / "missing.json") is None


def test_an_object_answer_is_sure_only_when_it_stands_out() -> None:
    engine, _ = h.engine(building=False, cutoff=-1.0)  # the cosine cut-off alone: always sure
    assert engine.search(SearchRequest(image=h.photo(90))).match is True
    engine.sure = SureRule(margin=0.3)
    assert engine.search(SearchRequest(image=h.photo(90))).match is True
    engine.sure = SureRule(margin=5.0)
    assert engine.search(SearchRequest(image=h.photo(90))).match is False


def test_buildings_keep_their_cut_off() -> None:
    engine, _ = h.engine(building=True, cutoff=-1.0)
    engine.sure = SureRule(margin=5.0)
    assert engine.search(SearchRequest(image=h.photo(90))).match is True


def hit(confidences: tuple[float, ...], right_rank: int) -> SiteHit:
    n = len(confidences)
    return SiteHit(
        bucket="object",
        photo_type=PhotoType.OBJECT,
        entities=tuple(f"e{i}" for i in range(n)),
        confidences=confidences,
        best_images=tuple(f"i{i}" for i in range(n)),
        right_rank=right_rank,
    )


def searched(set_id: str, hits: list[SiteHit]) -> Searched:
    ids = tuple(f"{set_id}-{i}" for i in range(len(hits)))
    return Searched(set_id, ids, ids, tuple(hits))


def test_the_smallest_margin_keeping_both_promises_is_chosen() -> None:
    # Right answers stand out by 0.2; wrong ones by 0.04; T8 searches by up to 0.06.
    t5 = [hit((0.9, 0.7), 1)] * 7 + [hit((0.8, 0.76), 2)] * 3
    t8 = [hit((0.8, 0.8 - 0.01 * i), 0) for i in range(7)] * 2
    rows = sure_rows([searched("T5", t5), searched("T8", t8)])
    chosen = choose(rows)
    assert chosen.right["T5"] == 1.0
    assert chosen.t8_sure <= 0.10
    assert chosen.margin == pytest.approx(0.065)
