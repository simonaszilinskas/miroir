from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from miroir.config import Settings
from miroir.eval.results import set_score, write_result
from miroir.eval.run import search_queries
from miroir.eval.scoring import Gallery, score
from miroir.eval.sets import Query, write_gallery, write_set
from miroir.eval.vectors import Route, RunInfo, read_vectors, write_vectors
from miroir.eval.viewer import GALLERY, build_page, find_model_runs
from miroir.records.build import build
from miroir.records.model import Image, Record
from miroir.records.store import read_images, read_records
from tests.fakes import jpeg

GOOD = "A1-good-224-screening-20k-publishable"
BAD = "A2-bad-224-screening-20k"
BAD_QUERIES = "A2-bad-224-T1-tune-queries"
DIM = 16


def _info(option: str, model: str, gallery: str) -> RunInfo:
    return RunInfo(
        option=option,
        model=model,
        resolution=224,
        pooling="cls",
        gallery=gallery,
        created=date(2026, 10, 12),
        cost_usd=0.0,
        wall_time_s=1.0,
        route=Route.HOSTED,
    )


def _queries(set_id: str, records: Sequence[Record]) -> list[Query]:
    return [
        Query(
            query_id=f"{set_id}-{r.images[0].image_id}",
            set_id=set_id,
            entity_id=r.entity_id,
            query_image_id=r.images[0].image_id,
            expected_entity_ids=(r.entity_id,),
            exclude_image_ids=(r.images[0].image_id,),
        )
        for r in records
    ]


@pytest.fixture
def data(settings: Settings) -> list[Record]:
    """Two sets, a publishable gallery, a good model and a bad one whose queries ran apart."""
    build(settings)
    records = read_records(settings.records_parquet, settings.images_parquet)
    multi = [r for r in records if r.publishable and len(r.images) >= 2]
    private = next(r for r in records if not r.publishable)
    write_set(settings, "T1", [*_queries("T1", multi[:5]), *_queries("T1", [private])], [])
    write_set(settings, "T5", _queries("T5", multi[5:]), [])
    # The list holds one private image by mistake: the page must leave it out.
    write_gallery(
        settings,
        GALLERY,
        [i.image_id for r in records if r.publishable for i in r.images]
        + [private.images[0].image_id],
    )

    images = read_images(settings.images_parquet)
    entities = sorted({i.entity_id for i in images})
    rng = np.random.default_rng(7)
    centre = dict(zip(entities, rng.normal(size=(len(entities), DIM)), strict=True))
    ids = [i.image_id for i in images]
    good = np.stack([centre[i.entity_id] + 0.05 * rng.normal(size=DIM) for i in images])
    bad = rng.normal(size=(len(ids), DIM))
    write_vectors(settings.vectors_path(GOOD), ids, good, _info("A1", "org/good", GALLERY))
    t1_query_ids = {q.query_image_id for q in _queries("T1", multi[:5])}
    in_gallery = [n for n, i in enumerate(ids) if i not in t1_query_ids]
    in_queries = [n for n, i in enumerate(ids) if i in t1_query_ids]
    write_vectors(
        settings.vectors_path(BAD),
        [ids[n] for n in in_gallery],
        bad[in_gallery],
        _info("A2", "org/bad", "screening-20k"),
    )
    write_vectors(
        settings.vectors_path(BAD_QUERIES),
        [ids[n] for n in in_queries],
        bad[in_queries],
        _info("A2", "org/bad", "T1-tune-queries"),
    )
    return records


class _Fetch:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, images: Sequence[Image]) -> Mapping[str, bytes]:
        self.asked.extend(i.image_id for i in images)
        return {i.image_id: jpeg(640, 480) for i in images}


def _payload(page: str) -> dict[str, object]:
    start = page.index('type="application/json">') + len('type="application/json">')
    return json.loads(page[start : page.index("</script>", start)])


def test_page_shows_each_model_top_five_on_publishable_queries(
    settings: Settings, data: list[Record], tmp_path: Path
) -> None:
    models = find_model_runs(settings)
    assert [m.name for m in models] == [GOOD, BAD]
    fetch = _Fetch()

    page = build_page(
        settings, models, fetch, sets=("T1", "T5"), per_set=3, results=tmp_path / "none"
    )

    assert page.startswith("<title>Miroir Match Viewer</title>\n<style>")
    assert not re.search(r"<(!doctype|html|head|body|link)[\s>]", page, re.IGNORECASE)
    assert "http://" not in page
    payload = _payload(page)
    public = {i.image_id for r in data if r.publishable for i in r.images}
    assert fetch.asked
    assert set(fetch.asked) <= public
    images = payload["images"]
    assert isinstance(images, dict)
    assert all(str(uri).startswith("data:image/jpeg;base64,") for uri in images.values())

    queries = payload["queries"]
    assert isinstance(queries, list)
    assert len(queries) == 6  # 3 per set; the private T1 query is never picked
    entity_of = {i.image_id: i.entity_id for r in data for i in r.images}
    for q in queries:
        good, bad = q["answers"]
        assert good["rank"] == 1
        assert good["top"][0]["right"] is True
        assert len(good["top"]) == 5
        assert [h["right"] for h in bad["top"]] == [
            entity_of[h["image"]] == q["query"]["entity"] for h in bad["top"]
        ]
        assert q["query"]["image"] not in {h["image"] for a in q["answers"] for h in a["top"]}

    # The bad model's ranks match the scoring harness on the same gallery.
    gallery_ids = tuple(i for i in read_vectors(settings.vectors_path(BAD)).ids if i in public)
    bad_gallery = Gallery.from_vectors(models[1].gallery, entity_of, gallery_ids)
    shown = [
        Query(q["query_id"], "T1", "", q["query"]["image"], (q["query"]["entity"],), ())
        for q in queries
    ]
    asking = read_vectors(settings.vectors_path(BAD_QUERIES))
    t1 = [s for s in shown if s.query_id.startswith("T1")]
    expected = {
        s.query_id: s.rank for s in score(asking, bad_gallery, search_queries(t1)).per_query
    }
    for q in queries:
        if q["query_id"] in expected:
            assert q["answers"][1]["rank"] == expected[q["query_id"]]

    sets = payload["sets"]
    assert isinstance(sets, list)
    assert [s["set_id"] for s in sets] == ["T1", "T5"]
    assert {line["source"] for s in sets for line in s["scores"]} == {"computed here"}


def test_score_lines_come_from_result_files_when_the_run_was_scored(
    settings: Settings, data: list[Record], tmp_path: Path
) -> None:
    good = read_vectors(settings.vectors_path(GOOD))
    entity_of = {i.image_id: i.entity_id for i in read_images(settings.images_parquet)}
    asked = next(r for r in data if r.publishable and len(r.images) >= 2).images[0]
    one = search_queries(
        [Query("T1-x", "T1", "E", asked.image_id, (asked.entity_id,), (asked.image_id,))]
    )
    gallery = Gallery.from_vectors(good, entity_of)
    scored = set_score(score(good, gallery, one), test_set="T1", half="tune", gallery=GALLERY)
    results = tmp_path / "results"
    write_result(
        results / "R1" / "A-A1-good.json",
        round_id="R1",
        step="A",
        option="A1-good",
        vector_run=GOOD,
        query_vector_run=GOOD,
        vectors=good.info,
        scores=[scored],
        today=date(2026, 10, 12),
    )

    page = build_page(
        settings, find_model_runs(settings, [GOOD]), _Fetch(), sets=("T1",), results=results
    )

    (t1,) = _payload(page)["sets"]  # pyright: ignore[reportGeneralTypeIssues]
    assert [line["source"] for line in t1["scores"]] == ["eval/results/R1/A-A1-good.json"]
    assert t1["scores"][0]["queries"] == 1


def test_thumbnails_shrink_until_the_page_fits(settings: Settings, data: list[Record]) -> None:
    del data
    models = find_model_runs(settings, [GOOD])
    roomy = build_page(settings, models, _Fetch(), sets=("T1",), results=Path("/nonexistent"))
    tight = build_page(
        settings,
        models,
        _Fetch(),
        sets=("T1",),
        results=Path("/nonexistent"),
        max_bytes=len(roomy.encode()) - 1,
    )
    assert len(tight) < len(roomy)
    with pytest.raises(ValueError, match="show fewer queries"):
        build_page(settings, models, _Fetch(), sets=("T1",), max_bytes=1_000)
