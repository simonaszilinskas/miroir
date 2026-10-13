"""Modal runs, driven against fake remote calls: no Modal, no network, no model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pytest

from miroir.commons.model import CommonsSet
from miroir.config import Settings
from miroir.eval import amstertime
from miroir.eval.rerank import CandidateMatch, MatchChunk, MatchTask, QueryMatches, read_matches
from miroir.eval.sets import Query, write_gallery, write_set
from miroir.eval.vectors import Route, read_vectors
from miroir.jobs.checkpoint import Checkpoint
from miroir.jobs.modal_run import ModalRunner, load_modal_probe, output_runs, run_name
from miroir.jobs.modal_spec import (
    CHUNK_TIMEOUT_S,
    L4,
    MATCH_QUERIES,
    OPTIONS,
    R0_OPTIONS,
    ChunkResult,
    ModalOption,
    Resize,
    SuppliedImage,
    option,
)
from miroir.jobs.privacy import PrivacyError
from miroir.jobs.spend import BudgetError, Service, SpendKind
from miroir.records.build import build
from miroir.records.model import Image
from miroir.records.store import read_records, write_commons
from tests.fakes import FakeCommons, commons_photo

GEMMA = "A7-embeddinggemma-2-512"
SECONDS_PER_CALL = 100.0


@dataclass
class Clock:
    """Each remote call advances it by SECONDS_PER_CALL."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeCalls:
    clock: Clock
    calls: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    fail: bool = False

    supplied: list[str] = field(default_factory=list[str])
    """Ids of the images sent as bytes."""

    def embed(
        self,
        option: ModalOption,
        images: Sequence[Image],
        supplied: Sequence[SuppliedImage] = (),
    ) -> ChunkResult:
        self.calls.append((option.id, len(images) + len(supplied)))
        self.supplied.extend(s.image_id for s in supplied)
        self.clock.now += SECONDS_PER_CALL
        if self.fail:
            msg = "container died"
            raise RuntimeError(msg)
        ids = (*(i.image_id for i in images), *(s.image_id for s in supplied))
        vectors = np.random.default_rng(len(ids)).normal(size=(len(ids), 8)).astype(np.float32)
        return ChunkResult(ids, vectors, {}, load_s=30.0, embed_s=60.0)

    def embed_texts(
        self, option: ModalOption, ids: Sequence[str], texts: Sequence[str]
    ) -> ChunkResult:
        self.calls.append((f"texts {option.id}", len(texts)))
        self.clock.now += SECONDS_PER_CALL
        vectors = np.random.default_rng(len(ids)).normal(size=(len(ids), 8)).astype(np.float32)
        return ChunkResult(tuple(ids), vectors, {}, load_s=30.0, embed_s=1.0)

    def scale(self, option: ModalOption, containers: int) -> None:
        self.calls.append((f"scale {option.id}", containers))

    def fill_images(self, chunks: Sequence[Sequence[Image]]) -> dict[str, int]:
        self.calls.append(("fill", sum(len(c) for c in chunks)))
        self.clock.now += SECONDS_PER_CALL
        return {"ok": sum(len(c) for c in chunks)}

    def amstertime_download(self) -> int:
        self.calls.append(("download", 0))
        self.clock.now += SECONDS_PER_CALL
        return 2 * amstertime.PAIRS

    def match(self, tasks: Sequence[MatchTask]) -> MatchChunk:
        self.calls.append(("match", len(tasks)))
        self.clock.now += SECONDS_PER_CALL
        done = tuple(
            QueryMatches(
                t.query_id,
                t.image_id,
                tuple(CandidateMatch(c, 20, 10, 12) for c in t.candidates),
                query_s=0.01,
                match_s=0.2,
                candidates_s=0.5,
            )
            for t in tasks
        )
        return MatchChunk(done, load_s=5.0, missing=())

    def amstertime_embed(self, option: ModalOption) -> ChunkResult:
        self.calls.append((option.id, 2 * amstertime.PAIRS))
        self.clock.now += SECONDS_PER_CALL
        places = [f"{n:04d}" for n in range(amstertime.PAIRS)]
        ids = (*(f"old/{p}.jpg" for p in places), *(f"new/{p}.png" for p in places))
        vectors = np.random.default_rng(0).normal(size=(len(ids), 8)).astype(np.float32)
        return ChunkResult(ids, vectors, {}, load_s=60.0, embed_s=30.0)


@pytest.fixture
def built(settings: Settings) -> Settings:
    build(settings)
    return settings


def image_ids(settings: Settings, *, publishable: bool) -> list[str]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return [i.image_id for r in records if r.publishable == publishable for i in r.images]


def runner(
    settings: Settings, *, key_usage: float | None = None, commons: FakeCommons | None = None
) -> tuple[ModalRunner, FakeCalls]:
    clock = Clock()
    calls = FakeCalls(clock)
    fetcher = None if commons is None else commons.images(settings)
    return ModalRunner(
        settings, calls, key_usage_usd=key_usage, commons=fetcher, clock=clock
    ), calls


def test_options_name_their_model_and_machine() -> None:
    assert option(GEMMA).model == "google/embeddinggemma-2"
    assert option("R0-dinov2-giant-224").model == "facebook/dinov2-giant"
    with pytest.raises(ValueError, match="unknown Modal option"):
        option("A99")
    assert L4.usd_per_s == pytest.approx(0.000222 + 4 * 0.0000131 + 16 * 0.00000222)
    assert all(o.dtype != "float16" for o in OPTIONS.values())


def test_probe_then_run_writes_vectors_and_logs_modal_spend(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    r, calls = runner(built)

    probe, size = r.probe(GEMMA, "g")
    assert size == len(ids)
    assert probe.images == len(ids)  # fewer than 100 in the fixtures
    assert probe.cost_usd == pytest.approx(SECONDS_PER_CALL * L4.usd_per_s)
    assert probe.fixed_s == pytest.approx(SECONDS_PER_CALL - 60.0)
    assert load_modal_probe(built.probe_path(GEMMA)) == probe

    (done,) = r.embed_gallery(GEMMA, "g")
    loaded = read_vectors(built.vectors_path(run_name(OPTIONS[GEMMA], "g")))
    assert loaded.ids == done.ids
    assert loaded.info.route == Route.HOSTED
    assert loaded.info.option == "A7"
    assert "Modal L4" in loaded.info.notes
    assert r.ledger.total(service=Service.MODAL) == pytest.approx(2 * probe.cost_usd)
    assert r.ledger.total(SpendKind.PROBE) == pytest.approx(probe.cost_usd)
    assert [c[0] for c in calls.calls] == [GEMMA, GEMMA]


def test_words_are_embedded_without_a_probe_and_logged(built: Settings) -> None:
    r, calls = runner(built)
    done = r.embed_texts(GEMMA, "T7-words", ["T7-a", "T7-b"], ["église", "vase"], max_usd=1.0)
    assert done.ids == ("T7-a", "T7-b")
    assert done.info.pooling == "text"
    assert "'SearchQuery' prompt" in done.info.notes
    assert calls.calls == [(f"texts {GEMMA}", 2)]
    assert r.ledger.total(service=Service.MODAL) == pytest.approx(SECONDS_PER_CALL * L4.usd_per_s)
    with pytest.raises(ValueError, match="non-empty text"):
        r.embed_texts(GEMMA, "other", ["x"], [""], max_usd=1.0)
    with pytest.raises(ValueError, match="only EmbeddingGemma"):
        r.embed_texts("A3-megaloc-322", "other", ["x"], ["y"], max_usd=1.0)


def test_run_needs_a_probe(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, calls = runner(built)
    with pytest.raises(BudgetError, match="--probe"):
        r.embed_gallery(GEMMA, "g")
    assert calls.calls == []


def test_galleries_with_private_images_never_go_to_modal(built: Settings) -> None:
    mixed = [*image_ids(built, publishable=True)[:2], image_ids(built, publishable=False)[0]]
    write_gallery(built, "g", mixed)
    r, calls = runner(built)
    with pytest.raises(PrivacyError, match="not publishable"):
        r.probe(GEMMA, "g")
    assert calls.calls == []


def test_calls_are_refused_near_the_cap(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, _ = runner(built)
    r.probe(GEMMA, "g")
    near, calls = runner(built, key_usage=built.openrouter_budget_usd - 5.1)
    with pytest.raises(BudgetError, match="usable budget"):
        near.embed_gallery(GEMMA, "g")
    assert calls.calls == []


def test_probes_must_fit_their_worst_case(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, calls = runner(built)
    worst = L4.usd_per_s * CHUNK_TIMEOUT_S
    r.ledger.record(
        SpendKind.PROBE,
        option="A9",
        run="old",
        model="m",
        cost_usd=built.probe_budget_usd - worst / 2,
        items=100,
    )
    with pytest.raises(BudgetError, match="probes have"):
        r.probe(GEMMA, "g")
    assert calls.calls == []


def test_a_failed_call_still_costs(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, calls = runner(built)
    calls.fail = True
    with pytest.raises(RuntimeError, match="container died"):
        r.probe(GEMMA, "g")
    assert r.ledger.total(service=Service.MODAL) == pytest.approx(SECONDS_PER_CALL * L4.usd_per_s)


def test_r0_downloads_on_modal_and_writes_the_run(settings: Settings) -> None:
    r, calls = runner(settings)
    done = r.r0("dinov2-giant-224")
    assert [c[0] for c in calls.calls] == ["download", "R0-dinov2-giant-224"]
    assert done.info.gallery == "amstertime"
    assert done.info.pooling == "cls"
    loaded = read_vectors(settings.vectors_path("R0-dinov2-giant-224-amstertime"))
    assert len(loaded.ids) == 2 * amstertime.PAIRS
    assert done.info.cost_usd == pytest.approx(r.ledger.total(service=Service.MODAL))
    with pytest.raises(FileExistsError):
        r.r0("dinov2-giant-224")


def test_r0_options_follow_the_dinov3_paper() -> None:
    for opt in R0_OPTIONS.values():
        assert (opt.resolution, opt.pooling) == (224, "cls")
    assert R0_OPTIONS["dinov3-vit7b16-224"].gated


def test_a_stopped_run_resumes_without_paying_twice(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    r, calls = runner(built)
    r.probe(GEMMA, "g")
    run = run_name(OPTIONS[GEMMA], "g")
    first = sorted(ids)[:2]
    saved = np.ones((2, 8), dtype=np.float16) / np.sqrt(8)
    Checkpoint(built.checkpoint_path(run)).append(first, saved)

    (done,) = r.embed_gallery(GEMMA, "g")
    assert calls.calls[-1] == (GEMMA, len(ids) - 2)
    assert set(done.ids) == set(ids)
    assert not built.checkpoint_path(run).exists()


def test_max_usd_refuses_a_run_projected_above_it(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, calls = runner(built)
    r.probe(GEMMA, "g")
    with pytest.raises(BudgetError, match="over the"):
        r.embed_gallery(GEMMA, "g", max_usd=0.0001)
    assert len(calls.calls) == 1  # the probe only


def test_fill_images_sends_the_gallery_once_and_logs_its_cost(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    r, calls = runner(built)
    assert r.fill_images("g") == {"ok": len(ids)}
    assert calls.calls == [("fill", len(ids))]
    assert r.ledger.total(service=Service.MODAL) > 0


def test_dino_options_ask_for_the_bigger_machine() -> None:
    assert option("A2-dinov3-7b-256-cls").machine.memory_gib == 32
    assert option(GEMMA).machine == L4


def test_cls_and_gem_share_a_probe(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, calls = runner(built)
    r.probe("A1-dinov2-l-224-cls", "g")
    r.embed_gallery("A1-dinov2-l-224-gem", "g")
    assert [c[0] for c in calls.calls] == ["A1-dinov2-l-224-cls", "A1-dinov2-l-224-gem"]


def test_cls_gem_writes_a_cls_run_and_a_gem_run(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    r, _ = runner(built)
    nocrop = "A2-dinov3-l-256-nocrop"
    r.probe(nocrop, "g")
    written = r.embed_gallery(nocrop, "g")
    names = list(output_runs(option(nocrop), "g"))
    assert names == [f"{nocrop}-cls-g", f"{nocrop}-gem-g"]
    assert [w.info.pooling for w in written] == ["cls", "gem"]
    cls, gem = (read_vectors(built.vectors_path(n)) for n in names)
    assert cls.dim == gem.dim == 4  # the fake's 8 columns, split in two
    assert "whole" in cls.info.notes
    assert "one pass" in gem.info.notes
    assert set(cls.ids) == set(ids)


def test_a_whole_image_option_never_borrows_a_cropped_probe(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    r, _ = runner(built)
    r.probe("A2-dinov3-l-256-cls", "g")
    assert option("A2-dinov3-l-256-nocrop").resize is Resize.WHOLE
    with pytest.raises(BudgetError):
        r.embed_gallery("A2-dinov3-l-256-nocrop", "g")


def test_matching_saves_each_answer_resumes_and_stops_at_its_limit(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    count = MATCH_QUERIES + 1
    tasks = [MatchTask(f"q{n}", ids[0], tuple(ids[1:3])) for n in range(count)]
    r, calls = runner(built)
    with pytest.raises(BudgetError, match="limit"):
        r.match("m", tasks, max_usd=SECONDS_PER_CALL * L4.usd_per_s / 2)
    assert len(read_matches(built.matches_path("m"))) == MATCH_QUERIES  # first call kept
    assert r.match("m", tasks, max_usd=1.0) == count
    assert calls.calls == [("match", MATCH_QUERIES), ("match", 1)]
    assert r.ledger.total(SpendKind.RUN, run="m") == pytest.approx(
        2 * SECONDS_PER_CALL * L4.usd_per_s
    )


def test_private_images_are_never_matched_on_modal(built: Settings) -> None:
    private = image_ids(built, publishable=False)[0]
    r, calls = runner(built)
    with pytest.raises(PrivacyError):
        r.match("m", [MatchTask("q", private, ())], max_usd=1.0)
    assert calls.calls == []


def _commons_sets(settings: Settings) -> tuple[str, str, str]:
    """T2 with a Commons photo and T5 with a private and a public image; returns their ids."""
    photo = commons_photo("File:a.jpg", CommonsSet.T2, "E1")
    write_commons([photo], settings.commons_parquet)
    public = image_ids(settings, publishable=True)[0]
    private = image_ids(settings, publishable=False)[0]
    write_set(settings, "T2", [Query("T2-a", "T2", "E1", photo.image_id, ("E1",), ())], [])
    t5 = [Query(f"T5-{i}", "T5", "E", i, ("E",), (i,)) for i in (public, private)]
    write_set(settings, "T5", t5, [])
    write_gallery(settings, "g", image_ids(settings, publishable=True))
    return photo.image_id, public, private


def test_query_run_sends_commons_photos_as_bytes(built: Settings) -> None:
    photo, public, _private = _commons_sets(built)
    commons = FakeCommons()
    r, calls = runner(built, commons=commons)
    r.probe(GEMMA, "g")

    done = r.embed_queries(GEMMA, ["T2", "T5"])

    assert set(done.ids) == {photo, public}  # the private image never went
    assert done.info.gallery == "T2-T5-tune-queries"
    assert calls.supplied == [photo]
    assert len(commons.calls) == 1
    assert r.ledger.total(SpendKind.RUN, run=run_name(OPTIONS[GEMMA], "T2-T5-tune-queries")) > 0


def test_query_run_needs_a_commons_fetcher_for_commons_photos(built: Settings) -> None:
    _commons_sets(built)
    r, _calls = runner(built)
    r.probe(GEMMA, "g")
    with pytest.raises(ValueError, match="Commons fetcher"):
        r.embed_queries(GEMMA, ["T2"])


def test_parallel_chunks_reuse_earlier_vectors_and_may_send_private_images(
    built: Settings,
) -> None:
    ids = image_ids(built, publishable=True) + image_ids(built, publishable=False)
    write_gallery(built, "g", ids)
    write_gallery(built, "old", ids[:3])
    clock = Clock()
    calls = FakeCalls(clock)
    r = ModalRunner(built, calls, clock=clock, all_images=True)
    r.probe(GEMMA, "g")
    (old,) = r.embed_gallery(GEMMA, "old")

    (done,) = r.embed_gallery(GEMMA, "g", workers=2, reuse=[run_name(OPTIONS[GEMMA], "old")])
    assert (f"scale {GEMMA}", 2) in calls.calls
    assert calls.calls[-1] == (GEMMA, len(ids) - 3)  # the three old ones are not sent again
    assert set(done.ids) == set(ids)
    row = {i: n for n, i in enumerate(done.ids)}
    for n, i in enumerate(old.ids):
        assert np.array_equal(done.vectors[row[i]], old.vectors[n])
    assert "reused" in done.info.notes


def test_vectors_of_another_model_are_never_reused(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    r, _ = runner(built)
    r.probe(GEMMA, "g")
    (old,) = r.embed_gallery(GEMMA, "g")
    megaloc = "A3-megaloc-322"
    r.probe(megaloc, "g")
    with pytest.raises(ValueError, match="not reused"):
        r.embed_gallery(megaloc, "g", reuse=[run_name(OPTIONS[GEMMA], "g")])
    assert old.ids
