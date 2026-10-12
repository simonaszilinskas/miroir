from __future__ import annotations

import numpy as np
import pytest
from pydantic import SecretStr

from miroir.commons.images import CommonsImages
from miroir.commons.model import CommonsSet
from miroir.config import Settings
from miroir.eval.sets import Query, write_gallery, write_set
from miroir.eval.vectors import Route, read_vectors
from miroir.images.fetch import ImageCache, ImageFetcher
from miroir.jobs.embed import (
    OPTIONS,
    OpenRouterEmbedder,
    embed_gallery,
    embed_queries,
    embed_texts,
    probe,
    run_name,
)
from miroir.jobs.openrouter import OpenRouterClient, OpenRouterError
from miroir.jobs.privacy import PrivacyError
from miroir.jobs.spend import BudgetError, SpendKind
from miroir.records.build import build
from miroir.records.store import read_records, write_commons
from tests.fakes import (
    COST_PER_INPUT,
    NO_WAIT,
    FakeCommons,
    FakeOpenRouter,
    FakePop,
    commons_photo,
)


@pytest.fixture
def built(settings: Settings) -> Settings:
    build(settings)
    return settings


def image_ids(settings: Settings, *, publishable: bool) -> list[str]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return [i.image_id for r in records if r.publishable == publishable for i in r.images]


def embedder(
    settings: Settings,
    fake: FakeOpenRouter,
    option: str = "A9",
    pop: FakePop | None = None,
    commons: CommonsImages | None = None,
) -> OpenRouterEmbedder:
    client = OpenRouterClient(SecretStr("sk-test"), transport=fake.transport(), retry=NO_WAIT)
    cache = ImageCache(settings.image_cache_dir, 50_000_000)
    fetcher = ImageFetcher((pop or FakePop()).client(), cache, retry=NO_WAIT, concurrency=2)
    return OpenRouterEmbedder(
        settings,
        OPTIONS[option],
        client,
        fetcher,
        commons=commons,
        request_concurrency=2,
        batch_poll_s=0.0,
    )


def test_non_publishable_images_never_leave(built: Settings) -> None:
    fake = FakeOpenRouter()
    e = embedder(built, fake)
    private = image_ids(built, publishable=False)
    mixed = [*image_ids(built, publishable=True)[:2], private[0]]
    with pytest.raises(PrivacyError, match="not publishable"):
        e.embed_images(mixed, kind=SpendKind.RUN, run="r", limit_usd=1.0)
    assert fake.embedding_bodies == []


def test_unknown_image_ids_are_refused(built: Settings) -> None:
    with pytest.raises(KeyError):
        embedder(built, FakeOpenRouter()).check_publishable(["ffffffffffffffff"])


def test_full_run_refuses_without_a_probe(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    fake = FakeOpenRouter()
    with pytest.raises(BudgetError, match="--probe"):
        embed_gallery(embedder(built, fake), "g")
    assert fake.embedding_bodies == []


def test_probe_then_run_writes_vectors_and_logs_spend(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    fake = FakeOpenRouter()
    e = embedder(built, fake)

    result, size = probe(e, "g")
    assert size == len(ids)
    assert result.images == len(ids)  # fewer than 100 in the fixtures
    assert result.cost_usd == pytest.approx(COST_PER_INPUT * len(ids))
    assert built.probe_path("A9").exists()
    # Gemini takes at most 6 images per request.
    assert max(fake.request_sizes) <= 6

    done = embed_gallery(e, "g")
    loaded = read_vectors(built.vectors_path(run_name(OPTIONS["A9"], "g")))
    assert loaded.ids == tuple(sorted(ids))  # write_gallery sorts
    assert loaded.info.route == Route.OPENROUTER
    assert loaded.info.resolution == 512
    assert done.info.cost_usd == pytest.approx(COST_PER_INPUT * len(ids))
    assert e.ledger.total(SpendKind.PROBE) == pytest.approx(result.cost_usd)
    assert e.ledger.total() == pytest.approx(2 * COST_PER_INPUT * len(ids))


def test_run_refused_when_key_spend_is_near_the_cap(built: Settings) -> None:
    write_gallery(built, "g", image_ids(built, publishable=True))
    probe(embedder(built, FakeOpenRouter()), "g")
    near_cap = FakeOpenRouter(key_usage=built.openrouter_budget_usd)
    with pytest.raises(BudgetError, match="usable budget"):
        embed_gallery(embedder(built, near_cap), "g")
    assert near_cap.embedding_bodies == []


def test_missing_images_are_counted_and_left_out(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    gone = next(i for i in read_records(built.records_parquet, built.images_parquet)[0].images)
    pop = FakePop(missing_on_bucket={gone.url}, missing_on_iiif={"%2F"})
    e = embedder(built, FakeOpenRouter(), pop=pop)
    probe(e, "g")
    done = embed_gallery(e, "g")
    assert gone.image_id not in done.ids
    assert len(done.ids) == len(ids) - 1
    assert "fetch_missing" in done.info.notes


def test_text_queries_go_through_the_batch_api(built: Settings) -> None:
    fake = FakeOpenRouter()  # gemini-embedding-2:batch works though it is not listed
    texts = {"q1": "une église romane", "q2": "un vase grec"}

    run = embed_texts(embedder(built, fake), "T7-tune-text", texts)

    assert run.ids == ("q1", "q2")
    assert fake.embedding_bodies == []
    assert fake.batch_bodies[0]["model"] == "google/gemini-embedding-2:batch"
    np.testing.assert_allclose(np.linalg.norm(run.vectors.astype(np.float32), axis=1), 1, atol=1e-3)


def query(set_id: str, n: int, image_id: str) -> Query:
    return Query(f"{set_id}-{n}", set_id, f"E{n}", image_id, (f"E{n}",), (image_id,))


def test_queries_keep_only_publishable_images(built: Settings) -> None:
    public = image_ids(built, publishable=True)[:2]
    private = image_ids(built, publishable=False)[:1]
    tune = [query("T5", n, i) for n, i in enumerate([*public, *private])]
    write_set(built, "T5", tune, [])
    write_gallery(built, "g", image_ids(built, publishable=True))
    fake = FakeOpenRouter()
    e = embedder(built, fake)
    probe(e, "g")

    run = embed_queries(e, ["T5"])

    assert run.ids == tuple(public)
    assert run.info.gallery == "T5-tune-queries"


def test_commons_query_photos_go_out_in_one_run_with_the_sets_queries(built: Settings) -> None:
    public = image_ids(built, publishable=True)[0]
    found = commons_photo("File:a.jpg", CommonsSet.T2, "E1")
    missing = commons_photo("File:b.jpg", CommonsSet.T8C, "E2")
    write_commons([found, missing], built.commons_parquet)
    write_set(built, "T5", [query("T5", 0, public)], [])
    write_set(built, "T2", [Query("T2-a", "T2", "E1", found.image_id, ("E1",), ())], [])
    write_set(built, "T8c", [Query("T8c-b", "T8c", "E2", missing.image_id, (), ())], [])
    write_gallery(built, "g", image_ids(built, publishable=True))
    fake, commons = FakeOpenRouter(), FakeCommons()
    e = embedder(built, fake, commons=commons.images(built))
    probe(e, "g")

    run = embed_queries(e, ["T5", "T2", "T8c"])

    assert set(run.ids) == {public, found.image_id, missing.image_id}
    assert run.info.gallery == "T5-T2-T8c-tune-queries"
    assert commons.calls == [found.thumb_url, missing.thumb_url]
    # A second run reads the photos from the cache: Wikimedia is asked once.
    again = FakeCommons()
    embed_queries(embedder(built, fake, commons=again.images(built)), ["T2"])
    assert again.calls == []


def test_an_unknown_commons_id_is_refused(built: Settings) -> None:
    e = embedder(built, FakeOpenRouter(), commons=FakeCommons().images(built))
    with pytest.raises(KeyError, match=r"commons\.parquet"):
        e.embed_images(["commons-0000000000000000"], kind=SpendKind.RUN, run="r", limit_usd=1.0)


def test_a_stopped_run_resumes_without_paying_twice(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    write_gallery(built, "g", ids)
    probe(embedder(built, FakeOpenRouter()), "g")

    dying = FakeOpenRouter(refuse_after=2)
    with pytest.raises(OpenRouterError, match="out of credit"):
        embed_gallery(
            embedder(
                built,
                dying,
            ),
            "g",
        )
    first = sum(dying.request_sizes)
    assert 0 < first < len(ids)

    fake = FakeOpenRouter()
    done = embed_gallery(embedder(built, fake), "g")

    assert sum(fake.request_sizes) == len(ids) - first
    assert sorted(done.ids) == sorted(ids)
    assert done.info.cost_usd == pytest.approx(COST_PER_INPUT * len(ids))
    assert not built.checkpoint_path(run_name(OPTIONS["A9"], "g")).exists()
