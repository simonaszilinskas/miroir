from __future__ import annotations

import pytest
from pydantic import SecretStr

from miroir.config import POP_ASSET_HOST, Settings
from miroir.images.fetch import ImageCache, ImageFetcher
from miroir.jobs.describe import (
    OPTIONS,
    DescribeOption,
    Describer,
    parse_description,
    probe,
    run,
)
from miroir.jobs.openrouter import OpenRouterClient, OpenRouterError
from miroir.jobs.privacy import PrivacyError
from miroir.jobs.spend import BudgetError, SpendKind
from miroir.records.build import build
from miroir.records.store import read_records
from tests.fakes import COST_PER_INPUT, NO_WAIT, FakeOpenRouter, FakePop


@pytest.fixture
def built(settings: Settings) -> Settings:
    build(settings)
    return settings


def image_ids(settings: Settings, *, publishable: bool) -> list[str]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return [i.image_id for r in records if r.publishable == publishable for i in r.images]


def describer(
    settings: Settings, fake: FakeOpenRouter, model: str = "qwen3-vl-30b-a3b"
) -> Describer:
    client = OpenRouterClient(SecretStr("sk-test"), transport=fake.transport(), retry=NO_WAIT)
    cache = ImageCache(settings.image_cache_dir, 50_000_000)
    fetcher = ImageFetcher(FakePop().client(), cache, retry=NO_WAIT, concurrency=2)
    return Describer(
        settings, OPTIONS[model], client, fetcher, request_concurrency=2, batch_poll_s=0.0
    )


def test_parse_reads_fenced_json_lists_and_empty_markers() -> None:
    text = (
        'Voici: ```json\n{"type": "Vase", "nom": "inconnu", "Commune": "Sèvres", '
        '"texte_visible": ["A", "", "B"], "confiance": "80%"}\n```'
    )
    d = parse_description(text)
    assert d is not None
    assert (d.kind, d.name, d.commune, d.visible_text) == ("Vase", "", "Sèvres", "A; B")
    assert d.confidence == pytest.approx(0.8)
    assert parse_description("je ne sais pas") is None
    assert parse_description('{"type": ') is None


def test_private_images_never_leave(built: Settings) -> None:
    fake = FakeOpenRouter()
    mixed = [*image_ids(built, publishable=True)[:1], image_ids(built, publishable=False)[0]]
    with pytest.raises(PrivacyError, match="not publishable"):
        describer(built, fake).ask(mixed, kind=SpendKind.RUN, limit_usd=1.0)
    assert fake.chat_bodies == []


def test_run_refuses_without_a_probe(built: Settings) -> None:
    fake = FakeOpenRouter()
    with pytest.raises(BudgetError, match="--probe"):
        run(describer(built, fake), image_ids(built, publishable=True), max_usd=1.0)
    assert fake.chat_bodies == []


def test_probe_then_run_saves_answers_once_and_logs_spend(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    fake = FakeOpenRouter()
    d = describer(built, fake)

    result = probe(d, ids[:2], "T1-tune")
    assert result.images == 2
    assert result.cost_usd == pytest.approx(2 * COST_PER_INPUT)
    body = fake.chat_bodies[0]
    assert body["model"] == "qwen/qwen3-vl-30b-a3b-instruct"
    assert body["response_format"] == {"type": "json_object"}

    run(d, ids, max_usd=1.0)
    assert len(fake.chat_bodies) == len(ids)  # the probe's two are not asked again
    answers = d.answers()
    assert set(answers) == set(ids)
    parsed = parse_description(answers[ids[0]].text)
    assert parsed is not None
    assert parsed.commune == "Beaune"
    assert d.ledger.total() == pytest.approx(len(ids) * COST_PER_INPUT)


def test_run_refuses_past_its_cap(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    d = describer(built, FakeOpenRouter())
    probe(d, ids[:1], "T1-tune")
    with pytest.raises(BudgetError, match="over"):
        run(d, ids, max_usd=COST_PER_INPUT / 2)


def test_failures_in_a_row_stop_the_run(built: Settings) -> None:
    ids = image_ids(built, publishable=True)
    with pytest.raises(OpenRouterError, match="in a row"):
        describer(built, FakeOpenRouter(refuse_after=0)).ask(ids, kind=SpendKind.RUN, limit_usd=1.0)


def test_batch_models_go_through_the_batch_api(built: Settings) -> None:
    ids = image_ids(built, publishable=True)[:3]
    fake = FakeOpenRouter(batch_models=("google/gemini-3.5-flash-lite:batch",))
    d = describer(built, fake, "gemini-3.5-flash-lite")

    done = d.ask(ids, kind=SpendKind.RUN, limit_usd=1.0)

    assert fake.chat_bodies == []
    sent = fake.batch_bodies[0]
    assert sent["endpoint"] == "/v1/chat/completions"
    assert sent["model"] == "google/gemini-3.5-flash-lite:batch"
    requests = sent["requests"]
    assert isinstance(requests, list)
    url = str(requests[0])
    # The Batch API refuses data URLs: the image goes as its public bucket URL.
    assert POP_ASSET_HOST in url
    assert "data:image" not in url
    assert done.answered == 3
    assert done.cost_usd == pytest.approx(COST_PER_INPUT / 2 * 3)
    assert set(d.answers()) == set(ids)


def test_options_have_safe_slugs() -> None:
    for option in OPTIONS.values():
        assert isinstance(option, DescribeOption)
        assert option.option == f"H1-{option.slug}"


def test_a_batch_left_by_a_stopped_run_is_collected_not_sent_again(built: Settings) -> None:
    ids = image_ids(built, publishable=True)[:3]
    fake = FakeOpenRouter(batch_models=("google/gemini-3.5-flash-lite:batch",), unseen_polls=2)
    d = describer(built, fake, "gemini-3.5-flash-lite")
    left = ids[:2]
    pending = d.answers_path.with_suffix(".pending")
    pending.parent.mkdir(parents=True)
    pending.write_text(f"batch-1\t{','.join(left)}\n", encoding="utf-8")
    fake.batch_bodies.append(
        {"endpoint": "/v1/chat/completions", "requests": [{"custom_id": i} for i in left]}
    )

    d.ask(ids, kind=SpendKind.RUN, limit_usd=1.0)

    assert set(d.answers()) == set(ids)
    sent = fake.batch_bodies[1]["requests"]
    assert isinstance(sent, list)
    assert [r["custom_id"] for r in sent] == [ids[2]]
    assert pending.read_text(encoding="utf-8") == ""
