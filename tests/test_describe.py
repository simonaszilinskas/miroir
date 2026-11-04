from __future__ import annotations

import pytest
from pydantic import SecretStr

from miroir.commons.model import CommonsSet
from miroir.config import POP_ASSET_HOST, Settings
from miroir.images.fetch import ImageCache, ImageFetcher
from miroir.jobs.describe import (
    FULL_SIZE,
    OPTIONS,
    DescribeOption,
    Describer,
    parse_description,
    probe,
    run,
    standard,
)
from miroir.jobs.openrouter import OpenRouterClient, OpenRouterError
from miroir.jobs.privacy import PrivacyError
from miroir.jobs.spend import BudgetError, RunSpend, SpendKind
from miroir.records.build import build
from miroir.records.store import read_records, write_commons
from tests.fakes import COST_PER_INPUT, NO_WAIT, FakeCommons, FakeOpenRouter, FakePop, commons_photo


@pytest.fixture
def built(settings: Settings) -> Settings:
    build(settings)
    return settings


def image_ids(settings: Settings, *, publishable: bool) -> list[str]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return [i.image_id for r in records if r.publishable == publishable for i in r.images]


def describer(
    settings: Settings,
    fake: FakeOpenRouter,
    model: str | DescribeOption = "qwen3-vl-30b-a3b",
    commons: FakeCommons | None = None,
) -> Describer:
    client = OpenRouterClient(SecretStr("sk-test"), transport=fake.transport(), retry=NO_WAIT)
    cache = ImageCache(settings.image_cache_dir, 50_000_000)
    fetcher = ImageFetcher(FakePop().client(), cache, retry=NO_WAIT, concurrency=2)
    return Describer(
        settings,
        OPTIONS[model] if isinstance(model, str) else model,
        client,
        fetcher,
        commons=None if commons is None else commons.images(settings),
        request_concurrency=2,
        batch_poll_s=0.0,
    )


def spend(d: Describer) -> RunSpend:
    return d.spend(SpendKind.RUN, limit_usd=1.0, per_item_usd=COST_PER_INPUT)


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
        (d := describer(built, fake)).ask(mixed, spend(d))
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
    d = describer(built, FakeOpenRouter(refuse_after=0))
    with pytest.raises(OpenRouterError, match="in a row"):
        d.ask(ids, spend(d))


def test_batch_models_go_through_the_batch_api(built: Settings) -> None:
    ids = image_ids(built, publishable=True)[:3]
    fake = FakeOpenRouter(batch_models=("google/gemini-3.5-flash-lite:batch",))
    d = describer(built, fake, "gemini-3.5-flash-lite")

    done = d.ask(ids, spend(d))

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

    d.ask(ids, spend(d))

    assert set(d.answers()) == set(ids)
    sent = fake.batch_bodies[1]["requests"]
    assert isinstance(sent, list)
    assert [r["custom_id"] for r in sent] == [ids[2]]
    assert pending.read_text(encoding="utf-8") == ""


GEMINI = "gemini-3.5-flash-lite"
BATCH = ("google/gemini-3.5-flash-lite:batch",)


def test_a_batch_stays_pending_until_its_answers_are_saved(
    built: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = image_ids(built, publishable=True)[:3]
    fake = FakeOpenRouter(batch_models=BATCH)
    d = describer(built, fake, GEMINI)

    def crash(_answers: object) -> None:
        raise KeyboardInterrupt  # the Mac stops between the batch's answer and its saving

    monkeypatch.setattr(d, "_save", crash)
    with pytest.raises(KeyboardInterrupt):
        d.ask(ids, spend(d))
    pending = d.answers_path.with_suffix(".pending")
    assert pending.read_text(encoding="utf-8").startswith("batch-1\t")  # kept: collect again
    assert d.ledger.total() == pytest.approx(COST_PER_INPUT / 2 * 3)  # and its cost is logged
    monkeypatch.undo()
    d.ask(ids, spend(d))
    assert set(d.answers()) == set(ids)
    assert len(fake.batch_bodies) == 1  # collected again, never sent twice
    assert pending.read_text(encoding="utf-8") == ""


def test_an_answer_cut_short_by_a_crash_is_asked_again(built: Settings) -> None:
    ids = image_ids(built, publishable=True)[:2]
    fake = FakeOpenRouter()
    d = describer(built, fake)
    d.ask(ids[:1], spend(d))
    with d.answers_path.open("ab") as out:
        out.write(b'{"image_id": "')
    assert set(d.answers()) == {ids[0]}
    d.ask(ids, spend(d))
    assert set(d.answers()) == set(ids)


def test_commons_query_photos_are_asked_on_the_standard_endpoint(built: Settings) -> None:
    photo = commons_photo("File:a.jpg", CommonsSet.T2, "E1")
    write_commons([photo], built.commons_parquet)
    fake = FakeOpenRouter(batch_models=BATCH)
    d = describer(built, fake, commons=FakeCommons())
    assert d.ask([photo.image_id], spend(d)).answered == 1
    batch = describer(built, fake, GEMINI, commons=FakeCommons())
    with pytest.raises(ValueError, match="--no-batch"):
        batch.ask([photo.image_id], spend(batch))


def test_batch_and_standard_runs_of_one_model_never_share_a_probe(built: Settings) -> None:
    ids = image_ids(built, publishable=True)[:2]
    fake = FakeOpenRouter(batch_models=BATCH)
    batch = describer(built, fake, GEMINI)
    measured = probe(batch, ids, "T1-tune")
    assert measured.resolution == FULL_SIZE  # the batch sends the image URL, as POP serves it
    no_batch = describer(built, fake, standard(OPTIONS[GEMINI]))
    assert no_batch.option.option != batch.option.option
    with pytest.raises(BudgetError, match="--probe"):
        run(no_batch, ids, max_usd=1.0)
