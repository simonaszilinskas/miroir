"""The privacy gate inside the GPU machine's calls: every caller goes through it."""

from __future__ import annotations

import pickle
from dataclasses import replace

import httpx
import pytest

from miroir.config import Settings
from miroir.jobs.gpu_run import PaidCall, paid_call
from miroir.jobs.privacy import ImageGate, PrivacyError, PublishableImages
from miroir.jobs.remote_spec import CHUNK_TIMEOUT_S, OPTIONS
from miroir.jobs.scaleway import OFFERS
from miroir.jobs.scaleway import Machine as GpuMachine
from miroir.jobs.scw_run import ScalewayCalls
from miroir.jobs.spend import Budget, SpendKind, SpendLedger
from miroir.records.build import build
from miroir.records.model import Image
from miroir.records.store import read_records

GEMMA = "A7-embeddinggemma-2-512"


@pytest.fixture
def built(settings: Settings) -> Settings:
    build(settings)
    return settings


def images(settings: Settings, *, publishable: bool) -> list[Image]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return [i for r in records if r.publishable == publishable for i in r.images]


def calls(settings: Settings, sent: list[str], *, all_images: bool = False) -> ScalewayCalls:
    def worker(request: httpx.Request) -> httpx.Response:
        name, _args = pickle.loads(request.content)
        sent.append(name)
        return httpx.Response(200, content=pickle.dumps([]))

    gate = ImageGate(lambda: PublishableImages.load(settings), all_images=all_images)
    machine = GpuMachine("s", "miroir-test", OFFERS[0], "127.0.0.1", 0.0, 3600.0)
    return ScalewayCalls("http://worker.test", machine, gate, transport=httpx.MockTransport(worker))


def test_a_paid_call_cannot_send_a_private_image(built: Settings) -> None:
    sent: list[str] = []
    remote = calls(built, sent)
    private = images(built, publishable=False)[0]
    paid = PaidCall(SpendLedger(built.spend_ledger), Budget.from_settings(built))
    with pytest.raises(PrivacyError, match="not publishable"):
        paid_call(
            paid,
            lambda: remote.embed(OPTIONS[GEMMA], [private]),
            machine=remote.machine,
            timeout_s=CHUNK_TIMEOUT_S,
            kind=SpendKind.RUN,
            option="A7",
            run="A7-test",
            model="m",
            items=1,
        )
    with pytest.raises(PrivacyError):
        remote.fill_images([[private]])
    assert sent == []


def test_the_full_sample_index_may_send_every_known_image(built: Settings) -> None:
    sent: list[str] = []
    private = images(built, publishable=False)[0]
    calls(built, sent, all_images=True).fill_images([[private]])
    assert sent == ["fill_images"]
    unknown = replace(private, image_id="nope")
    with pytest.raises(KeyError, match="neither"):
        calls(built, sent, all_images=True).fill_images([[unknown]])


def test_the_website_calls_are_the_only_raw_calls(built: Settings) -> None:
    sent: list[str] = []
    remote = calls(built, sent)
    with pytest.raises(ValueError, match="not a website call"):
        remote.site_call("embed")
    remote.site_call("search", "index")
    assert sent == ["search"]
