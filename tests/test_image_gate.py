"""The privacy gate inside the GPU machine's calls: every caller goes through it."""

from __future__ import annotations

import pickle

import httpx
import pytest

from miroir.config import Settings
from miroir.eval.wire import CaptionPhoto, RoomPhoto, SiteMatchTask
from miroir.jobs.gpu_run import PaidCall, paid_call
from miroir.jobs.privacy import ImageGate, PrivacyError, PublishableImages
from miroir.jobs.remote_spec import CHUNK_TIMEOUT_S
from miroir.jobs.scaleway import OFFERS
from miroir.jobs.scaleway import Machine as GpuMachine
from miroir.jobs.scw_run import ScalewayCalls
from miroir.jobs.spend import Budget, SpendKind, SpendLedger
from miroir.records.build import build
from miroir.records.store import read_records


@pytest.fixture
def built(settings: Settings) -> Settings:
    build(settings)
    return settings


def image_ids(settings: Settings, *, publishable: bool) -> list[str]:
    records = read_records(settings.records_parquet, settings.images_parquet)
    return [i.image_id for r in records if r.publishable == publishable for i in r.images]


def calls(settings: Settings, sent: list[str], *, all_images: bool = False) -> ScalewayCalls:
    def worker(request: httpx.Request) -> httpx.Response:
        name, _args = pickle.loads(request.content)
        sent.append(name)
        return httpx.Response(200, content=pickle.dumps([]))

    gate = ImageGate(lambda: PublishableImages.load(settings), all_images=all_images)
    machine = GpuMachine("s", "miroir-test", OFFERS[0], "127.0.0.1", 0.0, 3600.0)
    return ScalewayCalls("http://worker.test", machine, gate, transport=httpx.MockTransport(worker))


def test_an_eval_step_cannot_send_a_private_image_through_a_paid_call(built: Settings) -> None:
    """Step C, E1 and the archive call the machine through `paid_call` themselves."""
    sent: list[str] = []
    remote = calls(built, sent)
    private = image_ids(built, publishable=False)[0]
    paid = PaidCall(SpendLedger(built.spend_ledger), Budget.from_settings(built))
    with pytest.raises(PrivacyError, match="not publishable"):
        paid_call(
            paid,
            lambda: remote.detect_rooms(None, [RoomPhoto(private, b"jpeg")], {}),
            machine=remote.machine,
            timeout_s=CHUNK_TIMEOUT_S,
            kind=SpendKind.RUN,
            option="C",
            run="C-test",
            model="m",
            items=1,
        )
    with pytest.raises(PrivacyError):
        remote.read_captions("qwen", [CaptionPhoto(private, b"jpeg")])
    with pytest.raises(PrivacyError):
        remote.read_captions("qwen", [CaptionPhoto(f"{private}#cut", b"jpeg")])
    assert sent == []


def test_a_query_photo_sent_as_bytes_must_say_which_image_it_is(built: Settings) -> None:
    sent: list[str] = []
    remote = calls(built, sent)
    public = image_ids(built, publishable=True)[0]
    with pytest.raises(PrivacyError, match="without an image id"):
        remote.match_site([SiteMatchTask("q", (public,), jpeg=b"jpeg")])
    remote.match_site([SiteMatchTask("q", (public,), image_id=public)])
    assert sent == ["match_site"]


def test_the_full_sample_index_may_send_every_known_image(built: Settings) -> None:
    sent: list[str] = []
    private = image_ids(built, publishable=False)[0]
    calls(built, sent, all_images=True).read_captions("qwen", [CaptionPhoto(private, b"j")])
    assert sent == ["read_captions"]
    with pytest.raises(KeyError, match="neither"):
        calls(built, sent, all_images=True).read_captions("q", [CaptionPhoto("nope", b"j")])


def test_the_website_calls_are_the_only_raw_calls(built: Settings) -> None:
    sent: list[str] = []
    remote = calls(built, sent)
    with pytest.raises(ValueError, match="not a website call"):
        remote.site_call("embed")
    remote.site_call("search", "index")
    assert sent == ["search"]
