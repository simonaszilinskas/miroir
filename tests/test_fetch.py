from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage

from miroir.http import send
from miroir.images.fetch import (
    LONG_SIDE,
    RECIPE,
    FetchSkip,
    ImageCache,
    ImageFetcher,
    cache_key,
    iiif_url,
    to_jpeg,
)
from miroir.parallel import bounded_map
from miroir.records.model import Collection, Image, ImageType
from tests.fakes import NO_WAIT, FakePop, jpeg

ACCENTED = (
    "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/mnr/MNR00949/"
    "MNR%20949%20%C3%A9tiquette%20d%C3%A9chir%C3%A9e.jpg"
)


def image(url: str = ACCENTED, image_id: str = "aaaa000000000001") -> Image:
    return Image(
        image_id=image_id,
        ref="MNR00949",
        entity_id="MNR00949",
        collection=Collection.ROSE_VALLAND,
        url=url,
        image_type=ImageType.LABEL,
        position=9,
        is_cover=False,
    )


def fetcher(pop: FakePop, tmp_path: Path, max_bytes: int = 10_000_000) -> ImageFetcher:
    cache = ImageCache(tmp_path / "cache", max_bytes)
    return ImageFetcher(pop.client(), cache, retry=NO_WAIT, concurrency=2)


def test_iiif_url_encodes_the_whole_bucket_path() -> None:
    # Service id as POP's IIIF manifest for MNR00949 lists it.
    assert iiif_url(ACCENTED) == (
        "https://iiif.prd.cloud.culture.fr/iiif/3/"
        "mnr%2FMNR00949%2FMNR%20949%20%C3%A9tiquette%20d%C3%A9chir%C3%A9e.jpg"
        "/full/!512,512/0/default.jpg"
    )


def test_to_jpeg_shrinks_to_the_long_side_and_never_enlarges() -> None:
    with PILImage.open(io.BytesIO(to_jpeg(jpeg(2000, 1000)))) as big:
        assert big.size == (512, 256)
        assert big.format == "JPEG"
    with PILImage.open(io.BytesIO(to_jpeg(jpeg(300, 200)))) as small:
        assert small.size == (300, 200)


def test_fetch_resizes_and_then_serves_from_cache(tmp_path: Path) -> None:
    pop = FakePop()
    f = fetcher(pop, tmp_path)
    first = f.fetch(image())
    assert first.skip is None
    assert first.jpeg is not None
    second = f.fetch(image())
    assert second.jpeg == first.jpeg
    assert len(pop.calls) == 1


def test_bucket_404_falls_back_to_iiif(tmp_path: Path) -> None:
    pop = FakePop(missing_on_bucket={"MNR00949"})
    result = fetcher(pop, tmp_path).fetch(image())
    assert result.jpeg is not None
    assert pop.calls[1].startswith("https://iiif.prd.cloud.culture.fr/")


def test_missing_everywhere_is_skipped_not_raised(tmp_path: Path) -> None:
    pop = FakePop(missing_on_bucket={"MNR00949"}, missing_on_iiif={"MNR00949"})
    result = fetcher(pop, tmp_path).fetch(image())
    assert result.jpeg is None
    assert result.skip == FetchSkip.MISSING


def test_an_image_over_the_pixel_limit_is_skipped_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FakePop serves 1024 x 512 images; past twice the limit Pillow raises, not warns.
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 100_000)
    result = fetcher(FakePop(), tmp_path).fetch(image())
    assert result.jpeg is None
    assert result.skip == FetchSkip.UNREADABLE


def test_fetch_all_keeps_input_order(tmp_path: Path) -> None:
    images = [image(f"{ACCENTED}?{n}", f"aaaa00000000000{n}") for n in range(5)]
    got = [r.image_id for r in fetcher(FakePop(), tmp_path).fetch_all(images)]
    assert got == [i.image_id for i in images]


def test_cache_stays_under_its_cap(tmp_path: Path) -> None:
    cache = ImageCache(tmp_path / "cache", max_bytes=2_500)
    for n in range(10):
        cache.put(f"bb{n:014d}", b"x" * 1_000)
    assert cache.size_bytes <= 2_500
    files = list((tmp_path / "cache").rglob("*.jpg"))
    assert sum(p.stat().st_size for p in files) == cache.size_bytes
    assert cache.get("bb00000000000009") is not None


def test_send_retries_busy_answers_then_succeeds() -> None:
    answers = iter([httpx.Response(429), httpx.Response(503), httpx.Response(200)])
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: next(answers)))
    response = send(client, client.build_request("GET", "https://x.test/"), NO_WAIT)
    assert response.status_code == 200


def test_send_returns_the_last_busy_answer_when_out_of_tries() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(429)))
    response = send(client, client.build_request("GET", "https://x.test/"), NO_WAIT)
    assert response.status_code == 429


def test_bounded_map_keeps_order_and_reads_input_lazily() -> None:
    read: list[int] = []

    def numbers() -> Iterator[int]:
        for n in range(100):
            read.append(n)
            yield n

    out = bounded_map(lambda n: n * 2, numbers(), workers=2, window=4)
    assert [next(out) for _ in range(3)] == [0, 2, 4]
    assert len(read) <= 8
    assert list(out) == [n * 2 for n in range(3, 100)]


def test_an_uncapped_cache_never_evicts(tmp_path: Path) -> None:
    cache = ImageCache(tmp_path / "cache", None)
    for n in range(5):
        cache.put(f"{n:016x}", b"x" * 1000)
    assert all(cache.get(f"{n:016x}") is not None for n in range(5))


def _counting(*outcomes: httpx.Response | Exception) -> tuple[httpx.Client, list[str]]:
    sent: list[str] = []
    pending = iter(outcomes)

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.method)
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.Client(transport=httpx.MockTransport(answer)), sent


def test_a_paid_post_is_never_sent_twice_once_it_may_have_run() -> None:
    client, sent = _counting(httpx.ReadTimeout("slow"), httpx.Response(200))
    with pytest.raises(httpx.ReadTimeout):
        send(client, client.build_request("POST", "https://x.test/batches"), NO_WAIT)
    assert sent == ["POST"]
    client, sent = _counting(httpx.Response(502), httpx.Response(200))
    response = send(client, client.build_request("POST", "https://x.test/batches"), NO_WAIT)
    assert (response.status_code, sent) == (502, ["POST"])


def test_a_post_is_sent_again_when_it_provably_did_not_run() -> None:
    client, sent = _counting(
        httpx.ConnectError("refused"), httpx.Response(429), httpx.Response(200)
    )
    response = send(client, client.build_request("POST", "https://x.test/"), NO_WAIT)
    assert (response.status_code, sent) == (200, ["POST"] * 3)


def test_a_read_is_retried_on_a_dropped_answer() -> None:
    client, sent = _counting(httpx.ReadTimeout("slow"), httpx.Response(502), httpx.Response(200))
    response = send(client, client.build_request("GET", "https://x.test/"), NO_WAIT)
    assert (response.status_code, sent) == (200, ["GET"] * 3)


def test_a_stopped_map_cancels_what_has_not_started_and_saves_what_ran() -> None:
    started: list[int] = []
    saved: list[int] = []
    lock = threading.Lock()
    gate = threading.Event()

    def paid(n: int) -> int:
        with lock:
            started.append(n)
        if n == 0:
            raise RuntimeError("call 0 failed")
        gate.wait(5)
        return n

    out = bounded_map(paid, range(100), workers=3, window=6, salvage=saved.append)
    threading.Timer(0.2, gate.set).start()  # calls 1 and 2 finish after call 0 failed
    with pytest.raises(RuntimeError, match="call 0"):
        next(out)
    # Every call that ran beside call 0 was paid: each one is kept. Calls still queued when
    # the map stopped were never sent (before, the pool ran all six it held).
    assert sorted(saved) == sorted(n for n in started if n)
    assert len(started) <= 6


def test_a_consumer_that_stops_still_gets_the_calls_in_flight() -> None:
    saved: list[int] = []
    out = bounded_map(lambda n: n, range(100), workers=2, window=4, salvage=saved.append)
    assert next(out) == 0
    out.close()  # as when the consumer raises BudgetError
    assert all(n in {1, 2, 3} for n in saved)


def _open_jpeg(data: bytes) -> PILImage.Image:
    image = PILImage.open(io.BytesIO(data))
    image.load()
    return image


def _encoded(image: PILImage.Image, fmt: str, **options: object) -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt, **options)
    return out.getvalue()


def test_to_jpeg_turns_an_image_upright_as_its_exif_says() -> None:
    wide = PILImage.new("RGB", (40, 20), (200, 30, 30))
    exif = PILImage.Exif()
    exif[0x0112] = 6  # orientation: the camera was turned; show it rotated 90 degrees
    out = _open_jpeg(to_jpeg(_encoded(wide, "JPEG", exif=exif.tobytes()), 512))
    assert out.size == (20, 40)


def test_to_jpeg_puts_transparency_on_white_not_black() -> None:
    clear = PILImage.new("RGBA", (16, 16), (0, 0, 0, 0))
    for mode in ("RGBA", "LA"):
        out = _open_jpeg(to_jpeg(_encoded(clear.convert(mode), "PNG")))
        assert np.asarray(out).min() > 245


def test_to_jpeg_scales_16_bit_grey_instead_of_clipping_it() -> None:
    grey = PILImage.fromarray(np.full((8, 8), 32768, dtype=np.uint16))
    assert grey.mode.startswith("I")
    value = np.asarray(_open_jpeg(to_jpeg(_encoded(grey, "PNG"))))[4, 4, 0]
    assert 120 < value < 136  # half of white, not 255


def test_to_jpeg_shrinks_palette_images_smoothly() -> None:
    stripes = np.zeros((64, 64), dtype=np.uint8)
    stripes[:, ::2] = 255  # one-pixel stripes: NEAREST keeps them hard, LANCZOS greys them
    palette = PILImage.fromarray(stripes).convert("P")
    out = np.asarray(_open_jpeg(to_jpeg(_encoded(palette, "PNG"), 16)).convert("L"))
    assert 60 < out.mean() < 200
    assert out.std() < 40


def test_the_cache_path_names_the_size_and_recipe(tmp_path: Path) -> None:
    cache = ImageCache(tmp_path / "cache", None)
    cache.put("ab00000000000001", b"jpeg")
    assert (tmp_path / "cache" / cache_key("ab00000000000001")).exists()
    assert cache_key("ab00000000000001").startswith(f"{LONG_SIDE}-{RECIPE}/ab/")
    old = tmp_path / "cache" / "ab" / "ab00000000000002.jpg"  # made by the old recipe
    old.parent.mkdir()
    old.write_bytes(b"old")
    assert cache.get("ab00000000000002") is None
