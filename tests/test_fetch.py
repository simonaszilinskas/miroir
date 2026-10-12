from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import httpx
from PIL import Image as PILImage

from miroir.http import send
from miroir.images.fetch import FetchSkip, ImageCache, ImageFetcher, iiif_url, to_jpeg
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
