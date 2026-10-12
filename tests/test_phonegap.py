"""F1, F2 and G5: what turns a visitor's phone photo into a fair search of the archive."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
from PIL import Image as PILImage

from miroir.commons.api import CommonsFile
from miroir.commons.client import WikimediaClient
from miroir.commons.gallery import Excluded, GalleryFinder, Guard, order_for_gallery
from miroir.commons.model import Source, View
from miroir.commons.select import Candidate, only_commune
from miroir.eval.router import ROUTE_GRID
from miroir.eval.router3 import Bucket, SetGrid, bucket_of, one, pick_weights, three, two
from miroir.eval.scoring import QueryScore, ScoreResult
from miroir.images.colour import GREY_BELOW, already_grey, chroma_spread, greyscale
from miroir.images.phash import phash


def jpeg(pixels: np.ndarray) -> bytes:
    out = io.BytesIO()
    PILImage.fromarray(pixels.astype(np.uint8)).save(out, format="JPEG", quality=95)
    return out.getvalue()


def colour_photo(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 8, 3))
    return jpeg(np.kron(blocks, np.ones((32, 32, 1))))


def test_grey_and_tinted_prints_score_near_zero_colour_photos_above() -> None:
    ramp = np.tile(np.linspace(0, 255, 256), (256, 1))
    grey = jpeg(np.stack([ramp] * 3, axis=-1))
    sepia = jpeg(np.stack([ramp, ramp * 0.85, ramp * 0.65], axis=-1))
    assert chroma_spread(grey) < 0.5
    assert already_grey(grey)
    assert not already_grey(sepia)
    assert chroma_spread(colour_photo()) > GREY_BELOW
    assert already_grey(greyscale(colour_photo()))


def test_only_commune_catches_titles_that_name_the_town_alone() -> None:
    def titled(title: str) -> CommonsFile:
        return file(title=title)

    assert only_commune(titled("File:Torsac 3.jpg"), "Torsac")
    assert only_commune(titled("File:Mazangé (11).JPG"), "Mazangé")
    assert not only_commune(titled("File:Torsac - église Saint-Pierre.jpg"), "Torsac")
    assert not only_commune(titled("File:Torsac 3.jpg"), "")


def file(**changes: object) -> CommonsFile:
    base = CommonsFile(
        title="File:Eglise Saint-Pierre.jpg",
        page_url="https://commons.wikimedia.org/wiki/File:Eglise_Saint-Pierre.jpg",
        url="https://upload.wikimedia.org/x/Eglise_Saint-Pierre.jpg",
        thumb_url="https://thumb.wikimedia.org/x/960px-Eglise_Saint-Pierre.jpg",
        width=4000,
        height=3000,
        mime="image/jpeg",
        licence="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0",
        author="A. Visitor",
        credit="Own work",
        taken="2019:06:01 10:00:00",
        latitude=None,
        longitude=None,
        camera="Apple iPhone 11",
        categories=("Église Saint-Pierre",),
        description="",
    )
    return replace(base, **changes)


def candidate(name: str, view: View, author: str = "A", day: str = "2019:06:01") -> Candidate:
    f = file(
        title=f"File:{name}.jpg",
        thumb_url=f"https://thumb.wikimedia.org/x/960px-{name}.jpg",
        author=author,
        taken=f"{day} 10:00:00",
    )
    return Candidate(f, Source.CATEGORY, "Category:Église Saint-Pierre", view, named=True)


def test_gallery_order_puts_exteriors_and_new_visits_first() -> None:
    ranked = [
        candidate("nave", View.INTERIOR, "B"),
        candidate("front", View.EXTERIOR, "A"),
        candidate("front-again", View.EXTERIOR, "A"),
        candidate("side", View.EXTERIOR, "C"),
    ]
    names = [c.file.title for c in order_for_gallery(ranked)]
    assert names == ["File:front.jpg", "File:side.jpg", "File:nave.jpg", "File:front-again.jpg"]


class FakeThumbs(WikimediaClient):
    def __init__(self, thumbs: dict[str, bytes], cache: Path) -> None:
        super().__init__(httpx.Client(), cache)
        self._thumbs = thumbs

    def get_jpeg(self, url: str) -> bytes | None:
        return self._thumbs.get(url)


def test_gallery_pick_keeps_query_photos_their_copies_and_visits_out(tmp_path: Path) -> None:
    query, pop, fresh = colour_photo(1), colour_photo(2), colour_photo(3)
    picks = [
        candidate("copy-of-query", View.EXTERIOR, "X"),
        candidate("same-visit", View.EXTERIOR, "Q", "2020:01:01"),
        candidate("copy-of-pop", View.EXTERIOR, "Y"),
        candidate("fresh", View.EXTERIOR, "Z"),
    ]
    thumbs = {
        "https://thumb.wikimedia.org/x/500px-copy-of-query.jpg": query,
        "https://thumb.wikimedia.org/x/500px-copy-of-pop.jpg": pop,
        "https://thumb.wikimedia.org/x/500px-fresh.jpg": fresh,
    }
    guard = Guard(
        titles=frozenset(),
        hashes=(phash(query),),
        visits={"PA1": frozenset({("Q", "2020:01:01")})},
    )
    finder = GalleryFinder(FakeThumbs(thumbs, tmp_path), guard, lambda _: [phash(pop)])
    chosen = finder.pick("PA1", picks)
    assert [c.file.title for c, _ in chosen] == ["File:fresh.jpg"]
    assert finder.rejects == {
        Excluded.QUERY_COPY: 1,
        Excluded.QUERY_VISIT: 1,
        Excluded.POP_COPY: 1,
    }


def test_bucket_of_splits_buildings_by_colour() -> None:
    assert bucket_of(0.9, 0.0) is Bucket.ARCHIVE_BUILDING
    assert bucket_of(0.9, 12.0) is Bucket.COLOUR_BUILDING
    assert bucket_of(0.1, 0.0) is Bucket.OBJECT


def grid(set_id: str, buckets: list[Bucket], best: dict[Bucket, float]) -> SetGrid:
    """Each query is right (rank 1) only at its bucket's best weight, else rank 50."""
    results = tuple(
        ScoreResult(
            per_query=tuple(
                QueryScore(
                    f"{set_id}-{j}", 1 if w == best[b] else 50, 1.0 if w == best[b] else 0.02
                )
                for j, b in enumerate(buckets)
            ),
            skipped_no_answer=0,
            gallery_images=10,
            gallery_entities=10,
            ap_max_rank=None,
        )
        for w in ROUTE_GRID
    )
    n = len(buckets)
    return SetGrid(set_id, results, tuple(buckets), tuple(f"e{j}" for j in range(n)), ("q",) * n)


def test_each_bucket_gets_its_own_best_weight() -> None:
    best = {Bucket.ARCHIVE_BUILDING: 1.0, Bucket.COLOUR_BUILDING: 0.3, Bucket.OBJECT: 0.2}
    grids = [
        grid("T1", [Bucket.ARCHIVE_BUILDING] * 4, best),
        grid("T2", [Bucket.COLOUR_BUILDING] * 3 + [Bucket.OBJECT], best),
        grid("T5", [Bucket.OBJECT] * 4, best),
    ]
    assert pick_weights(grids, three) == best
    g4 = pick_weights(grids, two)
    assert g4[Bucket.ARCHIVE_BUILDING] == g4[Bucket.COLOUR_BUILDING]
    assert g4[Bucket.OBJECT] == 0.2
    assert len(set(pick_weights(grids, one).values())) == 1
    scored = grids[1].at(best)
    assert all(q.rank == 1 for q in scored.per_query)
