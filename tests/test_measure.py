"""Measuring a bundle: `site_search` ranks stored query vectors exactly as `Engine.search`."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from numpy.typing import NDArray

from miroir.eval.sets import Query
from miroir.eval.vectors import Route, RunInfo, VectorSet
from miroir.web import ranking
from miroir.web.calibrate import search_set
from miroir.web.engine import Engine, Projection
from miroir.web.measure import AskedSet, BundleVectors, SiteBundle, query_sets, site_search
from miroir.web.routing import Bucket, PhotoType, Routing
from miroir.web.store import ExactStore, quantize_rows
from tests import web_helpers as h

Floats = NDArray[np.float32]
GREY_BELOW = 2.0

# Images of `h.catalog()`, in row order: m1, m2 (PA1), j1, j2 (OBJ1), r1 (OBJ2), commons-c1 (PA1).
IDS = ("m1", "m2", "j1", "j2", "r1", "commons-c1")


def g5_routing() -> Routing:
    """The site's three buckets: grey buildings to EmbeddingGemma 2, colour ones to MegaLoc."""
    return Routing(
        buckets=(
            Bucket("archive-building", 1.0, 0.5, PhotoType.BUILDING, greyscale=True),
            Bucket("colour-building", 0.0, 0.5, PhotoType.BUILDING, greyscale=False),
            Bucket("object", 0.2, 0.5, PhotoType.OBJECT),
        ),
        grey_below=GREY_BELOW,
    )


class TwoGallery:
    """Both models' stored rows (MegaLoc as the store keeps it), cosines on the CPU."""

    def __init__(self, a7: Floats, a3: Floats) -> None:
        self.a7_matrix, self.a3_matrix = a7, a3

    def a7(self, query: Floats) -> Floats:
        return self.a7_matrix @ query

    def a3(self, query: Floats) -> Floats:
        return self.a3_matrix @ query

    def a7_row(self, row: int) -> Floats:
        return self.a7_matrix[row]

    def a3_row(self, row: int) -> Floats:
        return self.a3_matrix[row]


def _fit_projection(a3: Floats, dims: int) -> Projection:
    mean = a3.mean(axis=0)
    _, _, vt = np.linalg.svd(a3 - mean, full_matrices=False)
    return Projection(mean=mean.astype(np.float32), components=vt[:dims].astype(np.float32))


def _pair(
    *, building: bool, dims: int | None = None, exact: bool = False
) -> tuple[SiteBundle, Engine, Floats, Floats]:
    """The same small index as a measured bundle and as the site's engine."""
    a7 = h.unit_rows(6, 8, seed=2)
    a3 = h.unit_rows(6, 12, seed=3)
    projection = None if dims is None else _fit_projection(a3, dims)
    stored = a3 if projection is None else np.stack([projection.apply(v) for v in a3])
    store = ExactStore(*quantize_rows(a3)) if exact else None
    catalog, guess, routing = h.catalog(), h.guess(8, building), g5_routing()
    vectors = BundleVectors(
        ids=IDS,
        a7=a7.astype(np.float16),
        a3=stored.astype(np.float16),
        projection=projection,
        exact=store,
    )
    engine = Engine(
        catalog=catalog,
        routing=routing,
        guess=guess,
        encoders=h.RowEncoders(a7),
        gallery=TwoGallery(
            a7.astype(np.float16).astype(np.float32), stored.astype(np.float16).astype(np.float32)
        ),
        projection=projection,
        exact=store,
    )
    return SiteBundle(catalog, vectors, guess, routing), engine, a7, a3


def _asked(a7: Floats, a3: Floats, chroma: float, rows: tuple[int, ...]) -> AskedSet:
    return AskedSet(
        set_id="T",
        query_ids=tuple(f"q{r}" for r in rows),
        query_image_ids=tuple(IDS[r] for r in rows),
        a7=a7[list(rows)],
        a3=a3[list(rows)],
        hidden=tuple((IDS[r],) for r in rows),
        right=tuple(("PA1",) for _ in rows),
        chroma=(chroma,) * len(rows),
    )


@pytest.mark.parametrize(
    ("building", "chroma", "dims", "exact"),
    [
        (True, 0.5, None, False),
        (True, 30.0, None, False),
        (False, 30.0, None, False),
        (True, 30.0, 4, False),  # MegaLoc stored reduced by PCA
        (False, 30.0, 4, True),  # and shortlisted, exact rows from int8
    ],
)
def test_site_search_ranks_as_the_engine(
    building: bool, chroma: float, dims: int | None, exact: bool
) -> None:
    bundle, engine, a7, a3 = _pair(building=building, dims=dims, exact=exact)
    rows = (0, 2, 4)
    hits = site_search(bundle, _asked(a7, a3, chroma, rows), top=3)
    for row, hit in zip(rows, hits, strict=True):
        mask = np.ones(len(IDS), dtype=bool)
        mask[row] = False
        photo = engine.photo_of(engine.vector_cosines(a7[row], a3[row], chroma, mask, {}))
        top = ranking.top_entities(ranking.fuse(photo.signals), 3)
        assert hit.bucket == photo.bucket
        assert hit.entities == tuple(engine.catalog.entity_ids[int(e)] for e in top)
        assert hit.confidences == pytest.approx([photo.confidence(int(e)) for e in top], abs=1e-3)


def test_a_colour_building_photo_goes_to_the_colour_bucket() -> None:
    bundle, _, a7, a3 = _pair(building=True)
    grey, colour = (site_search(bundle, _asked(a7, a3, c, (0,)), top=1)[0] for c in (0.5, 30.0))
    assert (grey.bucket, colour.bucket) == ("archive-building", "colour-building")


def test_a_photo_without_its_chroma_is_refused_where_buckets_need_it() -> None:
    bundle, _, a7, a3 = _pair(building=True)
    with pytest.raises(ValueError, match="chroma"):
        bundle.routing.choose(PhotoType.BUILDING, None)
    assert bundle.routing.choose(PhotoType.OBJECT, None).name == "object"
    del a7, a3


INFO = RunInfo(
    option="A",
    model="m",
    resolution=1,
    pooling="cls",
    gallery="g",
    created=date(2026, 11, 4),
    cost_usd=0.0,
    wall_time_s=0.0,
    route=Route.HOSTED,
)


def _run(rows: Floats) -> VectorSet:
    return VectorSet(ids=IDS, vectors=rows.astype(np.float16), info=INFO)


def _query(image_row: int, entity: str, right: tuple[str, ...], exclude: tuple[str, ...] = ()):
    image = IDS[image_row]
    return Query(f"q-{image}", "T", entity, image, right, exclude)


def test_query_sets_hold_whole_vectors_and_the_bundle_projects_them() -> None:
    bundle, _, a7, a3 = _pair(building=True, dims=4)
    sets = query_sets(
        bundle.catalog,
        {"T1": [_query(0, "PA1", ("PA1",), ("m1",))], "T2": [_query(2, "OBJ1", ("OBJ1",))]},
        gallery=(_run(a7), _run(a3)),
        commons=(_run(a7), _run(a3)),
        chroma=dict.fromkeys(IDS, 30.0),
    )
    # Gallery-image sets and Commons sets alike: 12-d, the store's 4 are the search's business.
    assert sets["T1"].a3.shape == sets["T2"].a3.shape == (1, 12)
    assert sets["T1"].clusters == ("PA1",)
    for asked in sets.values():
        assert len(site_search(bundle, asked, top=2)) == 1


def test_a_not_found_query_hides_photos_added_to_its_entity_since_the_build() -> None:
    bundle, _, a7, a3 = _pair(building=True)
    # T8 listed PA1's POP photos when the set was built; the Commons photo came later (F2).
    lost = _query(0, "PA1", (), ("m1", "m2"))
    sets = query_sets(
        bundle.catalog,
        {"T8": [lost]},
        gallery=(_run(a7), _run(a3)),
        commons=None,
        chroma=dict.fromkeys(IDS, 0.5),
    )
    assert "commons-c1" in sets["T8"].hidden[0]
    hit = site_search(bundle, sets["T8"], top=3)[0]
    assert "PA1" not in hit.entities


def test_calibration_reads_the_top_confidence_and_bucket_of_each_search() -> None:
    bundle, _, a7, a3 = _pair(building=False)
    searched = search_set(bundle, _asked(a7, a3, 30.0, (2, 4)))
    hits = site_search(bundle, _asked(a7, a3, 30.0, (2, 4)), top=1)
    assert searched.buckets == ("object", "object")
    assert searched.confidence.tolist() == pytest.approx([h.confidences[0] for h in hits])
