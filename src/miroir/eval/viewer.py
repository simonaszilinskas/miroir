"""A results page that shows where models find the right record and where they miss.

For a seeded sample of tune-half queries per set (publishable images only), the page shows the
query image, one image of the right answer, and each model's top entities as thumbnails, marked
right or wrong, with title, collection and a link to the POP record. Each set opens with a
score line per model: hit@1/5/10 with 95% intervals, from eval/results when the run was scored
there, else computed here with the same scoring code.

Everything is one self-contained HTML file: thumbnails are embedded as data URIs, since the
page is published where outside images are blocked. Only NumPy search runs; no model.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import random
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from miroir.config import SEED, Settings
from miroir.eval.results import (
    ResultFormatError,
    RunResult,
    SetScore,
    read_result,
    results_dir,
    set_score,
)
from miroir.eval.run import search_queries
from miroir.eval.scoring import Gallery, Interval, score
from miroir.eval.sets import Query, load_queries, read_gallery
from miroir.eval.vectors import VectorFormatError, VectorSet, read_meta, read_vectors
from miroir.images.fetch import CONCURRENCY, ImageCache, ImageFetcher, count_skips
from miroir.records.model import Image, Record
from miroir.records.store import read_records

logger = logging.getLogger(__name__)

#: Sets shown, in page order.
SETS = ("T1", "T5", "T3")
SET_NAMES = {
    "T1": "Archive buildings",
    "T3": "Room photo crops",
    "T5": "Objects, other view",
}
QUERIES_PER_SET = 20
TOP_K = 5
#: The gallery every model is searched on: closed models may only see these images.
GALLERY = "screening-20k-publishable"
#: Gallery runs, best first. A run of a larger list is cut down to GALLERY.
_GALLERY_RUNS = ("screening-20k-publishable", "screening-20k", "full")
_QUERY_RUN_SUFFIX = "-tune-queries"
#: The published page must stay under this size.
MAX_PAGE_BYTES = 8_000_000
#: Thumbnail sizes tried in turn until the page fits: (longest side, JPEG quality).
THUMB_STEPS = ((220, 70), (180, 64), (150, 58), (120, 52))

_TEMPLATE = "viewer.html"
_DATA_MARK = "@@PAGE_DATA@@"

FetchImages = Callable[[Sequence[Image]], Mapping[str, bytes]]
"""Returns image bytes by `image_id` for the images it could fetch."""


class ModelRun:
    """One model as the page shows it: its gallery vectors and any runs holding query vectors."""

    def __init__(self, name: str, gallery: VectorSet, query_runs: Sequence[VectorSet] = ()) -> None:
        self.name = name
        """Folder of the gallery run."""
        self.gallery = gallery
        self._lookups = [(run, run.index()) for run in (*query_runs, gallery)]

    @property
    def label(self) -> str:
        info = self.gallery.info
        return f"{info.option} {info.model.rsplit('/', 1)[-1]}"

    def query_vector(self, image_id: str) -> NDArray[np.float32] | None:
        for run, rows in self._lookups:
            row = rows.get(image_id)
            if row is not None:
                return np.asarray(run.vectors[row], dtype=np.float32)
        return None


def find_model_runs(settings: Settings, names: Sequence[str] = ()) -> list[ModelRun]:
    """Model runs to show: `names` (gallery run folders), else one per model found on disk.

    Query runs (`<...>-T1-tune-queries`) of the same option and model are attached.
    """
    metas: dict[str, tuple[str, str, int, str]] = {}
    root = settings.vectors_dir
    for folder in sorted(root.iterdir()) if root.exists() else ():
        try:
            info = read_meta(folder).info
        except VectorFormatError:
            continue
        metas[folder.name] = (info.option, info.model, info.resolution, info.gallery)
    chosen = list(names) or _best_gallery_runs(metas)
    runs: list[ModelRun] = []
    for name in chosen:
        if name not in metas:
            msg = f"no finished vector run {name!r} under {root}"
            raise FileNotFoundError(msg)
        option, model, resolution, _ = metas[name]
        query_names = [
            other
            for other, (o, m, r, g) in metas.items()
            if (o, m, r) == (option, model, resolution) and g.endswith(_QUERY_RUN_SUFFIX)
        ]
        runs.append(
            ModelRun(
                name=name,
                gallery=read_vectors(settings.vectors_path(name)),
                query_runs=tuple(read_vectors(settings.vectors_path(q)) for q in query_names),
            )
        )
        logger.info("model %s: gallery run %s, query runs %s", option, name, query_names or "none")
    return runs


def _best_gallery_runs(metas: Mapping[str, tuple[str, str, int, str]]) -> list[str]:
    best: dict[tuple[str, str, int], tuple[int, str]] = {}
    for name, (option, model, resolution, gallery) in metas.items():
        if gallery not in _GALLERY_RUNS:
            continue
        key = (option, model, resolution)
        rank = (_GALLERY_RUNS.index(gallery), name)
        if key not in best or rank < best[key]:
            best[key] = rank
    return [name for _, name in sorted(best.values(), key=lambda r: r[1])]


# --- search -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hit:
    entity_id: str
    image_id: str
    """The entity's image closest to the query."""
    similarity: float


@dataclass(frozen=True, slots=True)
class Search:
    top: tuple[Hit, ...]
    rank: int | None
    """Rank of the best-placed right entity, from 1, as `scoring.score` counts it."""


class EntityIndex:
    """Exact cosine search over a gallery, ranking entities by their best image.

    Ties go to the lower entity id, as in `miroir.eval.scoring`.
    """

    def __init__(self, gallery: Gallery) -> None:
        names, codes = np.unique(np.asarray(gallery.entity_ids), return_inverse=True)
        self._names = [str(n) for n in names]
        self._code_of = {n: i for i, n in enumerate(self._names)}
        self._order = np.argsort(codes, kind="stable")
        sorted_codes = codes[self._order]
        self._starts = np.flatnonzero(np.r_[True, np.diff(sorted_codes) != 0])
        self._ends = np.r_[self._starts[1:], len(sorted_codes)]
        self._row = {image_id: i for i, image_id in enumerate(gallery.image_ids)}
        self._image_ids = gallery.image_ids
        self.gallery = gallery
        self._matrix = np.asarray(gallery.vectors, dtype=np.float32)

    @property
    def entities(self) -> frozenset[str]:
        return frozenset(self._names)

    def search(
        self, vector: NDArray[np.float32], hidden: Iterable[str], right: Iterable[str], k: int
    ) -> Search:
        sims = self._matrix @ vector
        rows = [self._row[i] for i in hidden if i in self._row]
        sims[rows] = -np.inf
        entity_scores = np.maximum.reduceat(sims[self._order], self._starts)
        ranking = np.argsort(-entity_scores, kind="stable")
        top: list[Hit] = []
        for code in ranking[:k]:
            if not np.isfinite(entity_scores[code]):
                break
            members = self._order[self._starts[code] : self._ends[code]]
            best = int(members[int(np.argmax(sims[members]))])
            top.append(
                Hit(
                    entity_id=self._names[int(code)],
                    image_id=self._image_ids[best],
                    similarity=float(sims[best]),
                )
            )
        position = np.empty_like(ranking)
        position[ranking] = np.arange(len(ranking))
        right_codes = [
            self._code_of[e]
            for e in right
            if e in self._code_of and np.isfinite(entity_scores[self._code_of[e]])
        ]
        rank = min(int(position[c]) for c in right_codes) + 1 if right_codes else None
        return Search(top=tuple(top), rank=rank)


# --- page data ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageModel:
    key: str
    label: str
    model: str
    run: str


@dataclass(frozen=True, slots=True)
class ScoreLine:
    model: str
    source: str
    """Result file it comes from, or "computed here"."""
    queries: int
    hit_at_1: tuple[float, float, float]
    hit_at_5: tuple[float, float, float]
    hit_at_10: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PageSet:
    set_id: str
    name: str
    shown: int
    scores: tuple[ScoreLine, ...]


@dataclass(frozen=True, slots=True)
class PageRecord:
    image: str
    entity: str
    title: str
    collection: str
    url: str


@dataclass(frozen=True, slots=True)
class PageHit:
    image: str
    entity: str
    title: str
    collection: str
    url: str
    right: bool
    similarity: float


@dataclass(frozen=True, slots=True)
class PageAnswer:
    model: str
    rank: int | None
    top: tuple[PageHit, ...]


@dataclass(frozen=True, slots=True)
class PageQuery:
    query_id: str
    set_id: str
    query: PageRecord
    answer: PageRecord | None
    """One image of a right entity, as the gallery holds it."""
    answers: tuple[PageAnswer, ...]


@dataclass(frozen=True, slots=True)
class PageData:
    title: str
    gallery: str
    gallery_images: int
    per_set: int
    seed: int
    models: tuple[PageModel, ...]
    sets: tuple[PageSet, ...]
    queries: tuple[PageQuery, ...]
    images: dict[str, str]
    """Thumbnail data URI by image id."""


class _Catalogue:
    """Records and images by id, and which images are publishable."""

    def __init__(self, records: Sequence[Record]) -> None:
        self.record_of_ref = {r.ref: r for r in records}
        self.images = {i.image_id: i for r in records for i in r.images}
        self.public = frozenset(i.image_id for r in records if r.publishable for i in r.images)
        self.entity_of = {i.image_id: i.entity_id for i in self.images.values()}

    def page_record(self, image_id: str) -> PageRecord:
        image = self.images[image_id]
        record = self.record_of_ref[image.ref]
        return PageRecord(
            image=image_id,
            entity=image.entity_id,
            title=record.title or record.ref,
            collection=record.collection.value,
            url=record.pop_url,
        )


@dataclass(frozen=True, slots=True)
class _Shown:
    key: str
    """Short key the page uses for the model: m0, m1..."""
    model: ModelRun
    index: EntityIndex


def build_page(
    settings: Settings,
    models: Sequence[ModelRun],
    fetch_images: FetchImages,
    *,
    sets: Sequence[str] = SETS,
    per_set: int = QUERIES_PER_SET,
    results: Path | None = None,
    max_bytes: int = MAX_PAGE_BYTES,
) -> str:
    """The whole page as HTML text."""
    if not models:
        msg = f"no model run to show: no finished gallery run under {settings.vectors_dir}"
        raise ValueError(msg)
    catalogue = _Catalogue(read_records(settings.records_parquet, settings.images_parquet))
    gallery_ids = _common_gallery(settings, models, catalogue)
    shown = [
        _Shown(
            f"m{n}",
            m,
            EntityIndex(Gallery.from_vectors(m.gallery, catalogue.entity_of, gallery_ids)),
        )
        for n, m in enumerate(models)
    ]
    entities = shown[0].index.entities
    stored = _stored_results(results or results_dir())

    page_sets: list[PageSet] = []
    page_queries: list[PageQuery] = []
    for set_id in sets:
        usable = _usable_queries(settings, set_id, models, catalogue, entities)
        picked = _sample(usable, per_set)
        lines = tuple(_score_line(s, set_id, usable, stored) for s in shown)
        page_sets.append(PageSet(set_id, SET_NAMES.get(set_id, set_id), len(picked), lines))
        for query in picked:
            page_queries.append(_page_query(query, shown, catalogue, gallery_ids))
        logger.info("%s: %d usable queries, showing %d", set_id, len(usable), len(picked))

    wanted = _image_ids(page_queries)
    private = [i for i in wanted if i not in catalogue.public]
    if private:
        msg = f"{len(private)} image(s) on the page are not publishable, e.g. {private[0]}"
        raise AssertionError(msg)
    originals = fetch_images([catalogue.images[i] for i in wanted])
    missing = len(wanted) - len(originals)
    if missing:
        logger.warning("%d of %d thumbnails could not be fetched", missing, len(wanted))
    page = PageData(
        title="Miroir Match Viewer",
        gallery=GALLERY,
        gallery_images=len(gallery_ids),
        per_set=per_set,
        seed=SEED,
        models=tuple(
            PageModel(s.key, s.model.label, s.model.gallery.info.model, s.model.name) for s in shown
        ),
        sets=tuple(page_sets),
        queries=tuple(page_queries),
        images={},
    )
    for side, quality in THUMB_STEPS:
        thumbs = {i: _data_uri(data, side, quality) for i, data in originals.items()}
        text = render(replace(page, images=thumbs))
        size = len(text.encode("utf-8"))
        if size <= max_bytes:
            logger.info("page: %d bytes, thumbnails %d px, quality %d", size, side, quality)
            return text
        logger.info("page: %d bytes at %d px is too big; trying smaller thumbnails", size, side)
    msg = f"the page stays over {max_bytes} bytes even with small thumbnails: show fewer queries"
    raise ValueError(msg)


def _common_gallery(
    settings: Settings, models: Sequence[ModelRun], catalogue: _Catalogue
) -> tuple[str, ...]:
    """Publishable images of GALLERY that every model has a vector for."""
    listed = read_gallery(settings, GALLERY)
    have = [set(m.gallery.ids) for m in models]
    kept = tuple(
        i
        for i in listed
        if i in catalogue.public and i in catalogue.images and all(i in h for h in have)
    )
    if len(kept) < len(listed):
        logger.info("gallery: %d of %d images have a vector in every run", len(kept), len(listed))
    return kept


def _usable_queries(
    settings: Settings,
    set_id: str,
    models: Sequence[ModelRun],
    catalogue: _Catalogue,
    entities: frozenset[str],
) -> list[Query]:
    """Tune queries with a publishable image, a vector in every model, and an answer to find."""
    usable: list[Query] = []
    for q in load_queries(settings, set_id, "tune"):
        if q.query_image_id not in catalogue.public or not q.expected_entity_ids:
            continue
        if not any(e in entities for e in q.expected_entity_ids):
            continue
        if any(m.query_vector(q.query_image_id) is None for m in models):
            continue
        usable.append(q)
    return sorted(usable, key=lambda q: q.query_id)


def _sample(queries: Sequence[Query], n: int) -> list[Query]:
    picked = random.Random(SEED).sample(list(queries), min(n, len(queries)))
    return sorted(picked, key=lambda q: q.query_id)


def _page_query(
    query: Query, shown: Sequence[_Shown], catalogue: _Catalogue, gallery_ids: Sequence[str]
) -> PageQuery:
    right = set(query.expected_entity_ids)
    hidden = (query.query_image_id, *query.exclude_image_ids)
    answers: list[PageAnswer] = []
    for entry in shown:
        vector = entry.model.query_vector(query.query_image_id)
        if vector is None:
            msg = f"{entry.model.name}: no vector for query image {query.query_image_id}"
            raise ValueError(msg)
        found = entry.index.search(vector, hidden, query.expected_entity_ids, TOP_K)
        hits = tuple(
            PageHit(
                **asdict(catalogue.page_record(h.image_id)),
                right=h.entity_id in right,
                similarity=round(h.similarity, 4),
            )
            for h in found.top
        )
        answers.append(PageAnswer(model=entry.key, rank=found.rank, top=hits))
    hidden_set = set(hidden)
    answer_image = next(
        (
            i
            for i in gallery_ids
            if catalogue.entity_of[i] in right
            and i not in hidden_set
            and catalogue.images[i].is_cover
        ),
        None,
    ) or next(
        (i for i in gallery_ids if catalogue.entity_of[i] in right and i not in hidden_set), None
    )
    return PageQuery(
        query_id=query.query_id,
        set_id=query.set_id,
        query=catalogue.page_record(query.query_image_id),
        answer=None if answer_image is None else catalogue.page_record(answer_image),
        answers=tuple(answers),
    )


def _image_ids(queries: Sequence[PageQuery]) -> list[str]:
    ids: dict[str, None] = {}
    for q in queries:
        ids[q.query.image] = None
        if q.answer is not None:
            ids[q.answer.image] = None
        for a in q.answers:
            for h in a.top:
                ids[h.image] = None
    return list(ids)


@contextmanager
def pop_images(settings: Settings) -> Generator[FetchImages]:
    """Fetch images from POP (512 px, IIIF fallback) through the shared image cache."""
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1024 * 1024)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as http:
        fetcher = ImageFetcher(http, cache)

        def fetch(images: Sequence[Image]) -> dict[str, bytes]:
            logger.info(
                "fetching %d thumbnails (cached ones are not downloaded again)", len(images)
            )
            done = list(fetcher.fetch_all(images))
            skips = count_skips(done)
            if skips:
                logger.warning("thumbnails left out: %s", dict(skips))
            return {f.image_id: f.jpeg for f in done if f.jpeg is not None}

        yield fetch


# --- scores -------------------------------------------------------------------------------


def _stored_results(directory: Path) -> list[tuple[Path, RunResult]]:
    out: list[tuple[Path, RunResult]] = []
    for path in sorted(directory.glob("R*/*.json")):
        try:
            out.append((path, read_result(path)))
        except ResultFormatError:
            logger.warning("skipped unreadable result file %s", path)
    return out


def _score_line(
    entry: _Shown,
    set_id: str,
    usable: Sequence[Query],
    stored: Sequence[tuple[Path, RunResult]],
) -> ScoreLine:
    key, model = entry.key, entry.model
    found = _stored_score(model.name, set_id, stored)
    if found is not None:
        path, scored = found
        source = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
        return _line(key, f"eval/{source}", scored)
    if not usable:
        return ScoreLine(key, "no queries", 0, (0, 0, 0), (0, 0, 0), (0, 0, 0))
    ids = tuple(dict.fromkeys(q.query_image_id for q in usable))
    vectors = np.stack([v for i in ids if (v := model.query_vector(i)) is not None])
    asking = VectorSet(ids=ids, vectors=vectors.astype(np.float16), info=model.gallery.info)
    result = score(asking, entry.index.gallery, search_queries(list(usable)))
    scored = set_score(result, test_set=set_id, half="tune", gallery=GALLERY)
    return _line(key, "computed here", scored)


def _stored_score(
    run: str, set_id: str, stored: Sequence[tuple[Path, RunResult]]
) -> tuple[Path, SetScore] | None:
    candidates = [
        (path, s)
        for path, result in stored
        if result.vector_run == run
        for s in result.scores
        if s.test_set == set_id and s.half == "tune"
    ]
    candidates.sort(key=lambda c: (c[1].gallery != GALLERY, str(c[0])))
    return candidates[0] if candidates else None


def _line(key: str, source: str, scored: SetScore) -> ScoreLine:
    return ScoreLine(
        model=key,
        source=source,
        queries=scored.queries,
        hit_at_1=_triple(scored.hit_at_1),
        hit_at_5=_triple(scored.hit_at_5),
        hit_at_10=_triple(scored.hit_at_10),
    )


def _triple(interval: Interval) -> tuple[float, float, float]:
    return (round(interval.value, 4), round(interval.low, 4), round(interval.high, 4))


# --- output -------------------------------------------------------------------------------


def _data_uri(data: bytes, side: int, quality: int) -> str:
    with PILImage.open(io.BytesIO(data)) as opened:
        opened.thumbnail((side, side), PILImage.Resampling.LANCZOS)
        rgb = opened.convert("RGB")
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def render(page: PageData) -> str:
    """Fill the template with the page data. `<` is escaped so the JSON cannot close its tag."""
    template = files("miroir.eval").joinpath(_TEMPLATE).read_text(encoding="utf-8")
    payload = json.dumps(asdict(page), ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    if template.count(_DATA_MARK) != 1:
        msg = f"{_TEMPLATE} must hold {_DATA_MARK} exactly once"
        raise AssertionError(msg)
    return template.replace(_DATA_MARK, payload).replace("@@PAGE_TITLE@@", html.escape(page.title))
