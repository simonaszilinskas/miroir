"""T1 archive prints: search the print without its mount, and use the caption it carries.

Every T1 tune query photo, as POP serves it (a print on a card with a typed caption), goes
through the website's own `Engine.search` on the full index, with the query's shoot hidden
(`SearchRequest.exclude_images`). The models do not run here: their outputs are computed on
Modal once and replayed.

- `run_remote` (Modal): the print cut from its mount (`images.mount`, on the Mac) and the
  whole photo, each embedded by EmbeddingGemma 2 and MegaLoc as the site embeds a photo;
  the caption read by each open model of `caption.CAPTION_MODELS`; the A7 text vector of
  each caption. Saved under `Settings.archive_dir`.
- `score`: the site's ranking for each way of using them (`ArchiveRule`), hit@1, @5, @10.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from numpy.typing import NDArray

from miroir.caption import CAPTION_MODELS, Gazetteer
from miroir.config import WEB_INDEX, Settings, eval_result_path
from miroir.eval.results import set_score, write_result
from miroir.eval.scoring import QueryScore, ScoreResult
from miroir.eval.sets import Query, load_queries
from miroir.eval.vectors import Route, RunInfo
from miroir.eval.wire import CaptionChunk, CaptionPhoto
from miroir.images.mount import crop_picture, find_picture
from miroir.jobs.modal_run import PaidCall, paid_call
from miroir.jobs.modal_spec import CHUNK_TIMEOUT_S, L4, OPTIONS, ChunkResult, ModalOption
from miroir.jobs.modal_spec import SuppliedImage as Supplied
from miroir.jobs.spend import SpendKind
from miroir.records.model import Image
from miroir.records.store import read_images
from miroir.web.archive import ArchiveRule
from miroir.web.bundle import SiteBundle
from miroir.web.engine import Engine, SearchRequest, prepare_photo
from miroir.web.nearby import EntityPlaces

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]

A7 = "A7-embeddinggemma-2-512"
A3 = "A3-megaloc-322"
PHOTO_VECTORS = "photo-vectors.npz"
#: Cosines of those photos with every image of the index, float16, computed once.
COSINES = "photo-cosines.npz"
#: Entities ranked per query; a right entity further down counts as a miss.
TOP = 100
ROUND = "R3"
STEP = "T1archive"


def _key(jpeg: bytes) -> str:
    return hashlib.sha256(jpeg).hexdigest()


# --- the query photos ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArchivePhoto:
    query: Query
    raw: bytes
    """As POP serves it (about 670 x 512): what a visitor's photo of the print would hold."""

    @property
    def whole(self) -> bytes:
        """What the site embeds for the whole photo."""
        return prepare_photo(self.raw)

    def cut(self) -> bytes | None:
        """What the site embeds when it finds the mount: the print alone."""
        picture = find_picture(self.raw)
        return None if picture is None else prepare_photo(crop_picture(self.raw, picture))


def archive_photos(settings: Settings) -> list[ArchivePhoto]:
    """T1 tune queries with their photo (fetched from POP once, then kept)."""
    folder = settings.archive_dir / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    url_of = {i.image_id: i.url for i in read_images(settings.images_parquet)}
    out: list[ArchivePhoto] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as http:
        for query in load_queries(settings, "T1", "tune"):
            path = folder / f"{query.query_image_id}.jpg"
            if not path.exists():
                response = http.get(url_of[query.query_image_id])
                if not response.is_success:
                    logger.warning("%s: HTTP %d", query.query_id, response.status_code)
                    continue
                path.write_bytes(response.content)
            out.append(ArchivePhoto(query, path.read_bytes()))
    return out


# --- on Modal ---------------------------------------------------------------------------------


class ArchiveCalls(Protocol):
    def embed(
        self, option: ModalOption, images: Sequence[Image], supplied: Sequence[Supplied] = ()
    ) -> ChunkResult: ...

    def embed_texts(
        self, option: ModalOption, ids: Sequence[str], texts: Sequence[str]
    ) -> ChunkResult: ...

    def read_captions(self, model: str, photos: Sequence[CaptionPhoto]) -> CaptionChunk: ...


def captions_path(settings: Settings, model: str) -> Path:
    return settings.archive_dir / f"captions-{model}.jsonl"


def caption_vectors_path(settings: Settings, model: str) -> Path:
    return settings.archive_dir / f"caption-vectors-{model}.npz"


def run_remote(
    settings: Settings, calls: ArchiveCalls, paid: PaidCall, models: Sequence[str]
) -> float:
    """Embed the whole photos and the cut prints, read the captions; returns the cost."""
    photos = archive_photos(settings)
    settings.archive_dir.mkdir(parents=True, exist_ok=True)
    cost = 0.0
    vectors_file = settings.archive_dir / PHOTO_VECTORS
    if not vectors_file.exists():
        jpegs = {_key(j): j for p in photos for j in (p.whole, p.cut()) if j is not None}
        supplied = [Supplied(k, j) for k, j in jpegs.items()]
        found: dict[str, ChunkResult] = {}
        for option_id in (A7, A3):
            option = OPTIONS[option_id]
            done = paid_call(
                paid,
                lambda option=option: calls.embed(option, [], supplied),
                machine=option.machine,
                timeout_s=CHUNK_TIMEOUT_S,
                kind=SpendKind.RUN,
                option=option_id,
                run=f"{STEP}-photos-{option_id}",
                model=option.model,
                items=len(supplied),
            )
            cost += done.cost_usd
            found[option_id] = done.result
        a7, a3 = found[A7], found[A3]
        if a7.ids != a3.ids:
            msg = "the two models returned the photos in different orders"
            raise RuntimeError(msg)
        np.savez(vectors_file, keys=np.asarray(a7.ids), a7=a7.vectors, a3=a3.vectors)
    for model in models:
        path = captions_path(settings, model)
        if not path.exists():
            asked = [CaptionPhoto(p.query.query_image_id, p.raw) for p in photos]
            done = paid_call(
                paid,
                lambda model=model, asked=asked: calls.read_captions(model, asked),
                machine=L4,
                timeout_s=CHUNK_TIMEOUT_S,
                kind=SpendKind.RUN,
                option="caption",
                run=f"{STEP}-captions-{model}",
                model=CAPTION_MODELS[model],
                items=len(asked),
            )
            cost += done.cost_usd
            lines = (
                json.dumps(
                    {"image_id": i, "caption": t, "seconds": round(s, 3)}, ensure_ascii=False
                )
                for i, t, s in zip(
                    done.result.ids, done.result.texts, done.result.seconds, strict=True
                )
            )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("%s: load %.0f s, $%.4f", model, done.result.load_s, done.cost_usd)
        texts = sorted({t for t in read_captions(settings, model).values() if t})
        out = caption_vectors_path(settings, model)
        if texts and not out.exists():
            option = OPTIONS[A7]
            done = paid_call(
                paid,
                lambda option=option, texts=texts: calls.embed_texts(option, texts, texts),
                machine=option.machine,
                timeout_s=CHUNK_TIMEOUT_S,
                kind=SpendKind.RUN,
                option=A7,
                run=f"{STEP}-caption-words-{model}",
                model=option.model,
                items=len(texts),
            )
            cost += done.cost_usd
            np.savez(out, texts=np.asarray(done.result.ids), a7=done.result.vectors)
    return cost


def read_captions(settings: Settings, model: str) -> dict[str, str]:
    """Image id: the caption line the model read ("" for none)."""
    lines = captions_path(settings, model).read_text(encoding="utf-8").splitlines()
    return {row["image_id"]: row["caption"] for row in map(json.loads, lines) if row}


def caption_seconds(settings: Settings, model: str) -> Floats:
    lines = captions_path(settings, model).read_text(encoding="utf-8").splitlines()
    return np.asarray([json.loads(x)["seconds"] for x in lines if x], dtype=np.float32)


# --- replaying the models ----------------------------------------------------------------


class _Replay:
    """The site's two interfaces, answering from vectors computed on Modal."""

    def __init__(
        self, photos: Mapping[str, tuple[Floats, Floats]], texts: Mapping[str, Floats]
    ) -> None:
        self._photos = photos
        self._texts = texts

    def image(self, jpeg: bytes) -> tuple[Floats, Floats]:
        return self._photos[_key(jpeg)]

    def text(self, text: str) -> Floats:
        return self._texts[text]


class _Captions:
    def __init__(self, captions: Mapping[str, str], raw_to_id: Mapping[str, str]) -> None:
        self._captions = captions
        self._raw_to_id = raw_to_id

    def read(self, jpeg: bytes) -> str:
        return self._captions.get(self._raw_to_id[_key(jpeg)], "")


class _NumpyGallery:
    """Cosines over the bundle's stored vectors. The photo queries' cosines are computed in
    one pass over the gallery (`preload`) and kept on disk; words are computed when asked."""

    def __init__(self, a7: NDArray[np.float16], a3: NDArray[np.float16]) -> None:
        self._a7 = a7
        self._a3 = a3
        self._cache: dict[tuple[str, bytes], Floats] = {}

    @staticmethod
    def _key(name: str, query: Floats) -> tuple[str, bytes]:
        return name, np.asarray(query, dtype=np.float16).tobytes()

    def preload(self, path: Path, a7: Floats, a3: Floats) -> None:
        """Cosines of every photo query (unit rows) with every image, from `path` when it
        holds them for this gallery, else computed and saved there."""
        c7 = c3 = None
        if path.exists():
            with np.load(path) as data:
                c7, c3 = data["a7"], data["a3"]
            if c7.shape != (len(a7), self._a7.shape[0]):  # another index's gallery
                c7 = c3 = None
        if c7 is None or c3 is None:
            c7 = _batch_cosines(self._a7, a7)
            c3 = _batch_cosines(self._a3, a3)
            np.savez(path, a7=c7, a3=c3)
        for j in range(len(a7)):
            self._cache[self._key("a7", a7[j])] = np.asarray(c7[j], dtype=np.float32)
            self._cache[self._key("a3", a3[j])] = np.asarray(c3[j], dtype=np.float32)

    def _cosines(self, name: str, matrix: NDArray[np.float16], query: Floats) -> Floats:
        key = self._key(name, query)
        if key not in self._cache:
            self._cache[key] = _batch_cosines(matrix, query[None, :])[0].astype(np.float32)
        return self._cache[key]

    def a7(self, query: Floats) -> Floats:
        return self._cosines("a7", self._a7, query)

    def a3(self, query: Floats) -> Floats:
        return self._cosines("a3", self._a3, query)


def _batch_cosines(matrix: NDArray[np.float16], queries: Floats) -> NDArray[np.float16]:
    out = np.empty((len(queries), matrix.shape[0]), dtype=np.float16)
    step = 16_384
    q = np.asarray(queries, dtype=np.float32).T
    for start in range(0, matrix.shape[0], step):
        block = np.asarray(matrix[start : start + step], dtype=np.float32)
        out[:, start : start + len(block)] = (block @ q).T
    return out


#: The ways of using a mounted print that `score` compares (results.md, T1 archive).
RULES: dict[str, ArchiveRule] = {
    "whole": ArchiveRule(crop=False),
    "crop": ArchiveRule(crop=True),
    "words": ArchiveRule(crop=False, words=0.5),
    "place": ArchiveRule(crop=False, place_bonus=1.0),
    "crop+words": ArchiveRule(crop=True, words=0.5),
    "crop+place": ArchiveRule(crop=True, place_bonus=1.0),
    "crop+place-5km": ArchiveRule(crop=True, place_bonus=1.0, place_scale_km=5.0),
    "crop+place+words": ArchiveRule(crop=True, place_bonus=1.0, words=0.5),
    "words-1": ArchiveRule(crop=False, words=1.0),
    "words-2": ArchiveRule(crop=False, words=2.0),
    "words-1+place-5km": ArchiveRule(crop=False, words=1.0, place_bonus=1.0, place_scale_km=5.0),
    "words-0.5+place-5km": ArchiveRule(crop=False, words=0.5, place_bonus=1.0, place_scale_km=5.0),
}


@dataclass(frozen=True, slots=True)
class RuleScore:
    label: str
    rule: ArchiveRule
    hits: Mapping[int, float]
    queries: int
    with_place: int
    """Queries whose caption named a commune the index knows."""
    right_place: int
    """... and the right one (the answer's commune)."""


def score(
    settings: Settings, model: str, rules: Mapping[str, ArchiveRule] = RULES
) -> list[RuleScore]:
    photos = archive_photos(settings)
    folder = settings.web_index_dir(WEB_INDEX)
    bundle = SiteBundle.load(folder)
    with np.load(settings.archive_dir / PHOTO_VECTORS) as data:
        keys = [str(k) for k in data["keys"]]
        vectors = {
            k: (np.asarray(v7, dtype=np.float32), np.asarray(v3, dtype=np.float32))
            for k, v7, v3 in zip(keys, data["a7"], data["a3"], strict=True)
        }
    captions = read_captions(settings, model)
    texts: dict[str, Floats] = {}
    words_file = caption_vectors_path(settings, model)
    if words_file.exists():
        with np.load(words_file) as data:
            texts = {
                str(t): np.asarray(v, dtype=np.float32)
                for t, v in zip(data["texts"], data["a7"], strict=True)
            }
    places = _entity_places(settings, bundle)
    raw_to_id = {_key(p.raw): p.query.query_image_id for p in photos}
    gallery = _NumpyGallery(bundle.vectors.a7, bundle.vectors.a3)
    unit7 = np.stack([v[0] / np.linalg.norm(v[0]) for v in vectors.values()])
    unit3 = np.stack([v[1] / np.linalg.norm(v[1]) for v in vectors.values()])
    gallery.preload(
        settings.archive_dir / COSINES, unit7.astype(np.float32), unit3.astype(np.float32)
    )
    engine = Engine(
        catalog=bundle.catalog,
        routing=bundle.routing,
        guess=bundle.guess,
        encoders=_Replay(vectors, texts),
        gallery=gallery,
        places=places,
        captions=_Captions(captions, raw_to_id),
    )
    assert engine.gazetteer is not None
    with_place, right_place = _place_counts(engine.gazetteer, photos, captions, bundle)
    out: list[RuleScore] = []
    for label, rule in rules.items():
        engine.archive = rule
        ranks: list[int] = []
        per_query: list[QueryScore] = []
        for photo in photos:
            q = photo.query
            response = engine.search(
                SearchRequest(
                    image=photo.raw,
                    limit=TOP,
                    exclude_images=(q.query_image_id, *q.exclude_image_ids),
                )
            )
            found = [c.entity_id for c in response.cards]
            rank = next((n + 1 for n, e in enumerate(found) if e in q.expected_entity_ids), 0)
            ranks.append(rank)
            per_query.append(QueryScore(q.query_id, rank or TOP + 1, 1.0 / rank if rank else 0.0))
        r = np.asarray(ranks)
        hits = {k: float(((r >= 1) & (r <= k)).mean()) for k in (1, 5, 10)}
        out.append(RuleScore(label, rule, hits, len(photos), with_place, right_place))
        _write(bundle, per_query, model=model, label=label, rule=rule)
        logger.info("%s: %s", label, ", ".join(f"hit@{k} {100 * v:.1f}" for k, v in hits.items()))
    return out


def _entity_places(settings: Settings, bundle: SiteBundle) -> EntityPlaces:
    from miroir.records.store import read_places  # noqa: PLC0415

    placed = {p.entity_id: (p.latitude, p.longitude) for p in read_places(settings.places_parquet)}
    ids = bundle.catalog.entity_ids
    lat = np.asarray([placed[e][0] if e in placed else np.nan for e in ids], dtype=np.float64)
    lon = np.asarray([placed[e][1] if e in placed else np.nan for e in ids], dtype=np.float64)
    return EntityPlaces(lat, lon)


def _place_counts(
    gazetteer: Gazetteer,
    photos: Sequence[ArchivePhoto],
    captions: Mapping[str, str],
    bundle: SiteBundle,
) -> tuple[int, int]:
    catalog = bundle.catalog
    named = right = 0
    for photo in photos:
        place = gazetteer.find(captions.get(photo.query.query_image_id, ""))
        if place is None:
            continue
        named += 1
        entity = catalog.entity_row.get(photo.query.entity_id)
        records = catalog.entity_records(entity) if entity is not None else []
        communes: set[str] = {r.commune for r in records if r.commune}
        right += place.commune in communes
    return named, right


def _write(
    bundle: SiteBundle,
    per_query: Sequence[QueryScore],
    *,
    model: str,
    label: str,
    rule: ArchiveRule,
) -> None:
    result = ScoreResult(
        per_query=tuple(per_query),
        skipped_no_answer=0,
        gallery_images=len(bundle.catalog.images),
        gallery_entities=bundle.catalog.entity_count,
        ap_max_rank=TOP,
    )
    notes = (
        f"T1 archive prints, the site's Engine.search on index {WEB_INDEX}, query shoot hidden. "
        f"Rule {rule}; captions read by {CAPTION_MODELS[model]}"
    )
    info = RunInfo(
        option=f"{STEP}-{label}",
        model=f"A7+A3, captions {CAPTION_MODELS[model]}",
        resolution=512,
        pooling="site",
        gallery=f"web-{WEB_INDEX}",
        created=datetime.now(UTC).date(),
        cost_usd=0.0,
        wall_time_s=0.0,
        route=Route.HOSTED,
        notes=notes,
    )
    write_result(
        eval_result_path(ROUND, STEP, f"{label}-{model}"),
        round_id=ROUND,
        step=STEP,
        option=f"{label}-{model}",
        vector_run=f"{A7}-full+{A3}-full",
        query_vector_run=f"{STEP}-photos",
        vectors=info,
        scores=[set_score(result, test_set="T1", half="tune", gallery=WEB_INDEX)],
        today=datetime.now(UTC).date(),
    )
