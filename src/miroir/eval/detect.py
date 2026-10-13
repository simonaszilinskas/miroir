"""Step C: object outlines on the Jeu de Paume room photos, scored end to end on T3.

Technical spec §7 (T3): detect on the room photo, search each box, hit if the artwork is in
the top 5. Here a box is searched exactly as the website searches a tapped outline: cut from
the photo as the page holds it (`ROOM_SIDE`), shrunk like any query photo, embedded by both
models on Modal, then ranked on the Mac against the site's own index bundle (full index,
routing and all, `miroir.web.bundle.site_search`).

Which outline would a person tap for a given artwork? The one that best covers it. The
export has no boxes, so we find where each T3 artwork hangs by matching its crops to the room
photo (`miroir.remote.rooms.locate_artworks`). Then, per tune query:

- **tapped**: the outline with the highest IoU with the artwork's place (at least `TAP_IOU`)
  is searched; its rank of the artwork is the query's rank. No such outline: a miss.
- **any box**: the best rank over every outline of the photo (a generous reading).
- **box recall**: an outline covers the artwork at IoU `RECALL_IOU` or more.

An artwork hung in two room photos counts its better photo. Queries whose artwork could not
be located fall back to "any box" and are counted.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from pydantic import ConfigDict, TypeAdapter

from miroir.config import WEB_INDEX, Settings, eval_result_path
from miroir.eval.results import set_score, write_result
from miroir.eval.scoring import QueryScore, ScoreResult
from miroir.eval.sets import load_queries
from miroir.eval.vectors import Route, RunInfo
from miroir.eval.wire import DetectChunk, DetectedRoom, Located, LocateTask, RoomPhoto
from miroir.images.fetch import Fetched, to_jpeg
from miroir.jobs.modal_run import PaidCall, paid_call
from miroir.jobs.modal_spec import CHUNK_TIMEOUT_S, L4, L4_DINO
from miroir.jobs.spend import SpendKind
from miroir.outlines import (
    DETECT_SIDE,
    MAX_OUTLINES,
    SITE_RULE,
    Box,
    DetectorOption,
    OutlineRule,
    detector,
    iou,
    pick_outlines,
)
from miroir.records.model import Record
from miroir.records.store import read_records
from miroir.web.bundle import SiteBundle, site_search

logger = logging.getLogger(__name__)

#: Longest side of a room photo as the page holds it (`KEEP_SIDE` in static/app.js).
ROOM_SIDE = 2048
#: An outline counts as the one a person taps for an artwork from this IoU.
TAP_IOU = 0.3
#: Box recall: an outline covers the artwork from this IoU (the usual detection bar).
RECALL_IOU = 0.5
#: Entities kept per searched box; an artwork further down counts as rank TOP + 1.
TOP = 50
LOCATED_FILE = "located.json"
ROOMS_DIR = "rooms"

Floats = NDArray[np.float32]


# --- the room photos and where the artworks hang -------------------------------------------


def room_scenes(records: Iterable[Record]) -> dict[str, str]:
    """Image id -> URL of every Jeu de Paume room photo an artwork links to (the T3 rule)."""
    records = list(records)
    linked = {ref for r in records for ref in r.linked_refs}
    return {
        image.image_id: image.url
        for r in records
        if r.ref.startswith("JDP-") and not r.domain and r.ref in linked
        for image in r.images
    }


def fetch_rooms(settings: Settings, scenes: Mapping[str, str]) -> list[RoomPhoto]:
    """The room photos at `ROOM_SIDE`, fetched from POP once and kept in the data dir."""
    folder = settings.detect_dir / ROOMS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    photos: list[RoomPhoto] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for image_id, url in sorted(scenes.items()):
            path = folder / f"{image_id}.jpg"
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(to_jpeg(response.content, ROOM_SIDE))
            photos.append(RoomPhoto(image_id, path.read_bytes()))
    return photos


@dataclass(frozen=True, slots=True)
class T3Item:
    query_id: str
    entity_id: str
    scenes: tuple[str, ...]


def t3_items(settings: Settings) -> list[T3Item]:
    return [
        T3Item(q.query_id, q.entity_id, q.scene_image_ids)
        for q in load_queries(settings, "T3", "tune")
    ]


def locate_tasks(items: Sequence[T3Item], records: Iterable[Record]) -> list[LocateTask]:
    """One task per (room photo, artwork): every crop of the artwork, matched to the photo."""
    crops: dict[str, list[str]] = {}
    for record in records:
        crops.setdefault(record.entity_id, []).extend(i.image_id for i in record.images)
    pairs = sorted({(scene, item.entity_id) for item in items for scene in item.scenes})
    return [LocateTask(scene, entity, tuple(crops.get(entity, ()))) for scene, entity in pairs]


_LOCATED = TypeAdapter(list[Located])


def save_located(settings: Settings, located: Sequence[Located]) -> Path:
    path = settings.detect_dir / LOCATED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_LOCATED.dump_json(list(located), indent=1) + b"\n")
    return path


def read_located(settings: Settings) -> dict[tuple[str, str], Box]:
    """(room photo, entity) -> where the artwork hangs, for the artworks found."""
    path = settings.detect_dir / LOCATED_FILE
    if not path.exists():
        msg = f"no {path}: run `miroir detect locate` first"
        raise FileNotFoundError(msg)
    return {
        (x.scene_id, x.entity_id): x.box
        for x in _LOCATED.validate_json(path.read_bytes())
        if x.box is not None
    }


# --- one detector's boxes, their vectors and their search results --------------------------


@dataclass(frozen=True, slots=True)
class RoomBoxes:
    __pydantic_config__ = ConfigDict(extra="forbid")

    image_id: str
    width: int
    height: int
    boxes: tuple[Box, ...]
    detect_s: float
    ranked: tuple[tuple[str, ...], ...] = ()
    """Per box, the site's top `TOP` entities for its crop."""


@dataclass(frozen=True, slots=True)
class Detections:
    __pydantic_config__ = ConfigDict(extra="forbid")

    label: str
    model: str
    index: str
    created: datetime
    load_s: float
    cost_usd: float
    gallery_images: int
    gallery_entities: int
    rooms: tuple[RoomBoxes, ...] = field(default_factory=tuple[RoomBoxes, ...])


_DETECTIONS = TypeAdapter(Detections)


def detections_path(settings: Settings, label: str) -> Path:
    return settings.detect_dir / f"{label}.json"


def rank_rooms(
    bundle: SiteBundle, chunk: DetectChunk, *, label: str, model: str, cost_usd: float
) -> Detections:
    """Search every box's crop with the site's ranking; keep each box's top entities."""
    rooms: list[RoomBoxes] = []
    for room in chunk.rooms:
        rooms.append(
            RoomBoxes(
                image_id=room.image_id,
                width=room.width,
                height=room.height,
                boxes=room.boxes,
                detect_s=room.detect_s,
                ranked=_rank(bundle, room),
            )
        )
        logger.info("%s: %d boxes searched", room.image_id, len(room.boxes))
    return Detections(
        label=label,
        model=model,
        index=WEB_INDEX,
        created=datetime.now(UTC),
        load_s=chunk.load_s,
        cost_usd=cost_usd,
        gallery_images=len(bundle.catalog.images),
        gallery_entities=bundle.catalog.entity_count,
        rooms=tuple(rooms),
    )


def _rank(bundle: SiteBundle, room: DetectedRoom) -> tuple[tuple[str, ...], ...]:
    if not room.boxes:
        return ()
    hits = site_search(
        bundle,
        _unit(room.a7),
        _unit(room.a3),
        hidden=[()] * len(room.boxes),
        right=[()] * len(room.boxes),
        top=TOP,
    )
    return tuple(h.entities for h in hits)


def _unit(rows: Floats) -> Floats:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def save_detections(settings: Settings, detections: Detections) -> Path:
    path = detections_path(settings, detections.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_DETECTIONS.dump_json(detections, indent=1) + b"\n")
    return path


def read_detections(settings: Settings, label: str) -> Detections:
    path = detections_path(settings, label)
    if not path.exists():
        msg = f"no {path}: run `miroir detect rooms --option {label}` first"
        raise FileNotFoundError(msg)
    return _DETECTIONS.validate_json(path.read_bytes())


# --- scoring --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutlineScore:
    rule: OutlineRule
    tapped: ScoreResult
    any_box: ScoreResult
    box_recall: float
    """Share of located tune artworks with an outline at IoU >= RECALL_IOU."""
    located: int
    """Tune artworks found in at least one of their room photos."""
    outlines_per_photo: float
    detect_ms: tuple[float, float]
    """Median and 95th percentile detection time per photo, warm GPU."""

    def hit_at(self, k: int, *, tapped: bool = True) -> float:
        scores = (self.tapped if tapped else self.any_box).per_query
        return float(np.mean([q.rank <= k for q in scores]))


def score_outlines(
    detections: Detections,
    items: Sequence[T3Item],
    located: Mapping[tuple[str, str], Box],
    rule: OutlineRule,
) -> OutlineScore:
    rooms = {r.image_id: r for r in detections.rooms}
    shown: dict[str, list[tuple[Box, tuple[str, ...]]]] = {}
    for room in detections.rooms:
        picked = set(pick_outlines(room.boxes, rule, width=room.width, height=room.height))
        shown[room.image_id] = [
            (b, r) for b, r in zip(room.boxes, room.ranked, strict=True) if b in picked
        ]
    tapped: list[QueryScore] = []
    any_box: list[QueryScore] = []
    recalled = 0
    found_any = 0
    for item in items:
        best_tap = best_any = TOP + 1
        was_located = covered = False
        for scene in item.scenes:
            if scene not in rooms:
                continue
            outlines = shown[scene]
            for _, ranked in outlines:
                best_any = min(best_any, _rank_of(item.entity_id, ranked))
            place = located.get((scene, item.entity_id))
            if place is None:
                continue
            was_located = True
            overlaps = [(iou(b, place), ranked) for b, ranked in outlines]
            covered = covered or any(o >= RECALL_IOU for o, _ in overlaps)
            best = max(overlaps, default=(0.0, ()), key=lambda x: x[0])
            if best[0] >= TAP_IOU:
                best_tap = min(best_tap, _rank_of(item.entity_id, best[1]))
        if not was_located:
            best_tap = best_any
        found_any += was_located
        recalled += covered
        tapped.append(QueryScore(item.query_id, best_tap, 1.0 / best_tap))
        any_box.append(QueryScore(item.query_id, best_any, 1.0 / best_any))
    times = [1000 * r.detect_s for r in detections.rooms]
    return OutlineScore(
        rule=rule,
        tapped=_result(tapped, detections),
        any_box=_result(any_box, detections),
        box_recall=recalled / found_any if found_any else 0.0,
        located=found_any,
        outlines_per_photo=float(np.mean([len(v) for v in shown.values()])),
        detect_ms=(float(np.percentile(times, 50)), float(np.percentile(times, 95))),
    )


def _rank_of(entity: str, ranked: Sequence[str]) -> int:
    return ranked.index(entity) + 1 if entity in ranked else TOP + 1


def _result(scores: list[QueryScore], detections: Detections) -> ScoreResult:
    return ScoreResult(
        per_query=tuple(scores),
        skipped_no_answer=0,
        gallery_images=detections.gallery_images,
        gallery_entities=detections.gallery_entities,
        ap_max_rank=TOP,
    )


#: The grid tuned on T3 tune: score thresholds are read from each detector's own scores.
OVERLAPS = (0.3, 0.5)
INSIDE = (0.8, 1.01)
MIN_SIDES = (0.0, 0.03)
SCORE_QUANTILES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True, slots=True)
class RuleScore:
    """One outline rule scored twice: every outline sent, and the first view (the
    `MAX_OUTLINES` surest, before "Afficher tous les objets")."""

    every: OutlineScore
    first: OutlineScore


def tune_rule(
    detections: Detections, items: Sequence[T3Item], located: Mapping[tuple[str, str], Box]
) -> list[RuleScore]:
    """Every rule of the grid, best first: tapped hit@5 with every outline, then in the first
    view, then fewer outlines (a photo with fewer, surer outlines is easier to read)."""
    scores = np.asarray([b.score for r in detections.rooms for b in r.boxes])
    thresholds = sorted({round(float(np.quantile(scores, q)), 4) for q in SCORE_QUANTILES})
    tried: list[RuleScore] = []
    for threshold, overlap, inside, side in product(thresholds, OVERLAPS, INSIDE, MIN_SIDES):
        rule = OutlineRule(min_score=threshold, overlap=overlap, inside=inside, min_side=side)
        first = replace(rule, max_outlines=MAX_OUTLINES)
        tried.append(
            RuleScore(
                score_outlines(detections, items, located, rule),
                score_outlines(detections, items, located, first),
            )
        )
    tried.sort(key=lambda s: (-s.every.hit_at(5), -s.first.hit_at(5), s.every.outlines_per_photo))
    return tried


def draw(photo: bytes, boxes: Sequence[Box], places: Sequence[Box] = ()) -> bytes:
    """The photo with outlines (white) and located artworks (red), to look at by eye."""
    from PIL import ImageDraw  # noqa: PLC0415

    with PILImage.open(io.BytesIO(photo)) as opened:
        image = opened.convert("RGB")
    pen = ImageDraw.Draw(image)
    for colour, group in (("white", boxes), ("red", places)):
        for box in group:
            pen.rectangle(box.pixels(image.width, image.height), outline=colour, width=3)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()


def summary(score: OutlineScore) -> dict[str, object]:
    return {
        "rule": json.loads(TypeAdapter(OutlineRule).dump_json(score.rule)),
        "tapped_hit@1": score.hit_at(1),
        "tapped_hit@5": score.hit_at(5),
        "tapped_hit@10": score.hit_at(10),
        "any_box_hit@5": score.hit_at(5, tapped=False),
        "box_recall@0.5": score.box_recall,
        "located": score.located,
        "outlines_per_photo": score.outlines_per_photo,
        "detect_ms_median": score.detect_ms[0],
        "detect_ms_p95": score.detect_ms[1],
    }


# --- the runs on Modal ----------------------------------------------------------------------


class SuppliesJpegs(Protocol):
    def fetch_all(self, image_ids: Iterable[str]) -> Iterable[Fetched]: ...


class RoomCalls(Protocol):
    """The remote calls of step C (`miroir.jobs.modal_app.ModalCalls`)."""

    def detect_rooms(
        self,
        option: DetectorOption | None,
        photos: Sequence[RoomPhoto],
        given: Mapping[str, tuple[Box, ...]] | None = None,
    ) -> DetectChunk: ...

    def locate_artworks(
        self, scenes: Sequence[RoomPhoto], tasks: Sequence[LocateTask]
    ) -> list[Located]: ...


#: The label of the "detector" whose boxes are the located artworks themselves: what a
#: perfect detector would give, cut and searched the same way.
LOCATED_LABEL = "C0-located"


def run_locate(settings: Settings, calls: RoomCalls, paid: PaidCall) -> list[Located]:
    """Find every T3 tune artwork in its room photos, save it, and draw it for a look."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    photos = fetch_rooms(settings, room_scenes(records))
    items = t3_items(settings)
    tasks = locate_tasks(items, records)
    done = paid_call(
        paid,
        lambda: calls.locate_artworks(photos, tasks),
        machine=L4,
        timeout_s=CHUNK_TIMEOUT_S,
        kind=SpendKind.RUN,
        option="C",
        run="C-locate",
        model="aliked+lightglue",
        items=len(tasks),
    )
    save_located(settings, done.result)
    look = settings.detect_dir / "look"
    look.mkdir(parents=True, exist_ok=True)
    for photo in photos:
        places = [x.box for x in done.result if x.scene_id == photo.image_id and x.box]
        (look / f"{photo.image_id}-located.jpg").write_bytes(draw(photo.jpeg, (), places))
    found = sum(x.box is not None for x in done.result)
    logger.info("located %d of %d (room, artwork) pairs, $%.4f", found, len(tasks), done.cost_usd)
    return done.result


def run_detector(settings: Settings, calls: RoomCalls, paid: PaidCall, label: str) -> Detections:
    """Detect on every room photo with option `label` (or embed the located artworks for
    `LOCATED_LABEL`), search every box with the site's ranking, save."""
    records = read_records(settings.records_parquet, settings.images_parquet)
    photos = fetch_rooms(settings, room_scenes(records))
    option: DetectorOption | None = None
    given: dict[str, tuple[Box, ...]] = {}
    if label == LOCATED_LABEL:
        for (scene, _), box in read_located(settings).items():
            given[scene] = (*given.get(scene, ()), box)
        model = "located crops (ALIKED + LightGlue)"
    else:
        option = detector(label)
        model = option.model
    done = paid_call(
        paid,
        lambda: calls.detect_rooms(option, photos, given),
        machine=L4_DINO,
        timeout_s=CHUNK_TIMEOUT_S,
        kind=SpendKind.RUN,
        option="C",
        run=f"C-{label}",
        model=model,
        items=len(photos),
    )
    bundle = SiteBundle.load(settings.web_index_dir(WEB_INDEX))
    detections = rank_rooms(bundle, done.result, label=label, model=model, cost_usd=done.cost_usd)
    save_detections(settings, detections)
    logger.info("%s: $%.4f, model load %.0f s", label, done.cost_usd, done.result.load_s)
    return detections


def write_scores(settings: Settings, label: str, *, round_id: str = "R3") -> list[RuleScore]:
    """Tune the outline rule for one detector on T3 tune; write its result files: `C-<label>`
    for the first view, `C-<label>-all` with every outline shown."""
    detections = read_detections(settings, label)
    items = t3_items(settings)
    tried = tune_rule(detections, items, read_located(settings))
    best = tried[0]
    for option, score in ((label, best.first), (f"{label}-all", best.every)):
        info = RunInfo(
            option=label.split("-", 1)[0],
            model=detections.model,
            resolution=DETECT_SIDE,
            pooling="boxes",
            gallery=f"web-{detections.index}",
            created=detections.created.date(),
            cost_usd=detections.cost_usd,
            wall_time_s=detections.load_s,
            route=Route.HOSTED,
            notes=(
                f"Modal L4. Outlines: {json.dumps(summary(score), ensure_ascii=False)}. Rank is "
                f"the tapped outline's (IoU >= {TAP_IOU} with the located artwork); {TOP + 1} = "
                f"not in its top {TOP} or no outline there."
            ),
        )
        write_result(
            eval_result_path(round_id, "C", option),
            round_id=round_id,
            step="C",
            option=option,
            vector_run=f"web-{detections.index}",
            query_vector_run=f"C-{label}-rooms",
            vectors=info,
            scores=[set_score(score.tapped, test_set="T3", half="tune", gallery=info.gallery)],
            today=datetime.now(UTC).date(),
        )
    detail = eval_result_path(round_id, "C", label).parent / "C-detail" / f"{label}.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        json.dumps(
            {
                "label": label,
                "model": detections.model,
                "chosen_every": summary(best.every),
                "chosen_first_view": summary(best.first),
                "any_box": {
                    "per_query": [[q.query_id, q.rank] for q in best.every.any_box.per_query]
                },
                "grid_top_10": [
                    {"every": summary(s.every), "first_hit@5": s.first.hit_at(5)}
                    for s in tried[:10]
                ],
                "grid_size": len(tried),
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    look = settings.detect_dir / "look"
    look.mkdir(parents=True, exist_ok=True)
    for room in detections.rooms:
        shown = pick_outlines(room.boxes, best.every.rule, width=room.width, height=room.height)
        photo = settings.detect_dir / ROOMS_DIR / f"{room.image_id}.jpg"
        if photo.exists():
            (look / f"{room.image_id}-{label}.jpg").write_bytes(draw(photo.read_bytes(), shown))
    return tried


#: Single-object photos checked for stray outlines, per set.
SINGLE_PER_SET = 20
_SINGLE = TypeAdapter(dict[str, list[RoomBoxes]])


def single_path(settings: Settings, label: str) -> Path:
    return settings.detect_dir / f"{label}-single.json"


def read_single(settings: Settings, label: str) -> dict[str, list[RoomBoxes]]:
    return _SINGLE.validate_json(single_path(settings, label).read_bytes())


def run_single_objects(
    settings: Settings,
    calls: RoomCalls,
    paid: PaidCall,
    label: str,
    commons: SuppliesJpegs,
) -> dict[str, list[int]]:
    """Outlines the site would draw on photos of one object (T5 museum views, T2b Commons
    photos, tune halves): ideally none, since "search the whole photo" already does the job.
    Returns, per set, the number of outlines on each photo."""
    option = detector(label)
    photos: dict[str, list[RoomPhoto]] = {"T5": [], "T2b": []}
    records = {
        i.image_id: i.url
        for r in read_records(settings.records_parquet, settings.images_parquet)
        for i in r.images
    }
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for q in load_queries(settings, "T5", "tune")[:SINGLE_PER_SET]:
            response = client.get(records[q.query_image_id])
            if response.is_success:
                photos["T5"].append(
                    RoomPhoto(q.query_image_id, to_jpeg(response.content, ROOM_SIDE))
                )
    ids = [q.query_image_id for q in load_queries(settings, "T2b", "tune")[:SINGLE_PER_SET]]
    photos["T2b"] = [RoomPhoto(f.image_id, f.jpeg) for f in commons.fetch_all(ids) if f.jpeg]
    every = [p for group in photos.values() for p in group]
    done = paid_call(
        paid,
        lambda: calls.detect_rooms(option, every, {}),
        machine=L4_DINO,
        timeout_s=CHUNK_TIMEOUT_S,
        kind=SpendKind.RUN,
        option="C",
        run=f"C-{label}-single",
        model=option.model,
        items=len(every),
    )
    rooms = {r.image_id: r for r in done.result.rooms}
    kept = {
        set_id: [
            RoomBoxes(
                p.image_id,
                rooms[p.image_id].width,
                rooms[p.image_id].height,
                rooms[p.image_id].boxes,
                rooms[p.image_id].detect_s,
            )
            for p in group
            if p.image_id in rooms
        ]
        for set_id, group in photos.items()
    }
    single_path(settings, label).write_bytes(_SINGLE.dump_json(kept, indent=1))
    out: dict[str, list[int]] = {}
    for set_id, group in photos.items():
        out[set_id] = [
            len(
                pick_outlines(
                    rooms[p.image_id].boxes,
                    SITE_RULE,
                    width=rooms[p.image_id].width,
                    height=rooms[p.image_id].height,
                )
            )
            for p in group
            if p.image_id in rooms
        ]
    logger.info("single objects, %s: %s ($%.4f)", label, out, done.cost_usd)
    return out
