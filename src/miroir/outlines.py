"""Object outlines (F4, experiments.md step C): the detectors we compare, and which of their
boxes the photo view shows.

A detector (`miroir.remote.detect`, Modal only) returns raw boxes: every box it is at least a
little sure of, in shares of the photo's width and height. `pick_outlines` keeps the ones worth
drawing (sure enough, not twice the same object, not the whole photo, at most `ALL_OUTLINES`)
and puts first the ones a visitor most likely means: big and near the photo's centre.
The same rule runs in the GPU container for the website and on the Mac when scoring T3, so the
outlines scored are the outlines shown.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ConfigDict

#: Step C's prompts, in experiments.md order.
PROMPTS: tuple[str, ...] = (
    "painting",
    "sculpture",
    "statue",
    "vase",
    "furniture",
    "building",
    "stained glass window",
)
#: Longest side, in pixels, of the photo a detector sees. The page sends this size.
DETECT_SIDE = 1024
#: Outlines drawn at first on one photo (functional spec §5: a person picks one by eye);
#: `MAX_OUTLINES` in static/app.js. 20 still reads at 375 px on the densest room photo.
MAX_OUTLINES = 20
#: Outlines sent: past the first `MAX_OUTLINES`, the page shows the rest on request ("Afficher
#: tous les objets"), for walls hung with dozens of pictures (docs/results.md, step C).
ALL_OUTLINES = 40


#: A box holding at least two other outlines this much (share of their area) is a group.
GROUP_INSIDE = 0.8


class DetectorKind(StrEnum):
    OWLV2 = "owlv2"
    GROUNDING_DINO = "grounding-dino"
    QWEN3_VL = "qwen3-vl"


@dataclass(frozen=True, slots=True)
class DetectorOption:
    """One detector of step C."""

    id: str
    kind: DetectorKind
    model: str
    raw_threshold: float
    """Lowest score kept from the model: low, so the outline rule can be tuned afterwards."""


DETECTORS: dict[str, DetectorOption] = {
    o.id: o
    for o in (
        DetectorOption(
            "C1-owlv2-base", DetectorKind.OWLV2, "google/owlv2-base-patch16-ensemble", 0.03
        ),
        DetectorOption(
            "C1-owlv2-large", DetectorKind.OWLV2, "google/owlv2-large-patch14-ensemble", 0.03
        ),
        DetectorOption(
            "C2-gdino-tiny", DetectorKind.GROUNDING_DINO, "IDEA-Research/grounding-dino-tiny", 0.1
        ),
        DetectorOption(
            "C2-gdino-base", DetectorKind.GROUNDING_DINO, "IDEA-Research/grounding-dino-base", 0.1
        ),
        DetectorOption("C4-qwen3-vl-8b", DetectorKind.QWEN3_VL, "Qwen/Qwen3-VL-8B-Instruct", 0.0),
    )
}


def detector(option_id: str) -> DetectorOption:
    found = DETECTORS.get(option_id)
    if found is None:
        msg = f"unknown detector {option_id!r}; known: {', '.join(DETECTORS)}"
        raise ValueError(msg)
    return found


@dataclass(frozen=True, slots=True)
class Box:
    """A box in shares of the photo's width (x) and height (y), 0 to 1."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            msg = f"not a box inside the photo: {self}"
            raise ValueError(msg)

    @property
    def area(self) -> float:
        """Share of the photo's area."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) in pixels of a photo of that size, at least 1 px wide."""
        left, top = int(self.x0 * width), int(self.y0 * height)
        right = max(left + 1, min(width, round(self.x1 * width)))
        bottom = max(top + 1, min(height, round(self.y1 * height)))
        return left, top, right, bottom


def clipped(
    x0: float, y0: float, x1: float, y1: float, *, score: float = 1.0, label: str = ""
) -> Box | None:
    """A box from coordinates that may spill over the photo's edge; None when nothing is left."""
    x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1, y1, score, label)


def intersection(a: Box, b: Box) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class OutlineRule:
    """Which raw boxes become outlines. Tuned on T3 tune (docs/results.md, step C)."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    min_score: float
    overlap: float = 0.5
    """A box overlapping a surer one by more than this (IoU) is the same object."""
    inside: float = 0.9
    """A box this much inside a surer one (share of its own area) is part of it."""
    min_side: float = 0.03
    """Shorter side of a box over the photo's shorter side: smaller is too small to tap."""
    whole: float = 0.8
    """A box this large (share of the photo) is the photo itself: "search the whole photo"."""
    lone: float = 0.4
    """When the surest box covers this much of the photo, show none: the photo shows one
    object (with its parts and stand), and "search the whole photo" does the job."""
    max_outlines: int = ALL_OUTLINES


def salience(box: Box) -> float:
    """How likely a visitor means this box: people frame what they care about, so big boxes
    near the photo's centre come first (docs/results.md, step C, order of the outlines)."""
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    near = 1 - math.hypot(cx - 0.5, cy - 0.5) / math.hypot(0.5, 0.5)
    return math.sqrt(box.area) * near


def pick_outlines(
    boxes: Sequence[Box], rule: OutlineRule, *, width: int, height: int
) -> tuple[Box, ...]:
    """The outlines to draw (see `OutlineRule`): the surest boxes, put in the order a visitor
    most likely means them (`salience`), boxes around several works last."""
    short = min(width, height)
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: -b.score):
        if len(kept) >= rule.max_outlines:
            break
        if box.score < rule.min_score or box.area >= rule.whole:
            continue
        side = min((box.x1 - box.x0) * width, (box.y1 - box.y0) * height)
        if side < rule.min_side * short:
            continue
        if any(
            iou(box, k) > rule.overlap or intersection(box, k) >= rule.inside * box.area
            for k in kept
        ):
            continue
        kept.append(box)
    if kept and kept[0].area >= rule.lone:
        return ()
    return order_outlines(kept, salience)


def order_outlines(outlines: Sequence[Box], key: Callable[[Box], float]) -> tuple[Box, ...]:
    """Single works by `key`, highest first; a box around several others (a whole wall)
    after them, so the first view shows works, not groups."""
    ordered = sorted(outlines, key=lambda b: -key(b))
    groups = {
        b
        for b in ordered
        if sum(
            intersection(other, b) >= GROUP_INSIDE * other.area
            for other in ordered
            if other is not b
        )
        >= 2
    }
    return tuple(b for b in ordered if b not in groups) + tuple(b for b in ordered if b in groups)


#: The rule the website uses, with the detector it was tuned for (docs/results.md, step C).
SITE_DETECTOR = "C2-gdino-tiny"
SITE_RULE = OutlineRule(min_score=0.2, overlap=0.3, inside=1.01, min_side=0.03)


@dataclass(frozen=True, slots=True)
class Outlines:
    """What the photo view draws (`POST /api/detect`)."""

    boxes: tuple[Box, ...]
    detect_ms: float = 0.0
    """Detection and the outline rule, on the GPU."""


def fake_outlines(data: bytes) -> Outlines:
    """Stand-in outlines for `miroir web dev` and tests (no model): three frames on a wall,
    placed from a hash of the photo so different photos get different outlines."""
    seed = int.from_bytes(hashlib.sha256(data).digest()[:2], "big") / 65535
    top = 0.15 + 0.2 * seed
    boxes = (
        Box(0.08, top, 0.3, top + 0.35, 0.9, "painting"),
        Box(0.4, top + 0.05, 0.58, top + 0.3, 0.8, "painting"),
        Box(0.68, top - 0.05, 0.92, top + 0.45, 0.7, "painting"),
    )
    return Outlines(boxes)
