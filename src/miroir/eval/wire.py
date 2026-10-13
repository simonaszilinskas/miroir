"""What the step C and site re-rank runs send to Modal and get back (`miroir.remote.rooms`,
`miroir.remote.rerank`).

Kept apart from the scoring (`miroir.eval.detect`, `miroir.eval.site_rerank`) so the
containers import only these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from miroir.outlines import Box


@dataclass(frozen=True, slots=True)
class RoomPhoto:
    """A room photo as the page holds it: at most `ROOM_SIDE` px on its long side."""

    image_id: str
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class DetectedRoom:
    """One detector's raw boxes on one room photo, and each box's crop searched as the site
    would: cut from the photo, shrunk like a query photo, embedded by both models."""

    image_id: str
    width: int
    height: int
    boxes: tuple[Box, ...]
    a7: NDArray[np.float32]
    """One EmbeddingGemma 2 vector per box, in box order."""
    a3: NDArray[np.float32]
    """One MegaLoc vector per box."""
    detect_s: float
    """Detection alone, on the GPU, model already warm."""


@dataclass(frozen=True, slots=True)
class DetectChunk:
    rooms: tuple[DetectedRoom, ...]
    load_s: float


@dataclass(frozen=True, slots=True)
class LocateTask:
    """Find where an artwork hangs in a room photo: match each of its crops to the photo."""

    scene_id: str
    entity_id: str
    image_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Located:
    scene_id: str
    entity_id: str
    image_id: str
    """The crop that matched best."""
    inliers: int
    box: Box | None
    """The crop's corners carried into the room photo by the homography; None when no crop
    matched well enough or the shape it gives is not a plausible picture."""


@dataclass(frozen=True, slots=True)
class SiteMatchTask:
    """One query photo and the best image of each of its top entities in the site's ranking."""

    query_id: str
    candidates: tuple[str, ...]
    image_id: str | None = None
    """A POP image, read from the image volume."""
    jpeg: bytes | None = None
    """Or the photo itself (a Commons photo, which the volume does not hold)."""


@dataclass(frozen=True, slots=True)
class SiteMatches:
    __pydantic_config__ = ConfigDict(extra="forbid")

    query_id: str
    candidates: tuple[str, ...]
    inliers: tuple[int, ...]
    seconds: float
    """What re-ranking added to the search on the GPU: reading the candidates, keypoints of
    the query and of every candidate, matching."""
    read_s: float = 0.0
    features_s: float = 0.0
    match_s: float = 0.0


@dataclass(frozen=True, slots=True)
class CaptionPhoto:
    """An archive photo sent to the caption reader (`miroir.remote.caption`)."""

    image_id: str
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class CaptionChunk:
    """What the caption reader read, one line per photo ("" for none), and how long each took
    once the model was warm."""

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    seconds: tuple[float, ...]
    load_s: float
