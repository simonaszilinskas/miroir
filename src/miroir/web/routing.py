"""How a photo search weighs its two models, and when it says "not found" (G4, T8).

The weights and cut-offs are data, not code: `routing.json` in the index bundle, written by
`miroir web build-index` (`miroir.web.bundle`). It holds a list of buckets; a query photo
goes to the first bucket whose conditions it meets:

- `photo_type`: "building" or "object", from the G4 guess (a logistic regression on the
  photo's EmbeddingGemma 2 vector, `guess.npz`);
- `greyscale`: true or false, from the photo's chroma spread (`miroir.images.colour`, the
  statistic step G5 routes on: below `grey_below` a photo is black and white or one tint).
  Lets an archive-photo bucket be switched on.

Each bucket gives EmbeddingGemma 2's weight (MegaLoc gets the rest) and the cut-off on the
top result's confidence below which the page says "Pas de correspondance sûre".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from miroir.images.colour import GREY_BELOW

ROUTING_FILE = "routing.json"
GUESS_FILE = "guess.npz"
SWITCH_AT = 0.5

Floats = NDArray[np.float32]


class PhotoType(StrEnum):
    BUILDING = "building"
    OBJECT = "object"


@dataclass(frozen=True, slots=True)
class Bucket:
    __pydantic_config__ = ConfigDict(extra="forbid")

    name: str
    a7_weight: float
    """EmbeddingGemma 2's share of the fused score; MegaLoc gets 1 - a7_weight."""
    cutoff: float
    """Below this confidence (the weighted cosines of the top result), say "not found"."""
    photo_type: PhotoType | None = None
    greyscale: bool | None = None
    note: str = ""
    """Where the weight and the cut-off come from."""

    def __post_init__(self) -> None:
        if not 0 <= self.a7_weight <= 1:
            msg = f"bucket {self.name}: a7_weight {self.a7_weight} is not in [0, 1]"
            raise ValueError(msg)

    def accepts(self, photo_type: PhotoType, chroma: float | None, grey_below: float) -> bool:
        if self.photo_type is not None and self.photo_type is not photo_type:
            return False
        if self.greyscale is None or chroma is None:
            return True
        return (chroma < grey_below) is self.greyscale


@dataclass(frozen=True, slots=True)
class TextWeights:
    """F2: words are searched two ways, mixed after min-max scaling over entities."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    a7: float = 0.9
    """EmbeddingGemma 2 text vector against the image vectors. 0.9 from step B on T7 tune
    (docs/results.md): hit@10 17.6 against 8.0 at 0.5; A7 alone scores 17.2 but loses
    names ("Pont du Gard", "Corot"), which T7 leaves out by design."""
    bm25: float = 0.1
    """French BM25 over record text (`miroir.eval.bm25`)."""
    refine: float = 0.5
    """Weight of the words when they refine a photo search (the photo counts 1). Left at 0.5:
    a commune as words lifts T1 and T2 most at 1.0 but lowers T5 (step B, results.md)."""


@dataclass(frozen=True, slots=True)
class Routing:
    __pydantic_config__ = ConfigDict(extra="forbid")

    buckets: tuple[Bucket, ...]
    grey_below: float = GREY_BELOW
    """Chroma spread under which a photo counts as black and white (`images.colour`)."""
    text: TextWeights = TextWeights()
    source: str = ""
    """How the file was made: index, sets, dates."""

    def __post_init__(self) -> None:
        if not self.buckets:
            msg = "routing needs at least one bucket"
            raise ValueError(msg)

    def choose(self, photo_type: PhotoType, chroma: float | None) -> Bucket:
        """The first bucket the photo fits. With no chroma spread, greyscale is not asked."""
        for bucket in self.buckets:
            if bucket.accepts(photo_type, chroma, self.grey_below):
                return bucket
        msg = f"no routing bucket takes a {photo_type} photo (chroma spread {chroma})"
        raise ValueError(msg)


_ROUTING = TypeAdapter(Routing)


def read_routing(path: Path) -> Routing:
    return _ROUTING.validate_json(path.read_bytes())


def write_routing(path: Path, routing: Routing) -> None:
    path.write_bytes(_ROUTING.dump_json(routing, indent=2) + b"\n")


@dataclass(frozen=True, slots=True)
class Guess:
    """G4's building-or-object guess: p(building) from a unit EmbeddingGemma 2 vector."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coef: NDArray[np.float64]
    bias: float

    def p_building(self, vector: Floats) -> float:
        z = (np.asarray(vector, dtype=np.float64) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.coef + self.bias))))

    def photo_type(self, vector: Floats) -> tuple[PhotoType, float]:
        p = self.p_building(vector)
        return (PhotoType.BUILDING if p > SWITCH_AT else PhotoType.OBJECT), p


def write_guess(path: Path, guess: Guess) -> None:
    with path.open("wb") as out:
        np.savez(
            out, mean=guess.mean, scale=guess.scale, coef=guess.coef, bias=np.array([guess.bias])
        )


def read_guess(path: Path) -> Guess:
    with np.load(path, allow_pickle=False) as data:
        return Guess(
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            coef=np.asarray(data["coef"], dtype=np.float64),
            bias=float(np.asarray(data["bias"]).reshape(-1)[0]),  # 0-d or 1-d
        )
