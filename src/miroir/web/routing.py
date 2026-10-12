"""How a photo search weighs its two models, and when it says "not found" (G4, T8).

The weights and cut-offs are data, not code: `routing.json` in the index bundle, written by
`miroir web build-index` (`miroir.web.bundle`). It holds a list of buckets; a query photo
goes to the first bucket whose conditions it meets:

- `photo_type`: "building" or "object", from the G4 guess (a logistic regression on the
  photo's EmbeddingGemma 2 vector, `guess.npz`);
- `greyscale`: true or false, from the photo's colourfulness (`colourfulness`, below
  `greyscale_max` counts as greyscale or sepia). Lets an archive-photo bucket be switched on.

Each bucket gives EmbeddingGemma 2's weight (MegaLoc gets the rest) and the cut-off on the
top result's confidence below which the page says "Pas de correspondance sûre".
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from pydantic import ConfigDict, TypeAdapter

ROUTING_FILE = "routing.json"
GUESS_FILE = "guess.npz"
SWITCH_AT = 0.5
#: Side of the copy colourfulness is measured on.
COLOUR_SIDE = 128

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

    def accepts(self, photo_type: PhotoType, colourfulness: float | None, grey_max: float) -> bool:
        if self.photo_type is not None and self.photo_type is not photo_type:
            return False
        if self.greyscale is None or colourfulness is None:
            return True
        return (colourfulness <= grey_max) is self.greyscale


@dataclass(frozen=True, slots=True)
class TextWeights:
    """F2: words are searched two ways, mixed after min-max scaling over entities."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    a7: float = 0.5
    """EmbeddingGemma 2 text vector against the image vectors."""
    bm25: float = 0.5
    """French BM25 over record text (`miroir.eval.bm25`)."""
    refine: float = 0.5
    """Weight of the words when they refine a photo search (the photo counts 1)."""


@dataclass(frozen=True, slots=True)
class Routing:
    __pydantic_config__ = ConfigDict(extra="forbid")

    buckets: tuple[Bucket, ...]
    greyscale_max: float = 12.0
    """Colourfulness at or below which a photo counts as greyscale or sepia."""
    text: TextWeights = TextWeights()
    source: str = ""
    """How the file was made: index, sets, dates."""

    def __post_init__(self) -> None:
        if not self.buckets:
            msg = "routing needs at least one bucket"
            raise ValueError(msg)

    def choose(self, photo_type: PhotoType, colourfulness: float | None) -> Bucket:
        """The first bucket the photo fits. With no colourfulness, greyscale is not asked."""
        for bucket in self.buckets:
            if bucket.accepts(photo_type, colourfulness, self.greyscale_max):
                return bucket
        msg = f"no routing bucket takes a {photo_type} photo (colourfulness {colourfulness})"
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
            bias=float(data["bias"][0]),
        )


def colourfulness(jpeg: bytes) -> float:
    """Hasler and Süsstrunk's colourfulness (2003) on a small copy: 0 for greyscale.

    Sepia prints score low too (one hue, little spread); colour photos score above 30.
    """
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        small = opened.convert("RGB")
        small.thumbnail((COLOUR_SIDE, COLOUR_SIDE))
        pixels = np.asarray(small, dtype=np.float64)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    spread = np.hypot(rg.std(), yb.std())
    centre = np.hypot(rg.mean(), yb.mean())
    return float(spread + 0.3 * centre)
