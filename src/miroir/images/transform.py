"""Crop, turn, mirror and grey a visitor's photo before an image search.

The agentic search (`miroir.web.chat_tools.image_search`) asks for these so it can aim a
search: one carved group of an altarpiece, one painting on a room's wall, a colour photo turned
black and white to meet Mémoire's archive prints. Plain image operations (PIL), no model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image as PILImage
from PIL import ImageOps

#: A crop's sides, as shares of the photo's, must be at least this (smaller finds nothing).
MIN_SIDE = 0.05
#: Padding added around a crop, as a share of its own width and height.
MAX_PAD = 0.3
DEFAULT_PAD = 0.05
#: Turns, in degrees clockwise.
TURNS = (0, 90, 180, 270)
#: Contrast stretch: the share of darkest and lightest pixels clipped on each side, in percent.
STRETCH_CUTOFF = 1.0
JPEG_QUALITY = 90

#: The 3 by 3 grid: each cell's box, its neighbours sharing a tenth of the photo so a part that
#: sits on a grid line stays whole in one cell (the crop slice test's 3 by 3 windows).
_THIRDS = ((0.0, 0.4), (0.3, 0.7), (0.6, 1.0))
_ROWS = ("top", "middle", "bottom")
_COLUMNS = ("left", "center", "right")
CELLS: dict[str, tuple[float, float, float, float]] = {
    f"{row}-{column}": (x0, y0, x1, y1)
    for row, (y0, y1) in zip(_ROWS, _THIRDS, strict=True)
    for column, (x0, x1) in zip(_COLUMNS, _THIRDS, strict=True)
}
CELL_NAMES_FR = {
    "top-left": "en haut à gauche",
    "top-center": "en haut au centre",
    "top-right": "en haut à droite",
    "middle-left": "au milieu à gauche",
    "middle-center": "au centre",
    "middle-right": "au milieu à droite",
    "bottom-left": "en bas à gauche",
    "bottom-center": "en bas au centre",
    "bottom-right": "en bas à droite",
}


class TransformError(ValueError):
    """A transform that cannot be done (a box outside the photo, too small, a bad turn)."""


@dataclass(frozen=True, slots=True)
class Transform:
    """What to do to the photo, in this order: crop, turn, mirror, grey."""

    crop: tuple[float, float, float, float] | None = None
    """Left, top, right, bottom in shares of the photo (0 to 1), padding included."""
    rotate: int = 0
    """Degrees clockwise: 0, 90, 180 or 270."""
    mirror: bool = False
    """Flip left to right."""
    grayscale: bool = False
    stretch: bool = False
    """With grayscale: spread the greys over the whole range (faded prints)."""

    @property
    def empty(self) -> bool:
        return self == Transform()


def padded(box: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    """`box` grown by `pad` of its own size on each side, kept inside the photo. Raises
    `TransformError` when it is not a box inside the photo or is too small."""
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        msg = f"box {list(box)} is not left < right and top < bottom, all within 0 to 1"
        raise TransformError(msg)
    if not 0.0 <= pad <= MAX_PAD:
        msg = f"pad must be 0 to {MAX_PAD}"
        raise TransformError(msg)
    w, h = (x1 - x0) * pad, (y1 - y0) * pad
    out = (max(0.0, x0 - w), max(0.0, y0 - h), min(1.0, x1 + w), min(1.0, y1 + h))
    if out[2] - out[0] < MIN_SIDE or out[3] - out[1] < MIN_SIDE:
        msg = f"the box is too small: each side must be at least {MIN_SIDE:g} of the photo"
        raise TransformError(msg)
    return out


def apply(data: bytes, transform: Transform, long_side: int) -> bytes:
    """The photo with `transform` done, as a JPEG at most `long_side` on its long side."""
    if transform.rotate not in TURNS:
        msg = f"rotate must be one of {', '.join(map(str, TURNS))}"
        raise TransformError(msg)
    with PILImage.open(io.BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if transform.crop is not None:
        x0, y0, x1, y1 = transform.crop
        w, h = image.size
        image = image.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    if transform.rotate:
        # PIL turns anticlockwise.
        image = image.rotate(-transform.rotate, expand=True)
    if transform.mirror:
        image = ImageOps.mirror(image)
    if transform.grayscale:
        grey = image.convert("L")
        if transform.stretch:
            grey = ImageOps.autocontrast(grey, cutoff=STRETCH_CUTOFF)
        image = grey.convert("RGB")
    image.thumbnail((long_side, long_side), PILImage.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()
