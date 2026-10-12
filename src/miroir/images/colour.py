"""Is a photo black and white? A pixel statistic, no model; and a grey copy of a photo.

An archive print is grey or carries one tint (sepia, a blue cast from the scan): its colour
barely changes from pixel to pixel. A colour photo's does. `chroma_spread` measures that
change: the spread of the two colour channels of YCbCr over the image. A grey or tinted
image scores near 0; a colour photo of a building well above `GREY_BELOW`.

`greyscale` makes the grey copy step F1 embeds (experiments.md: turn query and collection
images grey before embedding).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage

#: Side, in pixels, the image is shrunk to before measuring: enough for a global statistic.
_SIDE = 128
#: Below this spread (YCbCr units, 0 to 255) a photo counts as black and white. Mémoire prints
#: score 0 (stored grey); 1% of T2 phone photos score under 3.
GREY_BELOW = 2.0
JPEG_QUALITY = 90


def chroma_spread(jpeg: bytes) -> float:
    """Standard deviation of the colour (Cb, Cr) over the pixels: 0 for grey or one tint."""
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        small = opened.convert("RGB")
        small.thumbnail((_SIDE, _SIDE), PILImage.Resampling.BILINEAR)
        ycc = np.asarray(small.convert("YCbCr"), dtype=np.float64)
    cb, cr = ycc[..., 1].ravel(), ycc[..., 2].ravel()
    return float(np.sqrt(cb.var() + cr.var()))


def is_grey(jpeg: bytes, threshold: float = GREY_BELOW) -> bool:
    return chroma_spread(jpeg) < threshold


def greyscale(jpeg: bytes) -> bytes:
    """The same image in grey (ITU-R 601 luma), as an RGB JPEG so every model reads it."""
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        grey = opened.convert("L").convert("RGB")
    out = io.BytesIO()
    grey.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def already_grey(jpeg: bytes, tolerance: float = 1.0) -> bool:
    """Every pixel already grey: R, G and B differ from the luma by under `tolerance` on
    average, so a grey copy changes nothing but JPEG noise."""
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float64)
        luma = np.asarray(opened.convert("L"), dtype=np.float64)
    return float(np.abs(rgb - luma[..., None]).mean()) < tolerance
