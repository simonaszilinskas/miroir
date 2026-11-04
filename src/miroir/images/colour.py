"""Is a photo black and white? A pixel statistic, no model.

An archive print is grey or carries one tint (sepia, a blue cast from the scan): its colour
barely changes from pixel to pixel. A colour photo's does. `chroma_spread` measures that
change: the spread of the two colour channels of YCbCr over the image. A grey or tinted
image scores near 0; a colour photo of a building well above `GREY_BELOW`.
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


def chroma_spread(jpeg: bytes) -> float:
    """Standard deviation of the colour (Cb, Cr) over the pixels: 0 for grey or one tint."""
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        return chroma_of(opened)


def chroma_of(image: PILImage.Image) -> float:
    """`chroma_spread` of an opened image (left as it is)."""
    small = image.convert("RGB")
    small.thumbnail((_SIDE, _SIDE), PILImage.Resampling.BILINEAR)
    ycc = np.asarray(small.convert("YCbCr"), dtype=np.float64)
    cb, cr = ycc[..., 1].ravel(), ycc[..., 2].ravel()
    return float(np.sqrt(cb.var() + cr.var()))
