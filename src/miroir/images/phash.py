"""Perceptual hash: a 64-bit fingerprint that stays the same when an image is resized or
recompressed, so two copies of one scan differ by a few bits. No model: a grey 32 x 32
thumbnail, its 2-D DCT, and one bit per low frequency (above or below their median).
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

_SIDE = 32
_LOW = 8


def _dct_matrix(n: int) -> NDArray[np.float64]:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    matrix[0] /= np.sqrt(2)
    return matrix


_DCT = _dct_matrix(_SIDE)


def phash(image: bytes) -> int:
    """64-bit DCT hash of an encoded image (JPEG, PNG...)."""
    with PILImage.open(io.BytesIO(image)) as opened:
        grey = opened.convert("L").resize((_SIDE, _SIDE), PILImage.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.float64)
    low = (_DCT @ pixels @ _DCT.T)[:_LOW, :_LOW].flatten()
    bits = low > np.median(low[1:])  # the median leaves out the mean brightness term
    return int("".join("1" if b else "0" for b in bits), 2)


def distance(a: int, b: int) -> int:
    """Bits that differ between two hashes: 0 for copies, about 32 for unrelated images."""
    return (a ^ b).bit_count()
