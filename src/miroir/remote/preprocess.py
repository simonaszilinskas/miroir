"""DINO evaluation preprocessing in PIL and NumPy: the DINOv2 eval transform, written out.

Resize the shorter side to `crop * 256 / 224` (bicubic), centre-crop `crop` x `crop`, scale to
[0, 1], normalise with ImageNet mean and std. Matches torchvision's `Resize` + `CenterCrop`
as used by DINOv2's `make_classification_eval_transform` and the DINOv3 paper's AmsterTime
protocol (§D.8). No torch here, so it is unit-tested on the Mac.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_shorter_side(image: PILImage.Image, side: int) -> PILImage.Image:
    """Shorter side to `side`, aspect kept; the longer side is truncated as torchvision does."""
    width, height = image.size
    if width <= height:
        size = (side, int(side * height / width))
    else:
        size = (int(side * width / height), side)
    return image.resize(size, PILImage.Resampling.BICUBIC)


def center_crop(image: PILImage.Image, side: int) -> PILImage.Image:
    width, height = image.size
    top = round((height - side) / 2.0)
    left = round((width - side) / 2.0)
    return image.crop((left, top, left + side, top + side))


def dino_eval(image: PILImage.Image, crop: int) -> NDArray[np.float32]:
    """One image as a (3, crop, crop) float32 array, ready for a DINO model."""
    resized = resize_shorter_side(image.convert("RGB"), round(crop * 256 / 224))
    pixels = np.asarray(center_crop(resized, crop), dtype=np.float32) / 255.0
    return ((pixels - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
