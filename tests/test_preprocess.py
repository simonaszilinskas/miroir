"""DINO preprocessing: the eval transform written in PIL and NumPy (no torch on the Mac)."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from miroir.remote.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    center_crop,
    dino_eval,
    resize_shorter_side,
    squash,
)


def test_shorter_side_goes_to_256_and_aspect_is_kept() -> None:
    wide = PILImage.new("RGB", (1024, 512))
    assert resize_shorter_side(wide, 256).size == (512, 256)
    tall = PILImage.new("RGB", (300, 1000))
    assert resize_shorter_side(tall, 256).size == (256, 853)  # truncated, as torchvision does


def test_center_crop_takes_the_middle() -> None:
    image = PILImage.new("L", (10, 6))
    image.putpixel((5, 3), 255)
    cropped = center_crop(image, 2)
    assert cropped.size == (2, 2)
    assert cropped.getpixel((1, 1)) == 255


def test_dino_eval_shape_and_normalisation() -> None:
    grey = PILImage.new("L", (640, 480), 255)  # greyscale archive photos become RGB
    pixels = dino_eval(grey, 224)
    assert pixels.shape == (3, 224, 224)
    assert pixels.dtype == np.float32
    expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(pixels[:, 100, 100], expected, atol=1e-5)


def test_squash_ignores_the_aspect() -> None:
    assert squash(PILImage.new("RGB", (900, 300)), 322).shape == (3, 322, 322)
