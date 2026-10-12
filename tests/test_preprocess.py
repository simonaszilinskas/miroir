"""DINO preprocessing: the eval transform written in PIL and NumPy (no torch on the Mac)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from miroir.remote.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    center_crop,
    dino_eval,
    dino_whole,
    resize_shorter_side,
    squash,
    whole_size,
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


def test_whole_size_keeps_the_aspect_on_the_patch_grid() -> None:
    assert whole_size(512, 384, short=256, patch=16, max_aspect=2.0) == (336, 256)  # 341 -> 336
    assert whole_size(384, 512, short=224, patch=14, max_aspect=2.0) == (224, 294)  # 298.7 -> 294
    assert whole_size(300, 300, short=256, patch=16, max_aspect=2.0) == (256, 256)


def test_whole_size_caps_very_wide_images() -> None:
    assert whole_size(1500, 300, short=256, patch=16, max_aspect=2.0) == (512, 256)
    assert whole_size(300, 1500, short=224, patch=14, max_aspect=2.0) == (224, 448)


def test_whole_size_needs_a_short_side_on_the_grid() -> None:
    with pytest.raises(ValueError, match="multiple of the patch"):
        whole_size(512, 384, short=250, patch=16, max_aspect=2.0)


def test_dino_whole_keeps_the_sides_a_crop_would_cut() -> None:
    wide = PILImage.new("RGB", (512, 256), (0, 0, 0))
    wide.paste((255, 255, 255), (0, 0, 40, 256))  # a white band at the left edge
    pixels = dino_whole(wide, 256, 16, 2.0)
    assert pixels.shape == (3, 256, 512)
    white = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(pixels[:, 128, 5], white, atol=1e-2)
    assert not np.allclose(dino_eval(wide, 256)[:, 128, 5], white, atol=1e-2)
