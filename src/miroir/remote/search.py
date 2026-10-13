# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""The website's search models and gallery vectors on the GPU. Runs only inside Modal.

The query side of the R1 runs: the same encoder classes and options as the gallery
(`miroir.remote.models`, `OPTIONS`), so query vectors land in the same space as the stored
ones. EmbeddingGemma 2 reads photos with no prompt, as the gallery did, and words with its
"SearchQuery" prompt. Gallery vectors sit on the GPU in float16; a search is one product.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

from miroir.jobs.modal_spec import OPTIONS
from miroir.outlines import (
    DETECT_SIDE,
    SITE_DETECTOR,
    SITE_RULE,
    Outlines,
    detector,
    pick_outlines,
)
from miroir.remote.detect import detect_size, load_detector
from miroir.remote.models import DEVICE, MegaLocEncoder, SentenceEncoder
from miroir.web.archive import CaptionReader
from miroir.web.engine import A3_FILE, A7_FILE, Engine, load_engine

logger = logging.getLogger(__name__)

A7_OPTION = "A7-embeddinggemma-2-512"
A3_OPTION = "A3-megaloc-322"

Floats = NDArray[np.float32]


class QueryModels:
    """EmbeddingGemma 2 (photos and words) and MegaLoc (photos), loaded once."""

    def __init__(self) -> None:
        start = time.monotonic()
        self.a7 = SentenceEncoder(OPTIONS[A7_OPTION])
        self.a3 = MegaLocEncoder(OPTIONS[A3_OPTION])
        logger.info("models loaded in %.1f s", time.monotonic() - start)

    def image(self, jpeg: bytes) -> tuple[Floats, Floats]:
        with PILImage.open(io.BytesIO(jpeg)) as opened:
            image = opened.convert("RGB")
        return self.a7.encode([image])[0], self.a3.encode([image])[0]

    def text(self, text: str) -> Floats:
        return self.a7.encode_text([text])[0]


class GpuGallery:
    """Both models' gallery vectors on the GPU, in catalog row order."""

    def __init__(self, directory: Path) -> None:
        start = time.monotonic()
        self._a7 = self._load(directory / A7_FILE)
        self._a3 = self._load(directory / A3_FILE)
        logger.info(
            "gallery on GPU: %s and %s in %.1f s",
            tuple(self._a7.shape),
            tuple(self._a3.shape),
            time.monotonic() - start,
        )

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).to(DEVICE)

    @torch.inference_mode()
    def _cosines(self, matrix: torch.Tensor, query: Floats) -> Floats:
        q = torch.from_numpy(np.asarray(query, dtype=np.float16)).to(DEVICE)
        return (matrix @ q).float().cpu().numpy()

    def a7(self, query: Floats) -> Floats:
        return self._cosines(self._a7, query)

    def a3(self, query: Floats) -> Floats:
        return self._cosines(self._a3, query)


def load_search(directory: Path, captions: CaptionReader | None = None) -> Engine:
    """Everything a search container needs, loaded once at start."""
    gallery = GpuGallery(directory)
    return load_engine(directory, QueryModels(), gallery, captions)


class OutlineFinder:
    """F4 on the GPU: the site's detector, then the outline rule tuned in step C."""

    def __init__(self) -> None:
        self.detector = load_detector(detector(SITE_DETECTOR))
        blank = PILImage.new("RGB", (DETECT_SIDE, DETECT_SIDE * 3 // 4), "white")
        self.detector.detect(blank)  # warm the kernels before the first photo

    def find(self, jpeg: bytes) -> Outlines:
        start = time.monotonic()
        try:
            with PILImage.open(io.BytesIO(jpeg)) as opened:
                image = detect_size(ImageOps.exif_transpose(opened))
        except (OSError, PILImage.DecompressionBombError) as err:
            msg = f"cannot read the photo: {err}"
            raise ValueError(msg) from err
        boxes = self.detector.detect(image)
        shown = pick_outlines(boxes, SITE_RULE, width=image.width, height=image.height)
        return Outlines(shown, detect_ms=round(1000 * (time.monotonic() - start), 1))
