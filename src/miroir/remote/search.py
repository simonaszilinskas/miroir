# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""The website's search models and gallery vectors on the GPU. Runs only on a Scaleway machine.

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

from miroir.images.upload import open_upload
from miroir.jobs.remote_spec import OPTIONS
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
from miroir.web.crops import has_crops, read_numpy_crops
from miroir.web.engine import A3_FILE, A7_FILE, Engine, load_engine
from miroir.web.toolsearch import ToolHits, ToolQuery, ToolSearch

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
    """Both models' gallery vectors on the GPU in float16, in catalog row order. Without a
    GPU they stay in memory in float32: a CPU multiplies float16 many times slower."""

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
        matrix = np.load(path)
        if DEVICE == "cpu":
            return torch.from_numpy(matrix.astype(np.float32))
        return torch.from_numpy(matrix).to(DEVICE)

    @torch.inference_mode()
    def _cosines(self, matrix: torch.Tensor, query: Floats) -> Floats:
        q = torch.from_numpy(np.asarray(query)).to(DEVICE, dtype=matrix.dtype)
        return (matrix @ q).float().cpu().numpy()

    def a7(self, query: Floats) -> Floats:
        return self._cosines(self._a7, query)

    def a3(self, query: Floats) -> Floats:
        return self._cosines(self._a3, query)

    def a7_row(self, row: int) -> Floats:
        return self._a7[row].float().cpu().numpy()

    def a3_row(self, row: int) -> Floats:
        return self._a3[row].float().cpu().numpy()


class GpuCrops:
    """A bundle's short crop vectors (`miroir.web.crops`) on the GPU in int8, picked in
    float32 blocks. Copied up from the memory-mapped file block by block, so host memory
    never holds them whole."""

    #: Crop rows turned to float32 at once on the GPU.
    BLOCK = 1 << 19

    def __init__(self, directory: Path) -> None:
        start = time.monotonic()
        numpy = read_numpy_crops(directory, mmap=True)
        rows = numpy.coords
        self.coords = torch.empty(tuple(rows.shape), dtype=torch.int8, device=DEVICE)
        for at in range(0, rows.shape[0], self.BLOCK):
            block = np.ascontiguousarray(rows[at : at + self.BLOCK])
            self.coords[at : at + len(block)] = torch.from_numpy(block).to(DEVICE)
        self.scale = torch.from_numpy(numpy.scale).to(DEVICE)
        logger.info(
            "crops on GPU: %s int8 (%.1f GB) in %.0f s",
            tuple(rows.shape), rows.nbytes / 1e9, time.monotonic() - start,
        )  # fmt: skip

    @torch.inference_mode()
    def top_crops(self, query: Floats, count: int) -> NDArray[np.int64]:
        q = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(DEVICE)
        sims = torch.empty(self.coords.shape[0], dtype=torch.float32, device=DEVICE)
        for at in range(0, self.coords.shape[0], self.BLOCK):
            block = self.coords[at : at + self.BLOCK].float()
            sims[at : at + len(block)] = (block @ q) * self.scale[at : at + len(block)]
        best = torch.topk(sims, min(count, sims.shape[0]), sorted=False).indices
        return np.sort(best.cpu().numpy().astype(np.int64))


def load_search(directory: Path, captions: CaptionReader | None = None) -> Engine:
    """Everything a search container needs, loaded once at start. Without a GPU, crops stay
    in host memory and the CPU scores them."""
    gallery = GpuGallery(directory)
    crops = GpuCrops(directory) if has_crops(directory) and DEVICE != "cpu" else None
    return load_engine(directory, QueryModels(), gallery, captions, crops)


class OutlineFinder:
    """F4 on the GPU: the site's detector, then the outline rule tuned in step C."""

    def __init__(self) -> None:
        self.detector = load_detector(detector(SITE_DETECTOR))
        blank = PILImage.new("RGB", (DETECT_SIDE, DETECT_SIDE * 3 // 4), "white")
        self.detector.detect(blank)  # warm the kernels before the first photo

    def find(self, jpeg: bytes) -> Outlines:
        start = time.monotonic()
        image = detect_size(open_upload(jpeg).upright)
        boxes = self.detector.detect(image)
        shown = pick_outlines(boxes, SITE_RULE, width=image.width, height=image.height)
        return Outlines(shown, detect_ms=round(1000 * (time.monotonic() - start), 1))


#: A print's caption and a photo's outlines wait this long for their model while it loads.
MODEL_WAIT_S = 60.0


class SearchService:
    """Everything the site's searches need over one index bundle: the search models and the
    gallery vectors (`load_search`), the outline finder and the caption reader.

    The reader loads last, alone: see `BackgroundCaptions.load_now`. Without a GPU it is left
    out (Qwen3-VL 4B would take most of a minute per print on the CPU): prints are searched
    without their caption, and `degraded` says so. Used by the demo machine's site
    (`miroir.remote.site`) and by the job worker (`miroir web dev --scw`)."""

    def __init__(self, folder: Path) -> None:
        from miroir.caption import SITE_CAPTION_MODEL  # noqa: PLC0415
        from miroir.remote.caption import BackgroundCaptions  # noqa: PLC0415

        start = time.monotonic()
        self.degraded = DEVICE == "cpu"
        """No GPU: slower searches, no caption reading."""
        self.outlines = OutlineFinder()
        captions = (
            None
            if self.degraded
            else BackgroundCaptions(SITE_CAPTION_MODEL, MODEL_WAIT_S, start=False)
        )
        self.engine = load_search(folder, captions)
        if captions is not None:
            captions.load_now()
        # The conversation page's filters need every entity's facts (about 50 s on all of
        # POP). Gathered before the site listens: in a thread beside the first searches, it
        # held Python's lock and the first photo search took 27 s instead of 0.3 s.
        self._tools = ToolSearch(self.engine)
        self._tools.warm()
        logger.info("search service ready in %.0f s", time.monotonic() - start)

    def detect(self, jpeg: bytes) -> Outlines:
        return self.outlines.find(jpeg)

    def tool_search(self, query: ToolQuery) -> ToolHits:
        return self._tools.run(query)
