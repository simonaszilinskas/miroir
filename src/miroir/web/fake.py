"""Stand-in models for running the website on the Mac (`miroir web dev`) and in tests.

No model is loaded here (CLAUDE.md rule 1). A photo "embeds" as the stored vector of a
gallery image picked from a hash of its bytes, plus a little noise, so results look like a
real search (a coherent group of similar images) without being one. Words embed as a seeded
random vector; BM25, which needs no model, runs for real.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.outlines import Outlines, fake_outlines
from miroir.web.engine import (
    A7_FILE,
    Engine,
    SearchRequest,
    SearchResponse,
    load_engine,
)

Floats = NDArray[np.float32]
NOISE = 0.02


def _seed(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class FakeEncoders:
    """Photos map to a stored gallery vector; MegaLoc gets the same vector (see NumpyGallery)."""

    gallery: Floats

    def image(self, jpeg: bytes) -> tuple[Floats, Floats]:
        rng = np.random.default_rng(_seed(jpeg))
        row = int(rng.integers(0, self.gallery.shape[0]))
        noise = np.asarray(rng.normal(0.0, 1.0, self.gallery.shape[1]), dtype=np.float32)
        vector = np.asarray(self.gallery[row] + NOISE * noise, dtype=np.float32)
        return vector, vector

    def text(self, text: str) -> Floats:
        rng = np.random.default_rng(_seed(text.encode()))
        return np.asarray(rng.normal(0.0, 1.0, self.gallery.shape[1]), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class NumpyGallery:
    """Cosines on the CPU. MegaLoc reuses the EmbeddingGemma 2 matrix: its 8,448-d vectors
    are too big to multiply in float32 on the Mac for a demo."""

    a7_matrix: Floats

    def a7(self, query: Floats) -> Floats:
        return self.a7_matrix @ query

    def a3(self, query: Floats) -> Floats:
        return self.a7_matrix @ query


#: What the stand-in caption reader "reads" on any mounted print.
FAKE_CAPTION = "BEAUNE (Côte-d'Or) Eglise Notre-Dame, façade ouest, porte centrale"


class FakeCaptions:
    """Reads the same made-up caption on every mounted print."""

    def read(self, jpeg: bytes) -> str:
        del jpeg
        return FAKE_CAPTION


def fake_engine(directory: Path) -> Engine:
    matrix = np.asarray(np.load(directory / A7_FILE), dtype=np.float32)
    engine = load_engine(directory, FakeEncoders(matrix), NumpyGallery(matrix), FakeCaptions())
    engine.projection = None  # fake MegaLoc vectors live in the EmbeddingGemma 2 space
    return engine


@dataclass(frozen=True, slots=True)
class LocalBackend:
    """Runs an engine in a worker thread; `warm` has nothing to start; outlines are fake."""

    engine: Engine

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await asyncio.to_thread(self.engine.search, request)

    async def warm(self) -> None:
        return None

    async def detect(self, jpeg: bytes) -> Outlines:
        return fake_outlines(jpeg)
