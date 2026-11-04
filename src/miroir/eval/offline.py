"""The agentic search's backend offline (step H5 on test sets): the site's own engine and tool
searches over a built bundle, with each query photo's stored vectors standing in for the
models. No model runs here.

`OfflineWorld` loads the engine once (`miroir.web.engine.Engine`, its `ToolSearch`), as the
site does, and `OfflineBackend` serves one test query to the conversation
(`miroir.web.chat.ToolBackend`): the first look and every tool search carry the query's
stored vectors (`SearchRequest.vectors`, `ToolQuery.vectors`) and its hidden images (its own,
for T1 its whole shoot), so the scores are the site's, and a hidden image never scores nor
reaches the model.

What needs a model cannot run offline, and says so to the model: a search of a transformed
photo (a crop, a grey copy, a turn) or of a part the visitor framed, words embedded by the
text tower (the first look has none: the simulated visitor gives a photo only), and the
detector's outlines (none are listed). The caption of an archive mount is not read either.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.outlines import Outlines
from miroir.web.catalog import Catalog
from miroir.web.chat_tools import ArgumentError
from miroir.web.engine import Engine, SearchRequest, SearchResponse, load_engine
from miroir.web.measure import BundleVectors
from miroir.web.toolsearch import ToolHits, ToolQuery, ToolSearch

logger = logging.getLogger(__name__)

Floats = NDArray[np.float32]
#: Gallery rows cast to float32 at once.
CHUNK = 8192


class OfflineError(ArgumentError):
    """A search that needs a model. The tool answers the model with this message."""


NEEDS_MODEL = (
    "offline, only the whole photo can be searched by image: a transformed photo or a "
    "framed part needs a model, which does not run here. Nothing was searched; search the "
    "whole photo, by words, or with more_like."
)


def cosines(matrix: NDArray[np.float16] | Floats, query: Floats) -> Floats:
    """Cosine of a unit query with every row, in chunks (the rows are unit vectors)."""
    out = np.empty(matrix.shape[0], dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    for start in range(0, matrix.shape[0], CHUNK):
        out[start : start + CHUNK] = np.asarray(matrix[start : start + CHUNK], np.float32) @ q
    return out


@dataclass(frozen=True, slots=True)
class StoredGallery:
    """`GalleryVectors` over a bundle's stored matrices, on the CPU."""

    vectors: BundleVectors

    def a7(self, query: Floats) -> Floats:
        return cosines(self.vectors.a7, query)

    def a3(self, query: Floats) -> Floats:
        return cosines(self.vectors.a3, query)

    def a7_row(self, row: int) -> Floats:
        return np.asarray(self.vectors.a7[row], dtype=np.float32)

    def a3_row(self, row: int) -> Floats:
        return np.asarray(self.vectors.a3[row], dtype=np.float32)


class NoModels:
    """`Encoders` that refuse: every photo the agent searches comes with its stored vectors."""

    def image(self, jpeg: bytes) -> tuple[Floats, Floats]:
        del jpeg
        raise OfflineError(NEEDS_MODEL)

    def text(self, text: str) -> Floats:
        del text
        msg = "offline, words cannot be embedded: no model runs here"
        raise OfflineError(msg)


class OfflineWorld:
    """The engine and its tool searches, shared by every query of a run."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.tools = ToolSearch(engine)

    @classmethod
    def load(cls, folder: Path) -> OfflineWorld:
        """The bundle as the site loads it (`load_engine`), its models replaced by `NoModels`."""
        return cls(load_engine(folder, NoModels(), StoredGallery(BundleVectors.load(folder))))

    @property
    def catalog(self) -> Catalog:
        return self.engine.catalog

    def site_scores(self, query: StoredQuery) -> Floats:
        """The site's photo ranking of the query, per catalog entity (-inf when hidden): its
        first look before a caption or a position count, and the fallback order of the
        agent's answer."""
        request = SearchRequest(
            image=query.photo, vectors=(query.a7, query.a3), exclude_images=query.hidden
        )
        return np.asarray(self.engine.photo_scores(request), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class StoredQuery:
    """One test query as the offline backend serves it."""

    query_id: str
    photo: bytes
    """The query photo as the model sees it (fetched once; publishable or Commons)."""
    a7: Floats
    a3: Floats
    """Its stored vectors, whole unit rows as the models give them."""
    hidden: tuple[str, ...]
    """Images it may not find: its own, and for T1 its whole shoot."""


@dataclass(frozen=True, slots=True)
class OfflineBackend:
    """`miroir.web.chat.ToolBackend` for one test query, over a shared `OfflineWorld`."""

    world: OfflineWorld
    query: StoredQuery

    def _vectors(self, image: bytes | None) -> tuple[Floats, Floats] | None:
        if image is None:
            return None
        if image != self.query.photo:
            raise OfflineError(NEEDS_MODEL)
        return (self.query.a7, self.query.a3)

    async def search(self, request: SearchRequest) -> SearchResponse:
        asked = replace(
            request,
            vectors=self._vectors(request.image),
            exclude_images=(*request.exclude_images, *self.query.hidden),
        )
        return await asyncio.to_thread(self.world.engine.search, asked)

    async def tool_search(self, query: ToolQuery) -> ToolHits:
        asked = replace(
            query,
            vectors=self._vectors(query.image),
            exclude_images=(*query.exclude_images, *self.query.hidden),
        )
        return await asyncio.to_thread(self.world.tools.run, asked)

    async def detect(self, jpeg: bytes) -> Outlines:
        del jpeg
        return Outlines(())


def entity_order(catalog: Catalog) -> NDArray[np.int64]:
    """Catalog entity rows sorted by id: the column order scoring expects."""
    return np.argsort(np.asarray(catalog.entity_ids), kind="stable").astype(np.int64)
