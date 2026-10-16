"""Stand-in models for running the website on the Mac (`miroir web dev`) and in tests.

No model is loaded here (CLAUDE.md rule 1). A photo "embeds" as the stored vector of a
gallery image picked from a hash of its bytes, plus a little noise, so results look like a
real search (a coherent group of similar images) without being one. Words embed as a seeded
random vector; BM25, which needs no model, runs for real.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.jobs.openrouter import JSON, ChatAnswer, ToolCall
from miroir.outlines import Outlines, fake_outlines
from miroir.web.engine import (
    A7_FILE,
    Engine,
    SearchRequest,
    SearchResponse,
    load_engine,
)
from miroir.web.toolsearch import ToolHits, ToolQuery, ToolSearch

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
    engine.exact = None
    return engine


@dataclass(frozen=True, slots=True)
class LocalBackend:
    """Runs an engine in a worker thread; `warm` has nothing to start; outlines are fake."""

    engine: Engine
    tools: ToolSearch = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", ToolSearch(self.engine))

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await asyncio.to_thread(self.engine.search, request)

    async def tool_search(self, query: ToolQuery) -> ToolHits:
        return await asyncio.to_thread(self.tools.run, query)

    async def warm(self) -> None:
        return None

    async def detect(self, jpeg: bytes) -> Outlines:
        return fake_outlines(jpeg)


# --- stand-in language models for the conversation page (`miroir.web.chat`) ------------------


def _answer(text: str = "", calls: Sequence[tuple[str, Mapping[str, JSON]]] = ()) -> ChatAnswer:
    return ChatAnswer(
        text=text,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason="tool_calls" if calls else "stop",
        tool_calls=tuple(
            ToolCall(f"fake{n:05d}", name, json.dumps(args, ensure_ascii=False))
            for n, (name, args) in enumerate(calls)
        ),
    )


class ScriptedModel:
    """Answers from a list, in order; keeps every request body (tests)."""

    def __init__(self, answers: Sequence[ChatAnswer], model: str = "scripted") -> None:
        self._answers = list(answers)
        self._model = model
        self.bodies: list[dict[str, JSON]] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        self.bodies.append(dict(body))
        if not self._answers:
            msg = "the script has no answer left"
            raise AssertionError(msg)
        return self._answers.pop(0)


_LISTED = re.compile(r"^\d+\. ([^ |]+) \|", re.MULTILINE)


class FakeGuide:
    """A rule-following stand-in for the model, so the page runs on the Mac at no cost: one
    question when there is a photo (a request to frame part of it when the goal says
    "cadre"), an image search, a look at three thumbnails, then three results. A message from
    the visitor gets a text search of their words."""

    @property
    def model(self) -> str:
        return "fake-guide"

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:  # noqa: PLR0911 (one rule per return)
        messages = _dicts(body.get("messages"))
        tools = {str(_dict(t.get("function")).get("name")) for t in _dicts(body.get("tools"))}
        last = messages[-1]
        role = last.get("role")
        content = _content_text(last)
        listed = [i for m in messages if m.get("role") == "tool" for i in _ids(_content_text(m))]
        if role == "user" and "The visitor's goal" in content:
            if "ask_region" in tools and "cadre" in content:
                reason = {"reason": "Encadrez la partie de la photo qui vous intéresse."}
                return _answer("", [("ask_region", reason)])
            if "image_search" in tools:
                question: dict[str, JSON] = {
                    "question": "Que montre votre photo, en quelques mots ?",
                    "choices": ["Un bâtiment", "Un tableau", "Un objet"],
                }
                return _answer(
                    "Modèle factice : une question d’abord.", [("ask_visitor", question)]
                )
            words = _between(content, "typed words: «", "»") or "église"
            return _answer("", [("text_search", {"query": words})])
        if role == "user" and "The part the visitor framed" in content:
            return _answer("", [("image_search", {"source": "region", "model": "auto"})])
        if role == "tool" and ("The visitor answers" in content or "framed nothing" in content):
            return _answer("", [("image_search", {"model": "auto", "top_k": 10})])
        if role == "user" and "The visitor writes" in content:
            words = _between(content, "«", "»") or "église"
            return _answer("", [("text_search", {"query": words})])
        viewed = any("view_results" in _content_text(m) for m in messages[-3:])
        if role == "tool" and listed and not viewed:
            ids: list[JSON] = list(listed[-10:][:3])
            return _answer("", [("view_results", {"entity_ids": ids})])
        picks: list[JSON] = [
            {"id": i, "reason": "Choix du modèle factice"}
            for i in list(dict.fromkeys(reversed(listed)))[:3]
        ]
        return _answer(
            "Modèle factice : voici ce qui ressemble le plus.",
            [("show_results", {"results": picks})],
        )


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _dicts(value: object) -> list[dict[str, object]]:
    items: list[object] = list(value) if isinstance(value, list) else []  # pyright: ignore[reportUnknownArgumentType]
    return [_dict(v) for v in items if isinstance(v, dict)]  # pyright: ignore[reportUnknownArgumentType]


def _content_text(message: Mapping[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return " ".join(str(p.get("text", "")) for p in _dicts(content))


def _ids(text: str) -> list[str]:
    return _LISTED.findall(text)


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()
