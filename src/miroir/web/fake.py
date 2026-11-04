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

    def a7_row(self, row: int) -> Floats:
        return np.asarray(self.a7_matrix[row], dtype=np.float32)

    def a3_row(self, row: int) -> Floats:
        return np.asarray(self.a7_matrix[row], dtype=np.float32)


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


def _image_search(messages: Sequence[Mapping[str, object]]) -> dict[str, JSON]:
    """An image search: cropped to the first outline and in black and white when the
    detector found one, so the page shows a transformed search."""
    outlined = any("Outlines the detector found" in _content_text(m) for m in messages)
    if not outlined:
        why = "Les premiers résultats ne tranchent pas ; je relance par l’image."
        return {"model": "auto", "top_k": 10, "why": why}
    transform: dict[str, JSON] = {"outline": 1, "grayscale": True}
    why = "Je cherche la première œuvre repérée, en noir et blanc comme les tirages d’archives."
    return {"model": "auto", "top_k": 10, "transform": transform, "why": why}


#: The tools that search.
_SEARCHES = frozenset({"image_search", "text_search", "more_like"})


def _called(messages: Sequence[Mapping[str, object]], names: frozenset[str]) -> bool:
    """Whether the model has called one of these tools yet."""
    return any(
        _dict(c.get("function")).get("name") in names
        for m in messages
        for c in _dicts(m.get("tool_calls"))
    )


class FakeGuide:
    """A rule-following stand-in for the model, so the page runs on the Mac at no cost. It
    looks at the first look's top thumbnails, searches once (by image, or by the visitor's
    words), looks at what came back, then asks where the photo was taken (or asks to frame
    part of the photo when the goal says "cadre"). The answer starts a search of its words
    (near the visitor when they shared their position); it looks at the new results and shows
    them as close. A message from the visitor gets a text search of their first words. Every
    search and look says why."""

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
        searched = _called(messages, _SEARCHES)
        asked = _called(messages, frozenset({"ask_visitor", "ask_region"}))
        if role == "user" and "The visitor's goal" in content:
            if "ask_region" in tools and "cadre" in content:
                reason = {"reason": "Encadrez la partie de la photo qui vous intéresse."}
                return _answer("", [("ask_region", reason)])
            first = _ids(content)[:3]
            if first:
                look: dict[str, JSON] = {
                    "entity_ids": list[JSON](first),
                    "why": "Je regarde d’abord les premiers résultats de près.",
                }
                return _answer("", [("view_results", look)])
            if "image_search" in tools:
                return _answer("", [("image_search", _image_search(messages))])
            words = _between(content, "typed words: «", "»") or "église"
            return _answer("", [("text_search", _words_search(words))])
        if role == "user" and "The part the visitor framed" in content:
            search = {"source": "region", "model": "auto", "why": "Je cherche la partie encadrée."}
            return _answer("", [("image_search", search)])
        if role == "tool" and "The visitor answers" in content:
            return _answer("", [_after_answer(content)])
        if role == "tool" and "framed nothing" in content:
            why = "Toute la photo compte ; je relance par l’image, autrement."
            return _answer("", [("image_search", {"model": "mix", "top_k": 10, "why": why})])
        if role == "user" and "The visitor writes" in content:
            words = _between(content, "«", "»") or "église"
            return _answer("", [("text_search", _words_search(words))])
        thumbnails = "Thumbnails asked for" in content
        if thumbnails and not searched and "image_search" in tools:
            return _answer("", [("image_search", _image_search(messages))])
        looked = _looked_at(messages)
        fresh = [i for i in dict.fromkeys(listed[-10:]) if i not in looked][:3]
        if fresh and not thumbnails:
            why = "Je regarde de près les premiers résultats nouveaux."
            ids: list[JSON] = list[JSON](fresh)
            return _answer("", [("view_results", {"entity_ids": ids, "why": why})])
        if searched and not asked and "ask_visitor" in tools:
            return _answer("Je ne suis pas encore sûr : une précision m’aiderait.", [_QUESTION])
        return _answer("", [("show_results", _shown(listed))])


_REASONS = ("Même silhouette et mêmes matériaux", "Même type d’édifice", "Proche par le décor")

_QUESTION: tuple[str, Mapping[str, JSON]] = (
    "ask_visitor",
    {
        "question": "Savez-vous où la photo a été prise ? Une commune suffit.",
        "choices": ["Dans une ville", "Dans un village", "Dans un musée"],
        "about": "place",
        "offer_position": True,
    },
)


def _shown(listed: Sequence[str]) -> dict[str, JSON]:
    """The last results listed, latest first, as close."""
    picks: list[JSON] = [
        {"id": i, "reason": _REASONS[n % len(_REASONS)]}
        for n, i in enumerate(list(dict.fromkeys(reversed(listed)))[:3])
    ]
    summary = "Modèle factice : voici ce qui ressemble le plus."
    return {"found": "close", "summary": summary, "results": picks}


def _after_answer(content: str) -> tuple[str, Mapping[str, JSON]]:
    """The search that follows the visitor's answer."""
    if "shared their position" in content:
        why = "Vous êtes sur place : je cherche ce qui est près de vous."
        near: dict[str, JSON] = {"near": "visitor"}
        return "image_search", {"model": "A3", "top_k": 10, "filters": near, "why": why}
    answer = _between(content, "«", "»")
    if answer and "ne sais pas" not in answer:
        why = "Je cherche le lieu que vous donnez."
        return "text_search", {**_words_search(answer), "why": why}
    why = "Sans lieu, je compare autrement : la forme d’abord."
    return "image_search", {"model": "A7", "top_k": 10, "why": why}


def _looked_at(messages: Sequence[Mapping[str, object]]) -> set[str]:
    """Ids already passed to view_results."""
    out: set[str] = set()
    for m in messages:
        for c in _dicts(m.get("tool_calls")):
            function = _dict(c.get("function"))
            if function.get("name") != "view_results":
                continue
            args = _dict(json.loads(str(function.get("arguments") or "{}")))
            ids = args.get("entity_ids")
            items: list[object] = list(ids) if isinstance(ids, list) else []  # pyright: ignore[reportUnknownArgumentType]
            out.update(str(e) for e in items)
    return out


def _words_search(words: str) -> dict[str, JSON]:
    return {"query": " ".join(words.split()[:5]), "why": "Je cherche vos mots dans les notices."}


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
