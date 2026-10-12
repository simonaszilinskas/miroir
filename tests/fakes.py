"""Fake HTTP servers for POP images and OpenRouter. No network: tests run with --disable-socket."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from functools import cache

import httpx
from PIL import Image as PILImage

from miroir.config import POP_ASSET_HOST, POP_IIIF_IMAGE_HOST
from miroir.http import Retry

NO_WAIT = Retry(attempts=3, backoff_s=0.0, sleep=lambda _s: None)
COST_PER_INPUT = 0.0001


@cache
def jpeg(width: int, height: int) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), (120, 80, 40)).save(out, format="JPEG")
    return out.getvalue()


@dataclass
class FakePop:
    """Serves every asset URL as a 1024x512 JPEG, except paths listed as missing."""

    missing_on_bucket: set[str] = field(default_factory=set[str])
    missing_on_iiif: set[str] = field(default_factory=set[str])
    calls: list[str] = field(default_factory=list[str])

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        if url.startswith(POP_IIIF_IMAGE_HOST):
            if any(m in url for m in self.missing_on_iiif):
                return httpx.Response(404)
            return httpx.Response(200, content=jpeg(512, 256))
        if url.startswith(POP_ASSET_HOST):
            if any(m in url for m in self.missing_on_bucket):
                return httpx.Response(404)
            return httpx.Response(200, content=jpeg(1024, 512))
        return httpx.Response(500)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@dataclass
class FakeOpenRouter:
    """Answers /embeddings with one 4-d vector per input and a cost of 0.0001 per input."""

    key_usage: float = 0.0
    batch_models: tuple[str, ...] = ()
    embedding_bodies: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    batch_bodies: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    request_sizes: list[int] = field(default_factory=list[int])
    """Number of inputs in each /embeddings request."""
    refuse_after: int | None = None
    """Refuse every /embeddings request after this many, as if credit ran out."""
    fail_first: int = 0
    """Answer this many requests with 429 before serving."""
    polls: int = 0

    def handler(self, request: httpx.Request) -> httpx.Response:  # noqa: PLR0911 (one per route)
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(429, headers={"retry-after": "0"})
        path = request.url.path.removeprefix("/api/v1")
        if path == "/embeddings":
            if self.refuse_after is not None and len(self.request_sizes) >= self.refuse_after:
                return httpx.Response(402, json={"error": {"message": "out of credit"}})
            body = json.loads(request.content)
            self.embedding_bodies.append(body)
            self.request_sizes.append(len(body["input"]))
            return httpx.Response(200, json=_embedding_answer(len(body["input"]), cost=True))
        if path == "/key":
            return httpx.Response(200, json={"data": {"usage": self.key_usage, "limit": 50}})
        if path in ("/models", "/embeddings/models"):
            ids = ["google/gemini-embedding-2", *self.batch_models]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        if path == "/batches" and request.method == "POST":
            self.batch_bodies.append(json.loads(request.content))
            return httpx.Response(202, json={"id": "batch-1", "status": "validating"})
        if path == "/batches/batch-1":
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"id": "batch-1", "status": "in_progress"})
            requests = self.batch_bodies[-1]["requests"]
            assert isinstance(requests, list)
            results = [
                {
                    "custom_id": r["custom_id"],
                    "response": {"status_code": 200, "body": _embedding_answer(1, cost=False)},
                }
                for r in requests
            ]
            cost = COST_PER_INPUT / 2 * len(results)
            return httpx.Response(
                200,
                json={
                    "id": "batch-1",
                    "status": "completed",
                    "results": results,
                    "usage": {"cost": cost},
                },
            )
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _embedding_answer(count: int, *, cost: bool) -> dict[str, object]:
    usage: dict[str, object] = {"prompt_tokens": 258 * count, "total_tokens": 258 * count}
    if cost:
        usage["cost"] = COST_PER_INPUT * count
    return {
        "id": "gen-emb-1",
        "data": [
            {"object": "embedding", "embedding": [1.0, float(i), 0.0, 2.0], "index": i}
            for i in range(count)
        ],
        "usage": usage,
    }
