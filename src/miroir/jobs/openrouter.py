"""A small OpenRouter client: embeddings, the Batch API, and what the key has spent.

This module only speaks HTTP. Privacy and money rules live in `miroir.jobs.embed` and
`miroir.jobs.spend`, which every caller that sends images goes through.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from miroir.config import OPENROUTER_API_BASE
from miroir.http import Retry, send

logger = logging.getLogger(__name__)

BATCH_SUFFIX = ":batch"

#: Models whose `:batch` variant works although OpenRouter's model lists leave it out.
#: Checked 2026-10-12: text embeddings at half price. Image inputs are refused on any batch
#: embedding ("Multimodal embedding inputs are not supported on the Batch API").
UNLISTED_BATCH_MODELS = frozenset({"google/gemini-embedding-2"})

#: JSON values sent to OpenRouter. Inputs are built by the helpers below.
type JSON = str | int | float | bool | list[JSON] | dict[str, JSON] | None


class OpenRouterError(RuntimeError):
    """OpenRouter refused a request or answered with something we cannot read."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        """The HTTP status, when OpenRouter answered with an error status."""


class BatchStatus(StrEnum):
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in _FINISHED


_FINISHED = {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED}


# --- wire formats (only the fields we use) -----------------------------------


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Usage(_Wire):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None


class _ChatMessage(_Wire):
    content: str | None = None


class _ChatChoice(_Wire):
    message: _ChatMessage
    finish_reason: str | None = None


class _ChatResponse(_Wire):
    id: str = ""
    model: str = ""
    choices: list[_ChatChoice]
    usage: _Usage | None = None


class _EmbeddingItem(_Wire):
    embedding: list[float]
    index: int


class _EmbeddingResponse(_Wire):
    id: str = ""
    model: str = ""
    data: list[_EmbeddingItem]
    usage: _Usage | None = None


class _ResultResponse(_Wire):
    status_code: int
    body: dict[str, object] | None = None


class _BatchResult(_Wire):
    custom_id: str
    response: _ResultResponse | None = None
    error: dict[str, object] | None = None


class _Batch(_Wire):
    id: str
    status: BatchStatus
    model: str = ""
    usage: _Usage | None = None
    results: list[_BatchResult] | None = None
    error: dict[str, object] | None = None


class _KeyData(_Wire):
    usage: float | None = None
    limit: float | None = None
    limit_remaining: float | None = None


class _Key(_Wire):
    data: _KeyData


class _Model(_Wire):
    id: str


class _Models(_Wire):
    data: list[_Model]


# --- values handed to callers --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Embeddings:
    vectors: NDArray[np.float32]
    """One row per input, in input order."""
    cost_usd: float
    tokens: int
    generation_id: str


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    text: str
    cost_usd: float
    """0 for a single answer of a batch: the batch carries the cost."""
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    batch_id: str
    status: BatchStatus
    cost_usd: float
    bodies: Mapping[str, Mapping[str, object]]
    """Response body per `custom_id`, for requests that succeeded."""
    errors: Mapping[str, str]
    """Error per `custom_id`, for requests that failed (or the whole batch, under "*")."""


@dataclass(frozen=True, slots=True)
class KeySpend:
    usage_usd: float
    limit_usd: float | None


# --- input helpers ---------------------------------------------------------------


def image_input(data_url: str) -> dict[str, JSON]:
    """One embedding input holding one image (a data URL or an https URL)."""
    return {"content": [{"type": "image_url", "image_url": {"url": data_url}}]}


def jpeg_data_url(jpeg_b64: str) -> str:
    return f"data:image/jpeg;base64,{jpeg_b64}"


# --- the client -------------------------------------------------------------------


class OpenRouterClient:
    def __init__(
        self,
        api_key: SecretStr,
        *,
        transport: httpx.BaseTransport | None = None,
        retry: Retry | None = None,
        timeout_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = httpx.Client(
            base_url=OPENROUTER_API_BASE,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            transport=transport,
            timeout=timeout_s,
        )
        self._retry = retry or Retry()
        self._clock = clock
        self._batch_models: frozenset[str] | None = None

    def close(self) -> None:
        self._http.close()

    # embeddings

    def embed(
        self, model: str, inputs: Sequence[str | dict[str, JSON]], *, dimensions: int | None = None
    ) -> Embeddings:
        """Embed texts or image inputs with the standard endpoint. One vector per input."""
        body: dict[str, JSON] = {"model": model, "input": list(inputs), "encoding_format": "float"}
        if dimensions is not None:
            body["dimensions"] = dimensions
        response = self._parse(_EmbeddingResponse, self._call("POST", "/embeddings", body))
        return _embeddings(response, len(inputs))

    # chat

    def chat(self, model: str, body: Mapping[str, JSON]) -> ChatAnswer:
        """One chat completion. `body` holds `messages` and any other request fields."""
        payload: dict[str, JSON] = {**body, "model": model}
        response = self._parse(_ChatResponse, self._call("POST", "/chat/completions", payload))
        return _chat_answer(response, need_cost=True)

    def chat_batch(
        self,
        model: str,
        requests: Mapping[str, Mapping[str, JSON]],
        *,
        poll_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> tuple[dict[str, ChatAnswer], BatchOutcome]:
        """Chat completions `{custom_id: body}` through the Batch API (half price); waits.

        Images work here, unlike batch embeddings (checked 2026-10-12).
        """
        return self.wait_chat_batch(
            self.submit_batch("/v1/chat/completions", model, requests), poll_s=poll_s, sleep=sleep
        )

    def wait_chat_batch(
        self,
        batch_id: str,
        *,
        poll_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> tuple[dict[str, ChatAnswer], BatchOutcome]:
        """Wait for a chat batch submitted earlier; answers by `custom_id`."""
        outcome = self.wait_batch(batch_id, poll_s=poll_s, sleep=sleep)
        answers = {
            cid: _chat_answer(self._parse(_ChatResponse, body), need_cost=False)
            for cid, body in outcome.bodies.items()
        }
        return answers, outcome

    def batch_model(self, model: str) -> str | None:
        """`model:batch` when that half-price variant exists, else None.

        Embedding models take text only there; chat models also take images.
        """
        if model in UNLISTED_BATCH_MODELS:
            return model + BATCH_SUFFIX
        if self._batch_models is None:
            ids: set[str] = set()
            for path in ("/models", "/embeddings/models"):
                listed = self._parse(_Models, self._call("GET", path))
                ids.update(m.id for m in listed.data if m.id.endswith(BATCH_SUFFIX))
            self._batch_models = frozenset(ids)
        candidate = model + BATCH_SUFFIX
        return candidate if candidate in self._batch_models else None

    # Batch API

    def submit_batch(
        self, endpoint: str, model: str, requests: Mapping[str, Mapping[str, JSON]]
    ) -> str:
        """Submit `{custom_id: body}` to the Batch API; returns the batch id."""
        if not requests:
            msg = "a batch needs at least one request"
            raise ValueError(msg)
        body: dict[str, JSON] = {
            "endpoint": endpoint,
            "model": model,
            "requests": [{"custom_id": cid, "body": dict(b)} for cid, b in requests.items()],
        }
        batch = self._parse(_Batch, self._call("POST", "/batches", body))
        logger.info("submitted batch %s: %d request(s) to %s", batch.id, len(requests), model)
        return batch.id

    def wait_batch(
        self,
        batch_id: str,
        *,
        poll_s: float = 30.0,
        timeout_s: float = 24 * 3600.0,
        unseen_s: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> BatchOutcome:
        """Poll until the batch ends. Raises TimeoutError if it outlives `timeout_s`.

        A batch just submitted can answer 404 for a while (seen 2026-10-12); that is waited
        out for `unseen_s`.
        """
        start = self._clock()
        deadline = start + timeout_s
        while True:
            try:
                batch = self._parse(_Batch, self._call("GET", f"/batches/{batch_id}"))
            except OpenRouterError as err:
                if err.status != 404 or self._clock() > start + unseen_s:
                    raise
                logger.info("batch %s not visible yet", batch_id)
                sleep(poll_s)
                continue
            if batch.status.finished:
                return _outcome(batch)
            if self._clock() > deadline:
                msg = f"batch {batch_id} still {batch.status} after {timeout_s:.0f} s"
                raise TimeoutError(msg)
            logger.debug("batch %s: %s", batch_id, batch.status)
            sleep(poll_s)

    def embed_batch(
        self,
        model: str,
        inputs: Mapping[str, str | dict[str, JSON]],
        *,
        poll_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> tuple[dict[str, NDArray[np.float32]], BatchOutcome]:
        """Embed `{custom_id: input}` through the Batch API (half price); waits for the result."""
        requests: dict[str, dict[str, JSON]] = {
            cid: {"input": [value], "encoding_format": "float"} for cid, value in inputs.items()
        }
        outcome = self.wait_batch(
            self.submit_batch("/v1/embeddings", model, requests), poll_s=poll_s, sleep=sleep
        )
        vectors = {
            # The batch carries the cost; single results may not.
            cid: _embeddings(self._parse(_EmbeddingResponse, body), 1, need_cost=False).vectors[0]
            for cid, body in outcome.bodies.items()
        }
        return vectors, outcome

    # spend

    def key_spend(self) -> KeySpend:
        """What this key has spent in total, as OpenRouter counts it."""
        key = self._parse(_Key, self._call("GET", "/key")).data
        return KeySpend(usage_usd=float(key.usage or 0.0), limit_usd=key.limit)

    # plumbing

    def _call(self, method: str, path: str, body: Mapping[str, JSON] | None = None) -> object:
        request = self._http.build_request(method, path, json=body)
        response = send(self._http, request, self._retry)
        if response.status_code >= 400:
            msg = f"{method} {path}: HTTP {response.status_code}: {_error_text(response)}"
            raise OpenRouterError(msg, response.status_code)
        try:
            return response.json()
        except ValueError as err:
            msg = f"{method} {path}: answer is not JSON"
            raise OpenRouterError(msg) from err

    @staticmethod
    def _parse[M: BaseModel](model: type[M], payload: object) -> M:
        if (
            isinstance(payload, dict)
            and payload.get("error") is not None  # pyright: ignore[reportUnknownMemberType]
            and "data" not in payload
        ):
            # OpenRouter sometimes answers 200 with an error object.
            msg = f"OpenRouter error: {_ErrorBody.model_validate(payload).error}"
            raise OpenRouterError(msg)
        try:
            return model.model_validate(payload)
        except ValidationError as err:
            msg = f"unexpected answer from OpenRouter: {err}"
            raise OpenRouterError(msg) from err


def _embeddings(
    response: _EmbeddingResponse, expected: int, *, need_cost: bool = True
) -> Embeddings:
    if len(response.data) != expected:
        msg = f"asked for {expected} embeddings, got {len(response.data)}"
        raise OpenRouterError(msg)
    rows = sorted(response.data, key=lambda item: item.index)
    if [r.index for r in rows] != list(range(expected)):
        msg = "embedding indexes do not match the inputs"
        raise OpenRouterError(msg)
    vectors = np.asarray([r.embedding for r in rows], dtype=np.float32)
    usage = response.usage or _Usage()
    if usage.cost is None and need_cost:
        msg = "OpenRouter answered without a cost: cannot track spend"
        raise OpenRouterError(msg)
    return Embeddings(
        vectors=vectors,
        cost_usd=usage.cost or 0.0,
        tokens=usage.total_tokens,
        generation_id=response.id,
    )


def _chat_answer(response: _ChatResponse, *, need_cost: bool) -> ChatAnswer:
    if not response.choices:
        msg = "chat answer holds no choice"
        raise OpenRouterError(msg)
    usage = response.usage or _Usage()
    if usage.cost is None and need_cost:
        msg = "OpenRouter answered without a cost: cannot track spend"
        raise OpenRouterError(msg)
    choice = response.choices[0]
    return ChatAnswer(
        text=choice.message.content or "",
        cost_usd=usage.cost or 0.0,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        finish_reason=choice.finish_reason or "",
    )


def _outcome(batch: _Batch) -> BatchOutcome:
    bodies: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}
    for result in batch.results or []:
        response = result.response
        if response is not None and response.status_code == 200 and response.body is not None:
            bodies[result.custom_id] = response.body
        else:
            detail = result.error or (response.body if response else None)
            errors[result.custom_id] = str(detail)
    if batch.error is not None:
        errors["*"] = str(batch.error)
    cost = batch.usage.cost if batch.usage and batch.usage.cost is not None else 0.0
    logger.info(
        "batch %s %s: %d ok, %d failed, $%.4f",
        batch.id,
        batch.status,
        len(bodies),
        len(errors),
        cost,
    )
    return BatchOutcome(batch.id, batch.status, cost, bodies, errors)


class _ErrorBody(_Wire):
    error: object = None


def _error_text(response: httpx.Response) -> str:
    try:
        body = _ErrorBody.model_validate_json(response.content)
    except ValidationError:
        return response.text[:300]
    return str(body.error if body.error is not None else response.text)[:300]
