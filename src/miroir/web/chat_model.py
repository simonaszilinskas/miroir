"""The agentic search's model (step H5): Mistral Large 4, through OpenRouter or on Mistral's own
API. `ChatModel` is all the conversation needs of it.

A call is tried twice at most, the second time only when the first surely did not reach the
model (no connection, or a busy answer: 429, 502, 503). After a read timeout or a dropped
answer the model may have run, and been paid for: that call is not sent again, and the error
says it may have been billed (`ModelError.maybe_billed`). The turn has its own deadline
(`miroir.web.chat.TURN_S`).

The clients of `miroir.jobs` are synchronous, so each call runs in a thread of the model's own
pool (`MODEL_THREADS`): a slow call holds one of those, never the threads the searches and the
image work share. An async client would need async twins of `OpenRouterClient` and
`MistralClient`, which H4's batch runs share; not worth it for one request at a time per
conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Protocol

import httpx
from pydantic import SecretStr

from miroir.config import Settings
from miroir.http import Retry
from miroir.jobs.mistral import PRICES, MistralClient, MistralError
from miroir.jobs.openrouter import JSON, ChatAnswer, OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

#: The model through OpenRouter: open, takes images and tools; temperature 0. Its provider
#: now refuses reasoning "none" (HTTP 400, seen 2026-11-03), so it reasons a little, as on
#: Mistral's own API, with the same room for that thinking.
MODEL = "mistralai/mistral-large-4-0"
MODEL_FIELDS: dict[str, JSON] = {"temperature": 0.0, "top_p": 1.0, "reasoning": {"effort": "low"}}
#: Mistral Large 4 on Mistral's own API (released 2026-10-29). It always reasons; "low" keeps
#: a turn quick. Greedy sampling needs top_p 1.
MISTRAL_MODEL = "mistral-large-4"
MISTRAL_FIELDS: dict[str, JSON] = {"temperature": 0.0, "top_p": 1.0, "reasoning_effort": "low"}
#: Room for the thinking that comes before the answer.
MISTRAL_MAX_TOKENS = 3000
#: Seconds one call may wait for a byte of the answer.
TIMEOUT_S = 90.0
#: Tries per call, and the wait before the second.
ATTEMPTS = 2
RETRY_WAIT_S = 2.0
#: Answers that mean the request did not run: it may be sent once more.
RETRY_STATUS = frozenset({429, 502, 503})
#: Model calls at once, all conversations together.
MODEL_THREADS = 16
#: Characters of a request per prompt token, to price an answer that came without usage.
CHARS_PER_TOKEN = 3


class ModelError(RuntimeError):
    """The model did not answer: refused, down, or too slow."""

    def __init__(self, message: str, *, maybe_billed: bool = False) -> None:
        super().__init__(message)
        self.maybe_billed = maybe_billed
        """The request may have run, and been paid for, although no answer came back."""


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        """One chat completion; `body` holds messages, tools and the other request fields.
        Raises `ModelError` when no answer comes."""
        ...


class _Threaded:
    """Calls a synchronous client in the model's own threads, with the retry rule above."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(MODEL_THREADS, thread_name_prefix="chat-model")

    async def _ask(self, call: Callable[[], ChatAnswer]) -> ChatAnswer:
        loop = asyncio.get_running_loop()
        for attempt in range(1, ATTEMPTS + 1):
            last = attempt == ATTEMPTS
            try:
                return await loop.run_in_executor(self._pool, call)
            except (OpenRouterError, MistralError) as err:
                if err.status is None:  # a 200 we could not read: it ran
                    raise ModelError(str(err), maybe_billed=True) from err
                if err.status not in RETRY_STATUS or last:
                    raise ModelError(str(err)) from err
                reason = f"HTTP {err.status}"
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as err:
                if last:
                    raise ModelError(f"no connection: {err!r}") from err
                reason = type(err).__name__
            except httpx.HTTPError as err:
                raise ModelError(f"no answer: {err!r}", maybe_billed=True) from err
            logger.info("model call: %s; trying once more", reason)
            await asyncio.sleep(RETRY_WAIT_S)
        msg = "unreachable: the last attempt returns or raises"
        raise AssertionError(msg)


class OpenRouterModel(_Threaded):
    """Mistral Large 4 through OpenRouter. The key comes from `Settings`, never logged."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str = MODEL,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._client = OpenRouterClient(
            api_key, transport=transport, retry=Retry(attempts=1), timeout_s=TIMEOUT_S
        )
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        fields = {**body, **MODEL_FIELDS, "max_tokens": MISTRAL_MAX_TOKENS}
        answer = await self._ask(lambda: self._client.chat(self._model, fields, need_cost=False))
        return priced(answer, fields)


class MistralModel(_Threaded):
    """Mistral Large 4 on Mistral's API. The key comes from `Settings`, never logged."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str = MISTRAL_MODEL,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._client = MistralClient(
            api_key, transport=transport, retry=Retry(attempts=1), timeout_s=TIMEOUT_S
        )
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        fields = {**body, **MISTRAL_FIELDS, "max_tokens": MISTRAL_MAX_TOKENS}
        return await self._ask(lambda: self._client.chat(self._model, fields))


def priced(answer: ChatAnswer, body: Mapping[str, JSON]) -> ChatAnswer:
    """The answer with a cost: an OpenRouter answer that came without one (it was paid for all
    the same) is priced from its tokens at Mistral's prices, or from the request's size."""
    if answer.cost_usd > 0:
        return answer
    prompt, _, completion = PRICES[MISTRAL_MODEL]
    tokens_in = answer.prompt_tokens or len(json.dumps(body)) // CHARS_PER_TOKEN
    max_tokens = body.get("max_tokens")
    tokens_out = answer.completion_tokens or (max_tokens if isinstance(max_tokens, int) else 0)
    usd = (tokens_in * prompt + tokens_out * completion) / 1e6
    logger.warning("model answer without a cost: counted as $%.4f", usd)
    return replace(answer, cost_usd=usd)


def chat_model(settings: Settings) -> ChatModel | None:
    """The conversation page's model: OpenRouter when OPENROUTER_API_KEY is set, else Mistral's
    own API when MISTRAL_API_KEY is, else None (no key, no conversation page)."""
    if settings.openrouter_api_key is not None:
        return OpenRouterModel(settings.openrouter_api_key)
    if settings.mistral_api_key is not None:
        return MistralModel(settings.mistral_api_key)
    return None
