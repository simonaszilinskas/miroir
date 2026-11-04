"""The agentic search's model (step H5): Mistral Large 4, through OpenRouter or on Mistral's own
API. `ChatModel` is all the conversation needs of it."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Protocol

from pydantic import SecretStr

from miroir.config import Settings
from miroir.jobs.mistral import MistralClient
from miroir.jobs.openrouter import JSON, ChatAnswer, OpenRouterClient

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


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        """One chat completion; `body` holds messages, tools and the other request fields."""
        ...


class OpenRouterModel:
    """Mistral Large 4 through OpenRouter. The key comes from `Settings`, never logged."""

    def __init__(self, api_key: SecretStr, model: str = MODEL) -> None:
        self._client = OpenRouterClient(api_key, timeout_s=90.0)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        fields = {**body, **MODEL_FIELDS, "max_tokens": MISTRAL_MAX_TOKENS}
        return await asyncio.to_thread(self._client.chat, self._model, fields)


class MistralModel:
    """Mistral Large 4 on Mistral's API. The key comes from `Settings`, never logged."""

    def __init__(self, api_key: SecretStr, model: str = MISTRAL_MODEL) -> None:
        self._client = MistralClient(api_key, timeout_s=90.0)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        fields = {**body, **MISTRAL_FIELDS, "max_tokens": MISTRAL_MAX_TOKENS}
        return await asyncio.to_thread(self._client.chat, self._model, fields)


def chat_model(settings: Settings) -> ChatModel | None:
    """The conversation page's model: OpenRouter when OPENROUTER_API_KEY is set, else Mistral's
    own API when MISTRAL_API_KEY is, else None (no key, no conversation page)."""
    if settings.openrouter_api_key is not None:
        return OpenRouterModel(settings.openrouter_api_key)
    if settings.mistral_api_key is not None:
        return MistralModel(settings.mistral_api_key)
    return None
