"""A small client for Mistral's own chat API (api.mistral.ai), for the conversation page.

Mistral's answers carry no cost, so it is worked out here from the tokens and `PRICES`.
Mistral Large 4 always reasons (`reasoning_effort` low, medium or high; "none" is refused) and
then answers in chunks: a thinking chunk, then text. Only the text is kept as the answer; the
thinking goes to `ChatAnswer.thinking` for the logs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from miroir.config import MISTRAL_API_BASE
from miroir.http import Retry, send
from miroir.jobs.openrouter import JSON, ChatAnswer, WireToolCall, tool_call_of

logger = logging.getLogger(__name__)

#: Dollars per million tokens: (prompt, cached prompt, completion). Mistral's price page did
#: not list Large 4 on its release day (2026-10-29); these are the highest prices seen quoted
#: for it, so the spending caps err on the safe side. Check the console's bill and lower them.
PRICES: Mapping[str, tuple[float, float, float]] = {
    "mistral-large-4": (2.0, 0.2, 6.0),
    "mistral-large-4-0": (2.0, 0.2, 6.0),
}


class MistralError(RuntimeError):
    """Mistral refused a request, or answered with something we cannot read."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        """The HTTP status, when Mistral answered with an error status."""


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _PromptDetails(_Wire):
    cached_tokens: int | None = None


class _Usage(_Wire):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _PromptDetails | None = None


class _Chunk(_Wire):
    type: str = "text"
    text: str | None = None
    thinking: list[_Chunk] | None = None


class _Message(_Wire):
    content: str | list[_Chunk] | None = None
    tool_calls: list[WireToolCall] | None = None


class _Choice(_Wire):
    message: _Message
    finish_reason: str | None = None


class _Response(_Wire):
    model: str = ""
    choices: list[_Choice]
    usage: _Usage | None = None


class MistralClient:
    """Chat completions on Mistral's API, priced from `PRICES`. Retries as `miroir.http` does."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        transport: httpx.BaseTransport | None = None,
        retry: Retry | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._http = httpx.Client(
            base_url=MISTRAL_API_BASE,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            transport=transport,
            timeout=timeout_s,
        )
        self._retry = retry or Retry()

    def close(self) -> None:
        self._http.close()

    def chat(self, model: str, body: Mapping[str, JSON]) -> ChatAnswer:
        """One chat completion. `body` holds `messages` and any other request fields."""
        if model not in PRICES:
            msg = f"no price for {model}: add it to PRICES before spending on it"
            raise MistralError(msg)
        payload: dict[str, JSON] = {**body, "model": model}
        request = self._http.build_request("POST", "/chat/completions", json=payload)
        response = send(self._http, request, self._retry)
        if response.status_code >= 400:
            msg = f"POST /chat/completions: HTTP {response.status_code}: {response.text[:300]}"
            raise MistralError(msg, response.status_code)
        try:
            parsed = _Response.model_validate_json(response.content)
        except ValidationError as err:
            msg = f"unexpected answer from Mistral: {err}"
            raise MistralError(msg) from err
        return _answer(parsed, PRICES[model])


def _answer(response: _Response, prices: tuple[float, float, float]) -> ChatAnswer:
    if not response.choices:
        msg = "chat answer holds no choice"
        raise MistralError(msg)
    choice = response.choices[0]
    usage = response.usage or _Usage()
    details = usage.prompt_tokens_details
    cached = (details.cached_tokens or 0) if details is not None else 0
    prompt, cached_price, completion = prices
    cost = (
        (usage.prompt_tokens - cached) * prompt
        + cached * cached_price
        + usage.completion_tokens * completion
    ) / 1e6
    text, thinking = _split(choice.message.content)
    return ChatAnswer(
        text=text,
        cost_usd=cost,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        finish_reason=choice.finish_reason or "",
        tool_calls=tuple(tool_call_of(c) for c in choice.message.tool_calls or ()),
        cached_tokens=cached,
        thinking=thinking,
    )


def _split(content: str | list[_Chunk] | None) -> tuple[str, str]:
    """The answer's text and its thinking, each as one string."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return content, ""
    text = "".join(c.text or "" for c in content if c.type == "text")
    thinking = "\n".join(
        t.text or "" for c in content if c.type == "thinking" for t in c.thinking or ()
    )
    return text, thinking
