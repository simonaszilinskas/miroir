from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from miroir.jobs.mistral import MistralClient, MistralError
from tests.fakes import NO_WAIT

#: An answer as Mistral Large 4 sends it: a thinking chunk, then a tool call (2026-10-29).
ANSWER = {
    "model": "mistral-large-4",
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 200},
    },
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": "Search first."}]},
                    {"type": "text", "text": "Je cherche."},
                ],
                "tool_calls": [
                    {"id": "tDRPW92uI", "function": {"name": "text_search", "arguments": "{}"}}
                ],
            },
        }
    ],
}


def _client(handler: httpx.MockTransport) -> MistralClient:
    return MistralClient(SecretStr("k"), transport=handler, retry=NO_WAIT)


def test_chat_splits_thinking_from_text_and_prices_tokens() -> None:
    sent: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=ANSWER)

    answer = _client(httpx.MockTransport(handle)).chat("mistral-large-4", {"messages": []})
    assert sent[0]["model"] == "mistral-large-4"
    assert answer.text == "Je cherche."
    assert answer.thinking == "Search first."
    assert answer.tool_calls[0].name == "text_search"
    assert answer.cached_tokens == 200
    assert answer.cost_usd == pytest.approx((800 * 2.0 + 200 * 0.2 + 100 * 6.0) / 1e6)


def test_chat_refuses_a_model_without_a_price() -> None:
    client = _client(httpx.MockTransport(lambda _: httpx.Response(200, json=ANSWER)))
    with pytest.raises(MistralError, match="no price"):
        client.chat("mistral-small-latest", {"messages": []})


def test_chat_raises_on_http_errors() -> None:
    client = _client(httpx.MockTransport(lambda _: httpx.Response(400, json={"message": "no"})))
    with pytest.raises(MistralError, match="HTTP 400"):
        client.chat("mistral-large-4", {"messages": []})


def test_chat_reads_plain_text_object_arguments_and_no_usage() -> None:
    answer = {
        "choices": [
            {
                "message": {
                    "content": "Voici.",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "more_like", "arguments": {"id": "É1"}}}
                    ],
                }
            }
        ]
    }
    client = _client(httpx.MockTransport(lambda _: httpx.Response(200, json=answer)))
    got = client.chat("mistral-large-4", {"messages": []})
    assert (got.text, got.thinking, got.cost_usd, got.finish_reason) == ("Voici.", "", 0.0, "")
    assert json.loads(got.tool_calls[0].arguments) == {"id": "É1"}


@pytest.mark.parametrize(
    ("body", "error"),
    [({"choices": []}, "no choice"), ({"choices": "nope"}, "unexpected answer")],
)
def test_chat_raises_on_an_answer_it_cannot_use(body: dict[str, object], error: str) -> None:
    client = _client(httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    with pytest.raises(MistralError, match=error):
        client.chat("mistral-large-4", {"messages": []})
