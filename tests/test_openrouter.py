from __future__ import annotations

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from miroir.jobs.openrouter import OpenRouterClient, OpenRouterError, image_input
from tests.fakes import COST_PER_INPUT, NO_WAIT, FakeOpenRouter


def client(fake: FakeOpenRouter) -> OpenRouterClient:
    return OpenRouterClient(SecretStr("sk-test"), transport=fake.transport(), retry=NO_WAIT)


def test_embed_sends_images_as_image_url_items_and_reads_cost() -> None:
    fake = FakeOpenRouter()
    inputs = [image_input("data:image/jpeg;base64,AAAA"), image_input("data:image/jpeg;base64,BB")]
    result = client(fake).embed("google/gemini-embedding-2", inputs)
    assert result.vectors.shape == (2, 4)
    assert result.vectors.dtype == np.float32
    assert result.cost_usd == pytest.approx(2 * COST_PER_INPUT)
    sent = fake.embedding_bodies[0]
    assert sent["model"] == "google/gemini-embedding-2"
    assert sent["input"] == [
        {"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]},
        {"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BB"}}]},
    ]


def test_embed_retries_rate_limits() -> None:
    fake = FakeOpenRouter(fail_first=2)
    assert client(fake).embed("m", ["a"]).vectors.shape == (1, 4)


def test_refusals_raise_with_the_reason() -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "Insufficient credits"}})

    c = OpenRouterClient(SecretStr("k"), transport=httpx.MockTransport(refuse), retry=NO_WAIT)
    with pytest.raises(OpenRouterError, match="Insufficient credits"):
        c.embed("m", ["a"])


def test_an_answer_without_cost_is_refused() -> None:
    def no_cost(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0], "index": 0}]})

    c = OpenRouterClient(SecretStr("k"), transport=httpx.MockTransport(no_cost), retry=NO_WAIT)
    with pytest.raises(OpenRouterError, match="cost"):
        c.embed("m", ["a"])


def test_batch_model_only_when_listed() -> None:
    c = client(FakeOpenRouter(batch_models=("qwen/qwen3-embedding-8b:batch",)))
    assert c.batch_model("qwen/qwen3-embedding-8b") == "qwen/qwen3-embedding-8b:batch"
    assert c.batch_model("voyageai/voyage-multimodal-3.5") is None


def test_embed_batch_submits_polls_and_returns_vectors_by_id() -> None:
    fake = FakeOpenRouter()
    vectors, outcome = client(fake).embed_batch(
        "google/gemini-embedding-2:batch", {"q1": "une église", "q2": "un vase"}, poll_s=0.0
    )
    assert set(vectors) == {"q1", "q2"}
    assert outcome.cost_usd == pytest.approx(COST_PER_INPUT)
    sent = fake.batch_bodies[0]
    assert sent["endpoint"] == "/v1/embeddings"
    assert sent["requests"] == [
        {"custom_id": "q1", "body": {"input": ["une église"], "encoding_format": "float"}},
        {"custom_id": "q2", "body": {"input": ["un vase"], "encoding_format": "float"}},
    ]


def test_key_spend_reads_usage() -> None:
    assert client(FakeOpenRouter(key_usage=3.5)).key_spend().usage_usd == 3.5


def test_the_key_never_appears_in_errors() -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    c = OpenRouterClient(SecretStr("sk-secret"), transport=httpx.MockTransport(refuse))
    with pytest.raises(OpenRouterError) as err:
        c.key_spend()
    assert "sk-secret" not in str(err.value)


def test_gemini_embedding_batch_is_known_without_listing() -> None:
    fake = FakeOpenRouter()
    assert (
        client(fake).batch_model("google/gemini-embedding-2") == "google/gemini-embedding-2:batch"
    )
