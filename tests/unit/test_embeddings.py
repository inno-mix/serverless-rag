"""Unit tests for the Bedrock embedding wrapper (T1.3, TESTING.md §2).

Bedrock is stubbed rather than mocked with a library: the wrapper's whole job is
retry and parsing behaviour, and a hand-written stub makes the call sequence
explicit.
"""

import json

import numpy as np
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from shared import embeddings
from shared.embeddings import (
    ANTHROPIC_VERSION,
    MAX_ATTEMPTS,
    BedrockError,
    ModelResponseError,
    ModelUnavailableError,
    embed_texts,
    generate,
)

MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _completion(text: str = "hello", tokens_in: int = 12, tokens_out: int = 3) -> dict:
    """A well-formed Anthropic-on-Bedrock InvokeModel response payload."""
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }


class _Body:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode()


class _StubBedrock:
    """Returns queued outcomes; an Exception outcome is raised instead."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def invoke_model(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0) if self._outcomes else {"embedding": [0.0]}
        if isinstance(outcome, Exception):
            raise outcome
        return {"body": _Body(outcome)}


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _throttle() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "InvokeModel"
    )


def _validation_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": "bad input"}}, "InvokeModel"
    )


@pytest.fixture(autouse=True)
def _model_id(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_ID", MODEL_ID)
    monkeypatch.setenv("GENERATION_MODEL_ID", GENERATION_MODEL_ID)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_empty_input_never_calls_bedrock():
    client = _StubBedrock()
    assert embed_texts([], client=client) == []
    assert client.calls == []


def test_returns_one_float32_vector_per_text_in_order():
    client = _StubBedrock({"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]})
    vectors = embed_texts(["first", "second"], client=client)

    assert len(vectors) == 2
    assert all(v.dtype == np.float32 for v in vectors)
    assert np.allclose(vectors[0], [1.0, 2.0])
    assert np.allclose(vectors[1], [3.0, 4.0])


def test_sends_the_configured_model_id_and_input_text():
    client = _StubBedrock({"embedding": [1.0]})
    embed_texts(["hello"], client=client)

    (call,) = client.calls
    assert call["modelId"] == MODEL_ID
    assert json.loads(call["body"])["inputText"] == "hello"


def test_explicit_model_id_overrides_the_environment():
    client = _StubBedrock({"embedding": [1.0]})
    embed_texts(["hello"], client=client, model_id="override-model")
    assert client.calls[0]["modelId"] == "override-model"


def test_missing_model_id_env_var_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    with pytest.raises(BedrockError, match="EMBEDDING_MODEL_ID"):
        embed_texts(["hello"], client=_StubBedrock({"embedding": [1.0]}))


# --------------------------------------------------------------------------- #
# Retry behaviour (ARCHITECTURE.md §6)
# --------------------------------------------------------------------------- #


def test_throttling_is_retried_then_succeeds():
    client = _StubBedrock(_throttle(), {"embedding": [7.0]})
    sleeps = _Sleeps()

    (vector,) = embed_texts(["hello"], client=client, sleep=sleeps)

    assert np.allclose(vector, [7.0])
    assert len(client.calls) == 2
    assert len(sleeps.delays) == 1


def test_persistent_throttling_raises_model_unavailable_after_three_attempts():
    client = _StubBedrock(*[_throttle() for _ in range(MAX_ATTEMPTS)])
    sleeps = _Sleeps()

    with pytest.raises(ModelUnavailableError, match="after 3 attempts"):
        embed_texts(["hello"], client=client, sleep=sleeps)

    assert len(client.calls) == MAX_ATTEMPTS
    assert len(sleeps.delays) == MAX_ATTEMPTS - 1


def test_backoff_grows_and_stays_within_bounds():
    ceilings = [
        min(embeddings.MAX_DELAY_SECONDS, embeddings.BASE_DELAY_SECONDS * (2**attempt))
        for attempt in range(6)
    ]
    for attempt, ceiling in enumerate(ceilings):
        for _ in range(50):
            assert 0 <= embeddings._backoff_seconds(attempt) <= ceiling
    assert ceilings[-1] == embeddings.MAX_DELAY_SECONDS


def test_transport_errors_are_retried():
    client = _StubBedrock(
        EndpointConnectionError(endpoint_url="https://bedrock"), {"embedding": [1.0]}
    )
    (vector,) = embed_texts(["hello"], client=client, sleep=_Sleeps())
    assert np.allclose(vector, [1.0])


def test_non_retryable_errors_fail_immediately():
    client = _StubBedrock(_validation_error())
    sleeps = _Sleeps()

    with pytest.raises(BedrockError, match="ValidationException"):
        embed_texts(["hello"], client=client, sleep=sleeps)

    assert len(client.calls) == 1
    assert sleeps.delays == []


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {"inputTextTokenCount": 3},  # embedding key absent
        {"embedding": []},  # empty vector
        {"embedding": "not-a-list"},
        b"{not json",
    ],
)
def test_malformed_responses_raise_a_typed_error(payload):
    """A bad payload must not surface as KeyError or ValueError (TESTING.md §2)."""
    with pytest.raises(ModelResponseError):
        embed_texts(["hello"], client=_StubBedrock(payload))


def test_non_numeric_embedding_raises_a_typed_error():
    with pytest.raises(ModelResponseError):
        embed_texts(["hello"], client=_StubBedrock({"embedding": [1.0, "abc"]}))


# --------------------------------------------------------------------------- #
# Generation (T2.2)
# --------------------------------------------------------------------------- #


def test_generate_returns_text_and_token_usage():
    client = _StubBedrock(_completion("Revenue was $4.2M.", tokens_in=1450, tokens_out=210))
    result = generate(system="rules", user="context", max_tokens=1024, client=client)

    assert result.text == "Revenue was $4.2M."
    assert result.tokens_in == 1450
    assert result.tokens_out == 210


def test_generate_sends_the_bedrock_anthropic_request_shape():
    client = _StubBedrock(_completion())
    generate(system="rules", user="question", max_tokens=512, client=client)

    (call,) = client.calls
    body = json.loads(call["body"])
    assert call["modelId"] == GENERATION_MODEL_ID
    assert body["anthropic_version"] == ANTHROPIC_VERSION
    assert body["max_tokens"] == 512
    assert body["system"] == "rules"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "question"}]}]


def test_generate_pins_temperature_to_zero():
    """Grounded answering is not a creative task; repeat runs should match."""
    client = _StubBedrock(_completion())
    generate(system="rules", user="question", max_tokens=64, client=client)
    assert json.loads(client.calls[0]["body"])["temperature"] == 0.0


def test_generate_respects_the_max_tokens_cap():
    """NFR-4 caps generation at the configured maxTokens context value."""
    client = _StubBedrock(_completion())
    generate(system="rules", user="question", max_tokens=1024, client=client)
    assert json.loads(client.calls[0]["body"])["max_tokens"] <= 1024


def test_generate_joins_text_blocks_and_ignores_other_block_types():
    payload = {
        "content": [
            {"type": "text", "text": "first "},
            {"type": "thinking", "thinking": "should not leak"},
            {"type": "text", "text": "second"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    result = generate(system="s", user="u", max_tokens=64, client=_StubBedrock(payload))
    assert result.text == "first second"


def test_generate_defaults_missing_usage_to_zero():
    client = _StubBedrock({"content": [{"type": "text", "text": "answer"}]})
    result = generate(system="s", user="u", max_tokens=64, client=client)
    assert (result.tokens_in, result.tokens_out) == (0, 0)


def test_generate_explicit_model_id_overrides_the_environment():
    client = _StubBedrock(_completion())
    generate(system="s", user="u", max_tokens=64, client=client, model_id="override")
    assert client.calls[0]["modelId"] == "override"


def test_generate_missing_model_id_env_var_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("GENERATION_MODEL_ID", raising=False)
    with pytest.raises(BedrockError, match="GENERATION_MODEL_ID"):
        generate(system="s", user="u", max_tokens=64, client=_StubBedrock(_completion()))


def test_generate_shares_the_embedding_retry_policy():
    client = _StubBedrock(_throttle(), _completion("recovered"))
    sleeps = _Sleeps()

    result = generate(system="s", user="u", max_tokens=64, client=client, sleep=sleeps)

    assert result.text == "recovered"
    assert len(client.calls) == 2
    assert len(sleeps.delays) == 1


def test_generate_raises_model_unavailable_after_three_attempts():
    client = _StubBedrock(*[_throttle() for _ in range(MAX_ATTEMPTS)])
    with pytest.raises(ModelUnavailableError, match="after 3 attempts"):
        generate(system="s", user="u", max_tokens=64, client=client, sleep=_Sleeps())


@pytest.mark.parametrize(
    "payload",
    [
        {"usage": {"input_tokens": 1, "output_tokens": 1}},  # no content key
        {"content": "not-a-list"},
        {"content": []},  # no blocks at all
        {"content": [{"type": "text", "text": "   "}]},  # whitespace only
        {"content": [{"type": "tool_use", "id": "t1"}]},  # no text block
        b"[]",  # valid JSON, but not an object
    ],
)
def test_generate_malformed_responses_raise_a_typed_error(payload):
    with pytest.raises(ModelResponseError):
        generate(system="s", user="u", max_tokens=64, client=_StubBedrock(payload))


# --------------------------------------------------------------------------- #
# Client caching
# --------------------------------------------------------------------------- #


def test_client_is_created_once_and_reused(monkeypatch):
    """Module-level client caching is the intended warm-start pattern (CLAUDE.md)."""
    created: list[str] = []

    def fake_boto_client(service: str, *args: object, **kwargs: object) -> object:
        created.append(service)
        return object()

    monkeypatch.setattr(embeddings, "_client", None)
    monkeypatch.setattr(embeddings.boto3, "client", fake_boto_client)

    first, second = embeddings.get_client(), embeddings.get_client()

    assert first is second
    assert created == ["bedrock-runtime"]
