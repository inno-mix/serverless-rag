"""Bedrock wrappers for embedding and generation, with retry and backoff.

Isolates every Bedrock call behind one module so the rest of the codebase stays
unit-testable without AWS (TESTING.md §2). Model ids are read from the
environment, never hardcoded, so they can be swapped through `cdk.json` context
alone (ADR-005).

Embedding arrived in T1.3 and generation in T2.2. Both live here because
ARCHITECTURE.md §2 designates this module "Bedrock client wrappers" and both
share the same retry policy, the same error taxonomy, and the same
`InvokeModel` plumbing.
"""

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

#: Request format required by Anthropic models on Bedrock. This is a Bedrock
#: protocol constant, not a model version, so it is not `cdk.json` context.
ANTHROPIC_VERSION = "bedrock-2023-05-31"

#: Grounded question answering should not be creative: identical context and
#: question should give the same answer twice.
GENERATION_TEMPERATURE = 0.0

#: Attempts per call, per ARCHITECTURE.md §6 ("3 attempts").
MAX_ATTEMPTS = 3

#: Exponential backoff bounds, in seconds. Full jitter is applied on top.
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 8.0

#: Bedrock error codes that represent transient capacity problems.
RETRYABLE_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceQuotaExceededException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",
    }
)

_client: Any | None = None


class BedrockError(Exception):
    """Base class for every failure raised by this module."""


class ModelUnavailableError(BedrockError):
    """Bedrock stayed unavailable after exhausting retries.

    Handlers map this to `503 MODEL_UNAVAILABLE` (API_SPEC.md §6).
    """


class ModelResponseError(BedrockError):
    """Bedrock returned a response this module could not parse."""


@dataclass(frozen=True)
class Generation:
    """A completion together with the token counts API_SPEC.md §4 reports."""

    text: str
    tokens_in: int
    tokens_out: int


def get_client() -> Any:
    """Return the shared Bedrock runtime client.

    The client is created once per execution environment and reused, which is
    the intended pattern for warm Lambda invocations (CLAUDE.md).

    Returns:
        A `bedrock-runtime` boto3 client.
    """
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime")
    return _client


def embedding_model_id() -> str:
    """Read the embedding model id from the environment.

    Returns:
        The configured Bedrock model id.

    Raises:
        BedrockError: If `EMBEDDING_MODEL_ID` is unset, which means the Lambda
            was deployed without the context wiring described in ADR-005.
    """
    return _require_env("EMBEDDING_MODEL_ID")


def generation_model_id() -> str:
    """Read the generation model id from the environment.

    Returns:
        The configured Bedrock model id. In `us-east-1` this is an inference
        profile id rather than a bare model id (ADR-005 amendment).

    Raises:
        BedrockError: If `GENERATION_MODEL_ID` is unset.
    """
    return _require_env("GENERATION_MODEL_ID")


def _require_env(name: str) -> str:
    """Read a required model-id environment variable."""
    value = os.environ.get(name)
    if not value:
        raise BedrockError(f"{name} is not set")
    return value


def embed_texts(
    texts: list[str],
    *,
    client: Any | None = None,
    model_id: str | None = None,
    sleep: Any = time.sleep,
) -> list[np.ndarray]:
    """Embed each text with the configured Bedrock embedding model.

    Args:
        texts: Texts to embed. An empty list short-circuits without calling AWS.
        client: Bedrock runtime client. Defaults to the shared cached client;
            tests inject a stub.
        model_id: Override for the configured model id.
        sleep: Delay function used between retries, injectable for tests.

    Returns:
        One float32 vector per input text, in the same order.

    Raises:
        ModelUnavailableError: If Bedrock kept returning retryable errors.
        ModelResponseError: If a response was missing or had a malformed
            `embedding` field.
        BedrockError: For any non-retryable Bedrock failure.
    """
    if not texts:
        return []

    bedrock = client or get_client()
    resolved_model_id = model_id or embedding_model_id()
    return [_embed_one(bedrock, resolved_model_id, text, sleep) for text in texts]


def generate(
    *,
    system: str,
    user: str,
    max_tokens: int,
    client: Any | None = None,
    model_id: str | None = None,
    sleep: Any = time.sleep,
) -> Generation:
    """Generate an answer with the configured Bedrock generation model.

    Args:
        system: System prompt carrying the grounding rules (ARCHITECTURE.md §5).
        user: User turn, normally the retrieved context plus the question.
        max_tokens: Output cap. The caller passes the configured `maxTokens`
            context value, which NFR-4 requires generation to stay within.
        client: Bedrock runtime client. Defaults to the shared cached client;
            tests inject a stub.
        model_id: Override for the configured model id.
        sleep: Delay function used between retries, injectable for tests.

    Returns:
        The generated text and the token counts reported by the model.

    Raises:
        ModelUnavailableError: If Bedrock kept returning retryable errors.
        ModelResponseError: If the response carried no readable text.
        BedrockError: For any non-retryable Bedrock failure.
    """
    body = json.dumps(
        {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "temperature": GENERATION_TEMPERATURE,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
    )
    bedrock = client or get_client()
    response = _invoke(bedrock, model_id or generation_model_id(), body, sleep)
    return _parse_generation(_read_json(response))


def _embed_one(client: Any, model_id: str, text: str, sleep: Any) -> np.ndarray:
    """Embed a single text, retrying transient Bedrock failures."""
    body = json.dumps({"inputText": text, "normalize": True})
    return _parse_embedding(_read_json(_invoke(client, model_id, body, sleep)))


def _invoke(client: Any, model_id: str, body: str, sleep: Any) -> Any:
    """Call `InvokeModel`, retrying transient Bedrock failures.

    Shared by embedding and generation so both obey the same retry policy
    (ARCHITECTURE.md §6: three attempts, exponential backoff with jitter).
    """
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code not in RETRYABLE_ERROR_CODES:
                raise BedrockError(f"Bedrock rejected the request: {code}") from error
            last_error = error
        except BotoCoreError as error:
            last_error = error
        else:
            return response

        if attempt < MAX_ATTEMPTS - 1:
            sleep(_backoff_seconds(attempt))

    raise ModelUnavailableError(
        f"Bedrock unavailable after {MAX_ATTEMPTS} attempts"
    ) from last_error


def _backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff for the given zero-based attempt."""
    ceiling = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2**attempt))
    return random.uniform(0, ceiling)  # noqa: S311 - jitter, not cryptography


def _read_json(response: Any) -> dict:
    """Decode the streaming body of an `InvokeModel` response."""
    try:
        payload = json.loads(response["body"].read())
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ModelResponseError("Bedrock response body was not readable JSON") from error

    if not isinstance(payload, dict):
        raise ModelResponseError("Bedrock response body was not a JSON object")
    return payload


def _parse_embedding(payload: dict) -> np.ndarray:
    """Extract the embedding vector from a decoded Bedrock response."""
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ModelResponseError("Bedrock response did not contain an embedding")

    try:
        return np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ModelResponseError("Bedrock embedding was not numeric") from error


def _parse_generation(payload: dict) -> Generation:
    """Extract answer text and token usage from a decoded Bedrock response.

    Anthropic models return `content` as a list of typed blocks; only the text
    blocks are joined, so a future block type cannot leak into the answer.
    """
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise ModelResponseError("Bedrock response did not contain a content list")

    text = "".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ).strip()
    if not text:
        raise ModelResponseError("Bedrock response contained no text")

    usage = payload.get("usage") or {}
    return Generation(
        text=text,
        tokens_in=_as_token_count(usage.get("input_tokens")),
        tokens_out=_as_token_count(usage.get("output_tokens")),
    )


def _as_token_count(value: Any) -> int:
    """Coerce a reported token count to `int`, defaulting to 0 when absent."""
    return value if isinstance(value, int) else 0
