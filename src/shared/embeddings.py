"""Bedrock embedding wrapper with retry and backoff (T1.3).

Isolates every Bedrock call behind one module so the rest of the codebase stays
unit-testable without AWS (TESTING.md §2). The embedding model id is read from
the environment, never hardcoded, so it can be swapped through `cdk.json`
context alone (ADR-005).
"""

import json
import os
import random
import time
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

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
    model_id = os.environ.get("EMBEDDING_MODEL_ID")
    if not model_id:
        raise BedrockError("EMBEDDING_MODEL_ID is not set")
    return model_id


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


def _embed_one(client: Any, model_id: str, text: str, sleep: Any) -> np.ndarray:
    """Embed a single text, retrying transient Bedrock failures."""
    body = json.dumps({"inputText": text, "normalize": True})
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
            return _parse_embedding(response)

        if attempt < MAX_ATTEMPTS - 1:
            sleep(_backoff_seconds(attempt))

    raise ModelUnavailableError(
        f"Bedrock unavailable after {MAX_ATTEMPTS} attempts"
    ) from last_error


def _backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff for the given zero-based attempt."""
    ceiling = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2**attempt))
    return random.uniform(0, ceiling)  # noqa: S311 - jitter, not cryptography


def _parse_embedding(response: Any) -> np.ndarray:
    """Extract the embedding vector from a Bedrock InvokeModel response."""
    try:
        payload = json.loads(response["body"].read())
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ModelResponseError("Bedrock response body was not readable JSON") from error

    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ModelResponseError("Bedrock response did not contain an embedding")

    try:
        return np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ModelResponseError("Bedrock embedding was not numeric") from error
