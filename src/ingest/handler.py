"""S3 ObjectCreated handler: extract, chunk, embed, and store a document (T1.4).

The handler is a thin adapter (CLAUDE.md): it parses the S3 event, resolves
clients, and delegates to :func:`ingest_object`, which takes every dependency as
an argument so the pipeline can be unit-tested without AWS.

Failure handling follows ARCHITECTURE.md §6. Any failure marks the document
FAILED with a machine-readable reason, deletes any chunks already written so a
failed document never leaves orphans in the corpus, and posts a message to the
dead-letter queue.
"""

import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import boto3
import numpy as np
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from shared.chunking import chunk_text
from shared.embeddings import BedrockError, ModelUnavailableError, embed_texts
from shared.logging_utils import get_logger
from shared.storage import (
    ChunkRecord,
    DocumentStatus,
    delete_chunks,
    parse_s3_key,
    put_chunks,
    update_status,
)

#: File types the extractor understands (PRD FR-1.3).
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})

DEFAULT_TARGET_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

log = get_logger(__name__)

_s3: Any | None = None
_sqs: Any | None = None
_table: Any | None = None


class IngestError(Exception):
    """A failure that maps to a recorded document status."""

    code = "INTERNAL"


class UnsupportedTypeError(IngestError):
    """The uploaded file has an extension the extractor does not handle."""

    code = "UNSUPPORTED_TYPE"


class EmptyDocumentError(IngestError):
    """Extraction succeeded but produced no usable text."""

    code = "EMPTY_DOCUMENT"


class ExtractionError(IngestError):
    """The file matched a supported extension but could not be read."""

    code = "EXTRACTION_FAILED"


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting a single uploaded object."""

    doc_id: str
    status: DocumentStatus
    chunk_count: int = 0
    error_code: str | None = None


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded document.

    Args:
        filename: Original filename; only its extension is used to dispatch.
        data: Raw file bytes.

    Returns:
        The extracted text. Pages of a PDF are joined with blank lines so the
        chunker treats page breaks as paragraph breaks.

    Raises:
        UnsupportedTypeError: If the extension is not one of `.pdf`, `.txt`, `.md`.
        ExtractionError: If a supported file could not be parsed or decoded.
    """
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedTypeError(f"unsupported extension: {suffix or filename!r}")

    if suffix == ".pdf":
        return _extract_pdf(data)

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExtractionError(f"{filename} is not valid UTF-8") from error


def _extract_pdf(data: bytes) -> str:
    """Extract text from every page of a PDF."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, ValueError, OSError) as error:
        raise ExtractionError("could not read PDF") from error
    return "\n\n".join(page.strip() for page in pages if page.strip())


def ingest_object(
    *,
    table: Any,
    s3_client: Any,
    bucket: str,
    key: str,
    sqs_client: Any | None = None,
    dlq_url: str | None = None,
    embed: Callable[[list[str]], list[np.ndarray]] = embed_texts,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> IngestResult:
    """Run the full ingestion pipeline for one uploaded object.

    Args:
        table: DynamoDB table resource.
        s3_client: S3 client used to download the upload.
        bucket: Bucket holding the upload.
        key: Object key, following the `uploads/{doc_id}/{filename}` convention.
        sqs_client: SQS client used to post failures to the DLQ.
        dlq_url: Dead-letter queue URL; failures are not posted when unset.
        embed: Embedding function, injectable for tests.
        target_tokens: Chunk size passed to the chunker.
        overlap_tokens: Chunk overlap passed to the chunker.

    Returns:
        The outcome, including the reason when ingestion failed.

    Raises:
        ValueError: If `key` does not follow the upload convention, since there
            is then no document to mark as failed.
    """
    doc_id, filename = parse_s3_key(key)
    log.info("ingest.started", doc_id=doc_id, filename=filename, bucket=bucket)

    try:
        update_status(table, doc_id, DocumentStatus.PROCESSING)
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()

        text = extract_text(filename, body)
        chunks = chunk_text(text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        if not chunks:
            raise EmptyDocumentError(f"{filename} produced no text")

        vectors = embed([chunk.text for chunk in chunks])
        records = [
            ChunkRecord(
                index=chunk.index,
                text=chunk.text,
                token_count=chunk.token_count,
                vector=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        put_chunks(table, doc_id, records)
        update_status(table, doc_id, DocumentStatus.READY, chunk_count=len(records))
    except IngestError as error:
        return _fail(table, sqs_client, dlq_url, doc_id, error.code, str(error))
    except ModelUnavailableError as error:
        return _fail(table, sqs_client, dlq_url, doc_id, "MODEL_UNAVAILABLE", str(error))
    except BedrockError as error:
        return _fail(table, sqs_client, dlq_url, doc_id, "EMBEDDING_FAILED", str(error))
    except Exception as error:  # noqa: BLE001 - last resort, always recorded
        return _fail(table, sqs_client, dlq_url, doc_id, "INTERNAL", str(error))

    log.info("ingest.completed", doc_id=doc_id, filename=filename, chunk_count=len(records))
    return IngestResult(doc_id=doc_id, status=DocumentStatus.READY, chunk_count=len(records))


def _fail(
    table: Any,
    sqs_client: Any | None,
    dlq_url: str | None,
    doc_id: str,
    code: str,
    detail: str,
) -> IngestResult:
    """Record a failure, clean up partial writes, and notify the DLQ."""
    log.error("ingest.failed", doc_id=doc_id, reason=code, detail=detail)

    removed = delete_chunks(table, doc_id)
    if removed:
        log.info("ingest.partial_chunks_removed", doc_id=doc_id, removed=removed)
    update_status(table, doc_id, DocumentStatus.FAILED, error_msg=code)

    if sqs_client and dlq_url:
        sqs_client.send_message(
            QueueUrl=dlq_url,
            MessageBody=json.dumps({"doc_id": doc_id, "reason": code, "detail": detail}),
        )

    return IngestResult(doc_id=doc_id, status=DocumentStatus.FAILED, error_code=code)


def parse_event(event: dict) -> tuple[str, str]:
    """Pull the bucket and object key out of an EventBridge S3 event.

    Uploads reach this function through EventBridge rather than a direct bucket
    notification (ADR-006), so the payload is a single `Object Created` detail
    rather than a `Records` array. EventBridge delivers the key verbatim, so no
    URL-decoding is required.

    Args:
        event: The EventBridge event payload.

    Returns:
        A `(bucket, key)` pair.

    Raises:
        ValueError: If the payload is not a recognizable S3 `Object Created` event.
    """
    detail = event.get("detail") if isinstance(event, dict) else None
    if not isinstance(detail, dict):
        raise ValueError("event has no 'detail' object")

    bucket = detail.get("bucket", {}).get("name")
    key = detail.get("object", {}).get("key")
    if not bucket or not key:
        raise ValueError("event detail is missing bucket name or object key")
    return bucket, key


def handler(event: dict, context: Any = None) -> dict:  # noqa: ARG001 - Lambda signature
    """Lambda entry point for S3 `Object Created` events delivered by EventBridge.

    Args:
        event: The EventBridge event payload.
        context: Lambda context object, unused.

    Returns:
        A summary of the object processed in this invocation.
    """
    try:
        bucket, key = parse_event(event)
    except ValueError as error:
        log.error("ingest.unrecognized_event", detail=str(error))
        return {"processed": []}

    try:
        outcome = ingest_object(
            table=_get_table(),
            s3_client=_get_s3(),
            bucket=bucket,
            key=key,
            sqs_client=_get_sqs(),
            dlq_url=os.environ.get("DLQ_URL"),
            target_tokens=int(os.environ.get("CHUNK_TARGET_TOKENS", DEFAULT_TARGET_TOKENS)),
            overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", DEFAULT_OVERLAP_TOKENS)),
        )
    except ValueError:
        # The key is not one of ours, so there is no document to mark FAILED.
        log.error("ingest.unrecognized_key", bucket=bucket, key=key)
        return {"processed": []}

    return {
        "processed": [
            {
                "doc_id": outcome.doc_id,
                "status": str(outcome.status),
                "chunk_count": outcome.chunk_count,
                "error_code": outcome.error_code,
            }
        ]
    }


def _get_table() -> Any:
    """Return the cached DynamoDB table resource."""
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    return _table


def _get_s3() -> Any:
    """Return the cached S3 client."""
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _get_sqs() -> Any:
    """Return the cached SQS client."""
    global _sqs
    if _sqs is None:
        _sqs = boto3.client("sqs")
    return _sqs
