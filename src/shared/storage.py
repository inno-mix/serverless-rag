"""DynamoDB access patterns for document metadata and chunk vectors (T1.3).

Implements the single-table design in ARCHITECTURE.md §3:

===============  ==================  =======================
Entity           PK                  SK
===============  ==================  =======================
Document meta    ``DOC#{doc_id}``    ``META``
Chunk            ``DOC#{doc_id}``    ``CHUNK#{n:05d}``
===============  ==================  =======================

Document metadata additionally carries ``GSI1PK=DOCLIST`` and
``GSI1SK={uploaded_at}`` so the listing query returns documents newest-first.

Vectors are stored as packed float32 in a DynamoDB Binary attribute — roughly a
quarter the size of a list of Numbers, and far cheaper to parse. DynamoDB
base64-encodes Binary attributes on the wire, which is the "b64-packed float32"
described in ARCHITECTURE.md §3. A companion ``dim`` attribute records the
vector length so a corrupt or truncated item is detected on read.

Every function takes the table resource as its first argument rather than
reaching for a module-level client, so tests can pass a moto-backed table.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

import numpy as np
from boto3.dynamodb.conditions import Attr, Key
from boto3.dynamodb.types import Binary

#: Partition key of the GSI1 entry that lists every document.
DOC_LIST_PARTITION = "DOCLIST"

#: Sort-key prefix marking a chunk item.
CHUNK_PREFIX = "CHUNK#"

#: Sort key of a document's metadata item.
META_SORT_KEY = "META"

#: DynamoDB rejects batch writes larger than 25 items.
MAX_BATCH_SIZE = 25

#: Prefix for uploaded objects in the documents bucket. The key carries the
#: document id so the ingest Lambda, which only receives an S3 event, can find
#: the matching metadata item without a lookup table.
UPLOAD_PREFIX = "uploads/"


def build_s3_key(doc_id: str, filename: str) -> str:
    """Return the documents-bucket key for an upload.

    Args:
        doc_id: Document identifier.
        filename: Original filename, preserved so ingestion can dispatch on the
            extension and citations can show a human-readable name.

    Returns:
        The object key, of the form ``uploads/{doc_id}/{filename}``.
    """
    return f"{UPLOAD_PREFIX}{doc_id}/{filename}"


def parse_s3_key(key: str) -> tuple[str, str]:
    """Recover the document id and filename from an upload key.

    Args:
        key: An object key produced by :func:`build_s3_key`.

    Returns:
        A ``(doc_id, filename)`` pair.

    Raises:
        ValueError: If `key` does not follow the upload convention, which means
            the object was not created through the presign flow.
    """
    if not key.startswith(UPLOAD_PREFIX):
        raise ValueError(f"key does not start with {UPLOAD_PREFIX!r}: {key!r}")

    doc_id, separator, filename = key[len(UPLOAD_PREFIX) :].partition("/")
    if not separator or not doc_id or not filename:
        raise ValueError(f"key is not of the form uploads/<doc_id>/<filename>: {key!r}")
    return doc_id, filename


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document (PRD FR-1.5)."""

    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk ready to be written, with its embedding."""

    index: int
    text: str
    token_count: int
    vector: np.ndarray


@dataclass(frozen=True)
class StoredChunk:
    """A chunk read back from DynamoDB."""

    doc_id: str
    chunk_id: str
    text: str
    token_count: int
    vector: np.ndarray


@dataclass(frozen=True)
class DocumentSummary:
    """Document metadata as returned by `GET /documents` (API_SPEC.md §2)."""

    doc_id: str
    filename: str
    status: str
    uploaded_at: str
    chunk_count: int
    error_msg: str | None


def pack_vector(vector: np.ndarray) -> bytes:
    """Pack an embedding into little-endian float32 bytes.

    Args:
        vector: The embedding to pack.

    Returns:
        The raw float32 bytes, ready for a DynamoDB Binary attribute.
    """
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_vector(data: bytes | Binary) -> np.ndarray:
    """Restore an embedding packed by :func:`pack_vector`.

    Args:
        data: The stored bytes, or the `Binary` wrapper boto3 returns on read.

    Returns:
        A float32 vector. The array is copied so callers may mutate it safely.
    """
    raw = bytes(data.value) if isinstance(data, Binary) else bytes(data)
    return np.frombuffer(raw, dtype=np.float32).copy()


def chunk_id_for(index: int) -> str:
    """Return the citation-facing chunk id for a zero-based chunk index."""
    return f"chunk_{index:05d}"


def put_document_meta(
    table: Any,
    *,
    doc_id: str,
    filename: str,
    s3_key: str,
    uploaded_at: str | None = None,
    status: DocumentStatus = DocumentStatus.UPLOADED,
) -> str:
    """Create the metadata item for a newly uploaded document.

    Args:
        table: The DynamoDB table resource.
        doc_id: Document identifier.
        filename: Original filename supplied by the client.
        s3_key: Key of the uploaded object in the documents bucket.
        uploaded_at: ISO-8601 timestamp. Defaults to now, in UTC.
        status: Initial status.

    Returns:
        The `uploaded_at` timestamp actually stored, which doubles as the GSI1
        sort key.
    """
    timestamp = uploaded_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    table.put_item(
        Item={
            "PK": f"DOC#{doc_id}",
            "SK": META_SORT_KEY,
            "GSI1PK": DOC_LIST_PARTITION,
            "GSI1SK": timestamp,
            "filename": filename,
            "s3_key": s3_key,
            "status": str(status),
            "uploaded_at": timestamp,
            "chunk_count": 0,
        }
    )
    return timestamp


def update_status(
    table: Any,
    doc_id: str,
    status: DocumentStatus,
    *,
    chunk_count: int | None = None,
    error_msg: str | None = None,
) -> None:
    """Move a document to a new lifecycle state.

    Args:
        table: The DynamoDB table resource.
        doc_id: Document identifier.
        status: The new status.
        chunk_count: Number of chunks stored, set when a document becomes READY.
        error_msg: Failure reason, set when a document becomes FAILED.
    """
    expressions = ["#s = :s"]
    names: dict[str, str] = {"#s": "status"}
    values: dict[str, Any] = {":s": str(status)}

    if chunk_count is not None:
        expressions.append("chunk_count = :c")
        values[":c"] = chunk_count
    if error_msg is not None:
        expressions.append("error_msg = :e")
        values[":e"] = error_msg

    table.update_item(
        Key={"PK": f"DOC#{doc_id}", "SK": META_SORT_KEY},
        UpdateExpression="SET " + ", ".join(expressions),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def put_chunks(table: Any, doc_id: str, chunks: Sequence[ChunkRecord]) -> int:
    """Write a document's chunks and their vectors.

    Uses `batch_writer`, which buffers and flushes in DynamoDB's 25-item batches,
    so callers may pass any number of chunks.

    Args:
        table: The DynamoDB table resource.
        doc_id: Document identifier.
        chunks: The chunks to store.

    Returns:
        The number of chunks written.
    """
    with table.batch_writer() as batch:
        for chunk in chunks:
            vector = np.asarray(chunk.vector, dtype=np.float32)
            batch.put_item(
                Item={
                    "PK": f"DOC#{doc_id}",
                    "SK": f"{CHUNK_PREFIX}{chunk.index:05d}",
                    "text": chunk.text,
                    "vector": pack_vector(vector),
                    "dim": int(vector.size),
                    "token_count": int(chunk.token_count),
                }
            )
    return len(chunks)


def get_all_chunks(table: Any) -> list[StoredChunk]:
    """Read every chunk in the corpus for brute-force similarity search.

    This is the O(n) scan accepted by ADR-001. It is the migration trigger to
    watch: beyond roughly 50k chunks this scan, not the vector math, is what
    makes queries slow.

    Args:
        table: The DynamoDB table resource.

    Returns:
        Every stored chunk, in no guaranteed order.
    """
    chunks: list[StoredChunk] = []
    kwargs: dict[str, Any] = {"FilterExpression": Attr("SK").begins_with(CHUNK_PREFIX)}

    while True:
        page = table.scan(**kwargs)
        for item in page.get("Items", []):
            chunks.append(
                StoredChunk(
                    doc_id=item["PK"].removeprefix("DOC#"),
                    chunk_id=item["SK"].replace(CHUNK_PREFIX, "chunk_"),
                    text=item["text"],
                    token_count=_as_int(item.get("token_count", 0)),
                    vector=unpack_vector(item["vector"]),
                )
            )
        cursor = page.get("LastEvaluatedKey")
        if not cursor:
            return chunks
        kwargs["ExclusiveStartKey"] = cursor


def list_documents(table: Any) -> list[DocumentSummary]:
    """List every document, newest first.

    Args:
        table: The DynamoDB table resource.

    Returns:
        Document summaries ordered by `uploaded_at` descending, served from GSI1.
    """
    documents: list[DocumentSummary] = []
    kwargs: dict[str, Any] = {
        "IndexName": "GSI1",
        "KeyConditionExpression": Key("GSI1PK").eq(DOC_LIST_PARTITION),
        "ScanIndexForward": False,
    }

    while True:
        page = table.query(**kwargs)
        for item in page.get("Items", []):
            documents.append(
                DocumentSummary(
                    doc_id=item["PK"].removeprefix("DOC#"),
                    filename=item.get("filename", ""),
                    status=item.get("status", ""),
                    uploaded_at=item.get("uploaded_at", ""),
                    chunk_count=_as_int(item.get("chunk_count", 0)),
                    error_msg=item.get("error_msg"),
                )
            )
        cursor = page.get("LastEvaluatedKey")
        if not cursor:
            return documents
        kwargs["ExclusiveStartKey"] = cursor


def delete_document(table: Any, doc_id: str) -> bool:
    """Delete a document's metadata and all of its chunks.

    Idempotent: deleting an unknown `doc_id` is a no-op that returns `False`.
    Removing the S3 object is the caller's responsibility, since this module
    owns DynamoDB access only.

    Args:
        table: The DynamoDB table resource.
        doc_id: Document identifier.

    Returns:
        `True` if anything was deleted.
    """
    keys = _all_item_keys(table, doc_id)
    if not keys:
        return False

    with table.batch_writer() as batch:
        for key in keys:
            batch.delete_item(Key=key)
    return True


def delete_chunks(table: Any, doc_id: str) -> int:
    """Delete only a document's chunks, leaving its metadata in place.

    Used to clean up partial writes when ingestion fails part-way, so a FAILED
    document never leaves orphaned chunks in the corpus (ARCHITECTURE.md §6).

    Args:
        table: The DynamoDB table resource.
        doc_id: Document identifier.

    Returns:
        The number of chunk items deleted.
    """
    keys = [k for k in _all_item_keys(table, doc_id) if k["SK"].startswith(CHUNK_PREFIX)]
    if keys:
        with table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(Key=key)
    return len(keys)


def _all_item_keys(table: Any, doc_id: str) -> list[dict[str, str]]:
    """Collect the primary keys of every item belonging to a document."""
    keys: list[dict[str, str]] = []
    kwargs: dict[str, Any] = {
        "KeyConditionExpression": Key("PK").eq(f"DOC#{doc_id}"),
        "ProjectionExpression": "PK, SK",
    }

    while True:
        page = table.query(**kwargs)
        keys.extend({"PK": item["PK"], "SK": item["SK"]} for item in page.get("Items", []))
        cursor = page.get("LastEvaluatedKey")
        if not cursor:
            return keys
        kwargs["ExclusiveStartKey"] = cursor


def _as_int(value: Any) -> int:
    """Coerce a DynamoDB number (returned as `Decimal`) to `int`."""
    return int(value) if isinstance(value, Decimal | int) else 0
