"""Unit tests for the DynamoDB access layer (T1.3, TESTING.md §2).

Backed by moto, so the single-table design, the GSI, and DynamoDB's 25-item
batch limit are exercised for real without touching AWS.
"""

import boto3
import numpy as np
import pytest
from moto import mock_aws

from shared.storage import (
    ChunkRecord,
    DocumentStatus,
    chunk_id_for,
    delete_chunks,
    delete_document,
    get_all_chunks,
    list_documents,
    pack_vector,
    put_chunks,
    put_document_meta,
    unpack_vector,
    update_status,
)


@pytest.fixture(autouse=True)
def _aws_credentials(monkeypatch):
    """Moto refuses to run against ambient credentials; give it dummies."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def table():
    """A moto-backed table mirroring StorageStack's schema."""
    with mock_aws():
        resource = boto3.resource("dynamodb", region_name="us-east-1")
        yield resource.create_table(
            TableName="rag-store",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )


def _records(count: int, dim: int = 4) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            index=i,
            text=f"chunk text {i}",
            token_count=10 + i,
            vector=np.arange(dim, dtype=np.float32) + i,
        )
        for i in range(count)
    ]


# --------------------------------------------------------------------------- #
# Vector packing
# --------------------------------------------------------------------------- #


def test_vector_round_trip_preserves_values_and_dtype():
    original = np.array([0.1, -0.25, 3.5, 1e-4], dtype=np.float32)
    restored = unpack_vector(pack_vector(original))

    assert restored.dtype == np.float32
    assert np.allclose(original, restored)


def test_packing_downcasts_float64_input():
    original = np.array([0.5, 1.5], dtype=np.float64)
    restored = unpack_vector(pack_vector(original))

    assert restored.dtype == np.float32
    assert np.allclose(original, restored)


def test_unpacked_vector_is_writable():
    """np.frombuffer yields a read-only view; callers must get a real array."""
    restored = unpack_vector(pack_vector(np.ones(3, dtype=np.float32)))
    restored[0] = 9.0
    assert restored[0] == 9.0


def test_chunk_id_is_zero_padded_for_citations():
    assert chunk_id_for(7) == "chunk_00007"


# --------------------------------------------------------------------------- #
# Document metadata
# --------------------------------------------------------------------------- #


def test_put_document_meta_stores_a_listable_document(table):
    put_document_meta(table, doc_id="d_1", filename="report.pdf", s3_key="uploads/d_1.pdf")

    (document,) = list_documents(table)
    assert document.doc_id == "d_1"
    assert document.filename == "report.pdf"
    assert document.status == DocumentStatus.UPLOADED
    assert document.chunk_count == 0
    assert document.error_msg is None


def test_list_documents_returns_newest_first(table):
    put_document_meta(
        table, doc_id="old", filename="a", s3_key="k", uploaded_at="2026-01-01T00:00:00Z"
    )
    put_document_meta(
        table, doc_id="new", filename="b", s3_key="k", uploaded_at="2026-08-01T00:00:00Z"
    )
    put_document_meta(
        table, doc_id="mid", filename="c", s3_key="k", uploaded_at="2026-04-01T00:00:00Z"
    )

    assert [d.doc_id for d in list_documents(table)] == ["new", "mid", "old"]


def test_list_documents_is_empty_for_a_fresh_table(table):
    assert list_documents(table) == []


def test_update_status_records_success(table):
    put_document_meta(table, doc_id="d_1", filename="a.pdf", s3_key="k")
    update_status(table, "d_1", DocumentStatus.READY, chunk_count=42)

    (document,) = list_documents(table)
    assert document.status == DocumentStatus.READY
    assert document.chunk_count == 42


def test_update_status_records_failure_reason(table):
    put_document_meta(table, doc_id="d_1", filename="a.docx", s3_key="k")
    update_status(table, "d_1", DocumentStatus.FAILED, error_msg="UNSUPPORTED_TYPE")

    (document,) = list_documents(table)
    assert document.status == DocumentStatus.FAILED
    assert document.error_msg == "UNSUPPORTED_TYPE"


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


def test_put_chunks_round_trips_text_and_vectors(table):
    written = put_chunks(table, "d_1", _records(3))
    stored = sorted(get_all_chunks(table), key=lambda c: c.chunk_id)

    assert written == 3
    assert [c.chunk_id for c in stored] == ["chunk_00000", "chunk_00001", "chunk_00002"]
    assert all(c.doc_id == "d_1" for c in stored)
    assert stored[1].text == "chunk text 1"
    assert stored[1].token_count == 11
    assert np.allclose(stored[1].vector, np.arange(4, dtype=np.float32) + 1)


def test_put_chunks_batches_beyond_the_dynamodb_limit(table):
    """batch_writer must flush in 25-item batches without dropping anything."""
    count = 60
    put_chunks(table, "d_1", _records(count))
    assert len(get_all_chunks(table)) == count


def test_stored_dimension_matches_the_vector(table):
    put_chunks(table, "d_1", _records(1, dim=1024))
    item = table.get_item(Key={"PK": "DOC#d_1", "SK": "CHUNK#00000"})["Item"]
    assert int(item["dim"]) == 1024
    assert len(unpack_vector(item["vector"])) == 1024


def test_get_all_chunks_spans_documents_and_excludes_metadata(table):
    put_document_meta(table, doc_id="d_1", filename="a", s3_key="k")
    put_document_meta(table, doc_id="d_2", filename="b", s3_key="k")
    put_chunks(table, "d_1", _records(2))
    put_chunks(table, "d_2", _records(3))

    chunks = get_all_chunks(table)
    assert len(chunks) == 5
    assert {c.doc_id for c in chunks} == {"d_1", "d_2"}


def test_get_all_chunks_is_empty_for_a_fresh_table(table):
    assert get_all_chunks(table) == []


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #


def test_delete_document_removes_metadata_and_chunks(table):
    put_document_meta(table, doc_id="d_1", filename="a", s3_key="k")
    put_chunks(table, "d_1", _records(5))

    assert delete_document(table, "d_1") is True
    assert list_documents(table) == []
    assert get_all_chunks(table) == []


def test_delete_document_is_idempotent(table):
    put_document_meta(table, doc_id="d_1", filename="a", s3_key="k")
    assert delete_document(table, "d_1") is True
    assert delete_document(table, "d_1") is False


def test_delete_document_leaves_other_documents_untouched(table):
    put_document_meta(table, doc_id="keep", filename="a", s3_key="k")
    put_chunks(table, "keep", _records(2))
    put_document_meta(table, doc_id="drop", filename="b", s3_key="k")
    put_chunks(table, "drop", _records(2))

    delete_document(table, "drop")

    assert [d.doc_id for d in list_documents(table)] == ["keep"]
    assert {c.doc_id for c in get_all_chunks(table)} == {"keep"}


def test_delete_chunks_keeps_metadata_for_failure_reporting(table):
    """A FAILED document keeps its metadata but must leave no orphaned chunks."""
    put_document_meta(table, doc_id="d_1", filename="a", s3_key="k")
    put_chunks(table, "d_1", _records(4))

    assert delete_chunks(table, "d_1") == 4
    assert get_all_chunks(table) == []
    assert [d.doc_id for d in list_documents(table)] == ["d_1"]


def test_delete_chunks_on_unknown_document_is_a_no_op(table):
    assert delete_chunks(table, "missing") == 0
