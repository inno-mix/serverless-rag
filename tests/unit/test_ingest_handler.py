"""Unit tests for the ingestion pipeline (T1.4, TESTING.md §2).

Extraction is exercised against the committed fixtures; the pipeline runs
against moto-backed S3 and DynamoDB with a stubbed embedder, so no test touches
Bedrock or the network.
"""

import json
from pathlib import Path

import boto3
import numpy as np
import pytest
from moto import mock_aws

from ingest.handler import (
    ExtractionError,
    UnsupportedTypeError,
    extract_text,
    handler,
    ingest_object,
    parse_event,
)
from shared.embeddings import BedrockError, ModelUnavailableError
from shared.storage import (
    DocumentStatus,
    build_s3_key,
    get_all_chunks,
    list_documents,
    put_document_meta,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EVENTS = FIXTURES / "events"
BUCKET = "documents-bucket"
DLQ_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/ingest-dlq"


@pytest.fixture(autouse=True)
def _aws_credentials(monkeypatch):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def aws():
    """A moto environment with the documents bucket and the rag-store table."""
    with mock_aws():
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)

        table = boto3.resource("dynamodb", region_name="us-east-1").create_table(
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
        yield s3_client, table


class _RecordingSqs:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, **kwargs: object) -> dict:
        self.messages.append(kwargs)
        return {"MessageId": "m1"}


def _fake_embed(dim: int = 8):
    def embed(texts: list[str]) -> list[np.ndarray]:
        return [np.full(dim, float(i + 1), dtype=np.float32) for i, _ in enumerate(texts)]

    return embed


def _upload(s3_client, doc_id: str, filename: str, data: bytes) -> str:
    key = build_s3_key(doc_id, filename)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=data)
    return key


def _seed(table, doc_id: str, filename: str) -> None:
    put_document_meta(
        table, doc_id=doc_id, filename=filename, s3_key=build_s3_key(doc_id, filename)
    )


# --------------------------------------------------------------------------- #
# extract_text
# --------------------------------------------------------------------------- #


def test_extracts_every_page_of_a_pdf():
    text = extract_text("sample.pdf", (FIXTURES / "sample.pdf").read_bytes())
    assert "4.2 million dollars" in text
    assert "Wellington" in text  # second page


def test_extracts_plain_text():
    text = extract_text("sample.txt", (FIXTURES / "sample.txt").read_bytes())
    assert "three regional offices" in text


def test_extracts_markdown_verbatim():
    text = extract_text("sample.md", (FIXTURES / "sample.md").read_bytes())
    assert "# Acme Corporation Quarterly Report" in text


@pytest.mark.parametrize("filename", ["unsupported.docx", "archive.zip", "noextension"])
def test_unsupported_extensions_are_rejected(filename):
    with pytest.raises(UnsupportedTypeError):
        extract_text(filename, b"")


def test_extension_matching_is_case_insensitive():
    assert extract_text("SAMPLE.TXT", b"hello") == "hello"


def test_undecodable_text_file_raises_extraction_error():
    with pytest.raises(ExtractionError):
        extract_text("sample.txt", b"\xff\xfe\x00invalid")


def test_corrupt_pdf_raises_extraction_error():
    with pytest.raises(ExtractionError):
        extract_text("broken.pdf", b"definitely not a pdf")


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_pdf_upload_is_chunked_embedded_and_stored(aws):
    s3_client, table = aws
    _seed(table, "d_1", "sample.pdf")
    key = _upload(s3_client, "d_1", "sample.pdf", (FIXTURES / "sample.pdf").read_bytes())

    result = ingest_object(
        table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=_fake_embed()
    )

    assert result.status == DocumentStatus.READY
    assert result.chunk_count > 0

    chunks = get_all_chunks(table)
    assert len(chunks) == result.chunk_count
    assert all(c.doc_id == "d_1" for c in chunks)
    assert all(len(c.vector) == 8 for c in chunks)
    assert any("4.2 million dollars" in c.text for c in chunks)


def test_document_is_marked_ready_with_its_chunk_count(aws):
    s3_client, table = aws
    _seed(table, "d_1", "sample.txt")
    key = _upload(s3_client, "d_1", "sample.txt", (FIXTURES / "sample.txt").read_bytes())

    result = ingest_object(
        table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=_fake_embed()
    )

    (document,) = list_documents(table)
    assert document.status == DocumentStatus.READY
    assert document.chunk_count == result.chunk_count
    assert document.error_msg is None


def test_chunker_settings_are_honoured(aws):
    """Smaller target size must produce more chunks from the same document."""
    s3_client, table = aws
    body = ((FIXTURES / "sample.txt").read_text() * 40).encode()
    _seed(table, "d_1", "sample.txt")
    key = _upload(s3_client, "d_1", "sample.txt", body)

    coarse = ingest_object(
        table=table,
        s3_client=s3_client,
        bucket=BUCKET,
        key=key,
        embed=_fake_embed(),
        target_tokens=500,
        overlap_tokens=50,
    )
    fine = ingest_object(
        table=table,
        s3_client=s3_client,
        bucket=BUCKET,
        key=key,
        embed=_fake_embed(),
        target_tokens=60,
        overlap_tokens=10,
    )
    assert fine.chunk_count > coarse.chunk_count


# --------------------------------------------------------------------------- #
# Failure handling (ARCHITECTURE.md §6)
# --------------------------------------------------------------------------- #


def test_unsupported_type_marks_the_document_failed(aws):
    s3_client, table = aws
    _seed(table, "d_1", "unsupported.docx")
    key = _upload(s3_client, "d_1", "unsupported.docx", b"")
    sqs = _RecordingSqs()

    result = ingest_object(
        table=table,
        s3_client=s3_client,
        bucket=BUCKET,
        key=key,
        sqs_client=sqs,
        dlq_url=DLQ_URL,
        embed=_fake_embed(),
    )

    assert result.status == DocumentStatus.FAILED
    assert result.error_code == "UNSUPPORTED_TYPE"
    (document,) = list_documents(table)
    assert document.status == DocumentStatus.FAILED
    assert document.error_msg == "UNSUPPORTED_TYPE"


def test_empty_document_is_reported_as_such(aws):
    s3_client, table = aws
    _seed(table, "d_1", "blank.txt")
    key = _upload(s3_client, "d_1", "blank.txt", b"   \n\n  \t \n")

    result = ingest_object(
        table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=_fake_embed()
    )

    assert result.error_code == "EMPTY_DOCUMENT"
    assert list_documents(table)[0].error_msg == "EMPTY_DOCUMENT"


def test_failure_posts_a_message_to_the_dead_letter_queue(aws):
    s3_client, table = aws
    _seed(table, "d_1", "unsupported.docx")
    key = _upload(s3_client, "d_1", "unsupported.docx", b"")
    sqs = _RecordingSqs()

    ingest_object(
        table=table,
        s3_client=s3_client,
        bucket=BUCKET,
        key=key,
        sqs_client=sqs,
        dlq_url=DLQ_URL,
        embed=_fake_embed(),
    )

    (message,) = sqs.messages
    assert message["QueueUrl"] == DLQ_URL
    body = json.loads(message["MessageBody"])
    assert body["doc_id"] == "d_1"
    assert body["reason"] == "UNSUPPORTED_TYPE"


def test_failure_without_a_configured_dlq_is_still_recorded(aws):
    s3_client, table = aws
    _seed(table, "d_1", "unsupported.docx")
    key = _upload(s3_client, "d_1", "unsupported.docx", b"")

    result = ingest_object(
        table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=_fake_embed()
    )
    assert result.error_code == "UNSUPPORTED_TYPE"


def test_embedding_failure_leaves_no_partial_chunks(aws):
    """A FAILED document must never leave orphaned chunks in the corpus."""
    s3_client, table = aws
    _seed(table, "d_1", "sample.txt")
    key = _upload(s3_client, "d_1", "sample.txt", (FIXTURES / "sample.txt").read_bytes())

    def exploding_embed(texts: list[str]) -> list[np.ndarray]:
        raise ModelUnavailableError("bedrock is down")

    result = ingest_object(
        table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=exploding_embed
    )

    assert result.error_code == "MODEL_UNAVAILABLE"
    assert get_all_chunks(table) == []
    assert list_documents(table)[0].status == DocumentStatus.FAILED


def test_previously_written_chunks_are_removed_on_reingest_failure(aws):
    """A successful ingest followed by a failing one must not leave stale chunks."""
    s3_client, table = aws
    _seed(table, "d_1", "sample.txt")
    key = _upload(s3_client, "d_1", "sample.txt", (FIXTURES / "sample.txt").read_bytes())

    ingest_object(table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=_fake_embed())
    assert get_all_chunks(table)

    def exploding_embed(texts: list[str]) -> list[np.ndarray]:
        raise BedrockError("nope")

    ingest_object(table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=exploding_embed)

    assert get_all_chunks(table) == []
    assert list_documents(table)[0].error_msg == "EMBEDDING_FAILED"


def test_unexpected_errors_are_recorded_as_internal(aws):
    s3_client, table = aws
    _seed(table, "d_1", "sample.txt")
    key = _upload(s3_client, "d_1", "sample.txt", b"some text here.")

    def exploding_embed(texts: list[str]) -> list[np.ndarray]:
        raise RuntimeError("something unexpected")

    result = ingest_object(
        table=table, s3_client=s3_client, bucket=BUCKET, key=key, embed=exploding_embed
    )
    assert result.error_code == "INTERNAL"


def test_key_outside_the_upload_convention_is_rejected(aws):
    s3_client, table = aws
    with pytest.raises(ValueError, match="uploads/"):
        ingest_object(
            table=table,
            s3_client=s3_client,
            bucket=BUCKET,
            key="somewhere/else.pdf",
            embed=_fake_embed(),
        )


# --------------------------------------------------------------------------- #
# Event parsing
# --------------------------------------------------------------------------- #


def test_parses_the_committed_eventbridge_fixture():
    event = json.loads((EVENTS / "eventbridge_object_created.json").read_text())
    assert parse_event(event) == ("rag-documents-bucket", "uploads/d_8f3a1c/q3-report.pdf")


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"detail": {}},
        {"detail": {"bucket": {"name": "b"}}},
        {"detail": {"object": {"key": "k"}}},
        {"Records": [{"s3": {}}]},  # direct S3 notification shape is not used
    ],
)
def test_unrecognized_events_are_rejected(event):
    with pytest.raises(ValueError, match="detail|bucket name"):
        parse_event(event)


def test_handler_returns_empty_for_an_unrecognized_event():
    assert handler({"detail": {}}) == {"processed": []}
