"""Unit tests for the /documents routes (T2.3 acceptance criteria, TESTING.md §2).

Both DynamoDB and S3 are moto-backed, so the presigned policy is a real signed
policy and the delete cascade is verified against real stored objects.
"""

import base64
import json
import re

import boto3
import numpy as np
import pytest

from documents import handler as documents_handler
from documents.handler import (
    CONTENT_TYPES,
    MAX_FILENAME_CHARS,
    PRESIGN_EXPIRY_SECONDS,
    DocumentNotFoundError,
    InvalidFilenameError,
    UnsupportedTypeError,
    create_upload,
    delete_document_and_object,
    delete_handler,
    list_handler,
    new_doc_id,
    presign_handler,
    validate_filename,
)
from shared.storage import (
    ChunkRecord,
    DocumentStatus,
    build_s3_key,
    get_all_chunks,
    get_document,
    list_documents,
    put_chunks,
    put_document_meta,
)

BUCKET = "documents-bucket"
MAX_BYTES = 25 * 1024 * 1024


@pytest.fixture
def s3_client(aws):
    """A moto-backed S3 client with the documents bucket created."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    return client


def _policy_conditions(fields: dict) -> list:
    """Decode the signed policy document out of a presigned POST's fields."""
    return json.loads(base64.b64decode(fields["policy"]))["conditions"]


def _seed_document(table, doc_id: str, filename: str, chunks: int = 2) -> None:
    put_document_meta(
        table,
        doc_id=doc_id,
        filename=filename,
        s3_key=build_s3_key(doc_id, filename),
        uploaded_at=f"2026-08-04T10:{doc_id[-2:]}:00Z",
    )
    put_chunks(
        table,
        doc_id,
        [
            ChunkRecord(
                index=i,
                text=f"chunk {i}",
                token_count=5,
                vector=np.array([1.0, 0.0], dtype=np.float32),
            )
            for i in range(chunks)
        ],
    )


# --------------------------------------------------------------------------- #
# Filename validation (API_SPEC.md §1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix,content_type", sorted(CONTENT_TYPES.items()))
def test_supported_extensions_resolve_to_their_content_type(suffix, content_type):
    assert validate_filename(f"report{suffix}") == (f"report{suffix}", content_type)


def test_extension_matching_is_case_insensitive():
    assert validate_filename("REPORT.PDF")[1] == "application/pdf"


def test_surrounding_whitespace_is_trimmed():
    assert validate_filename("  report.pdf  ")[0] == "report.pdf"


@pytest.mark.parametrize("filename", ["notes.docx", "archive.zip", "noextension", "report.pdf.exe"])
def test_unsupported_extensions_are_rejected(filename):
    with pytest.raises(UnsupportedTypeError):
        validate_filename(filename)


@pytest.mark.parametrize("filename", [None, 42, [], {"a": 1}])
def test_non_string_filenames_are_rejected(filename):
    with pytest.raises(InvalidFilenameError):
        validate_filename(filename)


@pytest.mark.parametrize("filename", ["", "   ", "a" * (MAX_FILENAME_CHARS + 1) + ".pdf"])
def test_filename_length_bounds_are_enforced(filename):
    with pytest.raises(InvalidFilenameError):
        validate_filename(filename)


def test_maximum_length_filename_is_accepted():
    filename = "a" * (MAX_FILENAME_CHARS - 4) + ".pdf"
    assert validate_filename(filename)[0] == filename


@pytest.mark.parametrize(
    "filename",
    [
        "../../../etc/passwd.pdf",
        "nested/report.pdf",
        "..\\windows\\report.pdf",
        "/absolute.pdf",
        ".hidden.pdf",
    ],
)
def test_filenames_that_could_escape_the_upload_prefix_are_rejected(filename):
    """The filename becomes part of an object key, so separators are refused."""
    with pytest.raises(InvalidFilenameError):
        validate_filename(filename)


@pytest.mark.parametrize("filename", ["re\nport.pdf", "re\x00port.pdf", "report\x7f.pdf"])
def test_control_characters_are_rejected(filename):
    with pytest.raises(InvalidFilenameError):
        validate_filename(filename)


# --------------------------------------------------------------------------- #
# Document ids
# --------------------------------------------------------------------------- #


def test_doc_id_matches_the_documented_shape():
    assert re.fullmatch(r"d_[0-9a-f]{6}", new_doc_id())


def test_doc_ids_are_unique():
    assert len({new_doc_id() for _ in range(200)}) > 190


# --------------------------------------------------------------------------- #
# Presign (T2.3: "presign conditions")
# --------------------------------------------------------------------------- #


def test_presign_returns_the_documented_body(table, s3_client):
    body = create_upload(
        table=table,
        s3_client=s3_client,
        bucket=BUCKET,
        filename="q3-report.pdf",
        max_bytes=MAX_BYTES,
    )

    assert body.keys() == {"doc_id", "upload_url", "method", "fields", "expires_in", "max_bytes"}
    assert body["method"] == "POST"
    assert body["expires_in"] == PRESIGN_EXPIRY_SECONDS
    assert body["max_bytes"] == MAX_BYTES
    assert BUCKET in body["upload_url"]


def test_presigned_policy_enforces_the_size_limit(table, s3_client):
    """The reason this is a POST: only a POST policy can bound content length."""
    body = create_upload(
        table=table, s3_client=s3_client, bucket=BUCKET, filename="q3.pdf", max_bytes=MAX_BYTES
    )

    assert ["content-length-range", 1, MAX_BYTES] in _policy_conditions(body["fields"])


def test_presigned_policy_pins_the_content_type(table, s3_client):
    body = create_upload(
        table=table, s3_client=s3_client, bucket=BUCKET, filename="notes.md", max_bytes=MAX_BYTES
    )

    assert {"Content-Type": "text/markdown"} in _policy_conditions(body["fields"])
    assert body["fields"]["Content-Type"] == "text/markdown"


def test_presigned_policy_pins_the_exact_key(table, s3_client):
    body = create_upload(
        table=table, s3_client=s3_client, bucket=BUCKET, filename="q3.pdf", max_bytes=MAX_BYTES
    )

    expected = build_s3_key(body["doc_id"], "q3.pdf")
    assert body["fields"]["key"] == expected
    assert {"key": expected} in _policy_conditions(body["fields"])


def test_presign_records_the_document_as_uploaded(table, s3_client):
    body = create_upload(
        table=table,
        s3_client=s3_client,
        bucket=BUCKET,
        filename="q3-report.pdf",
        max_bytes=MAX_BYTES,
    )

    summary = get_document(table, body["doc_id"])
    assert summary is not None
    assert summary.filename == "q3-report.pdf"
    assert summary.status == DocumentStatus.UPLOADED
    assert summary.chunk_count == 0


def test_presign_rejects_an_unsupported_type_before_writing_anything(table, s3_client):
    with pytest.raises(UnsupportedTypeError):
        create_upload(
            table=table,
            s3_client=s3_client,
            bucket=BUCKET,
            filename="notes.docx",
            max_bytes=MAX_BYTES,
        )

    assert list_documents(table) == []


# --------------------------------------------------------------------------- #
# Delete cascade (T2.3: "delete cascade (chunks + S3 object)")
# --------------------------------------------------------------------------- #


def test_delete_removes_the_object_the_chunks_and_the_metadata(table, s3_client):
    _seed_document(table, "d_aa11bb", "q3.pdf", chunks=3)
    key = build_s3_key("d_aa11bb", "q3.pdf")
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=b"%PDF-1.4")

    delete_document_and_object(table=table, s3_client=s3_client, bucket=BUCKET, doc_id="d_aa11bb")

    assert get_document(table, "d_aa11bb") is None
    assert get_all_chunks(table) == []
    assert "Contents" not in s3_client.list_objects_v2(Bucket=BUCKET)


def test_delete_leaves_other_documents_untouched(table, s3_client):
    _seed_document(table, "d_aa11bb", "gone.pdf", chunks=2)
    _seed_document(table, "d_cc22dd", "kept.pdf", chunks=2)

    delete_document_and_object(table=table, s3_client=s3_client, bucket=BUCKET, doc_id="d_aa11bb")

    assert get_document(table, "d_cc22dd") is not None
    assert {chunk.doc_id for chunk in get_all_chunks(table)} == {"d_cc22dd"}


def test_delete_of_an_unknown_document_raises_not_found(table, s3_client):
    with pytest.raises(DocumentNotFoundError):
        delete_document_and_object(
            table=table, s3_client=s3_client, bucket=BUCKET, doc_id="d_missing"
        )


# --------------------------------------------------------------------------- #
# Handler adaptation (API_SPEC.md §§1–3, §6)
# --------------------------------------------------------------------------- #


@pytest.fixture
def api_env(monkeypatch, table, s3_client):
    """Environment and cached clients for handler-level tests."""
    monkeypatch.setenv("TABLE_NAME", "rag-store")
    monkeypatch.setenv("BUCKET_NAME", BUCKET)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(MAX_BYTES))
    monkeypatch.setenv("ALLOWED_ORIGIN", "https://example.cloudfront.net")
    monkeypatch.setattr(documents_handler, "_table", table)
    monkeypatch.setattr(documents_handler, "_s3", s3_client)


def test_presign_handler_returns_201(api_env):
    response = presign_handler({"body": json.dumps({"filename": "q3-report.pdf"})})
    body = json.loads(response["body"])

    assert response["statusCode"] == 201
    assert body["method"] == "POST"
    assert body["doc_id"].startswith("d_")


def test_presign_handler_sets_cors_headers(api_env):
    response = presign_handler({"body": json.dumps({"filename": "q3.pdf"})})
    assert response["headers"]["Access-Control-Allow-Origin"] == "https://example.cloudfront.net"


def test_presign_handler_rejects_an_unsupported_type(api_env):
    response = presign_handler({"body": json.dumps({"filename": "notes.docx"})})

    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["code"] == "UNSUPPORTED_TYPE"


@pytest.mark.parametrize("body", [{}, {"filename": ""}, {"filename": "a/b.pdf"}, None, "junk"])
def test_presign_handler_rejects_an_invalid_filename(api_env, body):
    event = {"body": body if isinstance(body, str) or body is None else json.dumps(body)}
    response = presign_handler(event)

    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["code"] == "INVALID_FILENAME"


def test_presign_handler_never_leaks_internal_detail(api_env, monkeypatch):
    def boom(**_):
        raise RuntimeError("bucket documents-bucket denied for arn:aws:iam::123")

    monkeypatch.setattr(documents_handler, "create_upload", boom)
    response = presign_handler({"body": json.dumps({"filename": "q3.pdf"})})

    assert response["statusCode"] == 500
    assert json.loads(response["body"])["error"]["code"] == "INTERNAL"
    assert "arn:aws:iam" not in response["body"]


def test_list_handler_returns_documents_newest_first(api_env, table):
    _seed_document(table, "d_aa0001", "older.pdf", chunks=1)
    _seed_document(table, "d_bb0002", "newer.pdf", chunks=1)

    response = list_handler({})
    documents = json.loads(response["body"])["documents"]

    assert response["statusCode"] == 200
    assert [d["filename"] for d in documents] == ["newer.pdf", "older.pdf"]


def test_list_handler_returns_every_documented_field(api_env, table):
    _seed_document(table, "d_aa0001", "q3.pdf", chunks=1)

    (document,) = json.loads(list_handler({})["body"])["documents"]

    assert document.keys() == {
        "doc_id",
        "filename",
        "status",
        "uploaded_at",
        "chunk_count",
        "error_msg",
    }


def test_list_handler_on_an_empty_corpus_returns_an_empty_list(api_env):
    assert json.loads(list_handler({})["body"]) == {"documents": []}


def test_list_handler_never_leaks_internal_detail(api_env, monkeypatch):
    def boom(_):
        raise RuntimeError("table rag-store unreachable")

    monkeypatch.setattr(documents_handler, "list_documents", boom)
    response = list_handler({})

    assert response["statusCode"] == 500
    assert "rag-store" not in response["body"]


def test_delete_handler_returns_204_with_no_body(api_env, table, s3_client):
    _seed_document(table, "d_aa11bb", "q3.pdf", chunks=2)
    s3_client.put_object(Bucket=BUCKET, Key=build_s3_key("d_aa11bb", "q3.pdf"), Body=b"x")

    response = delete_handler({"pathParameters": {"doc_id": "d_aa11bb"}})

    assert response["statusCode"] == 204
    assert response["body"] == ""
    assert get_document(table, "d_aa11bb") is None


def test_delete_handler_returns_404_for_an_unknown_document(api_env):
    response = delete_handler({"pathParameters": {"doc_id": "d_missing"}})

    assert response["statusCode"] == 404
    assert json.loads(response["body"])["error"]["code"] == "DOC_NOT_FOUND"


@pytest.mark.parametrize("event", [{}, {"pathParameters": None}, {"pathParameters": {}}])
def test_delete_handler_without_a_doc_id_returns_404(api_env, event):
    assert delete_handler(event)["statusCode"] == 404


def test_delete_handler_never_leaks_internal_detail(api_env, monkeypatch):
    def boom(**_):
        raise RuntimeError("s3 documents-bucket denied")

    monkeypatch.setattr(documents_handler, "delete_document_and_object", boom)
    response = delete_handler({"pathParameters": {"doc_id": "d_aa11bb"}})

    assert response["statusCode"] == 500
    assert "documents-bucket" not in response["body"]


def test_max_upload_bytes_reads_the_environment(api_env):
    assert documents_handler.max_upload_bytes() == MAX_BYTES


def test_clients_are_cached(monkeypatch, aws):
    monkeypatch.setenv("TABLE_NAME", "rag-store")
    monkeypatch.setattr(documents_handler, "_table", None)
    monkeypatch.setattr(documents_handler, "_s3", None)

    assert documents_handler._get_table() is documents_handler._get_table()
    assert documents_handler._get_s3() is documents_handler._get_s3()
