"""Presign, list, and delete handlers for the /documents routes (T2.3).

Three entry points share this module because they share the same validation and
response vocabulary, but each is deployed as its own Lambda so its role carries
only the permissions that route needs (NFR-9, ARCHITECTURE.md §7): presign can
write to the bucket but not read it, list can only read the table, and delete is
the only role that can remove an object.

Uploads use a **presigned POST**, not the presigned PUT originally sketched in
API_SPEC.md §1. A presigned PUT URL cannot carry a `content-length-range`
condition — only the POST policy document can — so PUT could not deliver the
"rejected at presign" behaviour ARCHITECTURE.md §6 requires. With POST, S3 itself
rejects an oversized or wrong-typed body and the bytes never reach the bucket
(ADR-002, amended).
"""

import os
import secrets
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any

import boto3

from shared.http import ApiError, error_response, json_response, parse_json_body, path_parameter
from shared.logging_utils import get_logger
from shared.storage import (
    DocumentStatus,
    build_s3_key,
    delete_document,
    get_document,
    list_documents,
    put_document_meta,
)

#: Document ids look like `d_8f3a1c` (API_SPEC.md §1): a short prefix and three
#: random bytes rendered as six hex characters.
DOC_ID_PREFIX = "d_"
DOC_ID_BYTES = 3

#: Presigned upload lifetime, in seconds, as documented in API_SPEC.md §1.
PRESIGN_EXPIRY_SECONDS = 300

#: Filename ceiling from API_SPEC.md §1.
MAX_FILENAME_CHARS = 255

#: Accepted extensions and the content type each is presigned for. The upload
#: policy pins the type, so a `.pdf` URL cannot be used to store something else.
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

log = get_logger(__name__)

_s3: Any | None = None
_table: Any | None = None


class UnsupportedTypeError(ApiError):
    """The filename's extension is not one this system can ingest."""

    status = 400
    code = "UNSUPPORTED_TYPE"


class InvalidFilenameError(ApiError):
    """The filename is missing, too long, or unsafe to use in an object key."""

    status = 400
    code = "INVALID_FILENAME"


class DocumentNotFoundError(ApiError):
    """No document exists with the requested id."""

    status = 404
    code = "DOC_NOT_FOUND"


def new_doc_id() -> str:
    """Return a fresh document id in the documented `d_8f3a1c` form."""
    return f"{DOC_ID_PREFIX}{secrets.token_hex(DOC_ID_BYTES)}"


def validate_filename(filename: Any) -> tuple[str, str]:
    """Check a client-supplied filename and resolve its content type.

    The filename becomes part of an S3 object key, so it is checked for
    separators and leading dots before anything else: a name like
    `../../secrets.pdf` must never be able to steer the key out of the
    `uploads/{doc_id}/` prefix that ingestion and deletion both rely on.

    Args:
        filename: The `filename` field from the request body.

    Returns:
        The accepted filename and the content type to presign it for.

    Raises:
        InvalidFilenameError: If the name is absent, too long, or unsafe.
        UnsupportedTypeError: If the extension is not `.pdf`, `.txt`, or `.md`.
    """
    if not isinstance(filename, str):
        raise InvalidFilenameError("A filename is required.")

    filename = filename.strip()
    if not filename or len(filename) > MAX_FILENAME_CHARS:
        raise InvalidFilenameError(
            f"Filename must be between 1 and {MAX_FILENAME_CHARS} characters."
        )
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise InvalidFilenameError("Filename must not contain path separators.")
    if any(character < " " or character == "\x7f" for character in filename):
        raise InvalidFilenameError("Filename must not contain control characters.")

    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in CONTENT_TYPES:
        raise UnsupportedTypeError("Only .pdf, .txt and .md files are supported.")
    return filename, CONTENT_TYPES[suffix]


def create_upload(
    *,
    table: Any,
    s3_client: Any,
    bucket: str,
    filename: Any,
    max_bytes: int,
    expires_in: int = PRESIGN_EXPIRY_SECONDS,
) -> dict[str, Any]:
    """Register a document and presign the upload that will carry its bytes.

    Args:
        table: DynamoDB table resource.
        s3_client: S3 client used to sign the upload policy.
        bucket: Documents bucket name.
        filename: The client-supplied filename, validated here.
        max_bytes: Upload ceiling signed into the policy (PRD G1: 25 MB).
        expires_in: Signature lifetime in seconds.

    Returns:
        The `201` body described in API_SPEC.md §1.

    Raises:
        InvalidFilenameError: If the filename is absent, too long, or unsafe.
        UnsupportedTypeError: If the extension is not supported.
    """
    filename, content_type = validate_filename(filename)
    doc_id = new_doc_id()
    key = build_s3_key(doc_id, filename)

    presigned = s3_client.generate_presigned_post(
        Bucket=bucket,
        Key=key,
        Fields={"Content-Type": content_type},
        Conditions=[
            {"Content-Type": content_type},
            # The size limit S3 enforces on our behalf. Zero-byte uploads are
            # excluded too, since they could only fail later as EMPTY_DOCUMENT.
            ["content-length-range", 1, max_bytes],
        ],
        ExpiresIn=expires_in,
    )

    put_document_meta(
        table,
        doc_id=doc_id,
        filename=filename,
        s3_key=key,
        status=DocumentStatus.UPLOADED,
    )
    log.info("documents.presigned", doc_id=doc_id, filename=filename, max_bytes=max_bytes)

    return {
        "doc_id": doc_id,
        "upload_url": presigned["url"],
        "method": "POST",
        "fields": presigned["fields"],
        "expires_in": expires_in,
        "max_bytes": max_bytes,
    }


def delete_document_and_object(
    *,
    table: Any,
    s3_client: Any,
    bucket: str,
    doc_id: str,
) -> None:
    """Remove a document's S3 object, its chunks, and its metadata.

    The object is deleted first. If the table write then fails, the document
    still lists and the caller can retry; the reverse order would leave an
    unreferenced object in the bucket with nothing left pointing at it.

    Args:
        table: DynamoDB table resource.
        s3_client: S3 client used to delete the uploaded object.
        bucket: Documents bucket name.
        doc_id: Document identifier.

    Raises:
        DocumentNotFoundError: If no document with that id exists.
    """
    summary = get_document(table, doc_id)
    if summary is None:
        raise DocumentNotFoundError("No document with that id.")

    s3_client.delete_object(Bucket=bucket, Key=build_s3_key(doc_id, summary.filename))
    delete_document(table, doc_id)
    log.info("documents.deleted", doc_id=doc_id, filename=summary.filename)


def presign_handler(event: dict, context: Any = None) -> dict:  # noqa: ARG001 - Lambda signature
    """Lambda entry point for `POST /documents`.

    Args:
        event: API Gateway proxy event.
        context: Lambda context object, unused.

    Returns:
        A `201` proxy response carrying the presigned upload.
    """
    try:
        body = create_upload(
            table=_get_table(),
            s3_client=_get_s3(),
            bucket=os.environ["BUCKET_NAME"],
            filename=parse_json_body(event).get("filename"),
            max_bytes=max_upload_bytes(),
        )
    except ApiError as error:
        log.error("documents.presign_rejected", code=error.code, detail=str(error))
        return error_response(error)
    except Exception as error:  # noqa: BLE001 - details stay in the logs, never in the body
        log.error("documents.presign_failed", detail=str(error))
        return error_response(ApiError("Unexpected error."))

    return json_response(201, body)


def list_handler(event: dict, context: Any = None) -> dict:  # noqa: ARG001 - Lambda signature
    """Lambda entry point for `GET /documents`.

    Args:
        event: API Gateway proxy event, unused beyond routing.
        context: Lambda context object, unused.

    Returns:
        A `200` proxy response listing every document, newest first.
    """
    try:
        documents = [asdict(summary) for summary in list_documents(_get_table())]
    except Exception as error:  # noqa: BLE001 - details stay in the logs, never in the body
        log.error("documents.list_failed", detail=str(error))
        return error_response(ApiError("Unexpected error."))

    return json_response(200, {"documents": documents})


def delete_handler(event: dict, context: Any = None) -> dict:  # noqa: ARG001 - Lambda signature
    """Lambda entry point for `DELETE /documents/{doc_id}`.

    Args:
        event: API Gateway proxy event.
        context: Lambda context object, unused.

    Returns:
        A `204` proxy response with no body, per API_SPEC.md §3.
    """
    doc_id = path_parameter(event, "doc_id")
    try:
        if doc_id is None:
            raise DocumentNotFoundError("No document with that id.")
        delete_document_and_object(
            table=_get_table(),
            s3_client=_get_s3(),
            bucket=os.environ["BUCKET_NAME"],
            doc_id=doc_id,
        )
    except ApiError as error:
        log.error("documents.delete_rejected", code=error.code, doc_id=doc_id)
        return error_response(error)
    except Exception as error:  # noqa: BLE001 - details stay in the logs, never in the body
        log.error("documents.delete_failed", doc_id=doc_id, detail=str(error))
        return error_response(ApiError("Unexpected error."))

    return json_response(204)


def max_upload_bytes() -> int:
    """Read the upload ceiling from the environment (PRD G1)."""
    return int(os.environ["MAX_UPLOAD_BYTES"])


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
