"""End-to-end smoke test against a deployed stack (T2.4, TESTING.md §4).

Runs the whole user journey: presign, upload, wait for indexing, ask an
answerable question, ask an unanswerable one, delete. It doubles as the
demo-readiness check, so it asserts structure — `found`, citations, status
transitions — and never the model's wording (TESTING.md §7).

Not part of the PR gate; it needs real AWS. Run it after a deploy:

    pytest tests/integration -q

Everything it needs is discovered from CloudFormation outputs, so there is
nothing to configure by hand. Only the standard library and boto3 are used:
`requests` is not a dependency of this project (CLAUDE.md).
"""

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import BotoCoreError, ClientError

API_STACK_NAME = "ApiStack"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"

#: PRD G1 promises indexing within two minutes; TESTING.md §4 allows three.
READY_TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 5

#: Answerable from the fixture, which states revenue of "4.2 million dollars".
ANSWERABLE_QUESTION = "What was the third quarter revenue?"
EXPECTED_EVIDENCE = "4.2 million"

#: Nothing in the fixture corpus is related to this (TESTING.md §4, step 5).
UNANSWERABLE_QUESTION = "What is the capital of Mongolia?"


def _configured_floor() -> float:
    """The deployed `minSimilarity`, read from the same place CDK reads it."""
    context = json.loads((Path(__file__).resolve().parents[2] / "cdk.json").read_text())["context"]
    return float(context["minSimilarity"])


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def outputs() -> dict[str, str]:
    """Read ApiStack's CloudFormation outputs, skipping if it is not deployed."""
    try:
        stacks = boto3.client("cloudformation").describe_stacks(StackName=API_STACK_NAME)
    except (ClientError, BotoCoreError) as error:
        pytest.skip(f"{API_STACK_NAME} is not deployed: {error}")

    return {
        output["OutputKey"]: output["OutputValue"]
        for output in stacks["Stacks"][0].get("Outputs", [])
    }


@pytest.fixture(scope="module")
def api_url(outputs) -> str:
    """Base URL of the deployed API, without a trailing slash."""
    return outputs["ApiUrl"].rstrip("/")


@pytest.fixture(scope="module")
def api_key(outputs) -> str:
    """The usage-plan API key value, read from API Gateway rather than stored."""
    key = boto3.client("apigateway").get_api_key(apiKey=outputs["ApiKeyId"], includeValue=True)
    return key["value"]


@pytest.fixture(scope="module")
def client(api_url, api_key):
    """A minimal JSON client for the deployed API."""
    return _ApiClient(api_url, api_key)


class _ApiClient:
    """Thin urllib wrapper returning `(status, decoded_body)`."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(  # noqa: S310 - fixed https API endpoint
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "x-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as error:
            return error.code, _decode(error.read())


def _decode(payload: bytes) -> dict:
    """Decode a response body, tolerating the empty body of a 204."""
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except ValueError:
        return {"raw": payload.decode(errors="replace")}


def _upload(presigned: dict, path: Path) -> int:
    """Upload a file with a presigned POST policy, returning the HTTP status.

    S3 expects `multipart/form-data` with the signed fields first and the file
    last; the policy is what enforces the size and content-type limits, so this
    deliberately sends exactly what a browser form would (ADR-002, amended).
    """
    boundary = f"----smoke{uuid.uuid4().hex}"
    content_type = presigned["fields"].get("Content-Type") or (
        mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )

    parts: list[bytes] = []
    for name, value in presigned["fields"].items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{path.name}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())

    request = urllib.request.Request(  # noqa: S310 - URL comes from our own API
        presigned["upload_url"],
        data=b"".join(parts),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as error:
        pytest.fail(f"upload rejected with {error.code}: {error.read().decode(errors='replace')}")


def _wait_for_status(client: _ApiClient, doc_id: str, target: str) -> dict:
    """Poll `GET /documents` until the document reaches `target`, or time out."""
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_seen = "UNKNOWN"

    while time.monotonic() < deadline:
        status_code, body = client.request("GET", "/documents")
        assert status_code == 200, body
        for document in body["documents"]:
            if document["doc_id"] != doc_id:
                continue
            last_seen = document["status"]
            if last_seen == target:
                return document
            if last_seen == "FAILED":
                pytest.fail(f"ingestion failed: {document.get('error_msg')}")
        time.sleep(POLL_INTERVAL_SECONDS)

    pytest.fail(
        f"{doc_id} did not reach {target} within {READY_TIMEOUT_SECONDS}s; last status {last_seen}"
    )


# --------------------------------------------------------------------------- #
# The journey
# --------------------------------------------------------------------------- #


def test_full_document_journey(client):
    """Upload a document, query it, and delete it (TESTING.md §4)."""
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}"

    status, presigned = client.request("POST", "/documents", {"filename": FIXTURE.name})
    assert status == 201, presigned
    doc_id = presigned["doc_id"]
    assert presigned["method"] == "POST"
    assert presigned["max_bytes"] == 25 * 1024 * 1024

    try:
        assert _upload(presigned, FIXTURE) in (200, 204)

        document = _wait_for_status(client, doc_id, "READY")
        assert document["chunk_count"] >= 1
        assert document["filename"] == FIXTURE.name
        assert document["error_msg"] is None

        _assert_answerable(client, doc_id)
        _assert_unanswerable(client)
    finally:
        status, _ = client.request("DELETE", f"/documents/{doc_id}")
        assert status == 204

    _, body = client.request("GET", "/documents")
    assert doc_id not in {document["doc_id"] for document in body["documents"]}


def _assert_answerable(client: _ApiClient, doc_id: str) -> None:
    """A question the fixture answers returns a grounded, cited answer."""
    status, body = client.request("POST", "/query", {"question": ANSWERABLE_QUESTION})

    assert status == 200, body
    assert body["found"] is True
    assert body["answer"].strip()
    assert body["citations"], "an answer must carry the evidence it used (PRD G2)"
    assert body["usage"]["tokens_out"] > 0
    assert isinstance(body["latency_ms"], int)

    citation = body["citations"][0]
    assert citation["doc_id"] == doc_id
    assert citation["filename"] == FIXTURE.name
    assert citation["score"] >= _configured_floor()

    # Retrieval, not phrasing: the evidence must contain the fact, however the
    # model chose to word its answer (TESTING.md §7).
    evidence = " ".join(c["text"] for c in body["citations"])
    assert EXPECTED_EVIDENCE in evidence


def _assert_unanswerable(client: _ApiClient) -> None:
    """An out-of-scope question is refused without reaching the model.

    Zero output tokens is the observable proof that ADR-003's similarity floor
    short-circuited generation, rather than the model being asked and declining.
    """
    status, body = client.request("POST", "/query", {"question": UNANSWERABLE_QUESTION})

    assert status == 200, body
    assert body["found"] is False
    assert body["citations"] == []
    assert body["usage"]["tokens_out"] == 0


# --------------------------------------------------------------------------- #
# Contract checks that need no corpus
# --------------------------------------------------------------------------- #


def test_requests_without_an_api_key_are_rejected(api_url):
    """API_SPEC.md §6: a missing key is a 403 from API Gateway itself."""
    request = urllib.request.Request(  # noqa: S310 - fixed https API endpoint
        f"{api_url}/documents", method="GET", headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=30)  # noqa: S310
    assert error.value.code == 403


def test_unsupported_file_types_are_rejected_at_presign(client):
    status, body = client.request("POST", "/documents", {"filename": "notes.docx"})

    assert status == 400
    assert body["error"]["code"] == "UNSUPPORTED_TYPE"


def test_an_invalid_question_is_rejected(client):
    status, body = client.request("POST", "/query", {"question": "a"})

    assert status == 400
    assert body["error"]["code"] == "INVALID_QUESTION"


def test_an_out_of_range_top_k_is_rejected(client):
    status, body = client.request("POST", "/query", {"question": "anything at all", "top_k": 99})

    assert status == 400
    assert body["error"]["code"] == "INVALID_TOP_K"


def test_deleting_an_unknown_document_returns_404(client):
    status, body = client.request("DELETE", "/documents/d_000000")

    assert status == 404
    assert body["error"]["code"] == "DOC_NOT_FOUND"
