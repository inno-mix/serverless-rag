# API_SPEC — Serverless RAG Document Q&A

REST API contract. All requests require header `x-api-key`. All responses are JSON with `Content-Type: application/json`. Errors use the standard error envelope (§6).

Base URL: CDK output `ApiStack.ApiUrl` (stage `v1`).

---

## 1. POST /documents — request upload

Request:
```json
{ "filename": "q3-report.pdf" }
```
Rules: extension must be `.pdf`, `.txt`, or `.md`; filename ≤ 255 chars.

Response `201`:
```json
{
  "doc_id": "d_8f3a1c",
  "upload_url": "https://<bucket>.s3.amazonaws.com/...",
  "method": "PUT",
  "expires_in": 300,
  "max_bytes": 26214400
}
```
Client then `PUT`s the raw file bytes to `upload_url`. The presigned URL enforces max size and content type.

Errors: `400 UNSUPPORTED_TYPE`, `400 INVALID_FILENAME`.

## 2. GET /documents — list documents

Response `200`:
```json
{
  "documents": [
    {
      "doc_id": "d_8f3a1c",
      "filename": "q3-report.pdf",
      "status": "READY",
      "uploaded_at": "2026-08-03T10:15:00Z",
      "chunk_count": 42,
      "error_msg": null
    }
  ]
}
```
`status` ∈ `UPLOADED | PROCESSING | READY | FAILED`. Sorted newest first.

## 3. DELETE /documents/{doc_id}

Response `204` (no body). Deletes S3 object, all chunks, and metadata. Idempotent: deleting a nonexistent `doc_id` returns `404 DOC_NOT_FOUND`.

## 4. POST /query

Request:
```json
{ "question": "What was Q3 revenue?", "top_k": 5 }
```
Rules: `question` 3–2000 chars; `top_k` optional, default 5, max 10.

Response `200` (answer found):
```json
{
  "answer": "Q3 revenue was $4.2M, up 12% year over year [chunk_00007].",
  "found": true,
  "citations": [
    {
      "chunk_id": "chunk_00007",
      "doc_id": "d_8f3a1c",
      "filename": "q3-report.pdf",
      "score": 0.83,
      "text": "…the chunk text used as evidence…"
    }
  ],
  "usage": { "tokens_in": 1450, "tokens_out": 210 },
  "latency_ms": 2870
}
```

Response `200` (not found — this is NOT an error):
```json
{
  "answer": "I couldn't find anything about that in your documents.",
  "found": false,
  "citations": [],
  "usage": { "tokens_in": 0, "tokens_out": 0 },
  "latency_ms": 640
}
```
Triggered when best similarity < `MIN_SIMILARITY`, corpus is empty, or the model replies `NOT_FOUND` (ADR-003). When generation was skipped, `usage` tokens are 0.

Errors: `400 INVALID_QUESTION`, `400 INVALID_TOP_K`.

## 5. Rate limiting

Usage plan: 5 requests/second, burst 10, quota 1000/day. Exceeding → `429` with `Retry-After` header (API Gateway default behavior).

## 6. Error envelope

Every non-2xx response:
```json
{
  "error": {
    "code": "UNSUPPORTED_TYPE",
    "message": "Only .pdf, .txt and .md files are supported."
  }
}
```

| HTTP | code | When |
|---|---|---|
| 400 | `UNSUPPORTED_TYPE` | bad file extension at presign |
| 400 | `INVALID_FILENAME` | filename fails validation |
| 400 | `INVALID_QUESTION` | question length out of bounds |
| 400 | `INVALID_TOP_K` | top_k out of bounds |
| 403 | — | missing/invalid API key (API GW default body) |
| 404 | `DOC_NOT_FOUND` | delete on unknown doc_id |
| 429 | — | throttled (API GW default body) |
| 500 | `INTERNAL` | unexpected failure; details in logs, never in the response |
| 503 | `MODEL_UNAVAILABLE` | Bedrock throttled/unavailable after retries |

## 7. CORS

Allow origin: the CloudFront distribution domain (CDK-wired, not `*` in deployed envs). Allow headers: `Content-Type, x-api-key`. Methods: `GET, POST, DELETE, OPTIONS`.
