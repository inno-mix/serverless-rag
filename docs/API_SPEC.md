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
  "upload_url": "https://<bucket>.s3.amazonaws.com/",
  "method": "POST",
  "fields": {
    "key": "uploads/d_8f3a1c/q3-report.pdf",
    "Content-Type": "application/pdf",
    "policy": "<base64 policy document>",
    "x-amz-algorithm": "AWS4-HMAC-SHA256",
    "x-amz-credential": "...",
    "x-amz-date": "...",
    "x-amz-signature": "..."
  },
  "expires_in": 300,
  "max_bytes": 26214400
}
```

The client then sends a `multipart/form-data` **POST** to `upload_url` containing every entry of `fields` followed by a `file` part carrying the bytes. The signed policy enforces the exact object key, the content type, and a `content-length-range` of 1 to `max_bytes`; S3 rejects anything outside those bounds with `400 EntityTooLarge` and the bytes never reach the bucket.

**Amendment (Phase 2, T2.3).** This was originally specified as a presigned **PUT**. A presigned PUT URL cannot carry a `content-length-range` condition — only the POST policy document can — so PUT could not deliver the "rejected at presign" behaviour `ARCHITECTURE.md` §6 requires. The `fields` object is new and `method` changed from `PUT` to `POST`; nothing else about the route changed. See ADR-002.

Errors: `400 UNSUPPORTED_TYPE`, `400 INVALID_FILENAME`.

Filename rules beyond the length and extension checks: the name becomes part of the object key, so path separators (`/`, `\`), a leading `.`, and control characters are rejected with `400 INVALID_FILENAME`.

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

Allow headers: `Content-Type, x-api-key`. Methods: `GET, POST, DELETE, OPTIONS`.

**Amendment (Phase 3, T3.1).** The deployed UI no longer makes cross-origin calls: CloudFront serves the app and proxies `/v1/*` to this API, so the browser's requests are same-origin and no preflight occurs (ADR-008). CORS therefore applies only to direct calls to the `execute-api` URL — a developer testing from `localhost`, say — and `allowedOrigin` defaults to `*` for that case. This is not a weakening: CORS governs which page may *attempt* a request, never whether the API accepts it, and every route still requires an `x-api-key` that a cross-origin caller has no way to obtain.

The documents bucket carries its own CORS rule allowing `POST` from any origin, because the browser uploads bytes straight to S3. The control on an upload is the signed policy (§1), which pins the key, the content type, and the size range.
