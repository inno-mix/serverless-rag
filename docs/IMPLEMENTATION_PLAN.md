# IMPLEMENTATION_PLAN — Serverless RAG Document Q&A

Build in this order. Each task lists acceptance criteria (AC) — a task is done only when all its ACs pass and the Definition of Done in `CLAUDE.md` is met. Do not start a phase before the previous one's ACs pass.

---

## Phase 0 — Skeleton, tooling, CI

### T0.1 Repo scaffold
- Create the repository layout from `ARCHITECTURE.md` §2, with empty modules and `__init__.py` files.
- `requirements.txt`: `aws-cdk-lib`, `constructs`, `pytest`, `pytest-cov`, `ruff`, `moto[dynamodb]`, `numpy`, `pypdf`, `boto3`.
- `cdk.json` with context keys: `embeddingModelId`, `generationModelId`, `minSimilarity` (0.35), `maxTokens` (1024), `maxUploadMb` (25). Leave model ID values as clearly-marked placeholders to be verified against current AWS docs.
- **AC:** `cdk synth` runs (empty app is fine); `pytest` runs with 0 tests collected without error; `ruff check .` passes.

### T0.2 CI pipeline
- `.github/workflows/ci.yml`: on PR → ruff + pytest (unit, infra) + `cdk synth`; on push to `main` → same, then `cdk deploy --all` using OIDC role auth (document the required GitHub secrets/vars in the workflow file header comment).
- **AC:** workflow YAML is valid; PR job has no deploy step; deploy job gated to `main`.

## Phase 1 — Storage + ingestion pipeline

### T1.1 StorageStack
- Docs bucket (private, versioned off, `RemovalPolicy.DESTROY`), DynamoDB single table `rag-store` (on-demand, PK/SK + GSI1 per `ARCHITECTURE.md` §3), SQS DLQ.
- **AC:** CDK assertions test verifies: table billing mode PAY_PER_REQUEST, GSI1 exists, bucket blocks public access.

### T1.2 Chunker (`src/shared/chunking.py`)
- `chunk_text(text: str, target_tokens: int = 500, overlap_tokens: int = 50) -> list[Chunk]`.
- Paragraph-aware: split on blank lines first, pack paragraphs into chunks up to target size, split oversize paragraphs on sentence boundaries, apply overlap between adjacent chunks. Token estimation: `len(text) // 4` heuristic is acceptable — document it.
- **AC:** unit tests cover: empty input → `[]`; single short paragraph → 1 chunk; long document → all chunks ≤ target+tolerance; overlap present between consecutive chunks; no chunk starts/ends mid-sentence for sentence-splittable input. Coverage ≥ 80%.

### T1.3 Embedding + storage wrappers (`src/shared/embeddings.py`, `storage.py`)
- `embed_texts(texts: list[str]) -> list[np.ndarray]` calling Bedrock (model ID from env), with retry/backoff (3 attempts, jitter) on throttling.
- `storage.py`: `put_document_meta`, `update_status`, `put_chunks (batch, vectors b64-packed float32)`, `get_all_chunks`, `list_documents`, `delete_document`.
- **AC:** unit tests with stubbed Bedrock verify retry behavior; moto-backed tests verify round-trip of pack/unpack vectors (`np.allclose`) and all storage functions.

### T1.4 Ingest Lambda (`src/ingest/handler.py`) + IngestStack
- S3 event → extract (pypdf for `.pdf`, utf-8 read for `.txt`/`.md`) → chunk → embed → store → status `READY`. Failures → status `FAILED` with reason, message to DLQ, and cleanup of any partially written chunks.
- Reject unsupported extensions with `FAILED / UNSUPPORTED_TYPE`. Empty extraction → `FAILED / EMPTY_DOCUMENT`.
- IngestStack wires S3 notification → Lambda, layer, env vars, DLQ, timeout 300s.
- **AC:** unit tests for the extract/dispatch logic with fixture files (small real PDF fixture, txt, md, unsupported `.docx`); CDK assertions test verifies event source mapping + env vars present.

## Phase 2 — Query pipeline + API

### T2.1 Vector search (`src/shared/vector_search.py`)
- `top_k(query_vec, chunks, k) -> list[ScoredChunk]` — vectorized NumPy cosine similarity (normalize once, matrix multiply), stable sort, ties broken by chunk order.
- **AC:** unit tests: known-geometry vectors return expected ranking and scores (hand-computed); k > n handled; zero-vector safe (no div-by-zero).

### T2.2 Query Lambda (`src/query/handler.py`)
- Flow per `ARCHITECTURE.md` §1 and ADR-003: embed question → retrieve → similarity floor check → prompt build (contract in §5) → Bedrock generation → parse citations → response per `API_SPEC.md`.
- `NOT_FOUND` model reply and below-floor case both map to the standard not-found response.
- **AC:** unit tests with stubbed Bedrock cover: happy path with citations; below-floor short-circuits generation (assert generation stub NOT called); model returns `NOT_FOUND`; empty corpus.

### T2.3 Documents Lambda + ApiStack
- `POST /documents` (presign with content-length + type conditions), `GET /documents`, `DELETE /documents/{doc_id}`, `POST /query`. REST API with API key + usage plan (5 rps / burst 10), CORS for the frontend origin.
- **AC:** CDK assertions: 4 routes exist, usage plan attached; moto/unit tests for presign conditions and delete cascade (chunks + S3 object).

### T2.4 Integration smoke test
- `tests/integration/test_smoke.py`: reads stack outputs → uploads fixture PDF via presign → polls status to `READY` (timeout 3 min) → queries a question answerable from the fixture → asserts non-empty answer with ≥ 1 citation → queries an unanswerable question → asserts not-found → deletes document.
- **AC:** passes against a freshly deployed stack.

## Phase 3 — Frontend, observability, polish

### T3.1 FrontendStack + static UI
- S3 + CloudFront (OAC), vanilla JS single-page app: upload, doc list with live status polling, question box, answer with expandable cited chunks.
- **AC:** `cdk deploy` outputs a CloudFront URL where the full flow works manually end-to-end.

### T3.2 ObservabilityStack
- Structured logging adopted in all handlers; EMF custom metrics for token usage; dashboard + alarms + $5 budget per `ARCHITECTURE.md` §8.
- **AC:** CDK assertions verify dashboard, 2 alarms, budget; a deployed query produces one JSON log line containing `latency_ms`, `top_score`, `tokens_out`.

### T3.3 README + docs polish
- Root `README.md`: what/why, architecture diagram, ADR-001 summary, quick start (`cdk deploy` to working demo), cost table, teardown instructions.
- **AC:** a reader can go from clone to demo using only the README.

## Phase 4 — Stretch (only on explicit request)

- T4.1 Textract fallback for scanned PDFs (Step Functions for the async job).
- T4.2 Cognito auth replacing API key.
- T4.3 Streaming responses (Lambda response streaming via function URL).
- T4.4 Golden-set eval harness (10 Q&A pairs on a fixture corpus) wired into CI as a regression gate.
