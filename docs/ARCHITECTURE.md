# ARCHITECTURE — Serverless RAG Document Q&A

How the system is designed and why. Requirements live in `PRD.md`. Build order lives in `IMPLEMENTATION_PLAN.md`.

---

## 1. High-level design

Two independent pipelines sharing one vector store.

```
INGESTION (async, event-driven)
  Client ──POST /documents──▶ API GW ──▶ Lambda (presign) ──▶ presigned URL
  Client ──PUT file─────────▶ S3 (docs bucket)
  S3 ObjectCreated ─────────▶ Lambda (ingest)
                                ├─ extract text (pypdf / plain)
                                ├─ chunk (paragraph-aware, ~500 tok, 50 overlap)
                                ├─ embed each chunk (Bedrock Titan Embed V2)
                                └─ write chunks+vectors ──▶ DynamoDB
                              on failure ──▶ status=FAILED + DLQ (SQS)

QUERY (sync, request/response)
  Client ──POST /query──▶ API GW ──▶ Lambda (query)
                                ├─ embed question (same model)
                                ├─ scan vectors, cosine sim in NumPy, top-k
                                ├─ if max sim < MIN_SIMILARITY → "not found" (no LLM call)
                                └─ Bedrock Claude (context + question) ──▶ answer + citations

FRONTEND:  S3 (static site) ──▶ CloudFront
```

## 2. Repository layout

```
serverless-rag/
├── app.py                      # CDK app entry point
├── cdk.json                    # CDK context: model IDs, region, thresholds
├── requirements.txt            # CDK + dev deps
├── infra/
│   ├── __init__.py
│   ├── storage_stack.py        # S3 buckets, DynamoDB table, DLQ
│   ├── ingest_stack.py         # ingest Lambda, S3 event source, alarms
│   ├── api_stack.py            # API GW, presign/query/list/delete Lambdas
│   ├── frontend_stack.py       # static site bucket + CloudFront
│   └── observability_stack.py  # dashboard, budget alarm
├── src/
│   ├── ingest/handler.py
│   ├── query/handler.py
│   ├── documents/handler.py    # presign, list, delete
│   └── shared/                 # Lambda layer (pure Python, no boto3 mocking needed)
│       ├── chunking.py
│       ├── embeddings.py       # Bedrock client wrappers
│       ├── vector_search.py    # cosine similarity, top-k
│       ├── storage.py          # DynamoDB access patterns
│       └── logging_utils.py    # structured JSON logging
├── frontend/                   # static assets (index.html, app.js, styles.css)
├── tests/
│   ├── unit/                   # chunking, similarity, prompt building
│   ├── infra/                  # CDK assertions
│   └── integration/            # smoke test against deployed stack
└── .github/workflows/ci.yml
```

## 3. Data model (DynamoDB, single table)

Table: `rag-store`, on-demand billing, PK/SK design:

| Entity        | PK               | SK                     | Attributes                                                             |
| ------------- | ---------------- | ---------------------- | ---------------------------------------------------------------------- |
| Document meta | `DOC#{doc_id}`   | `META`                 | `filename, status, uploaded_at, chunk_count, error_msg?, s3_key`       |
| Chunk         | `DOC#{doc_id}`   | `CHUNK#{n:05d}`        | `text, vector (List[float] or b64-packed float32), token_count, page?` |
| Doc listing   | GSI1PK=`DOCLIST` | GSI1SK=`{uploaded_at}` | projection: meta only                                                  |

Access patterns:

- List all documents → GSI1 query on `DOCLIST` (newest first).
- Get all chunks for similarity search → `Scan` with filter `begins_with(SK, "CHUNK#")`. Acceptable at MVP scale (ADR-001); vectors for a query fit in memory.
- Delete document → Query PK, BatchWrite delete all items, delete S3 object.

Vector storage note: store as base64-packed `float32` bytes (`struct`/`numpy.tobytes`) rather than a DynamoDB number list — ~4x smaller items, faster parse. Keep a `dim` attribute for safety.

## 4. Architecture Decision Records

### ADR-001: Brute-force vector search in Lambda instead of a vector database

- **Status:** Accepted
- **Context:** OpenSearch Serverless has a minimum OCU capacity floor (~$300+/mo regardless of traffic). Pinecone free tier adds a non-AWS dependency. Portfolio corpus is small (≤ ~50k chunks).
- **Decision:** Store vectors in DynamoDB; compute exact cosine similarity in the query Lambda with NumPy.
- **Consequences:** O(n) per query — fine below ~50k chunks (tens of ms of math; the bottleneck is the DynamoDB scan). Zero idle cost. Exact (not approximate) results.
- **Migration trigger:** corpus > ~50k chunks or p50 retrieval > 1 s → move to pgvector on Aurora Serverless v2 (scale-to-zero) or OpenSearch; `vector_search.py` is the single seam to swap.

### ADR-002: Presigned S3 upload instead of proxying files through API Gateway

- **Context:** API GW has a 10 MB payload limit and charges for transfer; Lambda has payload limits too.
- **Decision:** `POST /documents` returns a presigned PUT URL; client uploads directly to S3.
- **Consequences:** Supports 25 MB files, cheaper, and demonstrates the standard production pattern.

### ADR-003: Similarity floor before generation

- **Context:** RAG systems hallucinate when retrieval returns garbage; also every LLM call costs money.
- **Decision:** If best cosine score < `MIN_SIMILARITY` (context-configurable, default 0.35), return "not found" without calling Claude.
- **Consequences:** Prevents hallucinated answers on out-of-scope questions and saves generation cost.

### ADR-004: No LangChain / LlamaIndex

- **Decision:** Implement chunking, retrieval, and prompting directly.
- **Rationale:** Value comes from demonstrating understanding of the mechanics; fewer deps, smaller cold starts, nothing hidden behind abstractions.

### ADR-005: Model IDs as CDK context, not code

- **Decision:** `cdk.json` context keys: `embeddingModelId`, `generationModelId`, `minSimilarity`, `maxTokens`. Lambdas read them from environment variables set by CDK.
- **Rationale:** Bedrock model availability and naming change; swapping models must not require code edits. Verify current model IDs in the AWS Bedrock console/docs at build time.

## 5. Prompt contract (query Lambda)

System prompt requirements:

- Answer ONLY from the provided context chunks.
- Cite chunk IDs inline like `[chunk_003]` after each supported claim.
- If context is insufficient, reply exactly: `NOT_FOUND` (handler maps this to the standard not-found response).

Context format sent to the model:

```
<chunk id="chunk_003" source="report.pdf">
...chunk text...
</chunk>
```

## 6. Error handling

| Failure                                  | Behavior                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------- |
| Unsupported file type / extraction fails | status=FAILED with reason; message to SQS DLQ; no partial chunks left behind |
| Bedrock throttling                       | retry with exponential backoff + jitter (3 attempts), then fail gracefully   |
| Oversized file (> 25 MB)                 | rejected at presign (content-length condition on the presigned POST/PUT)     |
| Empty extraction (0 chars)               | FAILED with `EMPTY_DOCUMENT`                                                 |
| Query on empty corpus                    | not-found response, no LLM call                                              |

## 7. Security

- S3 buckets: block all public access; CloudFront OAC for the frontend bucket.
- API: API key + usage plan (throttle 5 rps, burst 10) at MVP.
- IAM: per-function roles; Bedrock `InvokeModel` scoped to the two configured model ARNs; DynamoDB scoped to the table + GSI.
- No secrets in code; nothing sensitive in the repo. `.env` files gitignored.

## 8. Observability

- Structured JSON logs via `logging_utils.py`: `request_id, doc_id, latency_ms, tokens_in, tokens_out, top_score`.
- CloudWatch dashboard (CDK): Lambda invocations/errors/p95 duration, API GW 4xx/5xx, DLQ depth, Bedrock token counters (custom metrics via EMF).
- Alarms: DLQ depth > 0; Lambda error rate > 5%; monthly AWS Budget at $5.
