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
- **Decision:** `POST /documents` returns a presigned URL; client uploads directly to S3.
- **Consequences:** Supports 25 MB files, cheaper, and demonstrates the standard production pattern.
- **Amendment (Phase 2, T2.3): presigned POST, not PUT.** §6 requires oversized files to be "rejected at presign (content-length condition)", but a presigned **PUT** URL cannot carry one. `generate_presigned_url` signs headers, and the only way to bound the body is to sign an exact `Content-Length`, which requires the client to declare its size up front and reports a mismatch as an opaque `403 SignatureDoesNotMatch`. Only `generate_presigned_post` produces a **policy document**, which is what can hold a `["content-length-range", 1, max_bytes]` condition alongside an exact-key and exact-content-type condition. The route therefore returns `"method": "POST"` plus a `fields` object, and the client posts `multipart/form-data` (API_SPEC.md §1). The alternative — presign a plain PUT and check the size in the ingest Lambda — was rejected because the oversized bytes would already be in the bucket by then, which is not what "rejected at presign" means.

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
- **Amendment (Phase 1):** In `us-east-1`, current Claude models are **not** invocable by their bare `anthropic.*` id — only `anthropic.claude-3-haiku-20240307-v1:0` still supports bare on-demand invocation. Every current model must be addressed through a regional or global **inference profile** (`us.anthropic.*` / `global.anthropic.*`); a bare id fails at runtime with _"on-demand throughput isn't supported"_. `generationModelId` therefore carries the `us.` prefix. Verified against the account with `aws bedrock list-inference-profiles`.

### ADR-006: EventBridge delivery for upload events instead of a direct S3 notification

- **Status:** Accepted (Phase 1, T1.4)
- **Context:** §1 describes `S3 ObjectCreated → Lambda (ingest)`. Implemented as a direct bucket notification, CDK places the notification resource in the **bucket's** stack (`StorageStack`), where it must reference the ingest function's ARN. `IngestStack` already references `StorageStack` for the table, bucket, and DLQ, so CloudFormation rejects the pair as a dependency cycle. This is inherent to the one-stack-per-concern split in §2, not an incidental wiring mistake.
- **Decision:** Enable `EventBridgeEnabled` on the documents bucket and match uploads with an EventBridge rule that lives in `IngestStack` and targets the ingest function. The rule filters on `detail.object.key` with the `uploads/` prefix.
- **Consequences:** References now point one way only (`IngestStack → StorageStack`), so the stacks stay split as §2 intends. The event payload changes shape: handlers read `detail.bucket.name` / `detail.object.key` rather than `Records[].s3`, and EventBridge delivers the key **un-encoded**, so no URL-decoding step is needed. EventBridge adds no idle cost. The EventBridge target deliberately carries **no** DLQ, since attaching one puts a queue policy referencing the rule ARN back into `StorageStack` and recreates the cycle; undelivered invocations are covered by the function's own dead-letter queue.
- **Alternative rejected:** Collapsing storage and ingestion into one stack. That removes the cycle but discards the stack separation §2 specifies.

### ADR-007: Lambda layer built without Docker

- **Status:** Accepted (Phase 1, T1.4)
- **Context:** §2 says shared code ships as a layer, but the layer also needs `numpy` (a binary wheel) and `pypdf`. CDK's normal answer is Docker bundling, which would make Docker a prerequisite for `cdk synth` — and therefore for the test suite and for the "clone and deploy" success metric in PRD §7.
- **Decision:** Implement CDK's `ILocalBundling` escape hatch: build the layer with the local `pip`, requesting `--platform manylinux2014_x86_64 --only-binary=:all:` so Linux wheels are resolved even on an arm64 developer machine. CDK falls back to the Docker image automatically if the local build returns `False`.
- **Consequences:** `cdk deploy` stays a single command with no Docker requirement. Tests set the `aws:cdk:bundling-stacks` context to `[]` so no suite ever shells out to pip. The layer asset hash covers both the shared sources and the pinned dependency specifiers, so bumping a pin invalidates the cached asset.

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
| Oversized file (> 25 MB)                 | rejected by S3 against the `content-length-range` condition in the presigned POST policy (ADR-002 amendment) |
| Empty extraction (0 chars)               | FAILED with `EMPTY_DOCUMENT`                                                 |
| Query on empty corpus                    | not-found response, no LLM call                                              |

## 7. Security

- S3 buckets: block all public access; CloudFront OAC for the frontend bucket.
- API: API key + usage plan (throttle 5 rps, burst 10) at MVP.
- IAM: per-function roles; DynamoDB scoped to the table + GSI. The `/documents` routes are three separate Lambdas rather than one router precisely so this is enforceable: only the delete role can remove an object, the list role is read-only, and the query role has no S3 access at all. `tests/infra/test_api_stack.py` asserts each of those.
- Filenames supplied by clients become part of an S3 object key, so `POST /documents` rejects path separators, leading dots, and control characters before signing anything (API_SPEC.md §1).
- Bedrock `InvokeModel` is scoped to the configured models, but that is **more than two ARNs** (amended in Phase 1). A regional inference profile is not itself the invocable resource: the caller needs the **inference-profile ARN plus the underlying foundation-model ARN in every region the profile can route to**. For `us.anthropic.claude-haiku-4-5-…` that is us-east-1, us-east-2, and us-west-2. Granting only the deployment region yields `AccessDeniedException` the first time Bedrock routes elsewhere. Still least privilege — no wildcard resources — just a wider ARN list than originally anticipated. A foundation-model ARN ends in the **complete** model id including its provider prefix (`foundation-model/amazon.titan-embed-text-v2:0`); only the `us.`/`global.` routing prefix of an inference-profile id is stripped to name the model behind it. `infra/bedrock_policy.py` is the single implementation of this rule, and `tests/infra/test_bedrock_policy.py` pins both cases.
- No secrets in code; nothing sensitive in the repo. `.env` files gitignored.

## 8. Observability

- Structured JSON logs via `logging_utils.py`: `request_id, doc_id, latency_ms, tokens_in, tokens_out, top_score`.
- CloudWatch dashboard (CDK): Lambda invocations/errors/p95 duration, API GW 4xx/5xx, DLQ depth, Bedrock token counters (custom metrics via EMF).
- Alarms: DLQ depth > 0; Lambda error rate > 5%; monthly AWS Budget at $5.
