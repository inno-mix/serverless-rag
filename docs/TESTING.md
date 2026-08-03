# TESTING — Serverless RAG Document Q&A

Test strategy across three layers. Unit and infra tests run in CI on every PR with **no AWS credentials and no network**. Integration tests run against a deployed stack.

---

## 1. Layers

| Layer | Location | Tooling | Runs | AWS access |
|---|---|---|---|---|
| Unit | `tests/unit/` | pytest, moto (DynamoDB), stubs (Bedrock) | every PR | none |
| Infra | `tests/infra/` | pytest + `aws_cdk.assertions` | every PR | none |
| Integration | `tests/integration/` | pytest + boto3 against deployed stack | post-deploy / manual | real |

## 2. Unit tests

Business logic lives in `src/shared/` as pure functions precisely so it can be tested without AWS.

### chunking.py (highest value — test hard)
- Empty string → `[]`; whitespace-only → `[]`.
- One short paragraph → exactly 1 chunk, no overlap artifacts.
- Multi-paragraph doc → every chunk ≤ target + tolerance; consecutive chunks share the configured overlap.
- Oversize single paragraph → split on sentence boundaries, never mid-sentence when sentence-splittable.
- Unicode / non-ASCII text survives round-trip intact.
- Property-style check: concatenating chunks minus overlaps reconstructs the original text (modulo normalized whitespace).

### vector_search.py
- Hand-computed 2D/3D vectors → exact expected scores (`pytest.approx`) and ranking.
- `k > n` returns n results; `k = 0` returns `[]`.
- Zero vector in corpus or query → score 0, no division error, no NaN in output.
- Determinism: equal scores keep stable chunk order.

### embeddings.py / generation wrapper
- Stub the Bedrock client (inject via parameter or monkeypatch the module-level client).
- Throttling exception → retried 3 times with backoff, then raises typed `ModelUnavailableError`.
- Response parsing: malformed model output → typed error, not KeyError.

### storage.py (moto)
- Vector pack/unpack round-trip: `np.allclose(original, unpacked)`, dtype float32, `dim` attribute correct.
- `put_chunks` batches correctly above 25 items (DynamoDB batch limit handling).
- `delete_document` removes meta + all chunks; idempotent on missing doc.
- `list_documents` returns newest-first via GSI1.

### Handlers
- Test the handler's parse/dispatch/format logic with fake events (S3 event fixture, API GW proxy event fixtures) and stubbed shared modules.
- Query handler: assert the generation stub is NOT called when best score < floor (ADR-003 regression guard — this is the single most important behavioral test in the suite).

## 3. Infra tests (CDK assertions)

One test module per stack. Assert existence + the properties that matter, not full-template snapshots (they're brittle):

- StorageStack: table `PAY_PER_REQUEST`, GSI1 present, bucket `PublicAccessBlockConfiguration` all true, DLQ exists.
- IngestStack: Lambda runtime `python3.12`, timeout 300, S3 event notification wired, env vars include `TABLE_NAME`, `EMBEDDING_MODEL_ID`.
- ApiStack: 4 methods exist, usage plan + API key attached, throttle settings 5/10.
- ObservabilityStack: dashboard exists, 2 alarms, budget amount 5 USD.
- Cross-cutting: a test that synthesizes the whole app and asserts NO resource of type `AWS::EC2::NatGateway`, `AWS::OpenSearchServerless::Collection`, or provisioned-concurrency config exists (cost guardrail as a test).

## 4. Integration smoke test

`tests/integration/test_smoke.py` — the full user journey, per IMPLEMENTATION_PLAN T2.4:

1. Read `ApiUrl` + API key from stack outputs / SSM.
2. Presign + upload `tests/fixtures/sample.pdf` (a small fixture whose content you control, containing a few known facts).
3. Poll `GET /documents` until `READY` (timeout 180 s, fail with last status).
4. `POST /query` with a question answerable from the fixture → assert `found=true`, ≥ 1 citation, citation `doc_id` matches.
5. `POST /query` with an unrelated question ("what is the capital of Mongolia") → assert `found=false` and `usage.tokens_out == 0`.
6. `DELETE` the document → assert 204, then `GET /documents` no longer lists it.

Keep it under ~4 minutes wall clock. This test doubles as the demo-readiness check.

## 5. Fixtures

- `tests/fixtures/sample.pdf` — 2–3 pages, generated in-repo (script committed) so its facts are known and stable.
- `tests/fixtures/sample.txt`, `sample.md`, `unsupported.docx` (empty bytes are fine — only the extension path is under test).
- `tests/fixtures/events/` — S3 event JSON, API GW proxy event JSON per route.

## 6. Coverage and gates

- Coverage gate in CI: `pytest --cov=src/shared --cov-fail-under=80 tests/unit`.
- CI order: ruff → unit → infra → `cdk synth`. Fail fast.
- Integration tests are NOT part of the PR gate (they need a deployed stack); run them in the post-deploy job on `main` and before any demo.

## 7. What we deliberately do not test

- Bedrock model output quality (non-deterministic) — the smoke test asserts structure (found flag, citations), not answer wording. The optional golden-set eval (Phase 4) addresses quality separately.
- AWS service behavior itself (presigned URL enforcement, API GW throttling) — trust the platform; test our configuration of it via CDK assertions.
