# PRD — Serverless RAG Document Q&A

**Status:** Draft v1.0
**Owner:** Micko Matamorosa
**Purpose of this doc:** Single source of truth for _what_ to build and _why_. For _how_, see `ARCHITECTURE.md`. For build order, see `IMPLEMENTATION_PLAN.md`. For agent coding rules, see `CLAUDE.md`.

---

## 1. Overview

A serverless web application where a user uploads documents (PDF, TXT, MD) and asks natural-language questions about them. The system retrieves the most relevant passages and generates a grounded answer **with citations** using Amazon Bedrock. Built entirely on AWS with **AWS CDK v2 (Python)** as infrastructure-as-code.

**Goal:** Event-driven serverless design, RAG implementation from first principles (no LangChain), cost-conscious architecture decisions, IaC, CI/CD, testing, and observability.

## 2. Problem statement

LLMs cannot answer questions about private documents, and pasting whole documents into prompts is expensive, slow, and hits context limits. RAG solves this by retrieving only the relevant chunks at query time.

## 3. Goals and non-goals

### Goals (MVP)

- G1: Upload a document (≤ 25 MB, PDF/TXT/MD) and have it automatically indexed within 2 minutes.
- G2: Ask a question and receive an answer grounded ONLY in uploaded documents, with citations (document name + chunk reference).
- G3: When no relevant content exists, respond with an explicit "not found in your documents" message — never hallucinate.
- G4: Entire stack deployable from scratch with `cdk deploy` in one command.
- G5: Idle cost ≈ $0/month; demo-traffic cost < $5/month.

### Non-goals (MVP)

- NG1: Multi-user auth / tenancy (single API key is sufficient; Cognito is a stretch goal).
- NG2: OCR of scanned documents (Textract integration is a stretch goal).
- NG3: Managed vector database (OpenSearch/Pinecone). See ADR-001 in `ARCHITECTURE.md`.
- NG4: Streaming responses (stretch goal).
- NG5: Conversation memory / multi-turn chat (stretch goal).

## 4. Users and use cases

| User                         | Use case                                                                  |
| ---------------------------- | ------------------------------------------------------------------------- |
| Hiring manager / interviewer | Opens live demo, uploads a sample PDF, asks questions, inspects citations |
| You (owner)                  | Deploys, tears down, demos; discusses tradeoffs in interviews             |
| AI coding agent              | Builds the system by following this doc set                               |

## 5. Functional requirements

### FR-1 Document ingestion

- FR-1.1: User obtains a presigned S3 URL via `POST /documents` and uploads directly to S3 (API Gateway must NOT proxy file bytes).
- FR-1.2: S3 `ObjectCreated` event triggers the ingestion Lambda.
- FR-1.3: Ingestion extracts text (pypdf for PDF; plain read for TXT/MD), chunks it (see FR-1.4), embeds each chunk via Bedrock Titan Text Embeddings V2, and writes chunks + vectors + metadata to DynamoDB.
- FR-1.4: Chunking: paragraph-aware splitting, target ~500 tokens per chunk, ~50-token overlap between adjacent chunks. Chunks must never split mid-sentence when avoidable.
- FR-1.5: Document status tracked in DynamoDB: `UPLOADED → PROCESSING → READY | FAILED` (with failure reason).
- FR-1.6: `GET /documents` lists documents with status; `DELETE /documents/{doc_id}` removes the S3 object and all chunks.

### FR-2 Query

- FR-2.1: `POST /query` accepts `{"question": str, "top_k": int (default 5, max 10)}`.
- FR-2.2: Embed the question with the SAME embedding model used at ingestion.
- FR-2.3: Retrieve top-k chunks by cosine similarity computed in Lambda (brute force over stored vectors).
- FR-2.4: If the best similarity score < `MIN_SIMILARITY` (default 0.35, configurable), return the "not found" response WITHOUT calling the generation model.
- FR-2.5: Otherwise call Claude on Bedrock with retrieved chunks, instructing it to answer only from context and cite chunk IDs.
- FR-2.6: Response includes: answer text, citations array (doc name, chunk id, similarity score), latency ms, and token usage.

### FR-3 Frontend (thin)

- FR-3.1: Static single-page UI on S3 + CloudFront: upload widget, document list with status, question box, answer view with expandable citations.
- FR-3.2: No framework build step required (vanilla JS or a single-file app is acceptable).

## 6. Non-functional requirements

| ID    | Requirement                                             | Target                                                                                         |
| ----- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| NFR-1 | Query latency (p50, excluding cold start)               | < 4 s                                                                                          |
| NFR-2 | Ingestion time for a 50-page PDF                        | < 2 min                                                                                        |
| NFR-3 | Idle monthly cost                                       | ≈ $0 (no always-on compute)                                                                    |
| NFR-4 | Generation cost cap                                     | `max_tokens ≤ 1024`, cheapest suitable Claude model on Bedrock                                 |
| NFR-5 | Budget guardrail                                        | AWS Budget alarm at $5/month deployed via CDK                                                  |
| NFR-6 | All infra in CDK Python; zero console-clicked resources | 100%                                                                                           |
| NFR-7 | Test coverage on `src/shared/` (chunking, similarity)   | ≥ 80%                                                                                          |
| NFR-8 | Observability                                           | Structured JSON logs; CloudWatch dashboard: invocations, errors, duration, Bedrock token usage |
| NFR-9 | Least privilege                                         | Each Lambda gets only the IAM actions it needs; no `*` resources except where AWS requires     |

## 7. Success metrics

- Live demo URL + public GitHub repo linked in resume.
- README explains ADR-001 (vector storage tradeoff) well enough to discuss in an interview.
- `git clone && cdk deploy` works for a stranger with an AWS account.
- CI pipeline green: lint + tests + `cdk synth` on every PR; deploy on merge to `main`.

## 8. Constraints and assumptions

- Region: `us-east-1` (broadest Bedrock model availability). **Verify Bedrock model access is enabled in the target account/region before first deploy.**
- Model IDs are configuration, not code — set in `cdk.json` context. Verify current IDs and pricing at build time; do not hardcode assumptions from training data.
- Corpus scale assumption for ADR-001: ≤ ~50k chunks total. Beyond that, brute-force search degrades; migration path documented in `ARCHITECTURE.md`.
- Python 3.12 for all Lambdas and CDK app.

## 9. Release plan

Phases and acceptance criteria live in `IMPLEMENTATION_PLAN.md`. Summary:

1. **P0** — CDK skeleton + CI
2. **P1** — Ingestion pipeline
3. **P2** — Query pipeline
4. **P3** — Frontend + hardening + observability
5. **P4** — Stretch: Textract, Cognito, streaming, eval harness

## 10. Open questions

- OQ-1: Add a tiny golden-set eval (10 Q&A pairs) to CI as a regression gate? (Recommended, P4.)
- OQ-2: Cognito for the live demo, or API key + rate limiting only?
- OQ-3: Response streaming via Lambda URLs, or keep simple request/response?
