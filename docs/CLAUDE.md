# CLAUDE.md — Agent instructions for this repository

You are building the Serverless RAG Document Q&A project. Read `PRD.md` (what/why), `ARCHITECTURE.md` (how), and `IMPLEMENTATION_PLAN.md` (order) before writing code. This file defines the rules you must follow while coding.

## Project facts

- Language: **Python 3.12** everywhere (CDK app and Lambdas).
- IaC: **AWS CDK v2 (Python)**. No Terraform, no SAM, no console-created resources.
- Region: `us-east-1`. Account/region come from the standard CDK environment, never hardcoded in stacks.
- AI: Amazon Bedrock only. Model IDs come from `cdk.json` context → Lambda env vars (**ADR-005**). Never hardcode a model ID in Python source. Verify current model IDs against AWS docs before setting context values.
- No LangChain, LlamaIndex, or other RAG frameworks (**ADR-004**). Allowed runtime deps: `boto3` (provided by runtime), `numpy`, `pypdf`. Ask before adding anything else.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# quality gates — run before every commit
ruff check . && ruff format --check .
pytest tests/unit tests/infra -q
cdk synth --quiet          # must succeed with no errors

# deploy / teardown
cdk deploy --all --require-approval never
cdk destroy --all

# integration smoke test (needs deployed stack; reads outputs)
pytest tests/integration -q
```

## Definition of done (every task)

1. Code + unit tests written; `pytest` green.
2. `ruff check` and `ruff format --check` clean.
3. `cdk synth` succeeds; infra changes covered by a CDK assertions test.
4. Acceptance criteria for the task in `IMPLEMENTATION_PLAN.md` are met.
5. No TODOs, placeholders, or commented-out code left behind.

## Coding conventions

- Type hints on all function signatures. Dataclasses (or plain dicts with TypedDict) for structured data — no pydantic (cold-start weight).
- Every Lambda handler is a thin adapter: parse event → call pure functions in `src/shared/` → format response. Business logic lives in `src/shared/` so it is unit-testable without AWS.
- Structured logging only, via `src/shared/logging_utils.py`. Never `print()`. Log line = single JSON object.
- Errors: raise typed exceptions in shared code; handlers map them to HTTP responses per `API_SPEC.md`. Never return a 200 with an error body.
- Docstrings: Google style, on all public functions in `src/shared/`.
- No global mutable state in handlers except cached clients (boto3 clients initialized at module level is correct and intended for warm starts).

## CDK conventions

- One stack per concern, as laid out in `ARCHITECTURE.md` §2. Cross-stack wiring via constructor props, not `Fn.import_value`.
- Every stack exposes needed values via `CfnOutput` (API URL, bucket names, table name) — the integration tests read these.
- IAM: use `grant_*` methods (`table.grant_read_write_data(fn)`); hand-written policy statements only for Bedrock `InvokeModel`, scoped to the two model ARNs.
- Lambda config defaults: `memory_size=512` (query: 1024), `timeout=30s` (ingest: 300s), `runtime=PYTHON_3_12`, log retention 2 weeks.
- `RemovalPolicy.DESTROY` + `auto_delete_objects=True` on buckets and the table — this is a teardown-friendly project, not production data.
- Shared code ships as a Lambda **layer** built from `src/shared/` so all functions import identical logic.

## Guardrails (do not violate)

- **Cost:** never add always-on resources (no OpenSearch, no provisioned concurrency, no NAT gateways, no VPCs unless unavoidable). DynamoDB stays on-demand. `max_tokens` for generation ≤ the `maxTokens` context value.
- **Safety:** the query pipeline must enforce the similarity floor (ADR-003) BEFORE any generation call.
- **Scope:** do not build stretch-goal features (Textract, Cognito, streaming) unless the current phase in `IMPLEMENTATION_PLAN.md` says so.
- **Secrets:** nothing sensitive in code, tests, fixtures, or git history.

## Testing rules

See `TESTING.md` for the full strategy. Non-negotiables:

- `src/shared/chunking.py` and `src/shared/vector_search.py` are pure and tested to ≥ 80% coverage.
- Bedrock and DynamoDB calls are wrapped in `src/shared/` modules; unit tests stub these wrappers (use `moto` for DynamoDB, plain stubs for Bedrock — never call real AWS in unit tests).
- Every new stack or resource change gets a `aws_cdk.assertions` test (resource exists + key properties).

## When uncertain

- Requirements conflict or something is ambiguous → check `PRD.md` first; if still unclear, stop and ask rather than assume.
- A library/model/API detail may have changed since your training data → check the official AWS docs rather than guessing.
- Keep diffs small and focused; one phase task per commit, imperative-mood commit messages (e.g. `Add paragraph-aware chunker with overlap`).
