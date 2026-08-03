# CLAUDE.md — Serverless RAG Document Q&A

**Full agent instructions live in [docs/CLAUDE.md](docs/CLAUDE.md). Read that file before writing code.**
This root file exists so the rules load automatically in every session; it is a summary, not a replacement.

## Planning docs — read in this order

| Doc | Role |
|---|---|
| [docs/PRD.md](docs/PRD.md) | What we're building and why; requirements and constraints |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, repo layout, data model, ADRs (decisions, not suggestions) |
| [docs/CLAUDE.md](docs/CLAUDE.md) | Full coding rules, conventions, and guardrails |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phased tasks with acceptance criteria — drives build order |
| [docs/API_SPEC.md](docs/API_SPEC.md) | Exact HTTP contract (read before Phase 2) |
| [docs/TESTING.md](docs/TESTING.md) | Test strategy, fixtures, coverage gates |

## Project facts

- **Python 3.12** everywhere. IaC is **AWS CDK v2 (Python)** — no Terraform, no SAM, no console-created resources.
- Region `us-east-1`, supplied via the `region` key in `cdk.json` context and never hardcoded in a stack.
- AI is **Amazon Bedrock only**. Model IDs come from `cdk.json` context → Lambda env vars (ADR-005); never hardcode one in Python.
- **No LangChain / LlamaIndex** (ADR-004). Allowed runtime deps: `boto3`, `numpy`, `pypdf`. Ask before adding anything else.

## Commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# quality gates — run before every commit
ruff check . && ruff format --check .
pytest tests/unit tests/infra -q --cov=src/shared --cov-fail-under=80
cdk synth --quiet

cdk deploy --all --require-approval never
cdk destroy --all
pytest tests/integration -q          # needs a deployed stack
```

## Definition of done (every task)

1. Code + unit tests written; `pytest` green.
2. `ruff check` and `ruff format --check` clean.
3. `cdk synth` succeeds; infra changes covered by an `aws_cdk.assertions` test.
4. The task's acceptance criteria in `docs/IMPLEMENTATION_PLAN.md` are met.
5. No TODOs, placeholders, or commented-out code left behind.

## Guardrails (do not violate)

- **Cost:** no always-on resources — no OpenSearch, no provisioned concurrency, no NAT gateways, no VPCs. DynamoDB stays on-demand. This is enforced as a test in `tests/infra/test_app_synth.py`.
- **Safety:** the query pipeline enforces the similarity floor (ADR-003) *before* any generation call.
- **Scope:** no stretch-goal features (Textract, Cognito, streaming) unless the current phase says so.
- **Secrets:** nothing sensitive in code, tests, fixtures, or git history.

## Working agreement

Work one phase of `docs/IMPLEMENTATION_PLAN.md` at a time, in order. After finishing a phase, stop and report what was built, which acceptance criteria pass (with test output), and any deviations — then wait for go-ahead. If PRD and ARCHITECTURE conflict, or a task is ambiguous, stop and ask rather than guess.
