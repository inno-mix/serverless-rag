# Serverless RAG Document Q&A — Documentation Set

Planning docs for building the project with an AI coding agent (e.g. Claude Code). Drop these into the root of a new repository and point the agent at them.

## Reading order

| File | Role | Who reads it |
|---|---|---|
| `PRD.md` | What to build and why; requirements, goals, constraints | You + agent (source of truth) |
| `ARCHITECTURE.md` | System design, repo layout, data model, ADRs | Agent, before any code |
| `CLAUDE.md` | Coding rules, commands, conventions, guardrails | Agent, every session |
| `IMPLEMENTATION_PLAN.md` | Phased tasks with acceptance criteria | Agent, drives build order |
| `API_SPEC.md` | Exact HTTP contract for all endpoints | Agent, during Phase 2 |
| `TESTING.md` | Test strategy per layer, fixtures, coverage gates | Agent, alongside every task |

## How to use with an agent

1. Create an empty repo, copy these six files into it, `git commit`.
2. First instruction: *"Read PRD.md, ARCHITECTURE.md and CLAUDE.md, then execute Phase 0 of IMPLEMENTATION_PLAN.md."*
3. Proceed one phase (or one task) at a time; require the acceptance criteria to pass before moving on.
4. Before Phase 1: enable Bedrock model access in your AWS account, verify the current embedding and generation model IDs in the AWS docs, and set them in `cdk.json` context.

## Conventions baked into this set

- AWS CDK v2 in Python 3.12; single command deploy/teardown.
- No RAG frameworks — chunking, retrieval, and prompting are implemented directly (ADR-004).
- Brute-force vector search in Lambda over DynamoDB-stored vectors (ADR-001) — the key cost decision; know it well for interviews.
- Cost guardrails are enforced as tests, not just prose (see TESTING.md §3, last bullet).

## Keeping the docs honest

When implementation reveals a better decision, update the relevant doc (add or amend an ADR) in the same commit as the code change. Stale planning docs are worse than none.
