# Security

This is a portfolio demonstration project, not production software. It is
deliberately built to be deployed, shown, and torn down. That shapes the threat
model below, and some of the trade-offs would be wrong for a real system.

## Reporting

Open a GitHub issue. There is no private disclosure process and no service
behind this repository — if a deployment of it is affected, the owner of that
AWS account is the only person who can act.

## What protects what

| Control | Enforced by |
|---|---|
| API requires `x-api-key` on every route | API Gateway, with a usage plan |
| The key never reaches the browser | CloudFront injects it as an origin header (ADR-008) |
| Rate limit 5 rps / burst 10 / 1000 per day | API Gateway usage plan |
| Uploads bounded to 25 MB and a fixed content type | Signed S3 POST policy — S3 rejects, the bytes never land |
| Object key cannot be steered out of `uploads/{doc_id}/` | Filename validation rejects separators, leading dots, control characters |
| Lambdas hold only the permissions their route needs | Per-function IAM roles, asserted in `tests/infra/test_api_stack.py` |
| Bedrock access scoped to two model IDs, no wildcards | `infra/bedrock_policy.py`, asserted in tests |
| Buckets are private | Block Public Access; CloudFront reads via Origin Access Control |
| Error responses never carry internal detail | `shared/http.py`, asserted in handler tests |
| Answers cannot be fabricated from weak retrieval | Similarity floor before generation (ADR-003), measured in `tests/eval` |

## Known limitations

These are accepted for a demo. Each would need addressing before this handled
anything real.

### A deployed instance is unauthenticated

The whole point of ADR-008 is that a visitor does not need a key — CloudFront
supplies it. So **anyone who has the CloudFront URL can upload documents and run
queries at the account owner's expense.** The usage plan caps this at 1000
requests a day, and the $5 budget alarm is the backstop, but a public URL is a
public spending endpoint.

If you deploy this somewhere discoverable, either keep the URL unpublished, put
Cognito in front of it (`T4.2` in the implementation plan), or expect to tear it
down after the demo.

### Uploaded documents are shared, not per-user

There is no tenancy. Every document is visible to every visitor, and any visitor
can delete any document. `NG1` in the PRD makes this explicit.

### Prompt injection is not mitigated

Retrieved chunks are user-supplied text placed into the model's context. A
document containing instructions ("ignore the above and…") may influence
answers. The system prompt asks the model to answer only from context, which is
guidance, not a boundary.

The blast radius is small — the model has no tools, no ability to act, and its
output is rendered as text, never as HTML — so the realistic outcome is a
misleading answer rather than code execution or data exfiltration. Do not point
this at documents you do not trust and then believe the answers.

### PDF parsing is attacker-reachable

`pypdf` parses files uploaded by anyone who can reach the demo. Parser
vulnerabilities are therefore directly reachable rather than theoretical: the
two open when this was last reviewed were both denial of service during text
extraction. CI runs `pip-audit` on every pull request and Dependabot watches for
new advisories, but keep the dependency current.

Impact is bounded by the ingest Lambda's 512 MB memory and 300 second timeout —
a malicious PDF costs compute and fails the document, rather than affecting
other requests.

### CORS is `*` on the API

Not a weakening. CORS governs which page may *attempt* a cross-origin request,
never whether the API accepts it, and every route still requires a key that a
cross-origin caller has no way to obtain. The deployed UI is same-origin behind
CloudFront and does not rely on this at all.

### The API key is visible inside the AWS account

CloudFront resolves it into the distribution config, so anyone with
`cloudfront:GetDistributionConfig` in the account can read it. That is an
account-internal audience, not the public one that shipping the key in the page
would create.

## If you deploy this

- **Set `alertEmail`** so the $5 budget and the CloudWatch alarms reach someone.
  Pass it at deploy time rather than committing it:
  `./scripts/deploy.sh --context alertEmail=you@example.com`
- **Tear it down when the demo is over.** `make destroy` removes everything and
  verifies by tag that nothing carrying `Project=serverless-rag` survived.
- **Do not reuse the AWS account** for anything you care about. Everything here
  is `RemovalPolicy.DESTROY`.
