"""Least-privilege `bedrock:InvokeModel` policy shared by the Lambda stacks.

Both the ingest function (embedding) and the query function (embedding plus
generation) need this, and the rule it encodes is subtle enough that two copies
would eventually disagree — so it lives in one place.

A regional inference profile is not itself the invocable resource. The caller
needs the **profile ARN and the underlying foundation-model ARN in every region
the profile can route to**; granting only the deployment region yields
`AccessDeniedException` the first time Bedrock routes elsewhere. That is why
this is the one place ARCHITECTURE.md §7 allows a hand-written policy statement
instead of a `grant_*` call, and why the ARN list is longer than the two ARNs
§7 originally anticipated. There are still no wildcard resources.
"""

import aws_cdk as cdk
from aws_cdk import aws_iam as iam

#: Regions the `us.` inference profiles fan out to. Verified against the account
#: with `aws bedrock get-inference-profile`; re-check when changing a model id.
PROFILE_ROUTED_REGIONS = ("us-east-1", "us-east-2", "us-west-2")

#: Prefixes that mark a model id as an inference profile rather than a bare
#: foundation model (ADR-005 amendment).
PROFILE_PREFIXES = ("us.", "global.")


def invoke_model_policy(stack: cdk.Stack, *model_ids: str) -> iam.PolicyStatement:
    """Build an `InvokeModel` statement scoped to the given models.

    Args:
        stack: The stack the ARNs are formatted against.
        *model_ids: Bedrock model ids from `cdk.json` context (ADR-005).

    Returns:
        A policy statement granting `bedrock:InvokeModel` on exactly those
        models, including the foundation-model ARNs an inference profile
        forwards to.
    """
    resources: list[str] = []
    for model_id in model_ids:
        for arn in _model_resources(stack, model_id):
            if arn not in resources:
                resources.append(arn)

    return iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=resources)


def _model_resources(stack: cdk.Stack, model_id: str) -> list[str]:
    """Return every ARN that must be granted for one configured model id."""
    is_profile = model_id.startswith(PROFILE_PREFIXES)
    # `us.anthropic.claude-…` addresses the foundation model `anthropic.claude-…`.
    foundation_model = model_id.split(".", 1)[-1] if is_profile else model_id
    regions = PROFILE_ROUTED_REGIONS if is_profile else (stack.region,)

    resources = [
        stack.format_arn(
            service="bedrock",
            region=region,
            account="",
            resource="foundation-model",
            resource_name=foundation_model,
        )
        for region in regions
    ]
    if is_profile:
        resources.append(
            stack.format_arn(
                service="bedrock",
                resource="inference-profile",
                resource_name=model_id,
            )
        )
    return resources
