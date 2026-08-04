"""Tests for the shared Bedrock IAM policy helper (ARCHITECTURE.md §7).

The rule this helper encodes is easy to get subtly wrong in a way that only
shows up as a runtime `AccessDeniedException`, so it is tested directly rather
than only through the stacks that use it.
"""

import aws_cdk as cdk
import pytest

from infra.bedrock_policy import PROFILE_ROUTED_REGIONS, invoke_model_policy

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
PROFILE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
FOUNDATION_BEHIND_PROFILE = "anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture
def stack() -> cdk.Stack:
    app = cdk.App()
    return cdk.Stack(
        app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1")
    )


def _resources(stack: cdk.Stack, *model_ids: str) -> list[str]:
    """Rendered ARNs, always as a list — CDK collapses a single entry to a string."""
    rendered = invoke_model_policy(stack, *model_ids).to_statement_json()["Resource"]
    return rendered if isinstance(rendered, list) else [rendered]


def test_only_invoke_model_is_granted(stack):
    action = invoke_model_policy(stack, EMBEDDING_MODEL_ID).to_statement_json()["Action"]
    assert action == "bedrock:InvokeModel"


def test_a_bare_model_keeps_its_provider_prefix(stack):
    """`amazon.titan-…` is the model id; stripping `amazon.` grants nothing."""
    (resource,) = _resources(stack, EMBEDDING_MODEL_ID)
    assert resource.endswith(f"foundation-model/{EMBEDDING_MODEL_ID}")


def test_a_bare_model_is_granted_in_the_deployment_region_only(stack):
    """An on-demand model does not fan out, so neither should its permission."""
    resources = _resources(stack, EMBEDDING_MODEL_ID)
    assert len(resources) == 1
    assert ":bedrock:us-east-1::" in resources[0]


def test_a_profile_grants_the_profile_arn(stack):
    resources = _resources(stack, PROFILE_MODEL_ID)
    assert any(f"inference-profile/{PROFILE_MODEL_ID}" in resource for resource in resources)


def test_a_profile_grants_the_underlying_model_in_every_routed_region(stack):
    """Granting only the deployment region 403s the first time Bedrock reroutes."""
    resources = _resources(stack, PROFILE_MODEL_ID)
    for region in PROFILE_ROUTED_REGIONS:
        assert any(
            f":bedrock:{region}::foundation-model/{FOUNDATION_BEHIND_PROFILE}" in resource
            for resource in resources
        ), region


def test_a_profile_strips_only_the_routing_prefix(stack):
    """`us.anthropic.claude-…` addresses the model `anthropic.claude-…`."""
    resources = _resources(stack, PROFILE_MODEL_ID)
    assert not any(f"foundation-model/{PROFILE_MODEL_ID}" in resource for resource in resources)
    assert any(FOUNDATION_BEHIND_PROFILE in resource for resource in resources)


def test_several_models_are_combined_without_duplicates(stack):
    resources = _resources(stack, EMBEDDING_MODEL_ID, PROFILE_MODEL_ID, EMBEDDING_MODEL_ID)
    assert len(resources) == len(set(resources))
    assert len(resources) == 1 + len(PROFILE_ROUTED_REGIONS) + 1


def test_no_wildcard_resource_is_ever_produced(stack):
    """NFR-9: no `*` resources except where AWS requires them, and this is not one."""
    for resource in _resources(stack, EMBEDDING_MODEL_ID, PROFILE_MODEL_ID):
        assert "*" not in resource
