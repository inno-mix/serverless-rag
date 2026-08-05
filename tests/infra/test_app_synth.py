"""Whole-app synthesis and cost-guardrail tests (TESTING.md §3)."""

import json
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk import cx_api

from app import build_app, resolve_env

REPO_ROOT = Path(__file__).resolve().parents[2]
CDK_CONTEXT: dict[str, object] = json.loads((REPO_ROOT / "cdk.json").read_text())["context"]

# Resource types that would break the "idle cost ~= $0" guarantee in PRD NFR-3.
# Asserted across every stack in the app, so the guardrail tightens automatically
# as stacks are added in later phases.
FORBIDDEN_RESOURCE_TYPES = (
    "AWS::EC2::NatGateway",
    "AWS::OpenSearchServerless::Collection",
    "AWS::OpenSearchService::Domain",
)


def _synth() -> cx_api.CloudAssembly:
    """Synthesize the real application using the committed cdk.json context.

    Asset bundling is switched off so the suite never shells out to pip or
    Docker; templates are identical either way.
    """
    return build_app(CDK_CONTEXT | {"aws:cdk:bundling-stacks": []}).synth()


def test_cdk_json_declares_required_context():
    """cdk.json must carry every key the stacks and Lambdas read (ADR-005)."""
    required = {
        "region",
        "embeddingModelId",
        "generationModelId",
        "minSimilarity",
        "maxTokens",
        "maxUploadMb",
        "chunkTargetTokens",
        "chunkOverlapTokens",
        "allowedOrigin",
        "alertEmail",
    }
    assert required <= CDK_CONTEXT.keys()


def test_max_tokens_stays_within_the_cost_guardrail():
    """NFR-4 caps generation output; the context value is what enforces it."""
    assert 0 < int(CDK_CONTEXT["maxTokens"]) <= 1024


def test_every_stack_in_the_plan_is_wired_up():
    """One stack per concern, as laid out in ARCHITECTURE.md §2."""
    assert {stack.stack_name for stack in _synth().stacks} == {
        "StorageStack",
        "IngestStack",
        "ApiStack",
        "FrontendStack",
        "ObservabilityStack",
    }


def test_no_personal_data_is_committed_in_context():
    """CLAUDE.md: nothing sensitive in code or git history.

    `alertEmail` is deliberately empty in the committed `cdk.json`; an operator
    sets their own address locally.
    """
    assert CDK_CONTEXT["alertEmail"] == ""


def test_generation_model_is_an_inference_profile():
    """Bare `anthropic.*` ids are not invocable on-demand in us-east-1.

    Current Claude models must be addressed through a regional or global
    inference profile, otherwise InvokeModel fails at runtime with
    "on-demand throughput isn't supported".
    """
    model_id = CDK_CONTEXT["generationModelId"]
    assert isinstance(model_id, str)
    assert model_id.startswith(("us.anthropic.", "global.anthropic.")), model_id


def test_app_synthesizes():
    assembly = _synth()
    assert assembly.stacks, "app must declare at least one stack"


def test_stacks_target_the_context_region():
    """Region comes from cdk.json, never from the operator's AWS CLI default."""
    for stack in _synth().stacks:
        assert stack.environment.region == "us-east-1"


def test_resolve_env_uses_context_region():
    app = cdk.App(context={"region": "us-east-1"})
    assert resolve_env(app).region == "us-east-1"


def test_resolve_env_rejects_missing_region():
    with pytest.raises(ValueError, match="region"):
        resolve_env(cdk.App(context={}))


def test_no_always_on_resources():
    """No always-on resource may appear in any synthesized template."""
    assembly = _synth()
    for stack in assembly.stacks:
        types = {r.get("Type") for r in stack.template.get("Resources", {}).values()}
        offenders = types & set(FORBIDDEN_RESOURCE_TYPES)
        assert not offenders, f"{stack.stack_name} contains always-on resources: {offenders}"


def test_no_provisioned_concurrency():
    """Lambda provisioned concurrency is an always-on cost (CLAUDE.md guardrails)."""
    assembly = _synth()
    for stack in assembly.stacks:
        for name, resource in stack.template.get("Resources", {}).items():
            if resource.get("Type") == "AWS::Lambda::Alias":
                props = resource.get("Properties", {})
                assert "ProvisionedConcurrencyConfig" not in props, (
                    f"{stack.stack_name}.{name} sets provisioned concurrency"
                )
