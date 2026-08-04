"""CDK assertions for ApiStack (T2.3 acceptance criteria, TESTING.md §3)."""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infra.api_stack import (
    DAILY_QUOTA,
    STAGE_NAME,
    THROTTLE_BURST_LIMIT,
    THROTTLE_RATE_LIMIT,
    ApiStack,
)
from infra.ingest_stack import IngestStack
from infra.storage_stack import StorageStack

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
ALLOWED_ORIGIN = "https://d111111abcdef8.cloudfront.net"


@pytest.fixture(scope="module")
def stack() -> ApiStack:
    # Bundling is skipped so the suite never shells out to pip or Docker.
    app = cdk.App(context={"aws:cdk:bundling-stacks": []})
    env = cdk.Environment(account="123456789012", region="us-east-1")
    storage = StorageStack(app, "StorageStack", env=env)
    ingest = IngestStack(
        app,
        "IngestStack",
        env=env,
        documents_bucket=storage.documents_bucket,
        table=storage.table,
        ingest_dlq=storage.ingest_dlq,
        embedding_model_id=EMBEDDING_MODEL_ID,
        chunk_target_tokens=500,
        chunk_overlap_tokens=50,
    )
    return ApiStack(
        app,
        "ApiStack",
        env=env,
        documents_bucket=storage.documents_bucket,
        table=storage.table,
        shared_layer=ingest.shared_layer,
        embedding_model_id=EMBEDDING_MODEL_ID,
        generation_model_id=GENERATION_MODEL_ID,
        min_similarity=0.35,
        max_tokens=1024,
        max_upload_mb=25,
        allowed_origin=ALLOWED_ORIGIN,
    )


@pytest.fixture(scope="module")
def template(stack) -> assertions.Template:
    return assertions.Template.from_stack(stack)


def _methods(template: assertions.Template) -> dict[str, dict]:
    """Map each non-OPTIONS method to its properties."""
    return {
        name: resource["Properties"]
        for name, resource in template.find_resources("AWS::ApiGateway::Method").items()
        if resource["Properties"]["HttpMethod"] != "OPTIONS"
    }


# --------------------------------------------------------------------------- #
# Routes (API_SPEC.md §§1–4)
# --------------------------------------------------------------------------- #


def test_the_four_documented_routes_exist(template):
    """POST /documents, GET /documents, DELETE /documents/{doc_id}, POST /query."""
    assert sorted(props["HttpMethod"] for props in _methods(template).values()) == [
        "DELETE",
        "GET",
        "POST",
        "POST",
    ]


def test_routes_are_attached_to_the_documented_paths(template):
    resources = {
        resource["Properties"]["PathPart"]
        for resource in template.find_resources("AWS::ApiGateway::Resource").values()
    }
    assert {"documents", "{doc_id}", "query"} <= resources


def test_every_route_requires_an_api_key(template):
    """API_SPEC.md: all requests require the `x-api-key` header."""
    for name, props in _methods(template).items():
        assert props["ApiKeyRequired"] is True, name


def test_every_route_is_a_lambda_proxy_integration(template):
    for name, props in _methods(template).items():
        assert props["Integration"]["Type"] == "AWS_PROXY", name


def test_the_stage_is_v1(template):
    template.has_resource_properties("AWS::ApiGateway::Stage", {"StageName": STAGE_NAME})


# --------------------------------------------------------------------------- #
# Rate limiting (API_SPEC.md §5)
# --------------------------------------------------------------------------- #


def test_usage_plan_carries_the_documented_throttle_and_quota(template):
    template.has_resource_properties(
        "AWS::ApiGateway::UsagePlan",
        {
            "Throttle": {
                "RateLimit": THROTTLE_RATE_LIMIT,
                "BurstLimit": THROTTLE_BURST_LIMIT,
            },
            "Quota": {"Limit": DAILY_QUOTA, "Period": "DAY"},
        },
    )


def test_an_api_key_is_attached_to_the_usage_plan(template):
    template.resource_count_is("AWS::ApiGateway::ApiKey", 1)
    template.resource_count_is("AWS::ApiGateway::UsagePlanKey", 1)


def test_the_stage_is_bound_to_the_usage_plan(template):
    (plan,) = template.find_resources("AWS::ApiGateway::UsagePlan").values()
    assert plan["Properties"]["ApiStages"]


# --------------------------------------------------------------------------- #
# CORS (API_SPEC.md §7)
# --------------------------------------------------------------------------- #


def _preflight(template: assertions.Template) -> list[dict]:
    return [
        resource["Properties"]
        for resource in template.find_resources("AWS::ApiGateway::Method").values()
        if resource["Properties"]["HttpMethod"] == "OPTIONS"
    ]


def test_preflight_is_answered_for_every_resource(template):
    """`/documents`, `/documents/{doc_id}`, `/query`, and the root CDK adds."""
    assert len(_preflight(template)) == 4


def test_preflight_does_not_require_an_api_key(template):
    """A browser cannot attach `x-api-key` to a preflight; requiring it breaks CORS."""
    for props in _preflight(template):
        assert props["ApiKeyRequired"] is False


def test_cors_allows_only_the_configured_origin(template):
    """A deployed environment must not answer `*` (API_SPEC.md §7)."""
    rendered = str(template.find_resources("AWS::ApiGateway::Method"))
    assert ALLOWED_ORIGIN in rendered


def test_the_configured_origin_reaches_every_function(template):
    for name, props in template.find_resources("AWS::Lambda::Function").items():
        variables = props["Properties"]["Environment"]["Variables"]
        assert variables["ALLOWED_ORIGIN"] == ALLOWED_ORIGIN, name


# --------------------------------------------------------------------------- #
# Functions and configuration
# --------------------------------------------------------------------------- #


def test_four_functions_are_deployed(template):
    """One role per route is what keeps the permissions separable (NFR-9)."""
    template.resource_count_is("AWS::Lambda::Function", 4)


def test_every_function_runs_python_312_on_the_shared_layer(template):
    for name, props in template.find_resources("AWS::Lambda::Function").items():
        assert props["Properties"]["Runtime"] == "python3.12", name
        assert props["Properties"]["Layers"], name


def test_the_query_function_gets_more_memory(template):
    memories = {
        props["Properties"]["Handler"]: props["Properties"]["MemorySize"]
        for props in template.find_resources("AWS::Lambda::Function").values()
    }
    assert memories["handler.handler"] == 1024
    assert memories["handler.list_handler"] == 512


def test_the_query_function_carries_its_configuration(template):
    variables = _environment(template, "handler.handler")
    assert variables["EMBEDDING_MODEL_ID"] == EMBEDDING_MODEL_ID
    assert variables["GENERATION_MODEL_ID"] == GENERATION_MODEL_ID
    assert variables["MIN_SIMILARITY"] == "0.35"
    assert variables["MAX_TOKENS"] == "1024"


def test_the_presign_function_receives_the_upload_ceiling_in_bytes(template):
    assert _environment(template, "handler.presign_handler")["MAX_UPLOAD_BYTES"] == str(
        25 * 1024 * 1024
    )


def test_log_retention_is_two_weeks(template):
    for name, props in template.find_resources("AWS::Logs::LogGroup").items():
        assert props["Properties"]["RetentionInDays"] == 14, name


def test_stack_exports_the_api_url_and_key_id(template):
    assert {"ApiUrl", "ApiKeyId"} <= template.find_outputs("*").keys()


def _environment(template: assertions.Template, handler: str) -> dict:
    """Environment variables of the function with the given handler."""
    for props in template.find_resources("AWS::Lambda::Function").values():
        if props["Properties"]["Handler"] == handler:
            return props["Properties"]["Environment"]["Variables"]
    raise AssertionError(f"no function with handler {handler}")


# --------------------------------------------------------------------------- #
# Least privilege (NFR-9, ARCHITECTURE.md §7)
# --------------------------------------------------------------------------- #


def _actions_for(template: assertions.Template, role_substring: str) -> set[str]:
    """Collect every IAM action granted to the role whose id contains the term."""
    actions: set[str] = set()
    for name, policy in template.find_resources("AWS::IAM::Policy").items():
        if role_substring not in name:
            continue
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            granted = statement["Action"]
            actions.update(granted if isinstance(granted, list) else [granted])
    return actions


def test_only_the_query_function_can_call_bedrock(template):
    for role in ("PresignFunction", "ListFunction", "DeleteFunction"):
        assert not any(action.startswith("bedrock:") for action in _actions_for(template, role))
    assert "bedrock:InvokeModel" in _actions_for(template, "QueryFunction")


def test_the_query_function_cannot_touch_the_bucket(template):
    assert not any(action.startswith("s3:") for action in _actions_for(template, "QueryFunction"))


def test_the_list_function_is_read_only(template):
    actions = _actions_for(template, "ListFunction")
    mutating = ("Put", "Delete", "Update", "Write")
    assert not any(verb in action for action in actions for verb in mutating)


def test_only_the_delete_function_can_delete_objects(template):
    assert "s3:DeleteObject*" in _actions_for(template, "DeleteFunction")
    assert "s3:DeleteObject*" not in _actions_for(template, "PresignFunction")


def test_bedrock_access_is_scoped_to_the_configured_models(template):
    """No wildcard resource, and every routed region is covered (§7 amendment)."""
    statements = [
        statement
        for policy in template.find_resources("AWS::IAM::Policy").values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if statement.get("Action") == "bedrock:InvokeModel"
    ]
    assert statements

    rendered = str(statements)
    assert "*" not in rendered
    assert f"foundation-model/{EMBEDDING_MODEL_ID}" in rendered
    for region in ("us-east-1", "us-east-2", "us-west-2"):
        assert f"bedrock:{region}::foundation-model/anthropic.claude-haiku" in rendered
    assert f"inference-profile/{GENERATION_MODEL_ID}" in rendered
