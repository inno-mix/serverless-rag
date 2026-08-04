"""CDK assertions for IngestStack (T1.4 acceptance criteria, TESTING.md §3)."""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infra import ingest_stack
from infra.ingest_stack import IngestStack
from infra.storage_stack import StorageStack
from shared import storage

# The real configured value from cdk.json. Ingestion embeds; it never generates,
# so passing a Claude id here would exercise a code path this stack never takes.
MODEL_ID = "amazon.titan-embed-text-v2:0"


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    # Asset bundling is skipped so the suite never shells out to pip or Docker;
    # this is CDK's documented switch for exactly that purpose.
    app = cdk.App(context={"aws:cdk:bundling-stacks": []})
    env = cdk.Environment(account="123456789012", region="us-east-1")
    storage_stack = StorageStack(app, "StorageStack", env=env)
    stack = IngestStack(
        app,
        "IngestStack",
        env=env,
        documents_bucket=storage_stack.documents_bucket,
        table=storage_stack.table,
        ingest_dlq=storage_stack.ingest_dlq,
        embedding_model_id=MODEL_ID,
        chunk_target_tokens=500,
        chunk_overlap_tokens=50,
    )
    return assertions.Template.from_stack(stack)


def test_function_runs_python_312_with_a_five_minute_timeout(template):
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"Runtime": "python3.12", "Timeout": 300, "MemorySize": 512, "Handler": "handler.handler"},
    )


def test_function_carries_the_required_environment(template):
    """TABLE_NAME and EMBEDDING_MODEL_ID must reach the Lambda (ADR-005)."""
    functions = template.find_resources("AWS::Lambda::Function")
    (properties,) = [f["Properties"] for f in functions.values()]
    variables = properties["Environment"]["Variables"]

    assert {"TABLE_NAME", "EMBEDDING_MODEL_ID", "DLQ_URL"} <= variables.keys()
    assert variables["EMBEDDING_MODEL_ID"] == MODEL_ID
    assert variables["CHUNK_TARGET_TOKENS"] == "500"
    assert variables["CHUNK_OVERLAP_TOKENS"] == "50"


def test_upload_events_are_routed_to_the_function(template):
    """Uploads reach the function through an EventBridge rule (ADR-006)."""
    template.has_resource_properties(
        "AWS::Events::Rule",
        {
            "EventPattern": assertions.Match.object_like(
                {
                    "source": ["aws.s3"],
                    "detail-type": ["Object Created"],
                    "detail": assertions.Match.object_like(
                        {"object": {"key": [{"prefix": "uploads/"}]}}
                    ),
                }
            ),
            "State": "ENABLED",
        },
    )


def test_rule_prefix_matches_the_upload_key_convention():
    """Drift guard: the stack duplicates the prefix rather than importing it."""
    assert ingest_stack.UPLOAD_PREFIX == storage.UPLOAD_PREFIX


def test_function_has_a_dead_letter_queue(template):
    template.has_resource_properties(
        "AWS::Lambda::Function", {"DeadLetterConfig": assertions.Match.any_value()}
    )


def test_shared_layer_targets_python_312(template):
    template.has_resource_properties(
        "AWS::Lambda::LayerVersion",
        {"CompatibleRuntimes": ["python3.12"], "CompatibleArchitectures": ["x86_64"]},
    )


def test_log_retention_is_two_weeks(template):
    template.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 14})


def test_bedrock_permission_is_scoped_to_the_configured_model(template):
    """No wildcard resource: InvokeModel is limited to the configured model."""
    policies = template.find_resources("AWS::IAM::Policy")
    statements = [
        statement
        for policy in policies.values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if statement.get("Action") == "bedrock:InvokeModel"
    ]
    assert statements, "no bedrock:InvokeModel statement found"

    for statement in statements:
        resources = statement["Resource"]
        resources = resources if isinstance(resources, list) else [resources]
        assert resources
        for resource in resources:
            rendered = str(resource)
            assert "*" not in rendered, rendered
            assert "bedrock" in rendered


def test_bedrock_permission_names_the_model_id_in_full(template):
    """Regression guard: the provider prefix is part of the model id.

    A foundation-model ARN ends in the complete id — `amazon.titan-embed-text-v2:0`,
    not `titan-embed-text-v2:0`. Dropping the provider silently produces a policy
    that grants nothing, and ingestion then fails with AccessDeniedException on
    its first embed call.
    """
    rendered = str(
        [
            statement
            for policy in template.find_resources("AWS::IAM::Policy").values()
            for statement in policy["Properties"]["PolicyDocument"]["Statement"]
            if statement.get("Action") == "bedrock:InvokeModel"
        ]
    )
    assert f"foundation-model/{MODEL_ID}" in rendered


def test_ingestion_is_not_granted_the_generation_model(template):
    """The ingest role embeds only; it has no reason to reach Claude (NFR-9)."""
    rendered = str(template.find_resources("AWS::IAM::Policy"))
    assert "anthropic" not in rendered
    assert "inference-profile" not in rendered


def test_stack_exports_wiring_outputs(template):
    outputs = template.find_outputs("*")
    assert {"IngestFunctionName", "SharedLayerArn"} <= outputs.keys()
