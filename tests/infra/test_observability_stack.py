"""CDK assertions for ObservabilityStack (T3.2 acceptance criteria, TESTING.md §3)."""

import json

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infra.api_stack import ApiStack
from infra.ingest_stack import IngestStack
from infra.observability_stack import (
    ERROR_RATE_THRESHOLD_PERCENT,
    METRIC_NAMESPACE,
    MONTHLY_BUDGET_USD,
    ObservabilityStack,
)
from infra.storage_stack import StorageStack

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
ALERT_EMAIL = "alerts@example.com"


def _build(alert_email: str) -> assertions.Template:
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
    api = ApiStack(
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
        allowed_origin="*",
    )
    stack = ObservabilityStack(
        app,
        "ObservabilityStack",
        env=env,
        functions={
            "ingest": ingest.function,
            "presign": api.presign_function,
            "list": api.list_function,
            "delete": api.delete_function,
            "query": api.query_function,
        },
        api=api.api,
        ingest_dlq=storage.ingest_dlq,
        alert_email=alert_email,
    )
    return assertions.Template.from_stack(stack)


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    return _build(ALERT_EMAIL)


def _dashboard_body(template: assertions.Template) -> str:
    (dashboard,) = template.find_resources("AWS::CloudWatch::Dashboard").values()
    return json.dumps(dashboard["Properties"]["DashboardBody"])


# --------------------------------------------------------------------------- #
# Dashboard (ARCHITECTURE.md §8)
# --------------------------------------------------------------------------- #


def test_a_dashboard_exists(template):
    template.has_resource_properties(
        "AWS::CloudWatch::Dashboard", {"DashboardName": "serverless-rag"}
    )


@pytest.mark.parametrize(
    "title",
    [
        "Lambda invocations",
        "Lambda errors",
        "Lambda duration (p95)",
        "API Gateway errors",
        "Bedrock tokens (EMF)",
        "Ingest dead-letter queue",
    ],
)
def test_dashboard_covers_every_documented_widget(template, title):
    assert title in _dashboard_body(template)


def test_dashboard_reads_the_custom_token_metrics(template):
    """The token counters come from EMF, not from a PutMetricData call."""
    body = _dashboard_body(template)
    assert METRIC_NAMESPACE in body
    assert "TokensIn" in body
    assert "TokensOut" in body


# --------------------------------------------------------------------------- #
# Alarms
# --------------------------------------------------------------------------- #


def test_exactly_two_alarms_are_created(template):
    """ARCHITECTURE.md §8: DLQ depth, and Lambda error rate."""
    template.resource_count_is("AWS::CloudWatch::Alarm", 2)


def test_the_dlq_alarm_fires_on_a_single_message(template):
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "MetricName": "ApproximateNumberOfMessagesVisible",
            "Threshold": 0,
            "ComparisonOperator": "GreaterThanThreshold",
        },
    )


def test_the_error_rate_alarm_uses_the_documented_threshold(template):
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "Threshold": ERROR_RATE_THRESHOLD_PERCENT,
            "ComparisonOperator": "GreaterThanThreshold",
        },
    )


def test_the_error_rate_alarm_is_a_ratio_not_a_count(template):
    """A busy period and a quiet one should not share an absolute threshold."""
    alarms = json.dumps(template.find_resources("AWS::CloudWatch::Alarm"))
    assert "100 * errors / invocations" in alarms


def test_an_idle_period_does_not_breach(template):
    """With no invocations the ratio is 0/0; the IF guard keeps it at zero."""
    alarms = json.dumps(template.find_resources("AWS::CloudWatch::Alarm"))
    assert "IF(invocations > 0" in alarms
    for alarm in template.find_resources("AWS::CloudWatch::Alarm").values():
        assert alarm["Properties"]["TreatMissingData"] == "notBreaching"


def test_every_alarm_notifies_the_topic(template):
    for alarm in template.find_resources("AWS::CloudWatch::Alarm").values():
        assert alarm["Properties"]["AlarmActions"]


# --------------------------------------------------------------------------- #
# Budget (PRD NFR-5)
# --------------------------------------------------------------------------- #


def test_a_five_dollar_monthly_budget_exists(template):
    template.has_resource_properties(
        "AWS::Budgets::Budget",
        {
            "Budget": {
                "BudgetType": "COST",
                "TimeUnit": "MONTHLY",
                "BudgetLimit": {"Amount": MONTHLY_BUDGET_USD, "Unit": "USD"},
            }
        },
    )


def test_the_budget_notifies_the_configured_address(template):
    budget = json.dumps(template.find_resources("AWS::Budgets::Budget"))
    assert ALERT_EMAIL in budget


# --------------------------------------------------------------------------- #
# Optional alerting
# --------------------------------------------------------------------------- #


def test_an_email_subscription_is_created_when_configured(template):
    template.has_resource_properties(
        "AWS::SNS::Subscription", {"Protocol": "email", "Endpoint": ALERT_EMAIL}
    )


def test_guardrails_still_exist_without_an_email():
    """A clone that has not set alertEmail still gets the alarms and the budget."""
    template = _build("")

    template.resource_count_is("AWS::CloudWatch::Alarm", 2)
    template.resource_count_is("AWS::SNS::Subscription", 0)
    template.resource_count_is("AWS::Budgets::Budget", 1)
    template.resource_count_is("AWS::CloudWatch::Dashboard", 1)


def test_no_personal_address_is_baked_into_the_defaults():
    """CLAUDE.md: nothing sensitive in code or git history."""
    template = _build("")
    assert "@" not in json.dumps(template.find_resources("AWS::Budgets::Budget"))


def test_stack_exports_the_dashboard_and_topic(template):
    assert {"DashboardName", "AlarmTopicArn"} <= template.find_outputs("*").keys()


def test_the_namespace_matches_the_one_the_lambda_writes():
    """A drift here would leave the token widgets permanently empty."""
    from shared.logging_utils import METRIC_NAMESPACE as EMITTED_NAMESPACE

    assert METRIC_NAMESPACE == EMITTED_NAMESPACE
