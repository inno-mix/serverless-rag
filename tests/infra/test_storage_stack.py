"""CDK assertions for StorageStack (T1.1 acceptance criteria)."""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infra.storage_stack import StorageStack


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    app = cdk.App()
    stack = StorageStack(
        app, "StorageStack", env=cdk.Environment(account="123456789012", region="us-east-1")
    )
    return assertions.Template.from_stack(stack)


def test_table_uses_on_demand_billing(template):
    """PAY_PER_REQUEST keeps idle cost at zero (PRD NFR-3)."""
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {"TableName": "rag-store", "BillingMode": "PAY_PER_REQUEST"},
    )


def test_table_has_pk_sk_key_schema(template):
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ]
        },
    )


def test_table_has_gsi1(template):
    """GSI1 serves the newest-first document listing (ARCHITECTURE.md §3)."""
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "GlobalSecondaryIndexes": assertions.Match.array_with(
                [
                    assertions.Match.object_like(
                        {
                            "IndexName": "GSI1",
                            "KeySchema": [
                                {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                                {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                            ],
                            "Projection": {"ProjectionType": "INCLUDE"},
                        }
                    )
                ]
            )
        },
    )


def test_bucket_blocks_all_public_access(template):
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            }
        },
    )


def test_bucket_is_not_versioned(template):
    """Versioning is off per T1.1; it would retain deleted uploads and cost money."""
    bucket = next(iter(template.find_resources("AWS::S3::Bucket").values()))
    assert "VersioningConfiguration" not in bucket["Properties"]


def test_bucket_and_table_are_destroyed_with_the_stack(template):
    """Teardown-friendly project, not a system of record (CLAUDE.md)."""
    for resource_type in ("AWS::S3::Bucket", "AWS::DynamoDB::Table"):
        for resource in template.find_resources(resource_type).values():
            assert resource["DeletionPolicy"] == "Delete"
            assert resource["UpdateReplacePolicy"] == "Delete"


def test_ingest_dlq_exists(template):
    template.resource_count_is("AWS::SQS::Queue", 1)
    template.has_resource_properties("AWS::SQS::Queue", {"MessageRetentionPeriod": 1209600})


def test_stack_exports_wiring_outputs(template):
    """Integration tests read these outputs (CLAUDE.md, CDK conventions)."""
    outputs = template.find_outputs("*")
    assert {"DocumentsBucketName", "TableName", "IngestDlqUrl"} <= outputs.keys()
