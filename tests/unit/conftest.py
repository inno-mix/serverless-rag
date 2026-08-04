"""Shared fixtures for the unit suite.

The moto-backed table lives here because three modules need it — storage,
the query handler, and the documents handler — and a table whose schema drifts
from `StorageStack` would make every one of those suites lie.
"""

import boto3
import pytest
from moto import mock_aws


@pytest.fixture(autouse=True)
def _aws_credentials(monkeypatch):
    """Moto refuses to run against ambient credentials; give it dummies.

    Autouse, so no test can reach real AWS by forgetting to ask (TESTING.md §1).
    """
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def aws():
    """An active moto context, for tests wiring more than one service."""
    with mock_aws():
        yield


@pytest.fixture
def table(aws):
    """A moto-backed table mirroring StorageStack's schema."""
    return _create_table()


def _create_table():
    """Create `rag-store` with the PK/SK + GSI1 design from ARCHITECTURE.md §3."""
    resource = boto3.resource("dynamodb", region_name="us-east-1")
    return resource.create_table(
        TableName="rag-store",
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
