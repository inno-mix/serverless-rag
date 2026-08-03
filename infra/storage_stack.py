"""S3 documents bucket, the DynamoDB single table, and the ingestion DLQ (T1.1)."""

import aws_cdk as cdk
from constructs import Construct


class StorageStack(cdk.Stack):
    """Durable storage shared by the ingestion and query pipelines.

    The documents bucket, the `rag-store` single table with GSI1, and the
    ingestion dead-letter queue are declared in T1.1 (ARCHITECTURE.md §3).
    This stack declares no resources yet; it exists so the CDK app has a
    synthesizable root, since the CDK CLI rejects an app containing no stacks.
    """

    def __init__(self, scope: Construct, construct_id: str, *, env: cdk.Environment) -> None:
        """Initialize the storage stack.

        Args:
            scope: The parent construct, normally the CDK `App`.
            construct_id: Stack identifier, used as the CloudFormation stack name.
            env: Target account and region, resolved from `cdk.json` context.
        """
        super().__init__(scope, construct_id, env=env)
