"""S3 documents bucket, the DynamoDB single table, and the ingestion DLQ (T1.1)."""

import aws_cdk as cdk
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from constructs import Construct

#: Document metadata attributes projected into GSI1 so `GET /documents` is
#: served entirely from the index (ARCHITECTURE.md §3, "projection: meta only").
GSI1_PROJECTED_ATTRIBUTES = [
    "filename",
    "status",
    "uploaded_at",
    "chunk_count",
    "error_msg",
    "s3_key",
]


class StorageStack(cdk.Stack):
    """Durable storage shared by the ingestion and query pipelines.

    Holds the private uploads bucket, the `rag-store` single table described in
    ARCHITECTURE.md §3, and the dead-letter queue that ingestion failures are
    routed to. Everything here is teardown-friendly: buckets and the table are
    destroyed with the stack, since this is a portfolio project and not a
    system of record (CLAUDE.md, CDK conventions).
    """

    def __init__(self, scope: Construct, construct_id: str, *, env: cdk.Environment) -> None:
        """Initialize the storage stack.

        Args:
            scope: The parent construct, normally the CDK `App`.
            construct_id: Stack identifier, used as the CloudFormation stack name.
            env: Target account and region, resolved from `cdk.json` context.
        """
        super().__init__(scope, construct_id, env=env)

        self.documents_bucket = s3.Bucket(
            self,
            "DocumentsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            # Upload events are delivered through EventBridge rather than a
            # direct bucket notification: a notification would put a resource
            # referencing the ingest function into this stack, while the ingest
            # stack already references this one, which CloudFormation rejects as
            # a dependency cycle (ADR-006).
            event_bridge_enabled=True,
        )

        self.table = dynamodb.Table(
            self,
            "RagStore",
            table_name="rag-store",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        self.table.add_global_secondary_index(
            index_name="GSI1",
            partition_key=dynamodb.Attribute(name="GSI1PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="GSI1SK", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.INCLUDE,
            non_key_attributes=GSI1_PROJECTED_ATTRIBUTES,
        )

        self.ingest_dlq = sqs.Queue(
            self,
            "IngestDlq",
            retention_period=cdk.Duration.days(14),
            enforce_ssl=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        cdk.CfnOutput(self, "DocumentsBucketName", value=self.documents_bucket.bucket_name)
        cdk.CfnOutput(self, "TableName", value=self.table.table_name)
        cdk.CfnOutput(self, "IngestDlqUrl", value=self.ingest_dlq.queue_url)
