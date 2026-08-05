"""REST API, presign/query/list/delete Lambdas, API key and usage plan (T2.3).

Four functions rather than one router, so each role carries only what its route
needs (NFR-9): the list function cannot delete, the query function cannot touch
the bucket at all, and only the query function can reach Bedrock.

The shared layer is built by `IngestStack` and passed in, so all five Lambdas in
the app run byte-identical shared code and the layer asset is built once.
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

from infra.bedrock_policy import invoke_model_policy
from infra.ingest_stack import ASSET_EXCLUDES

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: Usage plan limits from API_SPEC.md §5.
THROTTLE_RATE_LIMIT = 5
THROTTLE_BURST_LIMIT = 10
DAILY_QUOTA = 1000

#: Deployment stage, which API_SPEC.md names as part of the base URL.
STAGE_NAME = "v1"

#: Per-function sizing. CLAUDE.md sets 512 MB / 30 s as the default and gives
#: the query function 1024 MB, since it holds the whole corpus of vectors in
#: memory while scoring.
DEFAULT_MEMORY_MB = 512
QUERY_MEMORY_MB = 1024
DEFAULT_TIMEOUT_SECONDS = 30


class ApiStack(cdk.Stack):
    """The public REST API and the four Lambdas behind it."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env: cdk.Environment,
        documents_bucket: s3.IBucket,
        table: dynamodb.ITable,
        shared_layer: lambda_.ILayerVersion,
        embedding_model_id: str,
        generation_model_id: str,
        min_similarity: float,
        max_tokens: int,
        max_upload_mb: int,
        allowed_origin: str,
    ) -> None:
        """Initialize the API stack.

        Args:
            scope: The parent construct.
            construct_id: Stack identifier.
            env: Target account and region.
            documents_bucket: Bucket uploads are presigned against.
            table: The `rag-store` table.
            shared_layer: The layer built by IngestStack.
            embedding_model_id: Bedrock embedding model id (ADR-005).
            generation_model_id: Bedrock generation model id (ADR-005).
            min_similarity: The ADR-003 floor, passed to the query function.
            max_tokens: Generation output cap (NFR-4).
            max_upload_mb: Upload ceiling signed into the presigned policy.
            allowed_origin: CORS origin permitted by the API (API_SPEC.md §7).
        """
        super().__init__(scope, construct_id, env=env)

        common_env = {"TABLE_NAME": table.table_name, "ALLOWED_ORIGIN": allowed_origin}

        self.presign_function = self._function(
            "PresignFunction",
            source="documents",
            handler="handler.presign_handler",
            layer=shared_layer,
            environment={
                **common_env,
                "BUCKET_NAME": documents_bucket.bucket_name,
                "MAX_UPLOAD_BYTES": str(max_upload_mb * 1024 * 1024),
            },
        )
        self.list_function = self._function(
            "ListFunction",
            source="documents",
            handler="handler.list_handler",
            layer=shared_layer,
            environment=common_env,
        )
        self.delete_function = self._function(
            "DeleteFunction",
            source="documents",
            handler="handler.delete_handler",
            layer=shared_layer,
            environment={**common_env, "BUCKET_NAME": documents_bucket.bucket_name},
        )
        self.query_function = self._function(
            "QueryFunction",
            source="query",
            handler="handler.handler",
            layer=shared_layer,
            memory_size=QUERY_MEMORY_MB,
            environment={
                **common_env,
                "EMBEDDING_MODEL_ID": embedding_model_id,
                "GENERATION_MODEL_ID": generation_model_id,
                "MIN_SIMILARITY": str(min_similarity),
                "MAX_TOKENS": str(max_tokens),
            },
        )

        # Least privilege, one role at a time: presign writes metadata and signs
        # uploads, list only reads, delete is alone in being able to remove an
        # object, and only the query role can call Bedrock.
        table.grant_read_write_data(self.presign_function)
        documents_bucket.grant_put(self.presign_function)
        table.grant_read_data(self.list_function)
        table.grant_read_write_data(self.delete_function)
        documents_bucket.grant_delete(self.delete_function)
        table.grant_read_data(self.query_function)
        self.query_function.add_to_role_policy(
            invoke_model_policy(self, embedding_model_id, generation_model_id)
        )

        self.api = apigateway.RestApi(
            self,
            "RagApi",
            rest_api_name="serverless-rag",
            description="Document Q&A API (API_SPEC.md)",
            deploy_options=apigateway.StageOptions(
                stage_name=STAGE_NAME,
                throttling_rate_limit=THROTTLE_RATE_LIMIT,
                throttling_burst_limit=THROTTLE_BURST_LIMIT,
            ),
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=[allowed_origin],
                allow_headers=["Content-Type", "x-api-key"],
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            ),
            # Every route requires the key; the usage plan below is what makes
            # that key mean something (API_SPEC.md §5).
            default_method_options=apigateway.MethodOptions(api_key_required=True),
        )

        documents = self.api.root.add_resource("documents")
        documents.add_method("POST", apigateway.LambdaIntegration(self.presign_function))
        documents.add_method("GET", apigateway.LambdaIntegration(self.list_function))
        documents.add_resource("{doc_id}").add_method(
            "DELETE", apigateway.LambdaIntegration(self.delete_function)
        )
        self.api.root.add_resource("query").add_method(
            "POST", apigateway.LambdaIntegration(self.query_function)
        )

        # The key value is generated at deploy time and kept in Secrets Manager,
        # so it exists in neither the repository nor the browser. FrontendStack
        # reads the same reference to inject it as a CloudFront origin header,
        # which is what keeps the deployed demo self-serve without publishing a
        # credential (ADR-008).
        self.api_key_secret = secretsmanager.Secret(
            self,
            "ApiKeySecret",
            description="x-api-key value for the serverless-rag API",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            generate_secret_string=secretsmanager.SecretStringGenerator(
                # API Gateway keys must be at least 20 characters and accept no
                # whitespace; punctuation is excluded to keep the value safe in
                # a header without quoting.
                password_length=40,
                exclude_punctuation=True,
                exclude_characters=" ",
            ),
        )
        self.api_key_value = self.api_key_secret.secret_value.unsafe_unwrap()

        self.api_key = apigateway.ApiKey(
            self,
            "RagApiKey",
            description="Demo client key",
            value=self.api_key_value,
        )
        self.usage_plan = self.api.add_usage_plan(
            "RagUsagePlan",
            throttle=apigateway.ThrottleSettings(
                rate_limit=THROTTLE_RATE_LIMIT,
                burst_limit=THROTTLE_BURST_LIMIT,
            ),
            quota=apigateway.QuotaSettings(limit=DAILY_QUOTA, period=apigateway.Period.DAY),
        )
        self.usage_plan.add_api_stage(stage=self.api.deployment_stage)
        self.usage_plan.add_api_key(self.api_key)

        cdk.CfnOutput(self, "ApiUrl", value=self.api.url)
        cdk.CfnOutput(self, "ApiKeyId", value=self.api_key.key_id)
        cdk.CfnOutput(self, "ApiKeySecretArn", value=self.api_key_secret.secret_arn)

    def _function(
        self,
        construct_id: str,
        *,
        source: str,
        handler: str,
        layer: lambda_.ILayerVersion,
        environment: dict[str, str],
        memory_size: int = DEFAULT_MEMORY_MB,
    ) -> lambda_.Function:
        """Create one API Lambda with the project's standard configuration."""
        return lambda_.Function(
            self,
            construct_id,
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.X86_64,
            handler=handler,
            code=lambda_.Code.from_asset(str(_SRC_ROOT / source), exclude=ASSET_EXCLUDES),
            layers=[layer],
            memory_size=memory_size,
            timeout=cdk.Duration.seconds(DEFAULT_TIMEOUT_SECONDS),
            log_group=logs.LogGroup(
                self,
                f"{construct_id}Logs",
                retention=logs.RetentionDays.TWO_WEEKS,
                removal_policy=cdk.RemovalPolicy.DESTROY,
            ),
            environment=environment,
        )
