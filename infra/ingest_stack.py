"""Ingestion Lambda, its S3 event source, and the shared Lambda layer (T1.4)."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import jsii
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from constructs import Construct

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARED_SRC = _REPO_ROOT / "src" / "shared"

#: Key prefix the rule filters on. Must equal `shared.storage.UPLOAD_PREFIX`;
#: duplicated rather than imported so the CDK app does not need the Lambda
#: sources on its import path. `test_ingest_stack` asserts the two agree.
UPLOAD_PREFIX = "uploads/"

#: Runtime dependencies shipped in the shared layer. `boto3` is provided by the
#: Lambda runtime and is deliberately absent (CLAUDE.md dependency rule).
LAYER_DEPENDENCIES = ("numpy>=2.1.0,<3", "pypdf>=5.1.0,<6")

#: Lambdas run on x86_64, so wheels must be resolved for that platform even when
#: the layer is built on an arm64 developer machine.
PIP_PLATFORM = "manylinux2014_x86_64"
PYTHON_VERSION = "3.12"


@jsii.implements(cdk.ILocalBundling)
class _LocalPipBundling:
    """Build the layer with the local pip instead of Docker.

    CDK tries this first and falls back to the Docker image if it returns
    `False`, so `cdk deploy` works whether or not Docker is running. Platform
    wheels are requested explicitly, which is what makes a Docker-free build
    produce Linux-compatible binaries for numpy.
    """

    def try_bundle(self, output_dir: str, _options: Any = None, **_kwargs: Any) -> bool:
        """Populate `output_dir` with the layer contents.

        Args:
            output_dir: Directory CDK will turn into the layer asset.
            _options: Bundling options, passed positionally by jsii; unused.
            **_kwargs: Further bundling options; unused.

        Returns:
            `True` if the layer was built locally, `False` to fall back to Docker.
        """
        target = Path(output_dir) / "python"
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            *LAYER_DEPENDENCIES,
            "--platform",
            PIP_PLATFORM,
            "--python-version",
            PYTHON_VERSION,
            "--only-binary=:all:",
            "--target",
            str(target),
            "--quiet",
            "--disable-pip-version-check",
        ]
        try:
            subprocess.run(command, check=True)  # noqa: S603 - fixed argv, no shell
        except (subprocess.CalledProcessError, OSError):
            return False

        shutil.copytree(_SHARED_SRC, target / "shared", dirs_exist_ok=True)
        return True


def _layer_asset_hash() -> str:
    """Hash the shared sources together with the pinned dependency set.

    Including the dependency specifiers means bumping a pin invalidates the
    cached asset, which hashing the source tree alone would miss.
    """
    digest = hashlib.sha256()
    digest.update("|".join(LAYER_DEPENDENCIES).encode())
    for path in sorted(_SHARED_SRC.rglob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class IngestStack(cdk.Stack):
    """Event-driven ingestion: S3 upload in, embedded chunks out.

    Wires the documents bucket's ObjectCreated notifications to a Lambda that
    extracts, chunks, embeds, and stores a document. Failures are recorded on
    the document and posted to the dead-letter queue (ARCHITECTURE.md §6).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env: cdk.Environment,
        documents_bucket: s3.IBucket,
        table: dynamodb.ITable,
        ingest_dlq: sqs.IQueue,
        embedding_model_id: str,
        chunk_target_tokens: int,
        chunk_overlap_tokens: int,
    ) -> None:
        """Initialize the ingestion stack.

        Args:
            scope: The parent construct.
            construct_id: Stack identifier.
            env: Target account and region.
            documents_bucket: Bucket receiving uploads, from StorageStack.
            table: The `rag-store` table, from StorageStack.
            ingest_dlq: Dead-letter queue for ingestion failures.
            embedding_model_id: Bedrock embedding model id, from context (ADR-005).
            chunk_target_tokens: Target chunk size passed to the chunker.
            chunk_overlap_tokens: Chunk overlap passed to the chunker.
        """
        super().__init__(scope, construct_id, env=env)

        self.shared_layer = lambda_.LayerVersion(
            self,
            "SharedLayer",
            description="Shared RAG logic plus numpy and pypdf",
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            compatible_architectures=[lambda_.Architecture.X86_64],
            removal_policy=cdk.RemovalPolicy.DESTROY,
            code=lambda_.Code.from_asset(
                str(_SHARED_SRC),
                asset_hash=_layer_asset_hash(),
                bundling=cdk.BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    local=_LocalPipBundling(),
                    command=[
                        "bash",
                        "-c",
                        (
                            "pip install "
                            + " ".join(f"'{dep}'" for dep in LAYER_DEPENDENCIES)
                            + " -t /asset-output/python --quiet"
                            " && cp -r /asset-input /asset-output/python/shared"
                        ),
                    ],
                ),
            ),
        )

        self.function = lambda_.Function(
            self,
            "IngestFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.X86_64,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(_REPO_ROOT / "src" / "ingest")),
            layers=[self.shared_layer],
            memory_size=512,
            timeout=cdk.Duration.seconds(300),
            log_group=logs.LogGroup(
                self,
                "IngestFunctionLogs",
                retention=logs.RetentionDays.TWO_WEEKS,
                removal_policy=cdk.RemovalPolicy.DESTROY,
            ),
            dead_letter_queue=ingest_dlq,
            environment={
                "TABLE_NAME": table.table_name,
                "EMBEDDING_MODEL_ID": embedding_model_id,
                "DLQ_URL": ingest_dlq.queue_url,
                "CHUNK_TARGET_TOKENS": str(chunk_target_tokens),
                "CHUNK_OVERLAP_TOKENS": str(chunk_overlap_tokens),
            },
        )

        documents_bucket.grant_read(self.function)
        table.grant_read_write_data(self.function)
        ingest_dlq.grant_send_messages(self.function)
        self.function.add_to_role_policy(
            _invoke_model_policy(self, embedding_model_id),
        )

        # EventBridge rather than a direct bucket notification, so the rule lives
        # here and nothing in StorageStack has to reference this stack (ADR-006).
        self.rule = events.Rule(
            self,
            "UploadCreatedRule",
            description="Route documents-bucket uploads to the ingest function",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [documents_bucket.bucket_name]},
                    "object": {"key": [{"prefix": UPLOAD_PREFIX}]},
                },
            ),
            # No dead_letter_queue on the target: that would attach a queue policy
            # to the DLQ in StorageStack referencing this rule's ARN, recreating
            # the cycle. Undelivered invocations are covered by the function's own
            # dead-letter queue, and handled failures are posted by the handler.
            targets=[targets.LambdaFunction(self.function)],
        )

        cdk.CfnOutput(self, "IngestFunctionName", value=self.function.function_name)
        cdk.CfnOutput(self, "SharedLayerArn", value=self.shared_layer.layer_version_arn)


def _invoke_model_policy(stack: cdk.Stack, model_id: str) -> Any:
    """Least-privilege `bedrock:InvokeModel` for one configured model.

    A regional inference profile is not itself invocable: the caller needs the
    profile ARN *and* the underlying foundation-model ARN in every region the
    profile can route to. Scoping to the requested region alone yields
    `AccessDeniedException` the first time Bedrock routes elsewhere, so the
    routed regions are granted explicitly. This is narrower than a wildcard and
    broader than the two ARNs ARCHITECTURE.md §7 anticipated.
    """
    from aws_cdk import aws_iam as iam

    resources = [
        stack.format_arn(
            service="bedrock",
            region=region,
            account="",
            resource="foundation-model",
            resource_name=model_id.split(".", 1)[-1] if "." in model_id else model_id,
        )
        for region in ("us-east-1", "us-east-2", "us-west-2")
    ]
    if model_id.startswith(("us.", "global.")):
        resources.append(
            stack.format_arn(
                service="bedrock",
                resource="inference-profile",
                resource_name=model_id,
            )
        )

    return iam.PolicyStatement(
        actions=["bedrock:InvokeModel"],
        resources=resources,
    )
