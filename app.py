#!/usr/bin/env python3
"""CDK application entry point for the Serverless RAG Document Q&A stack."""

import os

import aws_cdk as cdk

from infra.api_stack import ApiStack
from infra.frontend_stack import FrontendStack
from infra.ingest_stack import IngestStack
from infra.observability_stack import ObservabilityStack
from infra.storage_stack import StorageStack

#: Applied to every resource in every stack. `scripts/destroy.sh` sweeps for
#: this tag to prove teardown was complete, and it is the handle Cost Explorer
#: needs to attribute spend to this project.
PROJECT_TAG = "serverless-rag"


def resolve_env(app: cdk.App) -> cdk.Environment:
    """Build the deployment environment for every stack in this app.

    The region comes from `cdk.json` context rather than `CDK_DEFAULT_REGION`
    so that deployment always targets the region the project is designed for
    (PRD §8), independent of the operator's local AWS CLI default. The account
    is read from the standard CDK environment and may be `None`, which yields
    an environment-agnostic template — this is what allows `cdk synth` to run
    in CI with no AWS credentials (TESTING.md §1).

    Args:
        app: The CDK application whose context holds the target region.

    Returns:
        The environment to pass to each stack constructor.

    Raises:
        ValueError: If the `region` context key is missing from `cdk.json`.
    """
    region = app.node.try_get_context("region")
    if not region:
        raise ValueError("Missing required context key 'region' in cdk.json")
    return cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=region)


def _require_context(app: cdk.App, key: str) -> str:
    """Read a required `cdk.json` context value.

    Args:
        app: The CDK application carrying the context.
        key: The context key to read.

    Returns:
        The configured value as a string.

    Raises:
        ValueError: If the key is absent, which means `cdk.json` is incomplete.
    """
    value = app.node.try_get_context(key)
    if value is None or value == "":
        raise ValueError(f"Missing required context key {key!r} in cdk.json")
    return str(value)


def build_app(context: dict[str, object] | None = None) -> cdk.App:
    """Construct the fully wired CDK application.

    Kept separate from `main` so the infrastructure tests can synthesize the
    real app rather than a parallel copy of its wiring.

    Args:
        context: Context overrides. Left `None` under the CDK CLI, which
            supplies `cdk.json` context itself; the tests pass the parsed
            `cdk.json` context explicitly, since a bare `App` does not read it.

    Returns:
        The application with every stack attached, ready to synthesize.
    """
    app = cdk.App(context=context)
    env = resolve_env(app)

    # Tag everything. This is what makes teardown verifiable: `scripts/destroy.sh`
    # sweeps for anything still carrying the tag, and Cost Explorer can be
    # filtered by it to prove the idle bill really is zero.
    cdk.Tags.of(app).add("Project", PROJECT_TAG)

    embedding_model_id = _require_context(app, "embeddingModelId")

    storage = StorageStack(app, "StorageStack", env=env)
    ingest = IngestStack(
        app,
        "IngestStack",
        env=env,
        documents_bucket=storage.documents_bucket,
        table=storage.table,
        ingest_dlq=storage.ingest_dlq,
        embedding_model_id=embedding_model_id,
        chunk_target_tokens=int(_require_context(app, "chunkTargetTokens")),
        chunk_overlap_tokens=int(_require_context(app, "chunkOverlapTokens")),
    )
    api = ApiStack(
        app,
        "ApiStack",
        env=env,
        documents_bucket=storage.documents_bucket,
        table=storage.table,
        # Built once in IngestStack so every Lambda in the app runs identical
        # shared code, rather than two copies of the same asset.
        shared_layer=ingest.shared_layer,
        embedding_model_id=embedding_model_id,
        generation_model_id=_require_context(app, "generationModelId"),
        min_similarity=float(_require_context(app, "minSimilarity")),
        max_tokens=int(_require_context(app, "maxTokens")),
        max_upload_mb=int(_require_context(app, "maxUploadMb")),
        allowed_origin=_require_context(app, "allowedOrigin"),
    )
    FrontendStack(
        app,
        "FrontendStack",
        env=env,
        api=api.api,
        # A deploy-time reference, not the key itself: CloudFront resolves it
        # and forwards it to the origin, so it never reaches a browser (ADR-008).
        api_key_value=api.api_key_value,
    )
    ObservabilityStack(
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
        # Optional: alarms and the budget exist either way, but only deliver
        # when an address is configured.
        alert_email=str(app.node.try_get_context("alertEmail") or ""),
    )
    return app


def main() -> None:
    """Synthesize the application: storage, ingestion, API, frontend, observability."""
    build_app().synth()


if __name__ == "__main__":
    main()
