#!/usr/bin/env python3
"""CDK application entry point for the Serverless RAG Document Q&A stack."""

import os

import aws_cdk as cdk

from infra.storage_stack import StorageStack


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
    StorageStack(app, "StorageStack", env=env)
    return app


def main() -> None:
    """Synthesize the application.

    Only `StorageStack` is wired up in Phase 0, and it declares no resources
    yet — the ingest, API, frontend, and observability stacks arrive from T1.4
    onwards (IMPLEMENTATION_PLAN.md).
    """
    build_app().synth()


if __name__ == "__main__":
    main()
