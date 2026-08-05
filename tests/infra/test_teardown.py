"""Teardown-safety guardrails (PRD G4, CLAUDE.md: teardown-friendly by design).

`cdk destroy` leaving something behind is a silent failure: no error, no output,
just a resource that keeps existing and — for a few resource types — keeps
billing. These tests fail the build instead.

They synthesize the real application, so a stack added later is covered without
anyone remembering to extend this file.
"""

import json
from pathlib import Path

import pytest
from aws_cdk import cx_api

from app import PROJECT_TAG, build_app

REPO_ROOT = Path(__file__).resolve().parents[2]
CDK_CONTEXT: dict[str, object] = json.loads((REPO_ROOT / "cdk.json").read_text())["context"]

#: Resource types that hold data and must therefore be explicitly destroyable.
#: Anything here defaulting to RETAIN would survive teardown holding documents.
STATEFUL_TYPES = ("AWS::S3::Bucket", "AWS::DynamoDB::Table", "AWS::SQS::Queue")

#: Types that keep costing money while they merely exist, so a leftover is not
#: just untidy. A Secrets Manager secret is billed per month even while it sits
#: in its post-deletion recovery window.
BILLED_WHILE_IDLE_TYPES = ("AWS::SecretsManager::Secret",)


@pytest.fixture(scope="module")
def assembly() -> cx_api.CloudAssembly:
    return build_app(CDK_CONTEXT | {"aws:cdk:bundling-stacks": []}).synth()


def _resources(assembly: cx_api.CloudAssembly):
    """Yield `(stack_name, logical_id, resource)` for every resource in the app."""
    for stack in assembly.stacks:
        for logical_id, resource in stack.template.get("Resources", {}).items():
            yield stack.stack_name, logical_id, resource


def test_nothing_survives_destroy(assembly):
    """No resource may be marked RETAIN.

    CDK applies RETAIN silently in places — most notably the account-level
    CloudWatch role API Gateway creates by default, which is why `ApiStack`
    passes `cloud_watch_role=False`. A retained resource makes `cdk destroy`
    report success while leaving things in the account.
    """
    retained = [
        f"{stack}.{logical_id} ({resource['Type']})"
        for stack, logical_id, resource in _resources(assembly)
        if resource.get("DeletionPolicy", "Delete") != "Delete"
    ]
    assert not retained, "these survive `cdk destroy`:\n  " + "\n  ".join(retained)


def test_stateful_resources_are_explicitly_destroyable(assembly):
    """Buckets, tables, and queues default to RETAIN in CloudFormation.

    This project is a demo, not a system of record: the documented promise is
    that teardown really does remove the data (PRD G4).
    """
    for stack, logical_id, resource in _resources(assembly):
        if resource["Type"] in STATEFUL_TYPES:
            assert resource.get("DeletionPolicy") == "Delete", f"{stack}.{logical_id}"
            assert resource.get("UpdateReplacePolicy") == "Delete", f"{stack}.{logical_id}"


def test_buckets_empty_themselves_on_delete(assembly):
    """CloudFormation refuses to delete a non-empty bucket.

    Both buckets fill up in normal use — one with uploads, one with the site —
    so without auto-delete the very first teardown fails partway through.
    """
    for stack, logical_id, resource in _resources(assembly):
        if resource["Type"] != "AWS::S3::Bucket":
            continue
        tags = {tag["Key"]: tag["Value"] for tag in resource["Properties"].get("Tags", [])}
        assert tags.get("aws-cdk:auto-delete-objects") == "true", (
            f"{stack}.{logical_id} would block teardown once it holds an object"
        )


def test_every_resource_carries_the_project_tag(assembly):
    """The tag is what makes teardown verifiable rather than assumed.

    `scripts/destroy.sh` asks the tagging API what still carries it; an untagged
    resource is one the sweep cannot see.
    """
    untagged = []
    for stack, logical_id, resource in _resources(assembly):
        tags = resource["Properties"].get("Tags") if "Properties" in resource else None
        if tags is None:
            # Most resource types take tags, but not all do; only flag the ones
            # that declare a Tags property and left the project tag out of it.
            continue
        if isinstance(tags, list) and not any(
            tag.get("Key") == "Project" and tag.get("Value") == PROJECT_TAG for tag in tags
        ):
            untagged.append(f"{stack}.{logical_id} ({resource['Type']})")

    assert not untagged, "not visible to the teardown sweep:\n  " + "\n  ".join(untagged)


def test_log_groups_expire_rather_than_accumulate(assembly):
    """An unbounded log group is a slow, invisible cost."""
    for stack, logical_id, resource in _resources(assembly):
        if resource["Type"] == "AWS::Logs::LogGroup":
            assert resource["Properties"].get("RetentionInDays"), f"{stack}.{logical_id}"


def test_the_api_does_not_create_account_level_resources(assembly):
    """Account-wide singletons cannot be safely deleted, so they are not created.

    `AWS::ApiGateway::Account` is one per account per region; CDK retains it
    because deleting it would break unrelated APIs. Declining it keeps teardown
    complete, and costs nothing here — the dashboard reads API Gateway metrics,
    which need no CloudWatch role.
    """
    types = {resource["Type"] for _, _, resource in _resources(assembly)}
    assert "AWS::ApiGateway::Account" not in types
