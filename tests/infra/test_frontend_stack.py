"""CDK assertions for FrontendStack (T3.1 acceptance criteria, TESTING.md §3)."""

import json

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infra.api_stack import ApiStack
from infra.frontend_stack import API_PATH_PATTERN, FrontendStack
from infra.ingest_stack import IngestStack
from infra.storage_stack import StorageStack

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture(scope="module")
def template() -> assertions.Template:
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
    frontend = FrontendStack(
        app, "FrontendStack", env=env, api=api.api, api_key_value=api.api_key_value
    )
    return assertions.Template.from_stack(frontend)


def _distribution(template: assertions.Template) -> dict:
    (resource,) = template.find_resources("AWS::CloudFront::Distribution").values()
    return resource["Properties"]["DistributionConfig"]


def _api_behavior(template: assertions.Template) -> dict:
    config = _distribution(template)
    (behavior,) = [
        cache
        for cache in config.get("CacheBehaviors", [])
        if cache["PathPattern"] == API_PATH_PATTERN
    ]
    return behavior


def _api_origin(template: assertions.Template) -> dict:
    config = _distribution(template)
    behavior = _api_behavior(template)
    (origin,) = [o for o in config["Origins"] if o["Id"] == behavior["TargetOriginId"]]
    return origin


# --------------------------------------------------------------------------- #
# Site bucket
# --------------------------------------------------------------------------- #


def test_site_bucket_blocks_all_public_access(template):
    """CloudFront is the only public entry point; the bucket itself is private."""
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


def test_site_bucket_is_readable_only_through_the_distribution(template):
    """Origin Access Control, not a public policy or a legacy OAI."""
    policies = str(template.find_resources("AWS::S3::BucketPolicy"))
    assert "cloudfront.amazonaws.com" in policies
    assert "AWS:SourceArn" in policies


def test_origin_access_control_is_configured(template):
    template.resource_count_is("AWS::CloudFront::OriginAccessControl", 1)


def test_the_app_is_deployed_into_the_bucket(template):
    """A BucketDeployment is what puts frontend/ in front of CloudFront."""
    template.resource_count_is("Custom::CDKBucketDeployment", 1)


def test_a_deploy_invalidates_every_path(template):
    """Invalidating only the HTML would ship a stylesheet change into a void.

    None of the assets are content-hashed — `styles.css` keeps its name across
    deploys — and they carry no `Cache-Control`, so CloudFront holds them at its
    default 24-hour TTL. A deploy that invalidated just `/index.html` would
    report success while the edge kept serving yesterday's CSS.
    """
    (deployment,) = template.find_resources("Custom::CDKBucketDeployment").values()
    assert deployment["Properties"]["DistributionPaths"] == ["/*"]


def test_removed_files_are_pruned_from_the_bucket(template):
    """A renamed asset must not linger and keep being served."""
    (deployment,) = template.find_resources("Custom::CDKBucketDeployment").values()
    assert deployment["Properties"]["Prune"] is True


# --------------------------------------------------------------------------- #
# Distribution
# --------------------------------------------------------------------------- #


def test_the_distribution_serves_the_app_and_the_api(template):
    assert len(_distribution(template)["Origins"]) == 2


def test_the_default_root_object_is_the_app(template):
    assert _distribution(template)["DefaultRootObject"] == "index.html"


def test_viewers_are_redirected_to_https(template):
    assert (
        _distribution(template)["DefaultCacheBehavior"]["ViewerProtocolPolicy"]
        == "redirect-to-https"
    )


def test_api_requests_are_routed_to_api_gateway(template):
    origin = _api_origin(template)
    assert "execute-api" in json.dumps(origin["DomainName"])


def test_the_api_origin_path_is_empty(template):
    """Regression guard for a doubled stage prefix.

    `RestApiOrigin` defaults its origin path to the stage name, which would
    forward `/v1/query` to the API as `/v1/v1/query` and 403 on every call.
    """
    assert _api_origin(template).get("OriginPath", "") == ""


def test_the_api_key_is_injected_as_an_origin_header(template):
    """ADR-008: the key reaches the origin, never the browser."""
    headers = _api_origin(template)["OriginCustomHeaders"]
    (header,) = [h for h in headers if h["HeaderName"] == "x-api-key"]
    assert header["HeaderValue"]


def test_the_api_key_value_is_a_secrets_manager_reference(template):
    """The literal must not appear in the template, only a resolve directive."""
    headers = json.dumps(_api_origin(template)["OriginCustomHeaders"])
    assert "resolve:secretsmanager" in headers


def test_api_responses_are_never_cached(template):
    """Document status changes while the page polls it."""
    behavior = _api_behavior(template)
    # CachePolicy.CACHING_DISABLED is an AWS managed policy with a fixed id.
    assert behavior["CachePolicyId"] == "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"


def test_api_behavior_allows_every_method(template):
    """Uploads presign with POST and documents are removed with DELETE."""
    assert set(_api_behavior(template)["AllowedMethods"]) >= {
        "GET",
        "POST",
        "DELETE",
        "OPTIONS",
        "PUT",
        "PATCH",
        "HEAD",
    }


def test_api_behavior_forwards_viewer_headers_except_host(template):
    """API Gateway rejects a Host header that is not its own."""
    behavior = _api_behavior(template)
    # OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER, a managed policy id.
    assert behavior["OriginRequestPolicyId"] == "b689b0a8-53d0-40ab-baf2-68738e2966ac"


def test_unknown_paths_render_the_single_page_app(template):
    responses = {
        response["ErrorCode"]: response
        for response in _distribution(template).get("CustomErrorResponses", [])
    }
    assert {403, 404} <= responses.keys()
    for response in responses.values():
        assert response["ResponsePagePath"] == "/index.html"
        assert response["ResponseCode"] == 200


def test_stack_exports_the_site_url(template):
    assert {"SiteUrl", "SiteBucketName", "DistributionId"} <= template.find_outputs("*").keys()
