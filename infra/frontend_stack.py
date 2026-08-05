"""Static site bucket and CloudFront distribution with OAC (T3.1).

One distribution serves two origins: the private site bucket for the app, and
the REST API for `/v1/*`. That makes the browser's API calls same-origin, which
removes CORS from the picture entirely and lets CloudFront attach the API key as
an origin header so it never reaches the client (ADR-008).

The site bucket stays private and is reachable only through Origin Access
Control, so the CloudFront URL is the single public entry point.
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from constructs import Construct

_FRONTEND_SRC = Path(__file__).resolve().parents[1] / "frontend"

#: Path pattern routed to the API. It matches the deployment stage name, so a
#: request for `/v1/query` reaches the API's own `/v1/query`.
API_PATH_PATTERN = "/v1/*"

#: Paths invalidated on every deploy.
#:
#: This has to be everything, not just index.html. None of the assets are
#: content-hashed — `styles.css` keeps its name across deploys — and they are
#: served with no `Cache-Control`, so CloudFront holds them at its default TTL
#: of 24 hours. Invalidating only the HTML would leave a stylesheet change
#: invisible for a day while the deploy reported success.
INVALIDATION_PATHS = ["/*"]


class FrontendStack(cdk.Stack):
    """The public CloudFront entry point and the static app behind it."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env: cdk.Environment,
        api: apigateway.RestApi,
        api_key_value: str,
    ) -> None:
        """Initialize the frontend stack.

        Args:
            scope: The parent construct.
            construct_id: Stack identifier.
            env: Target account and region.
            api: The REST API to place behind the `/v1/*` behavior.
            api_key_value: Deploy-time reference to the API key, injected as an
                origin header so the browser never sees it.
        """
        super().__init__(scope, construct_id, env=env)

        self.site_bucket = s3.Bucket(
            self,
            "SiteBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        self.distribution = cloudfront.Distribution(
            self,
            "SiteDistribution",
            comment="serverless-rag demo",
            default_root_object="index.html",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            default_behavior=cloudfront.BehaviorOptions(
                # S3BucketOrigin.with_origin_access_control keeps the bucket
                # private and grants only this distribution read access.
                origin=origins.S3BucketOrigin.with_origin_access_control(self.site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            additional_behaviors={
                API_PATH_PATTERN: cloudfront.BehaviorOptions(
                    # `origin_path=""` matters: RestApiOrigin defaults it to the
                    # stage, which would forward `/v1/query` as `/v1/v1/query`.
                    origin=origins.RestApiOrigin(
                        api,
                        origin_path="",
                        custom_headers={"x-api-key": api_key_value},
                    ),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    # An API response is never cacheable here: document status
                    # changes while the page polls it.
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    # API Gateway rejects a forwarded Host header that is not its
                    # own, so everything except Host is passed through.
                    origin_request_policy=(
                        cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER
                    ),
                ),
            },
            error_responses=[
                # The app is a single page; unknown paths render it rather than
                # CloudFront's XML error document.
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                ),
            ],
        )

        self.deployment = s3_deployment.BucketDeployment(
            self,
            "SiteDeployment",
            sources=[s3_deployment.Source.asset(str(_FRONTEND_SRC))],
            destination_bucket=self.site_bucket,
            distribution=self.distribution,
            distribution_paths=INVALIDATION_PATHS,
        )

        cdk.CfnOutput(self, "SiteUrl", value=f"https://{self.distribution.domain_name}")
        cdk.CfnOutput(self, "SiteBucketName", value=self.site_bucket.bucket_name)
        cdk.CfnOutput(self, "DistributionId", value=self.distribution.distribution_id)
