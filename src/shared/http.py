"""API Gateway proxy request parsing and the API_SPEC.md response envelope.

Both API-facing Lambdas answer through this module so a route can never invent
its own error shape. `ApiError` carries the HTTP status and the machine-readable
code from the table in API_SPEC.md §6, which is what lets a handler map a typed
exception to a response without a per-route lookup table (CLAUDE.md: "raise
typed exceptions in shared code; handlers map them to HTTP responses").

CORS headers are attached here too, from `ALLOWED_ORIGIN`. API Gateway answers
the preflight `OPTIONS` itself; a proxy-integrated response still has to carry
the origin header on the real call (API_SPEC.md §7).
"""

import json
import os
from typing import Any

#: Fallback when `ALLOWED_ORIGIN` is unset. The deployed value is set from
#: `cdk.json` context and tightened to the CloudFront domain in T3.1.
DEFAULT_ALLOWED_ORIGIN = "*"


class ApiError(Exception):
    """A failure that maps to a specific status and code in API_SPEC.md §6."""

    status = 500
    code = "INTERNAL"

    def client_message(self) -> str:
        """Return the message safe to send to the caller.

        Internal failures deliberately reveal nothing: API_SPEC.md §6 requires
        details to stay in the logs and never reach the response body.
        """
        return str(self) or "Unexpected error."


def allowed_origin() -> str:
    """Return the CORS origin this deployment permits."""
    return os.environ.get("ALLOWED_ORIGIN") or DEFAULT_ALLOWED_ORIGIN


def cors_headers() -> dict[str, str]:
    """Return the response headers required by API_SPEC.md §7."""
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": allowed_origin(),
        "Access-Control-Allow-Headers": "Content-Type, x-api-key",
        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    }


def json_response(status: int, payload: Any = None) -> dict[str, Any]:
    """Build an API Gateway proxy response.

    Args:
        status: HTTP status code.
        payload: JSON-serializable body. `None` produces an empty body, which
            is what `204 No Content` requires (API_SPEC.md §3).

    Returns:
        The proxy response dictionary API Gateway expects.
    """
    return {
        "statusCode": status,
        "headers": cors_headers(),
        "body": "" if payload is None else json.dumps(payload),
    }


def error_response(error: ApiError) -> dict[str, Any]:
    """Render a typed error as the standard envelope from API_SPEC.md §6."""
    return json_response(
        error.status,
        {"error": {"code": error.code, "message": error.client_message()}},
    )


def parse_json_body(event: dict) -> dict:
    """Read and decode the JSON body of a proxy event.

    A body that is missing, undecodable, or not an object is reported as an
    empty dict rather than raising. Field validation in the handler then
    produces the route's own error code — `INVALID_QUESTION` on `/query`,
    `INVALID_FILENAME` on `/documents` — instead of a generic parse failure.
    That keeps the code table in API_SPEC.md §6 authoritative: every 400 the
    API can emit is one the contract already names.

    Args:
        event: The API Gateway proxy event.

    Returns:
        The decoded body, or an empty dict when there was nothing usable.
    """
    raw = event.get("body")
    if not raw:
        return {}

    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def path_parameter(event: dict, name: str) -> str | None:
    """Return a path parameter from a proxy event, or `None` if absent."""
    parameters = event.get("pathParameters") or {}
    value = parameters.get(name)
    return value if isinstance(value, str) and value else None
