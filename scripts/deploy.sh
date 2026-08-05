#!/usr/bin/env bash
#
# Deploy the whole stack, after checking the things that actually go wrong.
#
# The preflight exists because a `cdk deploy` failure arrives fifteen minutes
# in, as a CloudFormation rollback with a message several layers away from the
# cause. Every check below corresponds to a real failure mode: missing
# credentials, an unbootstrapped account, or — by far the most common — Bedrock
# model access that was never granted.
#
#   ./scripts/deploy.sh              deploy everything
#   ./scripts/deploy.sh --check      run the preflight and stop

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The CDK CLI runs `python3 app.py` from cdk.json, which needs aws-cdk-lib. Put
# the project venv first so the script works whether or not it was activated —
# otherwise `cdk` fails with ModuleNotFoundError several steps in.
[[ -x .venv/bin/python3 ]] && PATH="$PWD/.venv/bin:$PATH"

readonly BOLD=$'\033[1m' RED=$'\033[31m' GREEN=$'\033[32m' YELLOW=$'\033[33m' DIM=$'\033[2m' OFF=$'\033[0m'

CHECK_ONLY=false
CDK_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--check" ]]; then
    CHECK_ONLY=true
  else
    # Anything else is forwarded to `cdk deploy`, which is how a value can be
    # supplied without committing it — notably:
    #   ./scripts/deploy.sh --context alertEmail=you@example.com
    CDK_ARGS+=("$arg")
  fi
done

ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$1"; }
die()  { printf '  %s✗%s %s\n\n%s\n' "$RED" "$OFF" "$1" "${2:-}" >&2; exit 1; }
step() { printf '\n%s%s%s\n' "$BOLD" "$1" "$OFF"; }

context() { python3 -c "import json;print(json.load(open('cdk.json'))['context']['$1'])"; }

REGION="$(context region)"
EMBEDDING_MODEL="$(context embeddingModelId)"
GENERATION_MODEL="$(context generationModelId)"

step "Toolchain"
for tool in python3 node aws; do
  command -v "$tool" >/dev/null || die "$tool is not installed"
done
command -v cdk >/dev/null || command -v npx >/dev/null \
  || die "the AWS CDK CLI is not installed" "  npm install -g aws-cdk"
CDK=$(command -v cdk >/dev/null && echo cdk || echo "npx cdk")
ok "python $(python3 --version | cut -d' ' -f2), node $(node --version), $($CDK --version 2>/dev/null | head -1)"

step "AWS credentials"
IDENTITY=$(aws sts get-caller-identity --output json 2>/dev/null) \
  || die "no usable AWS credentials" "  aws configure    # or export AWS_PROFILE"
ACCOUNT=$(python3 -c "import json,sys;print(json.load(sys.stdin)['Account'])" <<<"$IDENTITY")
ok "account $ACCOUNT, deploying to $REGION"

step "CDK bootstrap"
# Deploying needs a staging bucket and roles in the target region. `--check`
# only reports: a preflight must not change the account.
if aws cloudformation describe-stacks --stack-name CDKToolkit --region "$REGION" >/dev/null 2>&1; then
  ok "account is bootstrapped in $REGION"
  BOOTSTRAP_NEEDED=false
else
  BOOTSTRAP_NEEDED=true
  if $CHECK_ONLY; then
    warn "not bootstrapped in $REGION — deploy will do it first (one-off, ~1 min)"
  else
    warn "not bootstrapped in $REGION — doing it now"
  fi
fi

step "Bedrock model access"
# Enabling access is a console-only action, and the model list showing a model
# does NOT mean this account may invoke it. So the check is an actual call.
check_model() {
  local id="$1" body="$2" label="$3"
  if aws bedrock-runtime invoke-model \
        --model-id "$id" --region "$REGION" \
        --cli-binary-format raw-in-base64-out \
        --body "$body" \
        --content-type application/json --accept application/json \
        /dev/null >/dev/null 2>&1; then
    ok "$label"
  else
    die "cannot invoke $id" \
"  Enable it in the Bedrock console, then re-run:
    https://${REGION}.console.aws.amazon.com/bedrock/home?region=${REGION}#/modelaccess

  A model appearing in the list is not the same as access being granted.
  If it is already enabled, check that your credentials allow bedrock:InvokeModel."
  fi
}
check_model "$EMBEDDING_MODEL" '{"inputText":"preflight"}' "embeddings — $EMBEDDING_MODEL"
check_model "$GENERATION_MODEL" \
  '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
  "generation — $GENERATION_MODEL"

if $CHECK_ONLY; then
  printf '\n%sPreflight passed.%s Run ./scripts/deploy.sh to deploy.\n' "$GREEN" "$OFF"
  exit 0
fi

if $BOOTSTRAP_NEEDED; then
  step "Bootstrapping $REGION"
  $CDK bootstrap "aws://$ACCOUNT/$REGION"
fi

step "Deploying"
printf '%sFive stacks. CloudFront takes most of the ~10 minutes.%s\n\n' "$DIM" "$OFF"
$CDK deploy --all --require-approval never --outputs-file cdk-outputs.json "${CDK_ARGS[@]+"${CDK_ARGS[@]}"}"

SITE_URL=$(python3 -c "
import json
try:
    print(json.load(open('cdk-outputs.json')).get('FrontendStack', {}).get('SiteUrl', ''))
except Exception:
    print('')
")

printf '\n%s%s Deployed.%s\n' "$BOLD" "$GREEN" "$OFF"
if [[ -n "$SITE_URL" ]]; then
  printf '\n  Open  %s%s%s\n' "$BOLD" "$SITE_URL" "$OFF"
  printf '\n%s  Upload a document, wait for READY, then ask a question.\n' "$DIM"
  printf '  CloudFront can take a few minutes to serve the first request.%s\n' "$OFF"
fi
printf '\n  Verify   %spytest tests/integration -q%s\n' "$BOLD" "$OFF"
printf '  Remove   %s./scripts/destroy.sh%s\n\n' "$BOLD" "$OFF"
