#!/usr/bin/env bash
#
# Remove everything this project created, and prove it.
#
# `cdk destroy` alone is not quite enough, for two reasons that are easy to miss
# because neither reports an error:
#
#   1. Secrets Manager does not delete a secret when CloudFormation asks. It
#      schedules deletion, keeps the secret for a recovery window of up to 30
#      days, and keeps billing for it.
#   2. Log groups created *by* a Lambda at runtime are not part of any template,
#      so nothing deletes them. The custom-resource functions CDK generates for
#      bucket auto-delete and asset deployment each leave one behind.
#
# So this destroys the stacks, sweeps both, and then verifies by asking the
# tagging API whether anything still carries the project tag.
#
#   ./scripts/destroy.sh             confirm, then remove everything
#   ./scripts/destroy.sh --yes       no confirmation
#   ./scripts/destroy.sh --dry-run   show what would be removed

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# `cdk destroy` runs `python3 app.py` from cdk.json, which needs aws-cdk-lib.
[[ -x .venv/bin/python3 ]] && PATH="$PWD/.venv/bin:$PATH"

readonly BOLD=$'\033[1m' RED=$'\033[31m' GREEN=$'\033[32m' YELLOW=$'\033[33m' DIM=$'\033[2m' OFF=$'\033[0m'
readonly PROJECT_TAG="serverless-rag"
readonly STACKS=(ObservabilityStack FrontendStack ApiStack IngestStack StorageStack)

ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
info() { printf '  %s·%s %s\n' "$DIM" "$OFF" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$1"; }
step() { printf '\n%s%s%s\n' "$BOLD" "$1" "$OFF"; }

DRY_RUN=false
ASSUME_YES=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y)  ASSUME_YES=true ;;
    *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

run() { if $DRY_RUN; then printf '  %swould run:%s %s\n' "$DIM" "$OFF" "$*"; else "$@"; fi; }

context() { python3 -c "import json;print(json.load(open('cdk.json'))['context']['$1'])"; }
REGION="$(context region)"

command -v aws >/dev/null || { printf 'aws CLI is required\n' >&2; exit 1; }
aws sts get-caller-identity >/dev/null 2>&1 || { printf 'no usable AWS credentials\n' >&2; exit 1; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
CDK=$(command -v cdk >/dev/null && echo cdk || echo "npx cdk")

# --- what is actually out there ------------------------------------------- #

step "Looking for deployed stacks"
LIVE=()
for stack in "${STACKS[@]}"; do
  if aws cloudformation describe-stacks --stack-name "$stack" --region "$REGION" >/dev/null 2>&1; then
    LIVE+=("$stack")
    info "$stack"
  fi
done

if [[ ${#LIVE[@]} -eq 0 ]]; then
  ok "no stacks deployed in $ACCOUNT / $REGION"
else
  if ! $DRY_RUN && ! $ASSUME_YES; then
    printf '\n%sThis deletes %d stack(s) in account %s, region %s.%s\n' \
      "$YELLOW" "${#LIVE[@]}" "$ACCOUNT" "$REGION" "$OFF"
    printf 'Uploaded documents and their embeddings go with them. This is not reversible.\n'
    read -r -p 'Type the account number to confirm: ' reply
    [[ "$reply" == "$ACCOUNT" ]] || { printf '\nAborted.\n'; exit 1; }
  fi

  step "Destroying stacks"
  printf '%sCloudFront is the slow one — several minutes on its own.%s\n\n' "$DIM" "$OFF"
  run $CDK destroy --all --force
fi

# --- the two things CloudFormation does not finish ------------------------ #

step "Secrets scheduled for deletion"
# CloudFormation schedules rather than deletes, and a scheduled secret is still
# billed. Force-delete removes it now and frees the name for a redeploy.
SECRETS=$(aws secretsmanager list-secrets \
  --include-planned-deletion \
  --filters "Key=tag-value,Values=$PROJECT_TAG" \
  --region "$REGION" \
  --query 'SecretList[].ARN' --output text 2>/dev/null || true)

if [[ -z "$SECRETS" ]]; then
  ok "none pending"
else
  for arn in $SECRETS; do
    info "force-deleting ${arn##*:}"
    run aws secretsmanager delete-secret \
      --secret-id "$arn" --force-delete-without-recovery \
      --region "$REGION" >/dev/null
  done
fi

step "Orphaned log groups"
# Our own function log groups are in the templates and are already gone. These
# belong to the custom-resource Lambdas CDK generates, which create their log
# group at runtime — so no template ever owned them.
FOUND_LOGS=false
for stack in "${STACKS[@]}"; do
  GROUPS=$(aws logs describe-log-groups \
    --log-group-name-prefix "/aws/lambda/$stack" \
    --region "$REGION" \
    --query 'logGroups[].logGroupName' --output text 2>/dev/null || true)
  for group in $GROUPS; do
    FOUND_LOGS=true
    info "$group"
    run aws logs delete-log-group --log-group-name "$group" --region "$REGION" 2>/dev/null || true
  done
done
$FOUND_LOGS || ok "none left behind"

# --- verification --------------------------------------------------------- #

step "Verifying"
if $DRY_RUN; then
  warn "dry run — nothing was deleted"
  exit 0
fi

REMAINING=$(aws resourcegroupstaggingapi get-resources \
  --tag-filters "Key=Project,Values=$PROJECT_TAG" \
  --region "$REGION" \
  --query 'ResourceTagMappingList[].ResourceARN' --output text 2>/dev/null || true)

if [[ -z "$REMAINING" ]]; then
  ok "nothing in $REGION still carries Project=$PROJECT_TAG"
else
  warn "still tagged (the tagging API lags a little; re-run to confirm):"
  for arn in $REMAINING; do printf '      %s\n' "$arn"; done
fi

rm -f cdk-outputs.json

printf '\n%s%s Done.%s\n' "$BOLD" "$GREEN" "$OFF"
printf '\n%s  Not touched, because it is shared with every other CDK app in the\n' "$DIM"
printf '  account: the CDKToolkit bootstrap stack and its asset bucket.\n'
printf '  Remove it with `aws cloudformation delete-stack --stack-name CDKToolkit`\n'
printf '  only if nothing else in %s uses CDK.%s\n\n' "$REGION" "$OFF"
