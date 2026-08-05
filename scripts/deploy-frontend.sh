#!/usr/bin/env bash
#
# Push frontend/ to the live site without a CloudFormation deploy.
#
# `cdk deploy FrontendStack` works, but it round-trips through CloudFormation
# and a custom resource for what is three static files — minutes, for a CSS
# tweak. This syncs straight to the site bucket and invalidates CloudFront,
# which takes seconds.
#
# The stack stays the source of truth: the next `cdk deploy` re-uploads from the
# committed asset and overwrites anything pushed here. So this is for iterating,
# not for shipping something that is not in git.
#
#   ./scripts/deploy-frontend.sh             sync, invalidate, wait
#   ./scripts/deploy-frontend.sh --dry-run   show what would change
#   ./scripts/deploy-frontend.sh --no-wait   return before the CDN catches up

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly BOLD=$'\033[1m' RED=$'\033[31m' GREEN=$'\033[32m' DIM=$'\033[2m' OFF=$'\033[0m'
readonly STACK="FrontendStack"
readonly SOURCE="frontend"

ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
info() { printf '  %s·%s %s\n' "$DIM" "$OFF" "$1"; }
die()  { printf '  %s✗%s %s\n\n%s\n' "$RED" "$OFF" "$1" "${2:-}" >&2; exit 1; }
step() { printf '\n%s%s%s\n' "$BOLD" "$1" "$OFF"; }

DRY_RUN=false
WAIT=true
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --no-wait) WAIT=false ;;
    *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

context() { python3 -c "import json;print(json.load(open('cdk.json'))['context']['$1'])"; }
REGION="$(context region)"

output() {
  aws cloudformation describe-stacks \
    --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null
}

# --- Where does it go ------------------------------------------------------ #

step "Locating the deployed site"
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" >/dev/null 2>&1 \
  || die "$STACK is not deployed in $REGION" "  Deploy it first:  ./scripts/deploy.sh"

BUCKET="$(output SiteBucketName)"
DISTRIBUTION="$(output DistributionId)"
SITE_URL="$(output SiteUrl)"
[[ -n "$BUCKET" && -n "$DISTRIBUTION" ]] || die "stack outputs are missing SiteBucketName/DistributionId"
ok "bucket $BUCKET"
ok "distribution $DISTRIBUTION"

# --- Do not ship something broken ------------------------------------------ #

step "Checking the sources"
[[ -d "$SOURCE" ]] || die "$SOURCE/ does not exist"
if command -v node >/dev/null; then
  # A syntax error here would take the whole page down, and unlike a CDK deploy
  # nothing else in this path would catch it.
  node --check "$SOURCE/app.js" || die "app.js has a syntax error — not uploading"
  ok "app.js parses"
else
  info "node not found, skipping the JavaScript syntax check"
fi
ok "$(find "$SOURCE" -type f | wc -l | tr -d ' ') files to sync"

# --- Sync ------------------------------------------------------------------ #

step "Syncing to S3"
SYNC_ARGS=(--delete)
$DRY_RUN && SYNC_ARGS+=(--dryrun)
aws s3 sync "$SOURCE/" "s3://$BUCKET/" --region "$REGION" "${SYNC_ARGS[@]}"

if $DRY_RUN; then
  printf '\n  %sdry run — nothing was uploaded or invalidated%s\n\n' "$DIM" "$OFF"
  exit 0
fi

# --- Invalidate ------------------------------------------------------------ #

step "Invalidating CloudFront"
# Everything, because none of the assets are content-hashed: styles.css keeps
# its name, so an unversioned path would keep serving the old bytes.
INVALIDATION=$(aws cloudfront create-invalidation \
  --distribution-id "$DISTRIBUTION" \
  --paths "/*" \
  --query 'Invalidation.Id' --output text)
ok "created $INVALIDATION"

if $WAIT; then
  info "waiting for it to complete (usually under a minute)"
  aws cloudfront wait invalidation-completed \
    --distribution-id "$DISTRIBUTION" --id "$INVALIDATION"
  ok "edge caches cleared"
else
  info "not waiting — the edge may serve the previous version briefly"
fi

printf '\n%s%s Frontend updated.%s\n' "$BOLD" "$GREEN" "$OFF"
[[ -n "$SITE_URL" ]] && printf '\n  %s%s%s\n' "$BOLD" "$SITE_URL" "$OFF"
printf '\n%s  This bypassed CloudFormation. The next `cdk deploy` re-uploads from\n' "$DIM"
printf '  the committed asset, so keep frontend/ in git.%s\n\n' "$OFF"
