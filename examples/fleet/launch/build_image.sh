#!/usr/bin/env bash
# Build + push the miles-fleet trainer image in-cluster (Job + dind) on the
# B300 cluster.
#
# Usage: ./build_image.sh [owner/repository] [git-ref] [expected-40-char-commit]
#
# The ref may be a fork branch or a GitHub pull-request ref.  The expected
# commit is checked after fetch so a moved branch can never silently change the
# image bytes.  The builder pushes only the commit-derived tag; it deliberately
# does not update the shared mutable `latest` tag.
# The builder reuses the cluster's img-build-secrets (GH_TOKEN with
# fleet-ai/platform read + ghcr push); no local token needed when it exists.
set -euo pipefail

KUBE_CONTEXT="${KUBE_CONTEXT:-nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6}"
KUBECTL=(kubectl --context "$KUBE_CONTEXT" -n fleet-train-jobs)
REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
SOURCE_REPOSITORY="${1:-fleet-ai/miles-fleet}"
REF="${2:-fleet-integration}"
EXPECTED_COMMIT="${3:-$(git -C "$REPO_DIR" rev-parse "$REF")}"
CACHE_TAG="${CACHE_TAG:-latest}"
BUILD_JOB_PREFIX="${BUILD_JOB_PREFIX:-miles-build}"

case "$SOURCE_REPOSITORY" in
  */*) ;;
  *) echo "SOURCE_REPOSITORY must be owner/repository" >&2; exit 1 ;;
esac
case "$EXPECTED_COMMIT" in
  ""|*[!0-9a-f]*) echo "EXPECTED_COMMIT must be exactly 40 lowercase hex characters" >&2; exit 1 ;;
esac
[ "${#EXPECTED_COMMIT}" -eq 40 ] || {
  echo "EXPECTED_COMMIT must be exactly 40 lowercase hex characters" >&2
  exit 1
}
SHA="${EXPECTED_COMMIT%${EXPECTED_COMMIT#????????}}"
BUILD_JOB="${BUILD_JOB_PREFIX}-${SHA}"
case "$BUILD_JOB" in
  ""|-*|*-|*[!a-z0-9-]*) echo "BUILD_JOB_PREFIX must produce a lowercase DNS-safe job name" >&2; exit 1 ;;
esac
[ "${#BUILD_JOB}" -le 63 ] || {
  echo "build job name exceeds 63 characters: $BUILD_JOB" >&2
  exit 1
}

if [ -n "${GH_TOKEN:-}" ]; then
  "${KUBECTL[@]}" create secret generic img-build-secrets \
    --from-literal=GH_TOKEN="$GH_TOKEN" --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -
else
  "${KUBECTL[@]}" get secret img-build-secrets >/dev/null
fi

if "${KUBECTL[@]}" get job "$BUILD_JOB" >/dev/null 2>&1; then
  echo "refusing to replace existing build job: $BUILD_JOB" >&2
  exit 1
fi

export BUILD_JOB SHA SOURCE_REPOSITORY REF EXPECTED_COMMIT CACHE_TAG
envsubst '$BUILD_JOB $SHA $SOURCE_REPOSITORY $REF $EXPECTED_COMMIT $CACHE_TAG' \
  < "$(dirname "$0")/build_job.yaml.tmpl" | "${KUBECTL[@]}" create -f -

echo "source:    ${SOURCE_REPOSITORY}@${EXPECTED_COMMIT} (fetched as ${REF})"
echo "build job: ${BUILD_JOB}"
echo "logs:      kubectl --context ${KUBE_CONTEXT} logs -f job/${BUILD_JOB} -c build -n fleet-train-jobs"
echo "image:     ghcr.io/fleet-ai/miles-fleet/trainer:${SHA} (on success)"
