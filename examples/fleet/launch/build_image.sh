#!/usr/bin/env bash
# Build + push the miles-fleet trainer image in-cluster (Job + dind) on the
# B300 cluster.
#
# Usage: ./build_image.sh [git-ref]   (default: fleet-integration)
# The builder reuses the cluster's img-build-secrets (GH_TOKEN with
# fleet-ai/platform read + ghcr push); no local token needed when it exists.
set -euo pipefail

KUBE_CONTEXT="${KUBE_CONTEXT:-nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6}"
KUBECTL=(kubectl --context "$KUBE_CONTEXT" -n fleet-train-jobs)
REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
REF="${1:-fleet-integration}"
SHA=$(git -C "$REPO_DIR" rev-parse --short=8 "$REF")
CACHE_TAG="${CACHE_TAG:-latest}"

if [ -n "${GH_TOKEN:-}" ]; then
  "${KUBECTL[@]}" create secret generic img-build-secrets \
    --from-literal=GH_TOKEN="$GH_TOKEN" --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -
else
  "${KUBECTL[@]}" get secret img-build-secrets >/dev/null
fi

"${KUBECTL[@]}" delete job "miles-build-${SHA}" --ignore-not-found --wait=true

export SHA REF CACHE_TAG
envsubst '$SHA $REF $CACHE_TAG' < "$(dirname "$0")/build_job.yaml.tmpl" | "${KUBECTL[@]}" apply -f -

echo "build job: miles-build-${SHA}"
echo "logs:      kubectl --context ${KUBE_CONTEXT} logs -f job/miles-build-${SHA} -c build -n fleet-train-jobs"
echo "image:     ghcr.io/fleet-ai/miles-fleet/trainer:${SHA} (on success)"
