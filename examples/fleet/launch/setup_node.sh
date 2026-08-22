#!/usr/bin/env bash
# Host-side setup for a Fleet + miles training node.
#
# Target: a single 8-GPU H200 VM with a real docker daemon (GCP a3-ultragpu-8g
# with the cu128/driver-570 deeplearning image). Fleet environments run as
# SIBLING containers of the miles container via a docker-socket mount, so this
# does NOT work on container-only providers (RunPod pods) without dind.
#
# Expects these files staged in ~/fleet-stage/ before running (scp'd from the
# operator's machine):
#   flt-linux-amd64                      flt built at the SAME platform commit
#                                        as the fleet_runtime wheel
#   fleet_runtime-*.whl                  built from platform/runtime
#   credentials.json                     ~/.config/fleet/credentials.json (SSO
#                                        token; expires in days, refresh per run)
#   datasets/                            prepared JSONL dirs (ade-bench/,
#                                        evaluation-benchmark/) from prepare_dataset.py
#   env.sh                               exports HF_TOKEN, WANDB_API_KEY
#
# Usage: bash setup_node.sh <miles-git-url> <miles-ref>
set -euxo pipefail

MILES_URL="${1:?miles git url}"
MILES_REF="${2:?miles ref}"
STAGE="$HOME/fleet-stage"
CONTAINER=miles-fleet

# ---------------------------------------------------------------- preflight
nvidia-smi -L
docker run --rm --gpus all ubuntu nvidia-smi -L | head -2
df -h "$(docker info -f '{{.DockerRootDir}}')"

# ------------------------------------------------------------ host: flt/.flt
sudo install -m 0755 "$STAGE/flt-linux-amd64" /usr/local/bin/flt
mkdir -p ~/.config/fleet ~/.flt
cp "$STAGE/credentials.json" ~/.config/fleet/credentials.json
flt auth status

# Pull both tasksets into the host store (mounted into the container below).
# This also writes the image-locations plans the SDK uses to resolve env images.
flt pull registry-alpha.fleetai.me/library/ade-bench:latest ade-bench
flt pull registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3 evaluation-benchmark

# Docker credentials for the env-image registry (same SSO token flt uses).
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.config/fleet/credentials.json'))['registries']['registry-alpha.fleetai.me']['token'])" 2>/dev/null || true)
if [ -n "$TOKEN" ]; then
  echo "$TOKEN" | docker login registry-alpha.fleetai.me --username fleet --password-stdin || true
fi

# ------------------------------------------------------- miles container up
docker pull radixark/miles:latest
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$HOME/.flt:/root/.flt" \
  -v "$HOME/.config/fleet:/root/.config/fleet" \
  -v "$STAGE:/root/fleet-stage" \
  -v "$HOME/outputs:/root/outputs" \
  radixark/miles:latest sleep infinity

# --------------------------------------------------- container: provisioning
docker exec "$CONTAINER" bash -euxc "
  # docker CLI (static) so the fleet SDK can drive the host daemon
  curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar -xz --strip-components=1 -C /usr/local/bin docker/docker
  docker ps >/dev/null

  install -m 0755 /root/fleet-stage/flt-linux-amd64 /usr/local/bin/flt

  # miles at the pinned integration ref
  cd /root/miles && git remote set-url origin '$MILES_URL' && git fetch origin '$MILES_REF' \
    && git checkout FETCH_HEAD && pip install -e . --no-deps

  pip install /root/fleet-stage/fleet_runtime-*.whl

  mkdir -p /root/datasets/fleet
  cp -r /root/fleet-stage/datasets/* /root/datasets/fleet/

  # sanity: SDK + integration import, docker reachable, store visible
  cd /root/miles && python -c '
import fleet_runtime
import examples.fleet.rollout, examples.fleet.session
print(\"fleet integration imports OK\")
'
"

# ------------------------------------------------ container: docker smoke
docker exec "$CONTAINER" bash -euxc "
  export FLEET_FLT=/usr/local/bin/flt FLEET_DOCKER_TIMEOUT_S=600
  cd /root/miles && python -m pytest examples/fleet/tests/integration -q -m docker -p no:cacheprovider
"

echo 'setup complete. Enter with: docker exec -it '"$CONTAINER"' bash'
echo 'then: source /root/fleet-stage/env.sh && export FLEET_FLT=/usr/local/bin/flt FLEET_DOCKER_TIMEOUT_S=600'
echo 'and:  cd /root/miles && python examples/fleet/launch/run_fleet_glm47_flash.py --dataset-dir /root/datasets/fleet/ade-bench ...'
