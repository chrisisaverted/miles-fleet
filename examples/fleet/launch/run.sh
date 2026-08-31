#!/usr/bin/env bash
# Code-side entrypoint, baked into the trainer image and invoked by the run
# payload's `command`. Does everything the run needs before training: Fleet
# credentials, taskset pull, dataset build, environment-image pre-pull, model
# prep — then execs the training launcher with the arguments it was given.
#
# Reads from the injected environment:
#   RUN_ID        run name; names the SFS directory and WandB group
#   TASKSET_REF   v2-registry taskset to pull for the local backend
#   FLEET_AUTHORITATIVE_SELECTION  immutable selection file baked into image
#   FLEET_AUTHORITATIVE_SELECTION_SHA256  exact digest of that selection
#   TASK_LIMIT    task sample cap for the dataset build (0 = whole taskset)
#   FLEET_BACKEND  local or fleet_authoritative_cyber_v1
#   FLEET_REWARD_OBJECTIVE  raw_capability_v1 or safe_success_v1
#   FLEET_CREDENTIALS_B64, FLEET_API_KEY, AWS_*   from run secrets
#
# Everything else (model, mode, lengths, batch) comes in as arguments and is
# passed through to run_fleet.py. --model-name is also parsed here for the
# shared model prep.
set -euo pipefail

: "${RUN_ID:?RUN_ID must be set}"
TASK_LIMIT="${TASK_LIMIT:-0}"
FLEET_BACKEND="${FLEET_BACKEND:-local}"
FLEET_REWARD_OBJECTIVE="${FLEET_REWARD_OBJECTIVE:-raw_capability_v1}"

RUN_DIR=/mnt/sfs/miles-fleet/${RUN_ID}
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

MODEL_NAME=""
prev=""
for a in "$@"; do
  [ "$prev" = "--model-name" ] && MODEL_NAME="$a"
  prev="$a"
done
test -n "$MODEL_NAME" || { echo "run.sh: --model-name not found in arguments"; exit 1; }

mkdir -p ~/.config/fleet
printf "%s" "$FLEET_CREDENTIALS_B64" | base64 -d > ~/.config/fleet/credentials.json
chmod 600 ~/.config/fleet/credentials.json

CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$CODE_DIR"

if [ "$FLEET_BACKEND" = "local" ]; then
  : "${TASKSET_REF:?TASKSET_REF must be set for the local backend}"
  # registry-alpha has multi-minute 502 outages; every pull retries 10x30s
  PULLED=""
  for A in $(seq 1 10); do
    flt pull "$TASKSET_REF" taskset && { PULLED=1; break; }
    echo "flt pull attempt $A failed, retrying"
    sleep 30
  done
  test -n "$PULLED" || { echo "flt pull failed after 10 attempts"; exit 1; }

  # Resolve the repository independently of whether the immutable taskset is
  # selected by @sha256:<digest> or the legacy mutable :tag form.
  REPO_PATH=$(python -m examples.fleet.launch.taskset_ref "$TASKSET_REF")
  NS=${REPO_PATH%%/*}
  NAME=${REPO_PATH#*/}
  ROOT=$(flt list | grep "^taskset " | tr -s " " | cut -d" " -f4)
  ROOT_BARE=${ROOT#sha256:}

  python -m examples.fleet.prepare_dataset \
    --taskset-ref taskset --output-dir "$RUN_DIR/data" --max-tasks "$TASK_LIMIT" \
    --backend "$FLEET_BACKEND" --reward-objective "$FLEET_REWARD_OBJECTIVE"

  for i in $(seq 1 90); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker version --format "dind server: {{.Server.Version}}"
  TOKEN=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/fleet/credentials.json')))['registries']['registry-alpha.fleetai.me']['token'])")
  printf "%s" "$TOKEN" | docker login registry-alpha.fleetai.me --username fleet --password-stdin
  mkdir -p ~/.flt/image-locations/sha256
  curl -fsS --retry 10 --retry-delay 30 --retry-all-errors -H "Authorization: Bearer $TOKEN" \
    "https://registry-alpha.fleetai.me/v1/namespaces/$NS/repositories/$NAME/versions/$ROOT/image-locations" \
    > ~/.flt/image-locations/sha256/$ROOT_BARE.json
  echo "image-locations plan written for $ROOT"

  python -c "import json,os; need=set(open('$RUN_DIR/data/images.txt').read().split()); src=json.load(open(os.path.expanduser('~/.flt/image-locations/sha256/$ROOT_BARE.json')))['image_sources']; hit=sorted(set(v['location'] for k,v in src.items() if k in need)); print('\n'.join(hit if hit else sorted(set(v['location'] for v in src.values()))))" > /tmp/pull_list.txt
  while read -r LOC; do
    echo "pre-pulling $LOC"
    PULLED=""
    for A in $(seq 1 10); do
      docker pull -q "$LOC" && { PULLED=1; break; }
      echo "pull attempt $A failed for $LOC, retrying"
      sleep 30
    done
    test -n "$PULLED" || { echo "pull failed after 10 attempts: $LOC"; exit 1; }
  done < /tmp/pull_list.txt
elif [ "$FLEET_BACKEND" = "fleet_authoritative_cyber_v1" ]; then
  : "${FLEET_AUTHORITATIVE_SELECTION:?FLEET_AUTHORITATIVE_SELECTION must be set}"
  : "${FLEET_AUTHORITATIVE_SELECTION_SHA256:?FLEET_AUTHORITATIVE_SELECTION_SHA256 must be set}"
  case "$FLEET_AUTHORITATIVE_SELECTION_SHA256" in
    *[!0-9a-f]*)
      echo "FLEET_AUTHORITATIVE_SELECTION_SHA256 must be exactly 64 lowercase hex characters" >&2
      exit 1
      ;;
  esac
  [ "${#FLEET_AUTHORITATIVE_SELECTION_SHA256}" -eq 64 ] || {
    echo "FLEET_AUTHORITATIVE_SELECTION_SHA256 must be exactly 64 lowercase hex characters" >&2
    exit 1
  }
  python -m examples.fleet.prepare_dataset \
    --authoritative-selection "$FLEET_AUTHORITATIVE_SELECTION" \
    --authoritative-selection-sha256 "$FLEET_AUTHORITATIVE_SELECTION_SHA256" \
    --output-dir "$RUN_DIR/data" --max-tasks "$TASK_LIMIT" \
    --backend "$FLEET_BACKEND" --reward-objective "$FLEET_REWARD_OBJECTIVE"
else
  echo "unsupported FLEET_BACKEND: $FLEET_BACKEND" >&2
  exit 1
fi

export FLEET_DOCKER_TIMEOUT_S=600
export MILES_SCRIPT_EXTERNAL_RAY=1

# Model prep is shared on SFS across runs, serialized by flock; prepare() is
# idempotent (Path.exists).
MODEL_DIR=/mnt/sfs/miles-fleet/models
exec 9>"$MODEL_DIR/.prep.lock"
flock -w 14400 9
python examples/fleet/launch/run_fleet.py \
  --model-name "$MODEL_NAME" --prepare-only \
  --model-dir "$MODEL_DIR" --data-dir "$RUN_DIR"
flock -u 9

exec python examples/fleet/launch/run_fleet.py \
  --skip-prepare \
  --run-id "$RUN_ID" \
  --dataset-dir "$RUN_DIR/data" \
  --model-dir "$MODEL_DIR" \
  --data-dir "$RUN_DIR" \
  --output-dir /mnt/sfs/miles-fleet \
  "$@"
