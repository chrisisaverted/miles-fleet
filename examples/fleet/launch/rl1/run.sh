#!/usr/bin/env bash
# Code-side entrypoint, baked into the trainer image and invoked by the run
# payload's `command`. Does everything the run needs before training: Fleet
# credentials, taskset pull, dataset build, environment-image pre-pull, model
# prep — then execs the training launcher with the arguments it was given.
#
# Reads from the injected environment:
#   RUN_ID        run name; names the SFS directory and WandB group
#   TASKSET_REF   v2-registry taskset to pull and train on
#   TASK_LIMIT    task sample cap for the dataset build (0 = whole taskset)
#   FLEET_CREDENTIALS_B64, AWS_*   from the run secret
#
# Everything else (model, mode, lengths, batch) comes in as arguments and is
# passed through to run_fleet.py. --model-name is also parsed here for the
# shared model prep.
set -euo pipefail

: "${RUN_ID:?RUN_ID must be set}"
: "${TASKSET_REF:?TASKSET_REF must be set}"
TASK_LIMIT="${TASK_LIMIT:-0}"

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

for i in $(seq 1 90); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker version --format "dind server: {{.Server.Version}}"

# registry-alpha has multi-minute 502 outages; every pull retries 10x30s
PULLED=""
for A in $(seq 1 10); do
  flt pull "$TASKSET_REF" taskset && { PULLED=1; break; }
  echo "flt pull attempt $A failed, retrying"
  sleep 30
done
test -n "$PULLED" || { echo "flt pull failed after 10 attempts"; exit 1; }

REPO_PATH=$(printf "%s" "$TASKSET_REF" | cut -d: -f1 | sed "s#^registry-alpha.fleetai.me/##")
NS=$(printf "%s" "$REPO_PATH" | cut -d/ -f1)
NAME=$(printf "%s" "$REPO_PATH" | cut -d/ -f2)
ROOT=$(flt list | grep "^taskset " | tr -s " " | cut -d" " -f4)
ROOT_BARE=${ROOT#sha256:}
TOKEN=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/fleet/credentials.json')))['registries']['registry-alpha.fleetai.me']['token'])")
printf "%s" "$TOKEN" | docker login registry-alpha.fleetai.me --username fleet --password-stdin
mkdir -p ~/.flt/image-locations/sha256
curl -fsS --retry 10 --retry-delay 30 --retry-all-errors -H "Authorization: Bearer $TOKEN" \
  "https://registry-alpha.fleetai.me/v1/namespaces/$NS/repositories/$NAME/versions/$ROOT/image-locations" \
  > ~/.flt/image-locations/sha256/$ROOT_BARE.json
echo "image-locations plan written for $ROOT"

cd /root/miles
python -m examples.fleet.prepare_dataset \
  --taskset-ref taskset --output-dir "$RUN_DIR/data" --max-tasks "$TASK_LIMIT"

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
