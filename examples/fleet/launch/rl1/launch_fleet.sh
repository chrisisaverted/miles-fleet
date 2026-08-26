#!/usr/bin/env bash
# Launch a Fleet training Job on rl1 with the miles-fleet trainer image.
#
# Prereqs: kubectl context for rl1; `flt auth status` valid (credentials ship
# to the pod); the ghcr-pull secret already on the cluster (or GH_TOKEN set).
#
# Usage: ./launch_fleet.sh [image-tag] [taskset-ref] [mode]
#   mode: normal (default) | debug_minimal | rollout_only
# Env overrides: JOB_NAME NUM_GPUS ROLLOUT_BATCH N_SAMPLES MAX_TURNS
#   CONCURRENCY TASK_LIMIT MAIN_MEM MAIN_MEM_LIM SCRIPT_EXTRA
# AWS_ACCESS_KEY_ID/SECRET ride the run secret when set (evaluation-benchmark
# seed blobs live on s3://theseus-envdata; ade-bench never needs them).
set -euo pipefail

KUBE_CONTEXT="${KUBE_CONTEXT:-fleet-training-rl1-us-east-1}"
KUBECTL=(kubectl --context "$KUBE_CONTEXT")
HERE="$(cd "$(dirname "$0")" && pwd)"

TAG="${1:-latest}"
export TASKSET_REMOTE_REF="${2:-registry-alpha.fleetai.me/library/ade-bench:latest}"
export MODE="${3:-normal}"
export MODEL_NAME="${MODEL_NAME:-glm4.7-flash}"
export IMAGE="ghcr.io/fleet-ai/miles-fleet/trainer:${TAG}"

export NUM_GPUS="${NUM_GPUS:-8}"  # full-node default; override for splits
export ROLLOUT_BATCH="${ROLLOUT_BATCH:-8}"
export N_SAMPLES="${N_SAMPLES:-8}"
export MAX_TURNS="${MAX_TURNS:-32}"
export CONCURRENCY="${CONCURRENCY:-8}"
export TASK_LIMIT="${TASK_LIMIT:-0}"
# Host RAM: checkpoint save is the peak (4x MegatronTrainRayActor.save_model
# at ~185GB RSS each, measured on miles-smoke-ade 2026-08-22; an 800Gi limit
# OOM-killed at that point). Request sized so two runs fit the 1900Gi flavor
# quota; the limit rides node-level overcommit for the brief save peak.
export MAIN_MEM="${MAIN_MEM:-925Gi}"
NODE_WORKLOAD="${NODE_WORKLOAD:-gpu-h200}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p5en.48xlarge}"
export NODE_WORKLOAD INSTANCE_TYPE
export MAIN_MEM_LIM="${MAIN_MEM_LIM:-1300Gi}"
export SCRIPT_EXTRA="${SCRIPT_EXTRA:-}"

ts_short=$(basename "${TASKSET_REMOTE_REF%%:*}" | tr '[:upper:]' '[:lower:]' | cut -c1-12)
rand5=$(od -An -N4 -tx4 /dev/urandom | tr -d ' \n' | cut -c1-5)
export JOB_NAME="${JOB_NAME:-miles-${ts_short}-$(date +%m%d)-${rand5}}"
export SECRET_NAME="${JOB_NAME}-secrets"
echo "run name: ${JOB_NAME} (model: ${MODEL_NAME})"

if [ -n "${GH_TOKEN:-}" ]; then
  "${KUBECTL[@]}" create secret docker-registry ghcr-pull -n fleet-train-jobs \
    --docker-server=ghcr.io --docker-username=x --docker-password="$GH_TOKEN" \
    --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -
else
  "${KUBECTL[@]}" get secret ghcr-pull -n fleet-train-jobs >/dev/null
fi

"${KUBECTL[@]}" create secret generic "$SECRET_NAME" -n fleet-train-jobs \
  --from-literal=FLEET_CREDENTIALS_B64="$(base64 < "$HOME/.config/fleet/credentials.json" | tr -d '\n')" \
  ${AWS_ACCESS_KEY_ID:+--from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"} \
  ${AWS_SECRET_ACCESS_KEY:+--from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"} \
  --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -

envsubst '$MODEL_NAME $JOB_NAME $SECRET_NAME $IMAGE $TASKSET_REMOTE_REF $TASK_LIMIT $NUM_GPUS $MODE $ROLLOUT_BATCH $N_SAMPLES $MAX_TURNS $CONCURRENCY $MAIN_MEM $MAIN_MEM_LIM $SCRIPT_EXTRA $NODE_WORKLOAD $INSTANCE_TYPE' \
  < "$HERE/job.yaml.tmpl" | "${KUBECTL[@]}" apply -f -

echo
echo "status: kubectl --context ${KUBE_CONTEXT} get job ${JOB_NAME} -n fleet-train-jobs -w"
echo "logs:   kubectl --context ${KUBE_CONTEXT} logs -f job/${JOB_NAME} -c miles -n fleet-train-jobs"
echo "sfs:    /mnt/sfs/miles-fleet/${JOB_NAME}/driver.log"
