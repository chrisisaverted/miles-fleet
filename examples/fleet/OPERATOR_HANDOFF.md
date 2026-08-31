# Qwen3.6 cyber smoke: reproducible operator handoff

This procedure prepares one queue-safe, two-round Qwen3.6-27B smoke from the
frozen 129-task Fleet selection. It separates read-only preparation from the
two state-changing gates: publishing the TaskSet and submitting the training
job. Do not cross either gate without explicit approval.

Never paste credentials into a run JSON, ledger, terminal transcript, pull
request, or task artifact. Use the normal authenticated clients and environment
only. Every value written below is either an immutable public identifier or a
redacted local path.

## 1. Pin the source inputs

Work from reviewed, clean commits of `fleet-ai/platform` and
`fleet-ai/miles-fleet`. Record their full 40-character commits. The Platform
commit must contain `fleet.taskdump.v7` frozen-selection support; the miles
commit must contain the `qwen3.6-27b` recipe.

Create the bounded selector from the already-frozen run configuration:

```bash
export CYBER_RUN_CONFIG='<redacted:path-to-qwen36-27b-rl-base-full-runnable.json>'
export SELECTION='/tmp/cysec1-2-selection.v1.json'

TEAM_ID="$(jq -er '.tasks.team_id' "$CYBER_RUN_CONFIG")"
jq -e --arg team "$TEAM_ID" '
  {
    schema: "fleet.taskdump.selection.v1",
    tasks: [
      .tasks.task_versions[] |
      {
        task_key,
        task_version_id,
        team_id: $team,
        environment_version_id,
        env_key,
        env_version,
        data_key,
        data_version
      }
    ]
  }
' "$CYBER_RUN_CONFIG" > "$SELECTION"

jq -e '
  .schema == "fleet.taskdump.selection.v1" and
  (.tasks | length) == 129 and
  ([.tasks[].task_version_id] | unique | length) == 129 and
  ([.tasks[] | [.team_id, .task_key]] | unique | length) == 129
' "$SELECTION" >/dev/null
```

Record `sha256sum "$SELECTION"` in the experiment ledger. Stop if the count,
uniqueness checks, or any subsequent pin check fails.

## 2. Export and audit the exact TaskDump (read-only)

Run from the pinned Platform checkout. Supply the read-only task database and
approved image-grounding access through the operator environment; the command
does not publish anything:

```bash
export TASKDUMP='/tmp/cysec1-2-taskdump.v7.jsonl.zst'
export IMAGE_RECEIPT='/tmp/cysec1-2-image-resolution.json'

env TASKS_EXPORT_DATABASE_URL='<redacted:read-only-database-url>' \
  uv run tools/dump_tasks.py \
  --frozen-selection-manifest "$SELECTION" \
  --resolve-live-image-id \
  --image-resolution-out "$IMAGE_RECEIPT" \
  --out "$TASKDUMP"
```

Audit the uncompressed records before importing. The exporter records the
SHA-256 of the selector's exact bytes, so formatting changes are detectable:

```bash
SELECTION_SHA="$(sha256sum "$SELECTION" | awk '{print $1}')"
zstd -dc "$TASKDUMP" | jq -sc \
  --arg selection_sha "$SELECTION_SHA" \
  --slurpfile selection "$SELECTION" '
  . as $rows |
  ($selection[0].tasks | sort_by(.task_version_id)) as $wanted |
  ($rows | map(select(.record == "task")) | sort_by(.task_version_id)) as $got |
  ($rows | map(select(.record == "manifest"))) as $manifests |
  ($manifests | length) == 1 and
  $manifests[0].schema == "fleet.taskdump.v7" and
  $manifests[0].task_count == 129 and
  $manifests[0].selection.task_count == 129 and
  $manifests[0].selection.sha256 == $selection_sha and
  ($got | length) == 129 and
  ([ $got[].task_version_id ] | unique | length) == 129 and
  ([range(0; 129) |
    $got[.].task_version_id == $wanted[.].task_version_id and
    $got[.].team_id == $wanted[.].team_id and
    $got[.].key == $wanted[.].task_key and
    $got[.].environment.version_id == $wanted[.].environment_version_id and
    $got[.].environment.env_key == $wanted[.].env_key and
    $got[.].environment.version == $wanted[.].env_version and
    ($got[.].seed == null or (
      $got[.].seed.data_key == $wanted[.].data_key and
      $got[.].seed.data_version == $wanted[.].data_version
    ))
  ] | all) and
  ([ $got[].verifier.version_id | select(. == null) ] | length) == 0
' >/dev/null
```

Record the TaskDump SHA-256, selector SHA-256, exporter commit, schema, and
count in the ledger. The Registry import must report 129 tasks and no lowering,
identity, or image-grounding errors.

## 3. Stage the TaskSet locally, then stop at the publish gate

Use the pinned Platform `flt` binary to import the file locally and inspect the
result. Exact CLI spelling can change with Platform revisions, so use that
commit's `flt export taskdump --help`, `flt pull file --help`, and
`flt push --help` as the authority. The intended flow is:

```bash
export EDITABLE_TASKSET='<redacted:absolute-local-taskset-directory>'
export LOCAL_TASKSET='cysec1-2-current-gen-candidate'

flt export "taskdump://$TASKDUMP" "$EDITABLE_TASKSET"
flt pull "file://$EDITABLE_TASKSET/taskset.yml" "$LOCAL_TASKSET"
flt list "$LOCAL_TASKSET"
flt inspect "$LOCAL_TASKSET" --format digest
flt push "$LOCAL_TASKSET" \
  fleet/cysec1-2-current-gen:v2026-08-30 \
  --dry-run \
  --no-latest-retag
```

Verify the local count is 129 and review the dry-run plan. Publishing is a
state change: do not remove `--dry-run` until the owner explicitly approves it.
After an approved push, record the returned immutable
`registry-alpha.fleetai.me/fleet/cysec1-2-current-gen@sha256:...` reference.
Never record or train from the mutable tag.

## 4. Build and inspect the trainer image, then stop at the submit gate

Build the exact miles commit through the established cluster image builder.
Building is itself a state change and the current shared builder also advances
its mutable `latest` image tag, so obtain explicit approval before invoking it.
Resolve the resulting tag to its registry digest and put only the digest form
in the run JSON:

```json
{
  "name": "chris-cyber-qwen36-27b-smoke-01",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer@sha256:<resolved-digest>",
  "command": "bash examples/fleet/launch/run.sh --model-name qwen3.6-27b --mode debug_minimal --num-nodes 1 --num-gpus-per-node 8 --max-turns 2 --max-concurrent-envs 4",
  "workers": 1,
  "gpus_per_worker": 8,
  "pool": "gpu-b300",
  "env": {
    "TASKSET_REF": "registry-alpha.fleetai.me/fleet/cysec1-2-current-gen@sha256:<approved-taskset-digest>",
    "TASK_LIMIT": "1"
  },
  "secrets": ["wandb-api"]
}
```

`TASK_LIMIT=1` is the smallest existing Fleet cyber slice; it does not invent
a new taskset. The launcher still performs two rollout/training rounds in
`debug_minimal` mode. Hash this exact JSON and record its SHA-256 in the ledger.

Render and review the queued job without applying it:

```bash
./examples/fleet/launch/submit_run.py '<redacted:path-to-run.json>' --dry-run \
  > /tmp/chris-cyber-qwen36-27b-smoke-01.rayjob.yaml
```

Confirm the render requests one GPU head pod and no additional worker pods,
eight B300 GPUs, pool
`gpu-b300`, queue `training-lq`, the immutable trainer image, and the immutable
TaskSet reference. Submission is a state change: run the command without
`--dry-run` only after explicit approval. Do not cancel or bypass the queue.

## 5. Capture success or failure evidence

Inside the running image, record exact versions for Python, Torch,
Transformers, SGLang, flash-linear-attention, Ray, and fleet-runtime. Retain the
driver log and metrics link. Success requires evidence of all of the following:

1. the pinned Qwen3.6 revision was loaded;
2. Fleet environment preparation and verifier-backed grading completed;
3. two rollout/training rounds completed;
4. optimizer step 2 completed; and
5. the step-2 checkpoint exists and has a recorded manifest digest.

Fill every placeholder in
`launch/experiment-ledger.template.json`, set an attestation to `true` only
when its evidence is retained, then validate and freeze the record. A failed
run is still a scientific result: use `terminal_status: "failed"`, retain its
last evidence, and leave unmet evidence attestations false.

```bash
jq -e '
  .schema == "fleet.miles.experiment-ledger.v1" and
  .data.task_count == 129 and
  .data.taskdump_schema == "fleet.taskdump.v7" and
  (.data.selection_manifest_sha256 | test("^sha256:[0-9a-f]{64}$")) and
  (.data.taskdump_sha256 | test("^sha256:[0-9a-f]{64}$")) and
  (.data.taskset_reference | test("@sha256:[0-9a-f]{64}$")) and
  (.runtime.trainer_image | test("@sha256:[0-9a-f]{64}$")) and
  (.launch.payload_sha256 | test("^sha256:[0-9a-f]{64}$")) and
  (.model.revision | test("^[0-9a-f]{40}$")) and
  (.source.miles_commit | test("^[0-9a-f]{40}$")) and
  (.source.taskdump_exporter_commit | test("^[0-9a-f]{40}$")) and
  (.execution.terminal_status == "succeeded" or .execution.terminal_status == "failed") and
  (.execution.optimizer_steps_completed | type) == "number" and
  .execution.optimizer_steps_completed >= 0 and
  ([.execution.checkpoints[].manifest_sha256 | test("^sha256:[0-9a-f]{64}$")] | all) and
  ([paths(scalars) as $p | getpath($p) | strings | select(contains("<required:"))] | length) == 0 and
  .attestation.all_placeholders_resolved == true and
  ([.attestation[] | booleans] | length) == 6 and
  (
    .execution.terminal_status == "failed" or
    (
      .execution.optimizer_steps_completed >= 2 and
      ([.execution.checkpoints[].optimizer_step] | index(2)) != null and
      .attestation.selection_matches_taskdump and
      .attestation.taskdump_matches_taskset and
      .attestation.image_digest_verified and
      .attestation.model_revision_verified and
      .attestation.two_rounds_and_checkpoint_verified
    )
  )
' '<redacted:path-to-completed-ledger.json>' >/dev/null

sha256sum '<redacted:path-to-completed-ledger.json>'
```

Store the completed ledger and its out-of-band SHA-256 beside the retained run
outputs. Never edit a completed ledger; create a new experiment name and ledger
for a rerun.
