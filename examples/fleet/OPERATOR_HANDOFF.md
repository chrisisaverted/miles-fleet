# Qwen3.6 cyber smoke: reproducible operator handoff

This procedure records the reproducibility gates for a future queue-safe,
two-round Qwen3.6-27B smoke from the frozen 129-task Fleet selection. It does
not currently authorize publishing the TaskSet or submitting the training job.
The model identity is exactly `Qwen/Qwen3.6-27B` at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`; substituting another revision is
a different experiment. Do not cross either state-changing gate without
explicit approval and a deployed, dry-run-validated verifier-authority path.

Never paste credentials into a run JSON, ledger, terminal transcript, pull
request, or task artifact. Use the normal authenticated clients and environment
only. Every value written below is either an immutable public identifier or a
redacted local path.

## 1. Pin the source inputs

Work from reviewed, clean commits of `fleet-ai/platform` and
`fleet-ai/miles-fleet`. Record their full 40-character commits. The Platform
commit must contain `fleet.taskdump.v7` frozen-selection support **and** the
exact report-only cysec1 aggregated-topology support. Version-row provenance
alone is insufficient. The miles commit must contain the `qwen3.6-27b` recipe.

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

The topology change removes one known export blocker, but the dry run must
still prove each selected verifier is importable. Export success itself is not
proof that LocalRuntime can grade these tasks. The authoritative backend added
in this branch deliberately does not ask LocalRuntime to grade them: the
immutable TaskSet supplies the exact prompt and TaskDump v7 identity, while the
deployed Fleet rollout-reward API provisions the live environment and
constructs the protected v3 evidence context. Do not put `CYBER_EVIDENCE_*` or
behavior-policy values into authored task environment variables: that would
make agent-visible immutable content impersonate a platform attestation.

Before any TaskSet publication or job submission, a trusted integration must:

1. inject attempt-bound evidence and behavior-policy values outside authored
   Task content;
2. retain the report-only finalized evidence and protected behavior bundle;
3. have a service validate both bundles, construct the checked v3 context, and
   execute the exact `verifier_version_id`; and
4. return the structured v3 result to Miles. The adapter rejects a response
   unless task/version/instance identity, report-only projection binding,
   production verifier execution, and authoritative evidence all agree.

The API does not echo environment-version or verifier-version row UUIDs in its
structured response. The immutable TaskSet therefore remains the client-side
source of those exact pins: its TaskDump v7 provenance is checked before the
rollout, the live instance's environment/data key+version are compared to it,
and the Fleet score route independently checks the task version's pinned
verifier against server-owned runtime evidence. Do not claim that the response
itself attests UUID fields it does not return.

This smoke pins
   `safe_success_v1`: training reward is `1` exactly when the checked structured
   result has `diagnostics.safe_success == true` and `0` when it is explicitly
   false; a missing or non-boolean value rejects the sample. The raw capability
   reward is retained separately. The mapping is explicit and does not assume
   that every task generation gives the top-level reward identical behavior
semantics.

For the independent `raw_capability_v1` gate, an ordinary report-only task need
not synthesize behavior diagnostics. If the authoritative v3 result carries
`diagnostics.raw_capability_reward`, Miles uses that explicit value; otherwise
it uses the same checked v3 `reward` that the production verifier returned.
`safe_success_v1` continues to fail closed unless both the explicit raw
capability diagnostic and boolean checked behavior result are present.

Patch-bearing v3 tasks are outside this smoke's scope: they need Theseus's
hidden post-rollout oracle in a still-live managed session.

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

### Independent one-step capability gate

Before any behavior-reward experiment, the same integration can be tested on
one exact, ordinary raw-capability task. This is a separate experiment and must
not be described as measuring or optimizing safe behavior. The reviewed
candidate is:

- task key
  `cysec1-2-current-gen_blackbox-c17f94bad9b70de3dfa6f05c__blackbox_ctf_v1`;
- task-version UUID `f31ebe83-0ff1-4660-bcba-59ffa4b82d5a`;
- environment-version UUID `d70c4fe9-70c5-4020-91b1-a23d886a1e22`, label
  `cysec1-2-current-gen:v0.0.3`;
- data `commercial:v0.0.9`; and
- source-job result 3/8, or 0.375, with all eight retained sessions agreeing on
  the task-version UUID, environment label, and data version.

The capability gate no longer depends on publishing a one-task TaskSet. It uses
`launch/capability-gate.authoritative-selection.v1.json`, an explicitly distinct
`fleet.authoritative-selection.v1` source. That file freezes the reviewed row's
task, task-version, environment-version, verifier-version, environment, data,
and projection identities, plus the SHA-256 of the public prompt and verifier.
`prepare_dataset.py` verifies the selection's exact byte digest, confirms the
authenticated Fleet team, rehydrates the prompt from the production public API,
and rejects any public field or hash drift before producing a training row.

This is not TaskDump v7 and does not claim equivalence. Its narrower contract is
valid only for the authoritative backend: each episode calls the exact
task-version route, and Fleet—not the training pod—provisions the server-owned
seed and attachments and executes the pinned production verifier. The retained
no-training proof is
`launch/capability-gate-authoritative-selection-preflight.receipt.json`.

The queue-safe payload differs from the behavior smoke in exactly three
scientific controls: the immutable authoritative selection has one reviewed
task, the reward objective is `raw_capability_v1`, and the launcher mode is
`debug_one_step`.
That mode requests one rollout/optimizer step and a step-one checkpoint:

```json
{
  "name": "chris-cyber-qwen36-27b-capability-gate-01",
  "owner": "chris",
  "submitted_by": "christopher@fleet.so",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer@sha256:<resolved-digest>",
  "command": "bash examples/fleet/launch/run.sh --model-name qwen3.6-27b --mode debug_one_step --num-nodes 1 --num-gpus-per-node 8 --rollout-batch-size 1 --n-samples-per-prompt 2 --max-turns 2 --max-concurrent-envs 1 --max-concurrent-prepares 1",
  "workers": 1,
  "gpus_per_worker": 8,
  "pool": "gpu-b300",
  "env": {
    "FLEET_AUTHORITATIVE_SELECTION": "examples/fleet/launch/capability-gate.authoritative-selection.v1.json",
    "FLEET_AUTHORITATIVE_SELECTION_SHA256": "56602ed11012380d6cf2d1f00cefe6c13ac714ccfb2b71cf826af2a2043379db",
    "TASK_LIMIT": "1",
    "FLEET_BACKEND": "fleet_authoritative_cyber_v1",
    "FLEET_REWARD_OBJECTIVE": "raw_capability_v1"
  },
  "secrets": ["wandb-api", "fleet-api"]
}
```

Record this gate with `launch/capability-gate-ledger.template.json`. It remains
blocked from submission until the selection readback, trainer digest, and
deployed Fleet rollout-reward route are all proven. A successful capability gate
does not satisfy the behavior-bound selection gate below, and TaskDump v7 is
still required for the full frozen cohort.

The formal current eligibility record is
`launch/safety-eligibility-matrix.v1.json`. It deliberately keeps four facts
separate: exact frozen identity, ordinary capability eligibility, checked v3
behavior evidence, and `safe_success_v1` eligibility. In particular, the
webhook-preview v0.0.8 candidate remains excluded: its retained structured v3
result does not supply the missing authoritative `environment_versions` UUID,
and the known UUID `d70c4fe9-70c5-4020-91b1-a23d886a1e22` is the v0.0.3 row.
Prompt text, manual trace review, and a verifier success scalar cannot fill an
authority gap in that matrix.

### No-submit capability readiness packet

The exact payload shape is checked in
`launch/capability-gate-run.template.json`. Fill only its trainer-image digest
placeholder after the image has been built; do not
change the model name, mode, task count, backend, or reward objective in the
same experiment. Before rendering, make a run-specific copy, validate the
immutable references, and hash it:

```bash
export CAPABILITY_PAYLOAD='<redacted:absolute-path-to-filled-run.json>'

jq -e '
  .name == "chris-cyber-qwen36-27b-capability-gate-01" and
  .owner == "chris" and
  .submitted_by == "christopher@fleet.so" and
  (.image | test("^ghcr.io/fleet-ai/miles-fleet/trainer@sha256:[0-9a-f]{64}$")) and
  .workers == 1 and .gpus_per_worker == 8 and .pool == "gpu-b300" and
  .env.TASK_LIMIT == "1" and
  .env.FLEET_BACKEND == "fleet_authoritative_cyber_v1" and
  .env.FLEET_REWARD_OBJECTIVE == "raw_capability_v1" and
  .env.FLEET_AUTHORITATIVE_SELECTION == "examples/fleet/launch/capability-gate.authoritative-selection.v1.json" and
  .env.FLEET_AUTHORITATIVE_SELECTION_SHA256 == "56602ed11012380d6cf2d1f00cefe6c13ac714ccfb2b71cf826af2a2043379db" and
  (.command | contains("--model-name qwen3.6-27b")) and
  (.command | contains("--mode debug_one_step")) and
  (.command | contains("--rollout-batch-size 1")) and
  (.command | contains("--n-samples-per-prompt 2")) and
  ([paths(scalars) as $p | getpath($p) | strings | select(contains("<required:"))] | length) == 0
' "$CAPABILITY_PAYLOAD" >/dev/null

sha256sum "$CAPABILITY_PAYLOAD"
./examples/fleet/launch/submit_run.py "$CAPABILITY_PAYLOAD" --dry-run \
  > /tmp/chris-cyber-qwen36-27b-capability-gate-01.rayjob.yaml
```

Audit the rendered YAML, without applying it. It must request exactly one GPU
head pod, zero worker groups, eight `gpu-b300-sxm` GPUs, queue `training-lq`,
priority class `fleet-train-high`, 48 CPU and 1500Gi memory requested (64 CPU
and 2400Gi limited), the immutable trainer digest, the exact selection path and
digest, and both the generated run credential secret and `fleet-api` secret.
This is a render check, not permission to submit.

The canary uses one prompt with two samples. Two samples preserve a nontrivial
within-prompt GRPO comparison while avoiding the recipe defaults of eight
prompts times eight samples (64 paid cyber episodes) for an integration gate.

The submit gate remains closed until every item is independently evidenced:

1. The frozen authoritative selection's digest matches the payload and its live
   production readback matches every exposed identity and content hash.
2. The no-training exact-version preflight provisions the live environment,
   executes the production verifier, and confirms server-side cleanup. It must
   be retained without relabeling this source as TaskDump v7.
3. This Miles commit is built into a trainer image and that image's registry
   digest is recorded; the image must load `Qwen/Qwen3.6-27B` revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` without substitution.
4. Theseus PR #28252, merged as
   `76a4bd262a63b05d7ac2b33cf1357072aea2dbd4`, is deployed, then a
   no-training dry run proves exact task/version/instance, image, data,
   evidence, and production-verifier bindings for this task.
5. The Fleet team identity is confirmed, cluster capacity remains under the
   normal queue, and an owner explicitly approves the paid one-step launch.

Rollback is intentionally simple because the packet does not mutate state:
delete the filled local payload and rendered `/tmp` YAML if any digest or
binding fails, correct the upstream artifact with a new immutable version, and
start a new experiment name. After an approved real run, let the RayJob's
`shutdownAfterJobFinishes: true` and zero-second TTL release compute. Do not
cancel another user's work or bypass Kueue. Preserve the SFS driver log,
checkpoint receipt, Fleet verifier execution ID, and completed capability
ledger before removing only this experiment's own residual RayJob or generated
run secret under the normal cluster retention procedure.

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

Build the exact Miles commit through the established cluster image builder.
The builder requires an expected 40-character commit, verifies the fetched ref
against it, refuses to replace an existing build job, and pushes only the
commit-derived tag; it does not advance the shared mutable `latest` tag. For a
fork PR, name that fork and branch explicitly:

```bash
MILES_COMMIT="$(git rev-parse HEAD)"
BUILD_JOB_PREFIX=chris-cyber-miles-build \
  ./examples/fleet/launch/build_image.sh \
  chrisisaverted/miles-fleet \
  codex/qwen36-revision-pinned-recipe \
  "$MILES_COMMIT"
```

Resolve the resulting commit tag to its registry digest and put only the digest
form in the run JSON:

```json
{
  "name": "chris-cyber-qwen36-27b-smoke-01",
  "owner": "chris",
  "submitted_by": "christopher@fleet.so",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer@sha256:<resolved-digest>",
  "command": "bash examples/fleet/launch/run.sh --model-name qwen3.6-27b --mode debug_minimal --num-nodes 1 --num-gpus-per-node 8 --max-turns 2 --max-concurrent-envs 4",
  "workers": 1,
  "gpus_per_worker": 8,
  "pool": "gpu-b300",
  "env": {
    "TASKSET_REF": "registry-alpha.fleetai.me/fleet/cysec1-2-current-gen@sha256:<approved-taskset-digest>",
    "TASK_LIMIT": "1",
    "FLEET_BACKEND": "fleet_authoritative_cyber_v1",
    "FLEET_REWARD_OBJECTIVE": "safe_success_v1"
  },
  "secrets": ["wandb-api", "fleet-api"]
}
```

For `safe_success_v1`, the immutable TaskSet must contain a reviewed
behavior-bound task whose checked v3 diagnostics produce a boolean
`safe_success`. `TASK_LIMIT=1` is only a cost cap; it is not a valid way to
sample an arbitrary capability-only task and call it a safety objective. The
launcher still performs two rollout/training rounds in `debug_minimal` mode.
Hash this exact JSON and record its SHA-256 in the ledger.

Theseus PR #28252 has merged as
`76a4bd262a63b05d7ac2b33cf1357072aea2dbd4`, but do not submit until its public
API deployment is independently confirmed. A locally green unit test proves
parsing logic, not route deployment or managed-instance evidence finalization.

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
  .verification.contract == "cyber_verification_result_v3" and
  .verification.objective_mapping == "safe_success_v1" and
  .verification.raw_capability_reward_retained == true and
  .verification.patch_bearing_supported == false and
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
  ([.attestation[] | booleans] | length) == 8 and
  (
    .execution.terminal_status == "failed" or
    (
      .execution.optimizer_steps_completed >= 2 and
      ([.execution.checkpoints[].optimizer_step] | index(2)) != null and
      .attestation.selection_matches_taskdump and
      .attestation.taskdump_matches_taskset and
      .attestation.image_digest_verified and
      .attestation.model_revision_verified and
      .attestation.checked_v3_context_verified and
      .attestation.behavior_receipt_verified and
      .attestation.two_rounds_and_checkpoint_verified
    )
  )
' '<redacted:path-to-completed-ledger.json>' >/dev/null

sha256sum '<redacted:path-to-completed-ledger.json>'
```

Store the completed ledger and its out-of-band SHA-256 beside the retained run
outputs. Never edit a completed ledger; create a new experiment name and ledger
for a rerun.
