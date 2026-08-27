# Runbook: Fleet training with miles on rl1

Operations manual for runs launched from this directory. What the code does
is in [README.md](README.md); this file is what to run, what to watch, and
what each failure looks like. Every failure in the triage table happened at
least once and the fix is the one that worked.

## 1. Prerequisites

- kubeconfig for `fleet-training-rl1-us-east-1`. All commands here pass
  `--context` explicitly, so your current context does not matter.
- Fleet registry login: `flt auth status`. The token expires after a few
  days and is copied into the pod at launch; an expired token fails the boot
  at `docker login` with 401.
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` exported, only for tasksets
  with s3 seed data (evaluation-benchmark yes, ade-bench no).
- The cluster secrets `ghcr-pull`, `img-build-secrets`, and `wandb-api`
  already exist; nothing to create per run.

## 2. Build the image

```bash
cd <repo root>
bash examples/fleet/launch/rl1/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char sha of that ref>
```

Builds run in-cluster and clone the ref from GitHub, so uncommitted local
changes never reach an image. Push before building. A build takes about
20 minutes; watch with the logs command the script prints. Rebuild only when
python under `examples/fleet/` changes; `rayjob.yaml.tmpl` and
`launch_fleet.sh` render at launch time from your local checkout.

## 3. Launch

```bash
# GLM-4.7-Flash on ade-bench (text tool use, H200 node, defaults)
JOB_NAME=<name> MODEL_NAME=glm4.7-flash \
bash examples/fleet/launch/rl1/launch_fleet.sh <image-sha> \
  registry-alpha.fleetai.me/library/ade-bench:latest normal

# Qwen3.8-27B on evaluation-benchmark (vision, B200 node)
JOB_NAME=<name> MODEL_NAME=qwen3.8-27b \
NODE_WORKLOAD=gpu-b200 INSTANCE_TYPE=p6-b200.48xlarge \
MAIN_MEM=1500Gi MAIN_MEM_LIM=1900Gi \
bash examples/fleet/launch/rl1/launch_fleet.sh <image-sha> \
  registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3 normal
```

Every run is a RayJob (head GPU pod with the env-container docker, plus
`NUM_NODES-1` GPU workers; `NUM_NODES=1` means zero workers). Gang admission
happens at tier-2 topology. Modes: `normal` trains (200 rollouts); `debug_minimal` proves the stack in
about 40 minutes (2-turn episodes, 2 steps); `rollout_only` runs full-length
episodes without training, for checking parse rates and rewards. Use
`debug_minimal` first on any new image, model, or node type.

Other env vars: `NUM_GPUS` (8), `TASK_LIMIT` (0 = whole taskset),
`ROLLOUT_BATCH` (8), `N_SAMPLES` (8), `MAX_TURNS` (32), `SCRIPT_EXTRA`
(extra launcher args, e.g. `--extra-args "--save-debug-rollout-data ..."`).

## 4. Node pools and queueing

| Pool | Nodes | Per-GPU memory | Fits |
|---|---|---|---|
| `gpu-h200` | 2x p5en.48xlarge (8x H200) | 141GB | GLM recipe; NOT the 27B at full context |
| `gpu-b200` | 2x p6-b200.48xlarge (8x B200 each) | 179GB | everything above plus the 27B at full context; 2-node gangs |

Jobs are admitted by Kueue (queue `training-lq`); a job stays `suspend: true`
until quota frees. `kubectl describe workload <name>` shows why a job is
pending; "insufficient unused quota" names the current occupant's pool. The
inventory command:

```bash
kubectl --context fleet-training-rl1-us-east-1 get nodes \
  -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,POOL:.metadata.labels.workload,GPUS:.status.allocatable.nvidia\.com/gpu'
```

Adding a GPU pool needs three things or the node is unusable: the
gpu-operator daemonset must tolerate the new `workload=<pool>` taint (else
0 GPUs register), the node needs the synthetic topology labels
(`topology.nebius.com/{gpu-cluster-id,tier-1,tier-2}`), and a Kueue flavor
must target the pool. The Kueue config is ArgoCD-owned from
`fleet-ai/theseus` `k8s/training/clusters/<cluster>/apps/kueue-config.yaml`;
kubectl edits get reverted. Precedents: theseus PR #25271 (h200), #26799 (b200).

## 5. Monitor

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/<name>   # submitter relays the driver
# persistent copy: /mnt/sfs/miles-fleet/<name>/driver.log  (append-only across
# relaunches of the same JOB_NAME: check timestamps before blaming a line)
# metrics: https://wandb.ai/thefleet/miles-run_fleet, group = JOB_NAME
```

Within the first cycle, verify: (1) engines pass health checks and episodes
generate (`POST /generate` traffic); (2) the first train step logs sane
values (loss near 0 with real grad_norm is CORRECT for single-epoch GRPO);
(3) engines resume after the train step; (4) at rollout 20, checkpoints
appear in `/mnt/sfs/miles-fleet/<name>/checkpoints`.

Timing at full context on one node (sync colocated): about 2h generation +
10 min train per rollout. Long silences between train metrics are normal;
judge liveness by generation traffic, not train logs.

Benign log noise: router health-check `TimedOut` spam while engines sleep
during the train phase; `ppo_kl=0` / `pg_clipfrac=0` / tiny loss (one-epoch
math); a wandb teardown traceback at exit. Real trouble: `writing off as
ABORTED` in volume, `torch_memory_saver ... out of memory`, `Xid`,
`no_available_workers` repeating, or a 45+ minute stretch with no generation
traffic while the job says Running (engine death without respawn is a known
miles bug; the job will not self-heal).

## 6. Checkpoints, resume, artifacts

Checkpoints save asynchronously every 20 rollouts to
`/mnt/sfs/miles-fleet/<JOB_NAME>/checkpoints`. Relaunching with the same
`JOB_NAME` resumes from the latest one. Rollout dumps (from
`--save-debug-rollout-data`) and the driver log live in the same directory.
SFS survives job deletion; nothing needs copying out before killing a job.

## 7. Failure triage

| Signature in the log | Cause | Fix |
|---|---|---|
| `docker login ... 401 Unauthorized` at boot | Fleet credential expired | `flt auth login registry-alpha.fleetai.me`, relaunch |
| `flt: registry API: status 502` or pull 502s at boot | registry-alpha outage (observed 4 to 20+ min) | boot retries 10x30s on every pull; if it still dies, wait for recovery and relaunch |
| `Chat template rendering failed: No user query found in messages`, every episode ABORTED | stock Qwen template raises on the tool-result render | recipe passes the fixed template via `--chat-template-path`; do not remove it |
| `FlashAttention v3 Backend requires SM>=80 and SM<=90` | fa3 engine backend on a Blackwell node | recipe uses `triton` on B200; keep it |
| `Only {trtllm_mha, fa4, triton} ... on Blackwell GPUs for hybrid GDN` | wrong engine backend for this family on B200 | same as above |
| `torch.OutOfMemoryError` in `loss.backward` or a Triton kernel | train step does not fit: longest sample x ~2.3GB/1K tokens + ~65GB fixed | run on B200 (179GB), or cap `max_context_len` in the recipe; never "fix" with `--fsdp-cpu-offload` (next row) |
| `torch_memory_saver ... func=resume ... out of memory`, all engines die after train step | `--fsdp-cpu-offload` with `--colocate`: trainer cannot release the GPU between phases | remove the flag (upstream miles bug, reproducible with pure miles + 32K packed micro-batches) |
| `ray.exceptions.OutOfMemoryError ... node running low on memory` | host RAM: `offload_train` parks ~1.15TB (8 x 142GB) during rollouts | raise `MAIN_MEM_LIM` (1900Gi on the 2TB nodes) |
| Job pending forever, workload says quota needed | another run holds the pool | `kubectl get workloads` to see the occupant; wait or coordinate |
| Rewards all zero plus reset warnings | not the resets: `env.reset()` failing is a platform no-op | look for parse failures, template errors, or verifier outages instead |

## 8. Known-good configurations

| Model | Node | Backend | Attention (engine/train) | Context | Validated |
|---|---|---|---|---|---|
| GLM-4.7-Flash | H200 | Megatron TP4 | fa3 default / flash | 30720 | ade-bench, 79+ rollouts |
| Qwen3.8-27B | B200 | FSDP | triton / sdpa | 30720 | evalbench, full cycles, reward 0.17 to 0.31 per 64-sample rollout |
