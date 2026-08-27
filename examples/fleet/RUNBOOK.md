# Runbook: Fleet training with miles on rl1

Runs on the rl1 cluster (EKS `fleet-training-rl1-us-east-1`) submitted from a
RunPayload JSON — the same contract as the SkyRL-Fleet rl1 integration and
the future runs API; `submit_run.py` is the API stand-in. What the training
code does is in [README.md](README.md).

## 1. How it works

```
researcher ──payload JSON──▶ submit_run.py ──RayJob──▶ Kueue (training-lq) ──▶ B200/H200 node(s)
                                                                                   │
                                              /mnt/sfs/miles-fleet/<name>/ ◀──────┘
                                              (driver.log, checkpoints, dumps)
```

Every run is one RayJob: a head GPU pod (which also hosts the environment
containers in a docker sidecar), `workers - 1` identical GPU worker pods, and
a small submitter on infra. Kueue gang-admits the pod set against the node
pool's quota; the job stays `suspend: true` until capacity frees, then the
entrypoint runs on the head: pull the taskset, build train/eval JSONL,
pre-pull the environment images, download the model to SFS once, then start
training on the KubeRay cluster. Only `/mnt/sfs` persists; the pods' own
disks are wiped when the job ends.

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get rayjob <name> -w
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs delete rayjob <name>
```

## 2. The run payload

Two validated examples live in [`launch/rl1/examples/`](launch/rl1/examples/):

```json
{
  "name": "miles-vl-qwen38-01",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer:6d6baa6b",
  "workers": 1,
  "gpus_per_worker": 8,
  "env": {
    "MODEL_NAME": "qwen3.8-27b",
    "TASKSET_REF": "registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3",
    "MODE": "normal",
    "TASK_LIMIT": "64",
    "MAX_TURNS": "32"
  },
  "secrets": ["wandb-api"]
}
```

| Field | Meaning | Constraints |
|---|---|---|
| `name` | run, RayJob, SFS dir, and WandB group name | DNS-safe label |
| `image` | trainer image from the build step | |
| `workers` | GPU pods, head included | >= 1 |
| `gpus_per_worker` | GPUs per pod | 1..8 |
| `env.MODEL_NAME` | recipe row in `launch/run_fleet.py` | `glm4.7-flash` or `qwen3.8-27b` |
| `env.TASKSET_REF` | v2-registry taskset | |
| `env.MODE` | `normal`, `debug_minimal`, `rollout_only` | |
| `env.TASK_LIMIT` | task sample cap | "0" = whole taskset |
| `env.MAX_TURNS` | episode turn cap | |
| `secrets` | extra pre-created secrets mounted as env | `wandb-api` always included |

There is no `command` field: the training command is composed from the
MODEL_NAME recipe row, and node placement plus memory sizing are model facts
the submitter derives (GLM → H200 pool; the 27B → B200 pool, 179GB/GPU,
1900Gi host limit for `offload_train`).

## 3. One-time setup

- kubeconfig for `fleet-training-rl1-us-east-1` (every command passes
  `--context` explicitly).
- Fleet registry login: `flt auth status`; re-login with
  `flt auth login registry-alpha.fleetai.me` when expired. The submitter
  copies the credential into a per-run secret, so an expired login fails the
  boot at `docker login`.
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` exported before submitting,
  only for tasksets with s3 seed data (evaluation-benchmark yes, ade-bench
  no); the submitter includes them in the run secret when set.
- The cluster secrets `ghcr-pull`, `img-build-secrets`, `wandb-api` exist.

## 4. Build the image

```bash
bash examples/fleet/launch/rl1/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char sha of that ref>
```

Builds run in-cluster and clone the ref from GitHub: push first, local
changes never reach an image. About 20 minutes. Rebuild only when python
under `examples/fleet/` changes; the RayJob template and `submit_run.py`
run from your local checkout.

## 5. Submit a run

```bash
./examples/fleet/launch/rl1/submit_run.py examples/fleet/launch/rl1/examples/tool-use-glm47.json
./examples/fleet/launch/rl1/submit_run.py examples/fleet/launch/rl1/examples/vision-qwen38-27b.json
./examples/fleet/launch/rl1/submit_run.py my-run.json --dry-run   # print the RayJob, apply nothing
```

Use `MODE: debug_minimal` first on any new image, model, or node type: it
proves the whole stack (episodes, train steps, engine handoffs) in about
40 minutes. `workers: 2` submits a 2-node gang (tier-2 topology; inter-node
NCCL currently rides TCP — no EFA resource on the nodes yet).

## 6. Monitor

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/<name>  # submitter relays the driver
# persistent copy: /mnt/sfs/miles-fleet/<name>/driver.log  (append-only across
# resubmits of the same name: check timestamps before blaming a line)
# metrics: https://wandb.ai/thefleet/miles-run_fleet, group = <name>
# checkpoints: /mnt/sfs/miles-fleet/<name>/checkpoints, async save every 20
# rollouts; resubmitting the same name resumes from the latest one
```

First-cycle checklist: (1) engines pass health checks and episodes generate
(`POST /generate` traffic); (2) the first train step logs sane values (loss
near 0 with real grad_norm is CORRECT for single-epoch GRPO); (3) engines
resume after the train step; (4) at rollout 20, checkpoints appear on SFS.

Timing at full context (sync colocated): about 2h generation + 10 min train
per rollout. Judge liveness by generation traffic, not train-log frequency.
Benign noise: router health-check `TimedOut` spam while engines sleep;
`ppo_kl=0` / `pg_clipfrac=0` / tiny loss; a wandb teardown traceback at exit.

Queueing: `kubectl describe workload <name>` says why a job is pending;
"insufficient unused quota" names the pool's current occupant. Pool
inventory:

```bash
kubectl --context fleet-training-rl1-us-east-1 get nodes \
  -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,POOL:.metadata.labels.workload,GPUS:.status.allocatable.nvidia\.com/gpu'
```

| Pool | Nodes | Per-GPU memory | Fits |
|---|---|---|---|
| `gpu-h200` | 2x p5en.48xlarge (8x H200) | 141GB | GLM recipe; NOT the 27B at full context |
| `gpu-b200` | 2x p6-b200.48xlarge (8x B200 each) | 179GB | everything, incl. the 27B at full context and 2-node gangs |

Adding a GPU pool needs three things or the node is unusable: a gpu-operator
toleration for the new `workload=<pool>` taint (else 0 GPUs register), the
synthetic topology labels on the nodes, and a Kueue flavor. Kueue config is
ArgoCD-owned from theseus `k8s/training/clusters/<cluster>/apps/`; kubectl
edits get reverted (precedents: theseus PR #25271, #26799, #26961).

## 7. Failure triage

Every row happened at least once; the fix is the one that worked.

| Signature in the log | Cause | Fix |
|---|---|---|
| `docker login ... 401 Unauthorized` at boot | Fleet credential expired | `flt auth login registry-alpha.fleetai.me`, resubmit |
| `flt: registry API: status 502` or pull 502s at boot | registry-alpha outage (observed 4 to 20+ min) | boot retries 10x30s on every pull; if it still dies, wait for recovery and resubmit |
| `Chat template rendering failed: No user query found in messages`, every episode ABORTED | stock Qwen template raises on the tool-result render | recipe passes the fixed template via `--chat-template-path`; do not remove it |
| `FlashAttention v3 Backend requires SM>=80 and SM<=90` | fa3 engine backend on a Blackwell node | recipe uses `triton` on B200; keep it |
| `Only {trtllm_mha, fa4, triton} ... on Blackwell GPUs for hybrid GDN` | wrong engine backend for this family on B200 | same as above |
| `torch.OutOfMemoryError` in `loss.backward` or a Triton kernel | train step does not fit: longest sample x ~2.3GB/1K tokens + ~65GB fixed | run on B200 (179GB), or cap `max_context_len` in the recipe; never "fix" with `--fsdp-cpu-offload` (next row) |
| `torch_memory_saver ... func=resume ... out of memory`, all engines die after train step | `--fsdp-cpu-offload` with `--colocate`: trainer cannot release the GPU between phases | remove the flag (upstream miles bug, reproducible with pure miles + 32K packed micro-batches) |
| `ray.exceptions.OutOfMemoryError ... node running low on memory` | host RAM: `offload_train` parks ~1.15TB (8 x 142GB) during rollouts | the submitter sizes memory from the model row; if it recurs, raise the row's limit |
| Job pending forever, workload says quota needed | another run holds the pool | `kubectl get workloads` to see the occupant; wait or coordinate |
| Rewards all zero plus reset warnings | not the resets: `env.reset()` failing is a platform no-op | look for parse failures, template errors, or verifier outages instead |

## 8. Known-good configurations

| Model | Node | Backend | Attention (engine/train) | Context | Validated |
|---|---|---|---|---|---|
| GLM-4.7-Flash | H200 | Megatron TP4 | fa3 default / flash | 30720 | ade-bench, 79+ rollouts |
| Qwen3.8-27B | B200 | FSDP | triton / sdpa | 30720 | evalbench, full cycles, reward 0.17 to 0.31 per 64-sample rollout |
