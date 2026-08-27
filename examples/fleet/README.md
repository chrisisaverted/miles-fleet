# miles on rl1

## 1. How it works

```
researcher ──payload JSON──▶ submit_run.py ──RayJob──▶ Kueue (training-lq) ──▶ GPU node(s)
                                                                                   │
                                              /mnt/sfs/miles-fleet/<name>/ ◀──────┘
                                              (driver.log, checkpoints, dumps)
```

A run is one RayJob: a head GPU pod (also hosts the environment containers
in a docker sidecar), `workers - 1` GPU worker pods, and a small submitter
on infra. Kueue holds the job suspended until the node pool has capacity,
then the entrypoint runs on the head: pull the taskset, build train/eval
JSONL, pre-pull the environment images, download the model to SFS once,
then exec the payload's command. Only `/mnt/sfs` persists; pod disks are
wiped when the job ends.

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get rayjob <name> -w
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs delete rayjob <name>
```

## 2. The run payload

Validated examples: [`launch/rl1/examples/`](launch/rl1/examples/).

```json
{
  "name": "miles-vl-qwen38-01",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer:6d6baa6b",
  "command": "python examples/fleet/launch/run_fleet.py --skip-prepare --model-dir /mnt/sfs/miles-fleet/models --output-dir /mnt/sfs/miles-fleet --num-gpus-per-node 8 --rollout-batch-size 8 --n-samples-per-prompt 8 --max-concurrent-envs 8 --model-name qwen3.8-27b --mode normal --num-nodes 1 --run-id miles-vl-qwen38-01 --dataset-dir /mnt/sfs/miles-fleet/miles-vl-qwen38-01/data --data-dir /mnt/sfs/miles-fleet/miles-vl-qwen38-01 --max-turns 32",
  "workers": 1,
  "gpus_per_worker": 8,
  "env": {
    "MODEL_NAME": "qwen3.8-27b",
    "TASKSET_REF": "registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3",
    "MODE": "normal",
    "TASK_LIMIT": "64",
    "RUN_ID": "miles-vl-qwen38-01"
  },
  "secrets": ["wandb-api"]
}
```

| Field | Meaning | Constraints |
|---|---|---|
| `name` | run, RayJob, SFS dir, and WandB group name | DNS-safe label |
| `image` | trainer image from the build step | |
| `command` | training invocation, runs on the head after the boot phase | no apostrophes; must agree with env (`--mode`, `--model-name`, `--num-nodes`) |
| `workers` | GPU pods, head included | >= 1; equals the command's `--num-nodes` |
| `gpus_per_worker` | GPUs per pod | 1..8 |
| `env.MODEL_NAME` | recipe row in `launch/run_fleet.py`; selects node pool and memory sizing | `glm4.7-flash` (H200) or `qwen3.8-27b` (B200) |
| `env.TASKSET_REF` | taskset the boot phase pulls | |
| `env.MODE` | run type | `normal`, `debug_minimal`, `rollout_only` |
| `env.TASK_LIMIT` | task sample cap for the dataset build | "0" = whole taskset |
| `env.RUN_ID` | must equal `name` | |
| `secrets` | extra pre-created secrets mounted as env | `wandb-api` always included |

## 3. One-time setup

- kubeconfig for `fleet-training-rl1-us-east-1`.
- Fleet registry login: `flt auth status`; re-login with
  `flt auth login registry-alpha.fleetai.me` when expired (expired login
  fails the boot at `docker login` with 401).
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` exported before submitting,
  for tasksets with s3 seed data (evaluation-benchmark yes, ade-bench no).

## 4. Build the image

```bash
bash examples/fleet/launch/rl1/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char sha of that ref>
```

Builds run in-cluster from the pushed ref: push first, local changes never
reach an image. About 20 minutes. Rebuild only when python under
`examples/fleet/` changes.

## 5. Submit a run

```bash
./examples/fleet/launch/rl1/submit_run.py examples/fleet/launch/rl1/examples/tool-use-glm47.json
./examples/fleet/launch/rl1/submit_run.py examples/fleet/launch/rl1/examples/vision-qwen38-27b.json
./examples/fleet/launch/rl1/submit_run.py my-run.json --dry-run   # print the RayJob, apply nothing
```

Use `MODE: debug_minimal` first on any new image, model, or node type
(~40 min end to end). `workers: 2` submits a 2-node gang.

## 6. Monitor

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/<name>
# persistent copy: /mnt/sfs/miles-fleet/<name>/driver.log (append-only across resubmits)
# metrics: https://wandb.ai/thefleet/miles-run_fleet, group = <name>
# checkpoints: /mnt/sfs/miles-fleet/<name>/checkpoints, saved every 20 rollouts;
# resubmitting the same name resumes from the latest one
```

First cycle, verify in order: episodes generate (`POST /generate` traffic);
the first train step logs sane values (loss near 0 with real grad_norm is
correct for single-epoch GRPO); engines resume after the train step; at
rollout 20, checkpoints appear on SFS. One rollout at full context is about
2h generation + 10 min train; judge liveness by generation traffic.

If a job stays pending: `kubectl describe workload <name>` names the missing
quota and the pool's current occupant.

Failures seen so far, with the fix that worked:

| Signature in the log | Cause | Fix |
|---|---|---|
| `docker login ... 401 Unauthorized` at boot | Fleet credential expired | `flt auth login`, resubmit |
| `flt: registry API: status 502` or pull 502s | registry-alpha outage (4 to 20+ min observed) | boot retries 10x30s; if it still dies, resubmit after recovery |
| `No user query found in messages`, every episode ABORTED | stock Qwen template rejects the tool-result render | recipe passes the fixed template; do not remove `--chat-template-path` |
| `FlashAttention v3 Backend requires SM>=80 and SM<=90` | fa3 on a Blackwell node | recipe uses `triton` on B200 |
| `torch.OutOfMemoryError` in `loss.backward` | train step does not fit the GPU | run the 27B on B200; do not "fix" with `--fsdp-cpu-offload` (next row) |
| `torch_memory_saver ... resume ... out of memory`, engines die after a train step | `--fsdp-cpu-offload` with `--colocate` | remove the flag (upstream miles bug) |
| `ray ... OutOfMemoryError ... node running low on memory` | host RAM during rollouts (~1.15TB for the 27B) | memory is sized per model in `submit_run.py`; raise the row's limit |
| Rewards all zero plus reset warnings | `env.reset()` failing is a platform no-op, not the cause | look for parse failures or template errors instead |
