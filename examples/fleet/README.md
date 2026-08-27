# miles on rl1

How to train a model on Fleet tasks using the rl1 cluster. You write one
JSON file describing the run and submit it with one command; everything else
happens on the cluster.

## 1. How it works

```
you ──run JSON──▶ submit_run.py ──▶ cluster queue ──▶ GPU machine(s)
                                                          │
                          /mnt/sfs/miles-fleet/<name>/ ◀──┘
                          (log, checkpoints, dumps)
```

Submitting creates a RayJob (a Kubernetes object that KubeRay, the cluster's
Ray operator, turns into pods). Three kinds of pods come up:

- a **head** pod on a GPU machine. It runs the setup and the training
  driver, and it also runs the task environments as docker containers.
- **worker** pods, one per additional GPU machine, when the run uses more
  than one. They only contribute their GPUs.
- a **submitter** pod on a small machine. It kicks off the head's work and
  relays its log output.

The cluster queue (Kueue) holds the whole set back until enough GPUs are
free, then starts all pods at once. On the head, setup runs first: pull the
taskset, turn it into training data, download the docker images the tasks
need, and download the model onto the shared filesystem (a one-time cost;
later runs reuse it). Then your command runs. Everything under `/mnt/sfs`
survives after the run; everything else on the pods is wiped.

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get rayjob <name> -w
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs delete rayjob <name>
```

## 2. The run JSON

Working examples: [`launch/rl1/examples/`](launch/rl1/examples/).

```json
{
  "name": "miles-vl-qwen38-01",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer:a871b291",
  "command": "bash examples/fleet/launch/rl1/run.sh --model-name qwen3.8-27b --mode normal --num-nodes 1 --num-gpus-per-node 8 --max-turns 32",
  "workers": 1,
  "gpus_per_worker": 8,
  "pool": "gpu-b200",
  "env": {
    "TASKSET_REF": "registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3",
    "TASK_LIMIT": "64"
  },
  "secrets": ["wandb-api"]
}
```

| Field | What it is |
|---|---|
| `name` | names everything about the run: the job, the folder on `/mnt/sfs`, the WandB group. Lowercase letters, digits, dashes. |
| `image` | which trainer image to run (from step 4) |
| `command` | what to run on the head. `run.sh` does the setup and then starts training with the arguments you give it. One rule: no apostrophes. |
| `workers` | how many 8-GPU machines the run uses. 1 for a single machine, 2 to span two. |
| `gpus_per_worker` | GPUs per machine, normally 8 |
| `pool` | which machines: `gpu-b200` (default, 179GB per GPU) or `gpu-h200` (141GB per GPU). The 27B model at full context only fits the B200s. |
| `env` | environment variables handed to every pod. `run.sh` reads `TASKSET_REF` (which taskset) and `TASK_LIMIT` (how many tasks to sample; "0" means all). `RUN_ID` is filled in from `name` automatically. |
| `secrets` | names of cluster secrets whose contents become environment variables. `wandb-api` is always added. Your Fleet login is packaged into a fresh secret at submit time. |

## 3. One-time setup

- kubeconfig for `fleet-training-rl1-us-east-1`.
- A valid Fleet login: check with `flt auth status`, renew with
  `flt auth login registry-alpha.fleetai.me`. The login expires after a few
  days; submitting with an expired one makes the run die early with a 401.
- For tasksets whose data lives on s3 (evaluation-benchmark yes, ade-bench
  no): export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` before
  submitting.

## 4. Build the image

```bash
bash examples/fleet/launch/rl1/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char sha of that ref>
```

The build runs on the cluster and clones the branch from GitHub, so push
first — local uncommitted changes never make it into an image. Takes about
20 minutes. You only need a new image when python under `examples/fleet/`
changes.

## 5. Submit a run

```bash
./examples/fleet/launch/rl1/submit_run.py examples/fleet/launch/rl1/examples/tool-use-qwen38.json
./examples/fleet/launch/rl1/submit_run.py examples/fleet/launch/rl1/examples/vision-qwen38-27b.json
./examples/fleet/launch/rl1/submit_run.py my-run.json --dry-run   # print what would be applied, apply nothing
```

On any new image, model, or machine type, run with `--mode debug_minimal`
in the command first: it does two tiny training rounds end to end in about
40 minutes and catches almost every problem a long run would hit.

## 6. Watch a run

```bash
kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/<name>
# the same log persists at /mnt/sfs/miles-fleet/<name>/driver.log
#   (it is appended across resubmits of the same name — check timestamps)
# metrics: https://wandb.ai/thefleet/miles-run_fleet, group = <name>
# checkpoints: /mnt/sfs/miles-fleet/<name>/checkpoints, written every 20 rounds;
#   resubmitting the same name resumes from the latest one
```

In the first training round, check four things in order: task episodes are
being generated (`POST /generate` lines in the log); the first training step
prints numbers (a loss near zero next to a nonzero grad norm is expected
here, not a bug); generation starts again after the training step; and at
round 20 a checkpoint shows up on `/mnt/sfs`. One full round at maximum
context is roughly 2 hours of generation plus 10 minutes of training, so a
quiet training log usually means generation is busy, not that the run is
stuck.

If the job never starts: `kubectl describe workload <name>` says what it is
waiting for and which run currently holds the machines.

Failures we have actually hit, and what fixed them:

| What the log says | What it means | What to do |
|---|---|---|
| `docker login ... 401 Unauthorized` right after start | your Fleet login expired | `flt auth login`, resubmit |
| `registry API: status 502`, repeated | the Fleet registry is down (outages of 4 to 20+ minutes have happened) | setup retries for 5 minutes on its own; if the run still dies, resubmit after the registry recovers |
| `No user query found in messages`, every episode thrown away | the model's built-in chat template rejects how tool results are replayed | the launcher already substitutes a fixed template; do not remove `--chat-template-path` |
| `FlashAttention v3 Backend requires SM>=80 and SM<=90` | an attention library that only supports older GPUs, on a B200 | the launcher already uses `triton` on B200; keep it |
| `torch.OutOfMemoryError` in `loss.backward` | the training step needs more GPU memory than the machine has | run the 27B on the B200 pool; do not work around it with `--fsdp-cpu-offload` (next row) |
| `torch_memory_saver ... resume ... out of memory`, all engines die right after a training step | `--fsdp-cpu-offload` together with colocation: the trainer never gives the GPU back | remove the flag (bug in miles, reported upstream) |
| `ray ... node running low on memory` | the machine's RAM (not GPU memory) filled up; the 27B parks ~1.15TB there between training steps | pod memory is already sized per pool in `submit_run.py` |
| every reward is zero, log full of reset warnings | the reset warnings are a known harmless platform issue, not the cause | look for parse failures or template errors instead |
