# miles on the Fleet training cluster

Train a model on Fleet tasks: write one JSON file, submit it with one
command. The cluster is `fleetai-training` on Nebius: 24 machines, 8 B300
GPUs each (268GB per GPU), InfiniBand between machines.

## 1. How it works

```
you ──run JSON──▶ submit_run.py ──▶ cluster queue ──▶ GPU machine(s)
                                                          │
                          /mnt/sfs/miles-fleet/<name>/ ◀──┘
                          (log, checkpoints, dumps)
```

`submit_run.py` fills in for the jobs API until it exists: it validates the
JSON, packages your Fleet login into a cluster secret, and creates the
Kubernetes job. When the API ships, the JSON stays the same and the image
runs unchanged; only the submitting moves from this script to an API call.

A run is three kinds of pods: a **head** on a GPU machine (runs setup, the
training driver, and the task environments as docker containers),
**workers** on additional GPU machines (GPUs only), and a small
**submitter** that relays logs. The queue (Kueue) starts them all at once
when enough GPUs are free, and everything is torn down the moment the run
ends. Setup on the head pulls the taskset, builds the training data,
downloads the task docker images and the model (the model lands on the
shared filesystem once and is reused). Then your command runs. Everything
under `/mnt/sfs` survives the run.

```bash
kubectl --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 -n fleet-train-jobs get rayjob <name> -w
kubectl --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 -n fleet-train-jobs delete rayjob <name>
```

## 2. The code

Everything hangs off one miles hook: the launcher passes
`--custom-generate-function-path examples.fleet.rollout.generate`, and miles
calls that function once per sample instead of its own generation. Each call
runs one full episode against a Fleet task and returns Samples with tokens,
loss masks, and reward already set. That one hook is why everything lives
under `examples/fleet/` and miles itself is unmodified.

| File | What it does |
|---|---|
| `agent.py` | the agent loop: sample a turn, run the tool call, feed the observation back, close steps, submit, grade. Messages only; no tokens. |
| `recording.py` | turns the loop's messages into training data: token assembly, loss masks, image tensors, the Samples miles trains on |
| `rollout.py` | the `--custom-generate-function-path` entry point wiring the two together, plus the `--fleet-*` flags |
| `session.py` | one attempt against one Fleet task over the fleet-runtime SDK: open, call_tool, grade, close |
| `parser.py` | reads the model's tool calls out of its text |
| `prepare_dataset.py` | taskset to train/eval JSONL |
| `launch/run_fleet.py` | the training launcher; recipe table keyed by `--model-name` |
| `launch/run.sh` | on-cluster setup (credentials, taskset, images, model), then runs the launcher |
| `launch/submit_run.py` | run JSON to RayJob |
| `launch/examples/` | working run JSONs |

## 3. The run JSON

A run is five things (a name, an image, a command, how many GPUs, and env
variables) plus secrets. Working examples: [`launch/examples/`](launch/examples/).

```json
{
  "name": "miles-vl-qwen38-2node-01",
  "image": "ghcr.io/fleet-ai/miles-fleet/trainer:5f74853a",
  "command": "bash examples/fleet/launch/run.sh --model-name qwen3.8-27b --mode normal --num-nodes 2 --num-gpus-per-node 8 --max-turns 32 --max-concurrent-envs 16",
  "workers": 2,
  "gpus_per_worker": 8,
  "env": {
    "TASKSET_REF": "registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3",
    "TASK_LIMIT": "64"
  },
  "secrets": ["wandb-api"]
}
```

| Field | What it is |
|---|---|
| `name` | names the job, the folder on `/mnt/sfs`, and the WandB group. Lowercase letters, digits, dashes. |
| `image` | which trainer image to run (section 6) |
| `command` | what to run on the head. `run.sh` does setup, then starts training with these arguments. One rule: no apostrophes. |
| `workers` | how many 8-GPU machines |
| `gpus_per_worker` | GPUs per machine, normally 8 |
| `env` | environment variables for every pod. `run.sh` reads `TASKSET_REF` and `TASK_LIMIT` ("0" = all tasks). `RUN_ID` defaults to `name`. |
| `secrets` | cluster secrets exposed as environment variables. `wandb-api` is always added; your Fleet login becomes a fresh secret at submit time. |

## 4. One-time setup

- kubeconfig:
  `nebius mk8s cluster get-credentials --id mk8scluster-e04zw4ye1k7wczqdw6 --external`
- Fleet login: check `flt auth status`, renew with
  `flt auth login registry-alpha.fleetai.me`. It expires after a few days;
  an expired login kills the run at startup with a 401.
- For tasksets with s3 data (evaluation-benchmark yes, ade-bench no):
  export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

## 5. Pre-stage model weights (optional, saves GPU hours)

```bash
./examples/fleet/launch/prestage.py zai-org/GLM-5.3-Flash-BF16
```

Weights live on the shared filesystem and download only once; but by
default that download happens inside the training job, with the job's GPUs
sitting idle for its whole duration (1-2 hours for a 300B model). This
command does the same download as a small job on the CPU machines instead.
Run it when you plan a run with a model the cluster has not seen; a
training job submitted afterwards finds the weights ready. It is safe to
skip and safe to run concurrently with a training job: both sides take the
same lock.

## 6. Build the image

```bash
bash examples/fleet/launch/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char sha of that ref>
```

The build clones the branch from GitHub, so push first. About 20 minutes.
A new image is only needed when python under `examples/fleet/` changes.

The Dockerfile builds on a fixed version of the upstream miles image, not
the `dev-cu12` tag, because upstream overwrites that tag and one overwrite
broke the inference engines. If you change the pinned version, run
`--mode debug_minimal` on the new image before using it.

## 7. Submit a run

The two runs we train today, exactly as launched:

```bash
# text tool use: Qwen3.8-27B on ade-bench, 1 machine
./examples/fleet/launch/submit_run.py examples/fleet/launch/examples/tool-use-qwen38.json

# vision computer use: Qwen3.8-27B on evaluation-benchmark, 2 machines (export AWS keys first)
./examples/fleet/launch/submit_run.py examples/fleet/launch/examples/vision-qwen38-27b-2node.json

./examples/fleet/launch/submit_run.py my-run.json --dry-run   # print, apply nothing
```

On any new image, model, or machine type, first run with
`--mode debug_minimal` in the command: two tiny training rounds end to end,
about 40 minutes, catches almost everything a long run would hit.

## 8. Watch a run

```bash
kubectl --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 -n fleet-train-jobs logs -f job/<name>
# persistent copy: /mnt/sfs/miles-fleet/<name>/driver.log
# metrics: https://wandb.ai/thefleet/miles-run_fleet, group = <name>
# checkpoints: /mnt/sfs/miles-fleet/<name>/checkpoints, every 20 rounds;
#   resubmitting the same name resumes from the latest one
```

First round, check in order: episodes generating (`POST /generate` in the
log), first training step prints numbers (loss near zero with nonzero grad
norm is expected), generation resumes after the step, checkpoint at round
20. A round at full context is roughly 2h generation + 10min training, so a
quiet log usually means generation is busy. The log fills with engine
chatter over time; read progress from WandB.

If the job never starts, `kubectl describe workload <name>` says what it is
waiting for. If a pod sits Pending on a machine that looks free, something
outside the queue (a dev pod, an inference job) is holding it; delete the
run and resubmit so the queue picks another machine.

## 9. What has been validated

Smokes on image `trainer:5f74853a` (current); the production runs below launched on its predecessor `97ddfb89`.

| Model | Taskset | Machines | Result |
|---|---|---|---|
| Qwen3.8-27B | ade-bench (text) | 1 | debug_minimal passed; production run launched 2026-08-28 (`miles-tu-prod-01`) |
| Qwen3.8-27B | evaluation-benchmark (vision) | 2 | debug_minimal passed; traffic between machines confirmed on InfiniBand (`via NET/IB/GDRDMA`); production run launched 2026-08-28 (`miles-vl-prod-01`) |
