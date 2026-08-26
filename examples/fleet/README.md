# Fleet tasksets on miles

Trains models with GRPO on [Fleet](https://fleetai.com) v2 tasksets: each
episode runs in a real docker environment with tools, and the platform's own
verifiers produce the reward. Plugs into miles as a custom generate function;
no miles source changes.

| File | What it does |
|---|---|
| `rollout.py` | The episode loop miles calls: prompt, tool calls, screenshots, reward |
| `session.py` | Talks to the Fleet SDK and docker: boot env, run tools, grade |
| `parser.py` | Reads tool calls out of model text (Qwen, GLM, Kimi formats) |
| `content.py` | Turns tool results into message text and image payloads |
| `prepare_dataset.py` | Taskset registry content into train/eval JSONL |
| `templates/` | Chat templates vendored ahead of upstream (see Pins below) |
| `launch/run_fleet.py` | Training launcher; per-model config lives in its recipe table |
| `launch/rl1/` | Image build and job submission for the rl1 cluster |

## How an episode works

The model gets the task instructions and the environment's tool schemas. Each
turn it generates text, the parser extracts one tool call, the call runs in
the env container, and the result comes back as the next message. For vision
models, screenshots in tool results become image tokens: the processor's
patch tensors go to the inference engines with each request and into the
training batch (`multimodal_train_inputs`), with the same loss-mask
arithmetic as text. The episode ends when the model calls `fleet_submit`,
runs out of turns, or fills its context. Then the platform grades the
attempt: pass gives reward 1.0, fail gives 0.0 (or the average verifier score
with `--partial-reward`). Sampled tokens train with loss mask 1; tool results
are masked out. If an environment fails to boot or crashes, the episode is
written off and its batch group is resampled; the run keeps going.

## Run it on rl1

```bash
# 1. build the trainer image (in-cluster, pushes to ghcr)
bash examples/fleet/launch/rl1/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char-sha>

# 2. GLM-4.7-Flash on ade-bench (text tool use, H200 node)
JOB_NAME=my-run MODEL_NAME=glm4.7-flash \
bash examples/fleet/launch/rl1/launch_fleet.sh <sha> \
  registry-alpha.fleetai.me/library/ade-bench:latest normal

# 3. Qwen3.8-27B on evaluation-benchmark (GUI/vision, B200 node)
#    AWS creds are needed: this taskset's seeds live on s3
JOB_NAME=my-run MODEL_NAME=qwen3.8-27b \
NODE_WORKLOAD=gpu-b200 INSTANCE_TYPE=p6-b200.48xlarge \
MAIN_MEM=1500Gi MAIN_MEM_LIM=1900Gi \
bash examples/fleet/launch/rl1/launch_fleet.sh <sha> \
  registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3 normal
```

Modes: `normal` (train), `debug_minimal` (2-turn episodes, 2 steps, ~40 min
end to end), `rollout_only` (full-length episodes, no training; check parse
rate and rewards from the dumps). Other env vars: `NUM_GPUS` (default 8),
`TASK_LIMIT` (0 = whole taskset), `ROLLOUT_BATCH`, `N_SAMPLES`, `MAX_TURNS`.
Jobs go through the Kueue queue `training-lq`; unqueued GPU jobs on this
cluster get preempted. Check `flt auth status` before launching: expired
Fleet credentials fail the boot at `docker login`.

Watch a run:

```bash
kubectl -n fleet-train-jobs logs -f job/<name> -c miles     # live
# persistent copy: /mnt/sfs/miles-fleet/<name>/driver.log
# metrics: https://wandb.ai/thefleet/miles-run_fleet (group = JOB_NAME)
# checkpoints: /mnt/sfs/miles-fleet/<name>/checkpoints; relaunching with the
# same JOB_NAME resumes from the latest one
```

## Models

One recipe row per model in `launch/run_fleet.py`; the row holds the HF id,
backend, engine flags, tokenizer family, and memory-derived length caps.

| Recipe | Backend | Vision | Node | Notes |
|---|---|---|---|---|
| `glm4.7-flash` | Megatron TP4 | no | H200 | EAGLE speculative decoding on the engines |
| `qwen3.8-27b` | FSDP | yes | B200 | full 30720 context; needs the 179GB/GPU (the ~134GB/rank train step does not fit an H200) |

Runs:

| Model | Taskset | Run |
|---|---|---|
| GLM-4.7-Flash | ade-bench | [miles-train-ade](https://wandb.ai/thefleet/miles-run_fleet_glm47_flash/groups/miles-train-ade) |
| Qwen3.8-27B | evaluation-benchmark | [qwen-evalbench-b200](https://wandb.ai/thefleet/miles-run_fleet/groups/qwen-evalbench-b200) |

## Tests

```bash
pytest examples/fleet/tests -q                # no GPU, no network, no docker
pytest examples/fleet/tests -q -m network     # real GLM + Qwen3.8 tokenizers
FLEET_FLT=$(which flt) pytest examples/fleet/tests/integration -q -m docker  # real containers
```

## Pins and known limits

- `fleet-runtime` and the `flt` binary must come from the same
  `fleet-ai/platform` commit (currently `72e656c948ab`).
- `templates/qwen3.8_fixed.jinja` is vendored from miles PR #2760: Qwen3.8's
  real chat template (reasoning-effort system prefix included) with only the
  no-user-query guard removed. When that PR merges and the base pin moves,
  switch the recipe to `--fleet-tito-model qwen38` and delete the copy.
- `--use-rollout-routing-replay` stays off: miles overwrites instead of
  concatenating per-turn expert-routing data, so multi-turn replay would be
  wrong (upstream bug).
- Do not combine `--fsdp-cpu-offload` with `--colocate`: it disables the
  trainer's between-phase release, the engines then cannot re-acquire their
  memory after the first train step, and the run dies at engine resume.
  Reproducible in pure miles; upstream fix pending.
- Attention backends are per-architecture and set by the recipe row: fa3 is
  Hopper-only; on Blackwell the engines use `triton` (sglang allows only
  `trtllm_mha`/`fa4`/`triton` for this hybrid-GDN family) and the trainer
  uses `sdpa`.
- Host RAM: with `offload_train` (the default), the 27B trainer parks
  ~1.15TB on the host during rollouts; the pod memory limit must cover it
  (hence `MAIN_MEM_LIM=1900Gi` on the 2TB nodes).
