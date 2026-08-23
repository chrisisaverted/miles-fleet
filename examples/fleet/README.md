# Fleet tasksets on miles

Trains models with GRPO on [Fleet](https://fleetai.com) v2 tasksets: each
episode runs in a real docker environment with tools, and the platform's own
verifiers produce the reward. Plugs into miles as a custom generate function;
no miles source changes.

| File | What it does |
|---|---|
| `rollout.py` | The episode loop miles calls: prompt, tool calls, reward |
| `session.py` | Talks to the Fleet SDK and docker: boot env, run tools, grade |
| `parser.py` | Reads tool calls out of model text (Qwen, GLM, Kimi formats) |
| `content.py` | Turns tool results into message text |
| `prepare_dataset.py` | Taskset registry content into train/eval JSONL |
| `launch/run_fleet.py` | Training launcher; per-model config lives in its recipe table |
| `launch/rl1/` | Image build and job submission for the rl1 cluster |

## How an episode works

The model gets the task instructions and the environment's tool schemas. Each
turn it generates text, the parser extracts one tool call, the call runs in
the env container, and the result comes back as the next message. The episode
ends when the model calls `fleet_submit`, runs out of turns, or fills its
context. Then the platform grades the attempt: pass gives reward 1.0, fail
gives 0.0 (or the average verifier score with `--partial-reward`). Sampled
tokens train with loss mask 1; tool results are masked out. If an environment
fails to boot or crashes, the episode is written off and its batch group is
resampled; the run keeps going.

## Run it on rl1

```bash
# 1. build the trainer image (in-cluster, pushes to ghcr)
bash examples/fleet/launch/rl1/build_image.sh fleet-integration
# -> ghcr.io/fleet-ai/miles-fleet/trainer:<8-char-sha>

# 2. smoke: 2-turn episodes, 2 training steps, one checkpoint save (~30 min)
JOB_NAME=my-smoke bash examples/fleet/launch/rl1/launch_fleet.sh <sha> \
  registry-alpha.fleetai.me/library/ade-bench:latest debug_minimal

# 3. gate: full-length episodes, no training; check parse rate and rewards
JOB_NAME=my-gate bash examples/fleet/launch/rl1/launch_fleet.sh <sha> \
  registry-alpha.fleetai.me/library/ade-bench:latest rollout_only

# 4. train
JOB_NAME=my-run bash examples/fleet/launch/rl1/launch_fleet.sh <sha> \
  registry-alpha.fleetai.me/library/ade-bench:latest normal
```

Useful overrides (env vars): `MODEL_NAME` (default `glm4.7-flash`),
`NUM_GPUS` (default 8), `TASK_LIMIT` (0 = whole taskset), `ROLLOUT_BATCH`,
`N_SAMPLES`, `MAX_TURNS`. Jobs go through the Kueue queue `training-lq`;
unqueued GPU jobs on this cluster get preempted.

Watch a run:

```bash
kubectl -n fleet-train-jobs logs -f job/<name> -c miles     # live
# persistent copy: /mnt/sfs/miles-fleet/<name>/driver.log
# metrics: https://wandb.ai/thefleet/miles-run_fleet_glm47_flash
```

## Models

One recipe row per model in `launch/run_fleet.py`; the row holds the HF id,
Megatron settings, engine flags, and tokenizer family. Adding a model means
porting one row from its `scripts/run_*.py` recipe.

Runs so far (all GLM-4.7-Flash, 4x H200 on rl1):

| Taskset | Run | Outcome |
|---|---|---|
| ade-bench | [miles-train-ade](https://wandb.ai/thefleet/miles-run_fleet_glm47_flash/groups/miles-train-ade) | Training (200 rollouts planned); pass rate ~0.15-0.23, flat so far |
| evaluation-benchmark | [miles-train-evalbench](https://wandb.ai/thefleet/miles-run_fleet_glm47_flash/groups/miles-train-evalbench) | Stopped by design: the benchmark is computer-use (screenshot observations) and GLM-4.7-Flash is text-only, so the model ran blind. Pipeline proven end to end; resumes once vision support and a vision model land |
| ade-bench (gates) | [miles-smoke-ade / miles-rollout-ade](https://wandb.ai/thefleet/miles-run_fleet_glm47_flash) | debug_minimal smoke passed; rollout gate: 0.12 parse failures/episode, 20/32 episodes submitted, 6/32 passed |

## Tests

```bash
pytest examples/fleet/tests -q                # no GPU, no network, no docker
pytest examples/fleet/tests -q -m network     # real GLM tokenizer
FLEET_FLT=$(which flt) pytest examples/fleet/tests/integration -q -m docker  # real containers
```

## Pins and known limits

- `fleet-runtime` and the `flt` binary must come from the same
  `fleet-ai/platform` commit (currently `72e656c948ab`).
- `--use-rollout-routing-replay` stays off: miles overwrites instead of
  concatenating per-turn expert-routing data, so multi-turn replay would be
  wrong (upstream bug).
- Vision (screenshots into training) is not implemented yet; text only.
