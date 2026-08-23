# Fleet v2 tasksets on miles

Trains on [Fleet](https://fleetai.com) v2 platform tasksets: docker-backed
environments with MCP tools, graded by the platform's own verifiers at episode
end. Plugs in at the generate-function layer (like the HUD connector); no
miles source changes.

| Piece | File |
|---|---|
| Generate function (turn loop, TITO token assembly, steps) | `rollout.py` |
| Fleet SDK session (prepare/tools/grade, deadlines) | `session.py` |
| Tool-call parser (Qwen/GLM/Kimi grammars, try-all) | `parser.py` |
| Tool result -> message content projection | `content.py` |
| Taskset -> train/eval JSONL + `images.txt` | `prepare_dataset.py` |
| Launcher: recipes keyed by `--model-name` | `launch/run_fleet.py` |
| rl1 cluster launch (image build + run Job) | `launch/rl1/` |

## How it works

One episode = one Fleet attempt. The loop POSTs `/generate` with token ids and
appends the sampled tokens verbatim (loss mask 1); tool results come back as
`role:"tool"` messages tokenized incrementally by miles's TITO tokenizer (loss
mask 0), whose per-family subclasses own boundary quirks and keep-thinking
kwargs. History is never re-rendered. `fleet_submit` (injected into the tool
schemas) ends the episode; the platform grade becomes `Sample.reward`
(`report.ok` -> 1.0; verdict fail -> 0.0 or the per-verifier mean with
`--fleet-partial-reward`; grading infra failure -> 0.0 plus
`verifier_failed=1` in metadata, which distinguishes infra trouble from policy
regression).

Multi-step tasks are supported: `fleet_submit` closes the current step; a
`preserve` boundary appends the next step's prompt in-conversation, a `reset`
boundary starts a fresh conversation, returning several Samples for one
episode (grouped by `rollout_id`, terminal reward broadcast).

Environments are containers on the node that hosts the RolloutManager actor;
the loop bounds them with its own semaphore (`--fleet-max-concurrent-envs`),
a per-episode wall clock, and per-call deadlines, and it sets
`non_generation_time` so env wall-clock stays out of throughput metrics.
Aborted episodes (weight-update aborts, docker crashes) surface as ABORTED
samples; run with
`--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted`.

## Run it

```bash
# 1. prepare data (anywhere with the fleet_runtime wheel + flt store)
flt pull registry-alpha.fleetai.me/library/ade-bench:latest ade-bench
python -m examples.fleet.prepare_dataset --taskset-ref ade-bench --output-dir data/ade-bench

# 2. build the trainer image in-cluster (rl1)
bash examples/fleet/launch/rl1/build_image.sh fleet-integration

# 3. launch: debug_minimal smoke, rollout-only gate, then the run
MODEL_NAME=glm4.7-flash bash examples/fleet/launch/rl1/launch_fleet.sh <tag> <taskset-ref> debug_minimal
MODEL_NAME=glm4.7-flash bash examples/fleet/launch/rl1/launch_fleet.sh <tag> <taskset-ref> rollout_only
MODEL_NAME=glm4.7-flash bash examples/fleet/launch/rl1/launch_fleet.sh <tag> <taskset-ref> normal
```

## Models

One launcher, one `_Recipe` row per model (`launch/run_fleet.py`): the row
carries the model-coupled config (HF id, Megatron type + parallelism, sglang
engine flags, TITO tokenizer family); the Fleet rollout block is shared.
Adding a model = adding a row, ported from its `scripts/run_*.py` recipe.

| Model | Status with this connector |
|---|---|
| GLM-4.7-Flash | Probed end-to-end 2026-08-22 on rl1: debug_minimal (2 GRPO steps + save), rollout-only gate (0.12 parse failures/ep, 20/32 submitted, 6/32 pass), production runs on ade-bench and evaluation-benchmark |
| Qwen3.5-35B-A3B | Recipe row ported from `run_qwen3_5_35b_a3b_mtp.py`; never run here |

Surveyed as single-8xH200-node candidates (recipes exist in miles, no rows
yet): Qwen3-30B-A3B, Qwen3 dense 0.6B-32B, Qwen3.5/3.6/3.8 dense 4B-27B,
Qwen3-Next-80B-A3B (single-node topology), Nemotron-3-Nano 4B/30B-A3B,
Gemma-4 26B-A4B/31B, gpt-oss-20b. No Kimi or DeepSeek fits one node.

Pins: fleet-runtime and the `flt` binary must come from the SAME
`fleet-ai/platform` commit (validated pair: `72e656c948ab`). The recipe drops
`--use-rollout-routing-replay`: multi-turn routed-experts replay is
wrong-shaped upstream (assignment, not concat, in
`generate_endpoint_utils.update_sample_from_response`).

## Tests

```bash
pytest examples/fleet/tests -q                 # CPU: parser, content, dataset, session, loop
pytest examples/fleet/tests -q -m network      # real GLM-4.7-Flash tokenizer properties
FLEET_FLT=$(which flt) pytest examples/fleet/tests/integration -q -m docker  # real containers
```

Vision (computer_use screenshots into `multimodal_train_inputs`) is phase 2;
`--fleet-vision` currently raises. Text mode projects image blobs to digest
placeholders.
