"""GRPO on Fleet v2 tasksets — one launcher, recipes keyed by --model-name.

Follows the miles convention (run_qwen3_dense.py): near-duplicate recipes
merge behind a frozen _Recipe table; the model-coupled blocks (Megatron
parallelism, sglang engine flags, TITO tokenizer family, checkpoint
conversion type) come from the row, the Fleet rollout block is shared.

    python examples/fleet/launch/run_fleet.py \
        --model-name glm4.7-flash --dataset-dir <dir> --run-id <name>

Rows:
    glm4.7-flash     validated end-to-end with the Fleet connector (2026-08)

Deviations from the stock recipes, each with its reason:
- rollout block targets a prepared Fleet JSONL (see ../prepare_dataset.py).
- --use-rollout-routing-replay is OFF: update_sample_from_response assigns
  (not concatenates) routed experts per turn, so multi-turn replay data would
  be wrong-shaped (upstream TODO in generate_endpoint_utils.py).
- --dynamic-sampling-filter-path check_no_aborted: docker-crashed or
  timed-out episodes reject their group instead of training on it.
- --async-save --use-persistent-ckpt-worker: a synchronous 30-minute SFS save
  both stalls training and starves co-located env boots (measured 2026-08-22).
- prepare() is idempotent (Path.exists) so concurrent jobs share one
  converted checkpoint under the launch manifest's flock.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

ModelName = Literal["glm4.7-flash"]


@dataclass(frozen=True)
class _Recipe:
    hf_org: str
    hf_name: str
    megatron_model_type: str
    tito_model: str
    # perf/parallelism (expert parallel is min(8, num_gpus) at compose time,
    # so sub-node runs work; the stock recipes assume a full 8-GPU node)
    tp: int
    cp: int
    max_tokens_per_gpu: int
    # rollout engine: GPUs per sglang engine (None => one engine spanning
    # all GPUs; GLM-4.7-Flash fits TP1 on H200)
    rollout_gpus_per_engine: int | None
    sglang_extra: str
    train_extra: str


_RECIPES: dict[str, _Recipe] = {
    # as validated on rl1 (miles-train-ade / miles-train-evalbench)
    "glm4.7-flash": _Recipe(
        hf_org="zai-org",
        hf_name="GLM-4.7-Flash",
        megatron_model_type="glm4.7-flash",
        tito_model="glm47",
        tp=4,
        cp=1,
        max_tokens_per_gpu=32768,
        rollout_gpus_per_engine=1,
        sglang_extra=(
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
        ),
        train_extra="",
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: ModelName = "glm4.7-flash"
    mode: Literal["normal", "debug_minimal", "rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    dataset_dir: str = "/root/datasets/fleet/ade-bench"
    num_gpus_per_node: int = 8
    skip_prepare: bool = False
    prepare_only: bool = False
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    max_turns: int = 32
    max_concurrent_envs: int = 8
    partial_reward: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]


def prepare(args: ScriptArgs):
    """Idempotent: skips work whose output already exists, so concurrent jobs
    serialized by the launch manifest's flock share one prepared model."""
    recipe = args.recipe
    hf_dir = Path(args.model_dir) / recipe.hf_name
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    if not (hf_dir / "config.json").exists():
        U.exec_command_cpu(f"hf download {recipe.hf_org}/{recipe.hf_name} --local-dir {hf_dir}")
    if not (Path(args.model_dir) / f"{recipe.hf_name}_torch_dist").exists():
        U.convert_checkpoint(
            model_name=recipe.hf_name,
            megatron_model_type=recipe.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst=args.model_dir,
            hf_checkpoint=str(hf_dir),
            megatron_path=args.megatron_path,
        )


def execute(args: ScriptArgs):
    recipe = args.recipe
    ref_load_path = f"{args.model_dir}/{recipe.hf_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    debug = args.mode == "debug_minimal"
    few_steps = args.mode != "normal"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{recipe.hf_name} "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if debug else 20} "
        f"--save-retain-interval {2 if debug else 20} "
        "--async-save "
        "--use-persistent-ckpt-worker "
    )

    fleet_args = (
        "--custom-generate-function-path examples.fleet.rollout.generate "
        f"--fleet-tito-model {recipe.tito_model} "
        f"--fleet-max-turns {2 if debug else args.max_turns} "
        "--fleet-max-tokens-per-turn 4096 "
        f"--fleet-max-concurrent-envs {args.max_concurrent_envs} "
        "--fleet-max-concurrent-prepares 3 "
        "--fleet-episode-timeout-s 2400 "
        "--fleet-tool-output-max-chars 4000 "
        f"{'--fleet-partial-reward ' if args.partial_reward else ''}"
    )

    rollout_args = (
        f"--prompt-data {args.dataset_dir}/train.jsonl "
        "--input-key input "
        "--label-key label "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {2 if few_steps else 200} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {512 if debug else 24576} "
        "--rollout-max-context-len 30720 "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--over-sampling-batch-size {args.rollout_batch_size + args.rollout_batch_size // 2} "
        "--log-multi-turn "
        f"{fleet_args}"
    )

    perf_args = (
        f"--tensor-model-parallel-size {recipe.tp} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {recipe.cp} "
        f"--expert-model-parallel-size {min(8, args.num_gpus_per_node)} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {recipe.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    engine_gpus = recipe.rollout_gpus_per_engine or args.num_gpus_per_node
    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        "--sglang-mem-fraction-static 0.7 "
        f"{recipe.sglang_extra}"
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
    )
    if args.mode == "rollout_only":
        misc_args += "--debug-rollout-only "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{recipe.train_extra} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    if not args.prepare_only:
        execute(args)


if __name__ == "__main__":
    typer.run(main)
