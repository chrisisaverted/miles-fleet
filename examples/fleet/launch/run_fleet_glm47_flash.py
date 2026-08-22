"""GRPO on Fleet v2 tasksets with GLM-4.7-Flash.

A copy of scripts/run_glm47_flash.py with the rollout block swapped for the
Fleet generate function; the model/perf/optimizer blocks are the recipe's.
Run from the miles repo root, inside the miles container, after
examples/fleet/launch/setup_node.sh:

    python examples/fleet/launch/run_fleet_glm47_flash.py \
        --dataset-dir /root/datasets/fleet/ade-bench --run-id ade-bench-$(date +%m%d)

Differences from the stock recipe, each with its reason:
- rollout block targets a prepared Fleet JSONL (see ../prepare_dataset.py);
  metadata rides to the generate fn, prompts are rebuilt per episode.
- --use-rollout-routing-replay is OFF: update_sample_from_response assigns
  (not concatenates) routed experts per turn, so multi-turn replay data would
  be wrong-shaped (upstream TODO in generate_endpoint_utils.py).
- --dynamic-sampling-filter-path check_no_aborted: a docker-crashed or
  timed-out episode rejects its group instead of training on it.
- batch sizes sized to small tasksets (ade-bench: 31 train rows).
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal", "rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    dataset_dir: str = "/root/datasets/fleet/ade-bench"
    model_org: str = "zai-org"
    model_name: str = "GLM-4.7-Flash"
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int = 8
    hardware: Literal["H200", "B200"] = "H200"
    rollout_num_gpus_per_engine: int | None = None  # None => derive from hardware
    sglang_attention_backend: str | None = None
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


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} " f"--local-dir {args.model_dir}/{args.model_name}"
    )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    debug = args.mode == "debug_minimal"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if debug else 20} "
        f"--save-retain-interval {2 if debug else 20} "
    )

    fleet_args = (
        "--custom-generate-function-path examples.fleet.rollout.generate "
        "--fleet-tito-model glm47 "
        f"--fleet-max-turns {2 if debug else args.max_turns} "
        "--fleet-max-tokens-per-turn 4096 "
        f"--fleet-max-concurrent-envs {args.max_concurrent_envs} "
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
        f"--num-rollout {2 if debug else 200} "
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
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
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

    # GLM-4.7-Flash has 20 attention heads, so rollout TP must divide 20.
    rollout_num_gpus_per_engine = (
        args.rollout_num_gpus_per_engine
        if args.rollout_num_gpus_per_engine is not None
        else (2 if args.hardware == "B200" else 1)
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
        # EAGLE speculative decoding (MTP)
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
    )

    if args.sglang_attention_backend not in (None, "default"):
        sglang_args += f"--sglang-attention-backend {args.sglang_attention_backend} "

    if args.hardware == "B200" and args.sglang_attention_backend in (None, "default", "flashinfer"):
        sglang_args += "--sglang-flashinfer-mla-disable-ragged "

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
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
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
