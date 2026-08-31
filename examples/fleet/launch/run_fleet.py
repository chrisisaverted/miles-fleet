"""GRPO on Fleet v2 tasksets — one launcher, recipes keyed by --model-name.

Follows the miles convention (run_qwen3_dense.py): recipes live in a frozen
_Recipe table; the model-coupled blocks (engine flags, TITO tokenizer family,
chat template) come from the row, the Fleet rollout block is shared.

    python examples/fleet/launch/run_fleet.py \
        --model-name qwen3.8-27b --dataset-dir <dir> --run-id <name>

Rows:
    qwen3.6-27b      vision-capable; revision-pinned candidate for Fleet cyber
                     tasksets; requires a debug_minimal B300 validation run
    qwen3.8-27b      vision-capable; validated end-to-end with the Fleet
                     connector on text (ade-bench) and GUI (evaluation-
                     benchmark) tasksets (2026-08)

Deviations from the stock recipes, each with its reason:
- FSDP backend, not Megatron: miles's Megatron qwen3_5 spec is language-only
  (docs: "trains the text path only"); FSDP trains the full HF model
  including the vision towers, with no torch_dist conversion.
- rollout block targets a prepared Fleet JSONL (see ../prepare_dataset.py).
- --use-rollout-routing-replay is OFF: update_sample_from_response assigns
  (not concatenates) routed experts per turn, so multi-turn replay data would
  be wrong-shaped (upstream TODO in generate_endpoint_utils.py).
- --dynamic-sampling-filter-path check_no_aborted: docker-crashed or
  timed-out episodes reject their group instead of training on it.
- prepare() is idempotent (Path.exists) so concurrent jobs share one
  downloaded model under the launch manifest's flock.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

ModelName = Literal["qwen3.6-27b", "qwen3.8-27b"]


@dataclass(frozen=True)
class _Recipe:
    hf_org: str
    hf_name: str
    hf_revision: str | None
    tito_model: str
    max_tokens_per_gpu: int
    # rollout engine: GPUs per sglang engine (None => one engine spanning all)
    rollout_gpus_per_engine: int | None
    sglang_extra: str
    train_extra: str
    vision: bool = False  # screenshots into engine payload + train inputs
    sglang_mem_fraction: float = 0.7
    max_response_len: int = 24576
    max_context_len: int = 30720
    # Absolute or repo-relative path to a chat template that overrides the
    # TITO family's registered one (None = use the family resolution).
    chat_template: str | None = None


_RECIPES: dict[str, _Recipe] = {
    # Qwen3.6-27B has the same dense hybrid-GDN geometry as Qwen3.8-27B.
    # Keep the proven B300/FSDP resource recipe, but use Qwen3.6's distinct
    # TITO template and pin the exact reviewed checkpoint revision.
    "qwen3.6-27b": _Recipe(
        hf_org="Qwen",
        hf_name="Qwen3.6-27B",
        hf_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        tito_model="qwen36",
        vision=True,
        sglang_mem_fraction=0.8,
        max_response_len=24576,
        max_context_len=30720,
        max_tokens_per_gpu=9216,
        rollout_gpus_per_engine=1,
        sglang_extra="--sglang-attention-backend triton ",
        train_extra="--fleet-screenshot-max-dim 1024 ",
    ),
    # Vision-capable (Qwen3_5ForConditionalGeneration). Engine TP=1 (sglang
    # TP>1 garbage for this family on the pinned version, sglang#21039).
    # Memory at the full 30720 context: ~65GB fixed per rank (params + grads
    # + Adam fp32) + ~70GB activations (measured 2026-08-25) — fits a B300's
    # 268GB per GPU with wide headroom and no cpu offload (the offload's
    # engine-resume bug is reproduced in pure miles; fix pending upstream).
    "qwen3.8-27b": _Recipe(
        hf_org="Qwen",
        hf_name="Qwen3.8-27B",
        hf_revision=None,
        tito_model="qwen35",
        vision=True,
        sglang_mem_fraction=0.8,
        max_response_len=24576,
        max_context_len=30720,
        max_tokens_per_gpu=9216,
        rollout_gpus_per_engine=1,
        # triton, not fa3/flashinfer: fa3 is Hopper-only (SM<=90 assertion),
        # and sglang's registry allows only {trtllm_mha, fa4, triton} on
        # Blackwell for hybrid GDN models. triton JIT-compiles per-arch so it
        # cannot mismatch; fa4/trtllm_mha are perf candidates to profile later.
        sglang_extra="--sglang-attention-backend triton ",
        train_extra="--fleet-screenshot-max-dim 1024 ",
        # Qwen3.8's own fixed template, vendored verbatim from miles PR #2760
        # (branch jiajun/tito-qwen38-27b) until that lands and the base pin
        # moves: the true Qwen3.8 template with only the no-user-query guard
        # removed. It differs from qwen3.5_fixed materially: a reasoning-effort
        # system prefix (default xhigh) the model was trained with, and
        # preserve_thinking semantics. The qwen35 TITO family stays correct for
        # token merging (PR #2760: qwen38 uses the same Qwen3 boundary merge).
        chat_template="examples/fleet/templates/qwen3.8_fixed.jinja",
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: ModelName = "qwen3.8-27b"
    mode: Literal["normal", "debug_minimal", "debug_one_step", "rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    dataset_dir: str = "/root/datasets/fleet/ade-bench"
    num_gpus_per_node: int = 8
    skip_prepare: bool = False
    prepare_only: bool = False
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    max_turns: int = 32
    max_concurrent_envs: int = 8
    # Cold boots in flight, kept low on purpose: environments booting
    # together starve each other past their readiness budgets (measured at
    # 8), and waiting to boot costs an episode nothing because its wall
    # clock starts after the environment is up.
    max_concurrent_prepares: int = 4
    partial_reward: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]


def _hf_checkpoint_path(args: ScriptArgs) -> Path:
    """Return the model directory, including an immutable revision when set."""
    path = Path(args.model_dir) / args.recipe.hf_name
    if args.recipe.hf_revision:
        path /= args.recipe.hf_revision
    return path


def _validate_fsdp_batch_shape(args: ScriptArgs) -> None:
    """Reject rollout batches that cannot give every FSDP rank one sample.

    The Fleet recipe runs one FSDP data-parallel rank per requested GPU.
    Miles computes the local global-batch shard as ``global_batch_size //
    dp_size``.  A smaller batch therefore reaches rollout successfully but
    divides by zero at optimizer entry; a non-divisible batch also produces
    unequal raw shards.  Catch both before model allocation.
    """
    dp_size = args.num_nodes * args.num_gpus_per_node
    samples_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt
    if dp_size < 1:
        raise ValueError("FSDP data-parallel size must be positive")
    if samples_per_rollout < dp_size or samples_per_rollout % dp_size:
        raise ValueError(
            "Fleet FSDP rollout samples must be at least and divisible by "
            f"dp_size={dp_size}; got rollout_batch_size={args.rollout_batch_size} "
            f"* n_samples_per_prompt={args.n_samples_per_prompt} = {samples_per_rollout}"
        )


def prepare(args: ScriptArgs):
    """Idempotent: skips work whose output already exists, so concurrent jobs
    serialized by the launch manifest's flock share one downloaded model.
    FSDP loads the HF checkpoint directly; there is no conversion step."""
    recipe = args.recipe
    hf_dir = _hf_checkpoint_path(args)
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    if not (hf_dir / "config.json").exists():
        revision_arg = f" --revision {recipe.hf_revision}" if recipe.hf_revision else ""
        U.exec_command_cpu(f"hf download {recipe.hf_org}/{recipe.hf_name}{revision_arg} --local-dir {hf_dir}")


def execute(args: ScriptArgs):
    _validate_fsdp_batch_shape(args)
    recipe = args.recipe
    # Swap in a fixed chat template: the stock Qwen templates raise "No user
    # query found in messages" on the TITO suffix render ([dummy system, dummy
    # assistant, tool result]), which has no user turn; the fixed templates
    # drop that raise. miles wires this via --tito-model, but that flag
    # requires --use-session-server, which a custom generate fn doesn't use,
    # so pass the path through --chat-template-path directly. A recipe-level
    # chat_template wins over the TITO family's registered one.
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    if recipe.chat_template:
        fixed_template_path = str(Path(U.repo_base_dir) / recipe.chat_template)
    else:
        fixed_template_path, _ = resolve_fixed_chat_template(recipe.tito_model)
    hf_path = str(_hf_checkpoint_path(args))
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    debug = args.mode in {"debug_minimal", "debug_one_step"}
    one_step = args.mode == "debug_one_step"
    few_steps = args.mode != "normal"

    ckpt_args = (
        f"--hf-checkpoint {hf_path} "
        f"--ref-load {hf_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {1 if one_step else 2 if debug else 20} "
    )

    fleet_args = (
        f"{f'--chat-template-path {fixed_template_path} ' if fixed_template_path else ''}"
        "--custom-generate-function-path examples.fleet.rollout.generate "
        f"--fleet-tito-model {recipe.tito_model} "
        f"--fleet-max-turns {2 if debug else args.max_turns} "
        "--fleet-max-tokens-per-turn 4096 "
        f"--fleet-max-concurrent-envs {args.max_concurrent_envs} "
        f"--fleet-max-concurrent-prepares {args.max_concurrent_prepares} "
        "--fleet-episode-timeout-s 2400 "
        "--fleet-tool-output-max-chars 4000 "
        f"{'--fleet-partial-reward ' if args.partial_reward else ''}"
        f"{'--fleet-vision ' if recipe.vision else ''}"
    )

    rollout_args = (
        f"--prompt-data {args.dataset_dir}/train.jsonl "
        "--input-key input "
        "--label-key label "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {1 if one_step else 2 if few_steps else 200} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {512 if debug else recipe.max_response_len} "
        f"--rollout-max-context-len {recipe.max_context_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--over-sampling-batch-size {args.rollout_batch_size + args.rollout_batch_size // 2} "
        "--log-multi-turn "
        f"{fleet_args}"
    )

    perf_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
        # sdpa, not flash_attention_3: FA3 kernels are Hopper-only; sdpa
        # routes to cuDNN's Blackwell kernels in torch 2.11+cu129
        "--attn-implementation sdpa "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
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
    )

    engine_gpus = recipe.rollout_gpus_per_engine or args.num_gpus_per_node
    sglang_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} "
        f"--sglang-mem-fraction-static {recipe.sglang_mem_fraction} "
        f"{recipe.sglang_extra}"
    )

    misc_args = ""
    if args.num_nodes > 1:
        # The rollout manager drives env containers over DOCKER_HOST; only
        # the head pod carries the dind sidecar and flt store.
        misc_args += "--pin-rollout-manager-to-head "
    misc_args += (
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
        megatron_model_type=None,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    if not args.prepare_only:
        execute(args)


if __name__ == "__main__":
    typer.run(main)
