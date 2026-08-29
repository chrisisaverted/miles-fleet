"""GRPO on Fleet v2 tasksets — one launcher, recipes keyed by --model-name.

Follows the miles convention (run_qwen3_dense.py): recipes live in a frozen
_Recipe table; the model-coupled blocks (engine flags, TITO tokenizer family,
chat template) come from the row, the Fleet rollout block is shared.

    python examples/fleet/launch/run_fleet.py \
        --model-name qwen3.8-27b --dataset-dir <dir> --run-id <name>

Rows:
    qwen3.8-27b      vision-capable, FSDP; validated end-to-end with the Fleet
                     connector on text (ade-bench) and GUI (evaluation-
                     benchmark) tasksets (2026-08)
    glm5.3-flash     vision-capable, Megatron; ported verbatim from upstream
                     scripts/run_glm5_3_flash.py (its 6x4/8x4 GB300 shapes
                     re-expressed for 8-GPU nodes as 3x8/4x8). debug_minimal
                     is the only validated mode until budgets are measured.

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

ModelName = Literal["qwen3.8-27b", "glm5.3-flash"]


@dataclass(frozen=True)
class _Recipe:
    hf_org: str
    hf_name: str
    tito_model: str
    backend: str = "fsdp"  # "fsdp" | "megatron"
    megatron_model_type: str = ""
    # megatron only: parallelism and model-specific blocks, keyed by
    # (num_nodes, num_gpus_per_node); ported verbatim from the upstream
    # launcher so the argv stays diffable against it
    parallel_args_by_shape: dict | None = None
    misc_extra: str = ""
    env_extra: dict | None = None
    max_tokens_per_gpu: int = 8192
    # rollout engine: GPUs per sglang engine (None => one engine spanning all)
    rollout_gpus_per_engine: int | None = None
    sglang_extra: str = ""
    train_extra: str = ""
    vision: bool = False  # screenshots into engine payload + train inputs
    sglang_mem_fraction: float = 0.7
    max_response_len: int = 24576
    max_context_len: int = 30720
    # Absolute or repo-relative path to a chat template that overrides the
    # TITO family's registered one (None = use the family resolution).
    chat_template: str | None = None


_RECIPES: dict[str, _Recipe] = {
    # Vision-capable (Qwen3_5ForConditionalGeneration). Engine TP=1 (sglang
    # TP>1 garbage for this family on the pinned version, sglang#21039).
    # Memory at the full 30720 context: ~65GB fixed per rank (params + grads
    # + Adam fp32) + ~70GB activations (measured 2026-08-25) — fits a B300's
    # 268GB per GPU with wide headroom and no cpu offload (the offload's
    # engine-resume bug is reproduced in pure miles; fix pending upstream).
    "qwen3.8-27b": _Recipe(
        hf_org="Qwen",
        hf_name="Qwen3.8-27B",
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
    # GLM-5.3-Flash (320B-A18B, glm5_next hybrid). Ported verbatim from
    # upstream scripts/run_glm5_3_flash.py: Megatron backend, frozen vision
    # tower via the glm5_next plugin provider, trainer offloaded to local
    # disk between phases, engines at TP8/EP8 (one engine per 8-GPU node),
    # DSA on tilelang, radix cache off, KV bf16. Budgets are the upstream-
    # validated ones (4096-token responses); anything longer is unmeasured.
    # Template study 2026-08-28 (tokenizer-level, byte-exact merge checks):
    # the glm47 TITO family drives GLM-5.3 unchanged; the stock template has
    # no exception branches, is append-only over the vision dummy prefix,
    # and emits the 4.7-style inline arg_key/arg_value tool grammar the
    # parser handles. The template always prepends a "Reasoning Effort: Max"
    # system header (tunable via chat_template_kwargs reasoning_effort).
    "glm5.3-flash": _Recipe(
        # BF16 checkpoint, not the FP8 main repo: the converter's streaming
        # loader hands FP8 tensors on CPU to a GPU-only dequant kernel
        # (crash), and the whole-model-per-rank mode that avoids that path
        # needs 288GB GPUs (GB300); it OOMs at 260.9GB on our 267.7GB B300s.
        # BF16 needs no dequant, so the memory-safe auto-split conversion
        # works. Both failures observed on miles-glm53-smoke, 2026-08-29.
        hf_org="zai-org",
        hf_name="GLM-5.3-Flash-BF16",
        tito_model="glm47",
        backend="megatron",
        megatron_model_type="glm5.3-flash",
        vision=True,
        sglang_mem_fraction=0.7,
        max_response_len=4096,
        max_context_len=12288,
        max_tokens_per_gpu=8192,
        rollout_gpus_per_engine=8,
        parallel_args_by_shape={
            # upstream 6x4 shape on 8-GPU nodes
            (3, 8): (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 3 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            ),
            # 4x8: upstream's 8x4 row says EP16, but EP16 x PP4 = 64 does
            # not divide 32 GPUs (trainer init rejects it; attempt 9,
            # 2026-08-29). EP8 x PP4 = 32 divides exactly. This is also the
            # practical floor on 267.7GB B300s: the 3-node shape's per-rank
            # trainer state (233GB measured) leaves too little for
            # activations (OOM, attempts 7-8); 32 ranks fit with headroom.
            (4, 8): (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 4 "
                "--decoder-first-pipeline-num-layers 11 "
                "--decoder-last-pipeline-num-layers 12 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            ),
        },
        sglang_extra=(
            "--sglang-tp-size 8 "
            "--sglang-ep-size 8 "
            "--sglang-dp-size 1 "
            "--sglang-chunked-prefill-size 8192 "
            "--sglang-disable-radix-cache "
            "--sglang-dsa-prefill-backend tilelang "
            "--sglang-dsa-decode-backend tilelang "
            "--sglang-kv-cache-dtype bfloat16 "
            "--sglang-mm-attention-backend sdpa "
            "--router-health-success-threshold 1 "
            "--router-health-check-interval-secs 15 "
            "--router-health-failure-threshold 40 "
        ),
        train_extra="--fleet-screenshot-max-dim 1024 ",
        misc_extra=(
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--attention-softmax-in-fp32 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--update-weight-buffer-size 1073741824 "
            "--train-memory-margin-bytes 3221225472 "
            # cpu, not upstream's disk: their Grace hosts have little RAM so
            # they stream ~1.4TB/node of trainer state to NVMe; our x86
            # nodes have 2.7TB RAM but only ~956GB node disk, which filled
            # and killed the offload after step 0 (attempt 10, 2026-08-29).
            "--offload-train-target cpu "
            "--model-name glm5_next "
            "--qkv-format thd "
            "--rollout-health-check-interval 300 "
            "--rollout-health-check-timeout 300 "
            "--distributed-timeout-minutes 60 "
            # Upstream keeps engine WEIGHTS resident during training
            # (--offload-rollout-level kv_cache), affordable on its 288GB
            # GPUs. On our 267.7GB B300s that leaves the trainer ~3GB short
            # (OOM at 264.7GB, attempt 7, 2026-08-29). Omitting the flag
            # restores miles's default (offload kv_cache AND weight), which
            # frees the engine weight shard during the train phase.
            "--custom-model-provider-path "
            "miles_plugins.models.glm5_next.vision.glm5_next_vlm_model_provider "
            "--check-weight-update-equal "
            "--check-weight-update-skip-list visual. "
        ),
        env_extra={
            "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
            "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
            "PYTHONFAULTHANDLER": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TRITON_CACHE_DIR": "/tmp/triton_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
        },
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: ModelName = "qwen3.8-27b"
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


def prepare(args: ScriptArgs):
    """Idempotent: skips work whose output already exists, so concurrent jobs
    serialized by the launch manifest's flock share one downloaded model.
    FSDP loads the HF checkpoint directly; Megatron additionally needs the
    torch_dist conversion."""
    recipe = args.recipe
    hf_dir = Path(args.model_dir) / recipe.hf_name
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    if not (hf_dir / "config.json").exists():
        U.exec_command_cpu(f"hf download {recipe.hf_org}/{recipe.hf_name} --local-dir {hf_dir}")
    dist_dir = Path(args.model_dir) / f"{recipe.megatron_model_type}_torch_dist"
    tracker = dist_dir / "latest_checkpointed_iteration.txt"
    if recipe.backend == "megatron" and not (tracker.exists() and tracker.read_text().strip() == "release"):
        # No CONVERT_KEEP_PP1: whole-model-per-rank needs 288GB GPUs; the
        # converter's automatic pipeline split fits our 268GB B300s, and the
        # BF16 checkpoint avoids the FP8 dequant bug that split mode has.
        U.exec_command_gpu(
            "CUDA_DEVICE_MAX_CONNECTIONS=1 "
            f"PYTHONPATH={U.repo_base_dir}:/root/Megatron-LM "
            f"torchrun --nproc-per-node {args.num_gpus_per_node} "
            f"{U.repo_base_dir}/tools/convert_hf_to_torch_dist.py "
            f"{U.shell_safe_model_args(recipe.megatron_model_type)} "
            f"--hf-checkpoint {hf_dir} "
            f"--save {dist_dir}"
        )


def execute(args: ScriptArgs):
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
    hf_path = f"{args.model_dir}/{recipe.hf_name}"
    ref_load = hf_path if recipe.backend == "fsdp" else f"{args.model_dir}/{recipe.megatron_model_type}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    debug = args.mode == "debug_minimal"
    few_steps = args.mode != "normal"

    ckpt_args = (
        f"--hf-checkpoint {hf_path} "
        f"--ref-load {ref_load} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if debug else 20} "
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
        f"--num-rollout {2 if few_steps else 200} "
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

    if recipe.backend == "fsdp":
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
    else:
        shape = (args.num_nodes, args.num_gpus_per_node)
        parallel_args = (recipe.parallel_args_by_shape or {}).get(shape)
        assert parallel_args is not None, (
            f"{args.model_name} has no parallel config for {shape}; "
            f"supported: {sorted((recipe.parallel_args_by_shape or {}).keys())}"
        )
        perf_args = (
            f"{parallel_args}"
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--micro-batch-size 1 "
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

    misc_args = recipe.misc_extra
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
        megatron_model_type=recipe.megatron_model_type if recipe.backend == "megatron" else None,
        extra_env_vars=dict(recipe.env_extra or {}),
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    if not args.prepare_only:
        execute(args)


if __name__ == "__main__":
    typer.run(main)
