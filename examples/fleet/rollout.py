"""Fleet v2 tasksets on miles: custom generate function.

Wire with:
    --custom-generate-function-path examples.fleet.rollout.generate
    --prompt-data <prepared>.jsonl --input-key input --metadata-key metadata
    --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
    --fleet-tito-model qwen35  (per model family; miles --tito-model values)

Each dataset row's metadata carries the task identity (taskset_ref, task_key);
all behavior knobs are --fleet-* args registered via generate.add_arguments.

generate() composes two halves along miles's integration-shapes boundary:
    examples.fleet.agent      the message-level agent-environment loop
                              (the future --custom-agent-function-path body)
    examples.fleet.recording  token-in-token-out assembly, loss masks,
                              multimodal tensors, Sample construction
plus the process-level infrastructure that belongs to neither: the docker
concurrency semaphores, the leaked-network sweep, and the write-off policy
(an episode must never kill the run).

Reward is set on the Sample directly (miles skips its RM hook when reward is
already set). metadata carries per-episode metrics; `verifier_failed` spikes
mean grading infrastructure trouble, not policy regression.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from copy import deepcopy

from examples.fleet.agent import AgentConfig, EpisodeStats, run_agent
from examples.fleet.authoritative import AUTHORITATIVE_BACKEND, AuthoritativeCyberSession
from examples.fleet.recording import Recorder
from examples.fleet.session import FleetSession, SessionConfig, sweep_leaked_networks

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- state


_SWEPT = False
_ENV_SEMAPHORE: asyncio.Semaphore | None = None
_PREPARE_SEMAPHORE: asyncio.Semaphore | None = None


def _startup_once() -> None:
    global _SWEPT
    if not _SWEPT:
        _SWEPT = True
        sweep_leaked_networks()


def _env_semaphore(args) -> asyncio.Semaphore:
    """Bounds concurrent docker environments; miles's own semaphore sizes to
    the inference servers, not to what the docker daemon sustains."""
    global _ENV_SEMAPHORE
    if _ENV_SEMAPHORE is None:
        _ENV_SEMAPHORE = asyncio.Semaphore(args.fleet_max_concurrent_envs)
    return _ENV_SEMAPHORE


def _prepare_semaphore(args) -> asyncio.Semaphore:
    """Bounds concurrent env COLD BOOTS separately from running episodes:
    eight desktop stacks booting at once starve each other past their
    readiness budgets, while eight already-running episodes are cheap. Queue
    time here does not eat the episode wall clock (open precedes wait_for)."""
    global _PREPARE_SEMAPHORE
    if _PREPARE_SEMAPHORE is None:
        _PREPARE_SEMAPHORE = asyncio.Semaphore(args.fleet_max_concurrent_prepares)
    return _PREPARE_SEMAPHORE


def _session_config(args) -> SessionConfig:
    return SessionConfig(
        runtime_root=args.fleet_runtime_root,
        partial_reward=args.fleet_partial_reward,
        tool_output_max_chars=args.fleet_tool_output_max_chars,
        screenshot_max_dim=args.fleet_screenshot_max_dim or None,
        vision=args.fleet_vision,
        call_tool_timeout_s=args.fleet_call_tool_timeout_s,
        grade_timeout_s=args.fleet_grade_timeout_s,
    )


def _output(samples: list[Sample]) -> GenerateFnOutput:
    return GenerateFnOutput(samples=samples[0] if len(samples) == 1 else samples)


# ------------------------------------------------------------------ generate


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    state = input.state
    assert not args.partial_rollout, "fleet episodes own live container state; partial rollout is not supported"

    base_sample = deepcopy(input.sample)
    metadata = dict(base_sample.metadata or {})
    taskset_ref = metadata.get("taskset_ref")
    task_key = metadata.get("task_key")
    if not taskset_ref or not task_key:
        raise ValueError("dataset row metadata must carry taskset_ref and task_key (see prepare_dataset.py)")

    backend = metadata.get("fleet_backend", "local")
    if backend == AUTHORITATIVE_BACKEND:
        required = (
            "task_version_id",
            "environment_version_id",
            "verifier_version_id",
            "env_key",
            "env_version",
            "data_key",
            "data_version",
            "task_prompt",
            "reward_objective",
        )
        missing = [key for key in required if not metadata.get(key)]
        if missing:
            raise ValueError(f"authoritative dataset row is missing immutable binding fields: {missing}")
        session = AuthoritativeCyberSession(
            task_key=task_key,
            task_version_id=metadata["task_version_id"],
            instructions=metadata["task_prompt"],
            objective=metadata["reward_objective"],
            config=_session_config(args),
            expected_binding={key: metadata[key] for key in ("env_key", "env_version", "data_key", "data_version")},
        )
    elif backend == "local":
        _startup_once()
        session = FleetSession(taskset_ref, task_key, _session_config(args))
    else:
        raise ValueError(f"unsupported Fleet rollout backend {backend!r}")
    stats = EpisodeStats()
    if args.save_debug_trajectory_data is not None:
        stats.trajectory = []
    recorder = Recorder(args, state, base_sample, input.sampling_params)
    cfg = AgentConfig(max_turns=args.fleet_max_turns, episode_timeout_s=args.fleet_episode_timeout_s)

    try:
        async with _env_semaphore(args):
            if state.aborted:
                return _output(recorder.write_off())

            result = await run_agent(session, recorder, cfg, stats, prepare_gate=_prepare_semaphore(args))
            if result.grade is None:  # aborted mid-episode: nothing to train on
                return _output(recorder.write_off())
    except asyncio.CancelledError:
        recorder.write_off()
        raise
    except Exception as e:
        # An episode must never kill the run. Env prepare failures (an image
        # exceeding its own declared readiness budget on a loaded node killed
        # a 12h run on 2026-08-22), engine errors, or template drift become an
        # ABORTED write-off; the check_no_aborted filter rejects the group and
        # over-sampling replaces it.
        logger.warning("[%s] episode failed, writing off as ABORTED: %s", task_key, e)
        return _output(recorder.write_off(str(e)))
    finally:
        # Shielded so a cancelled episode still reaps its containers.
        try:
            await asyncio.shield(asyncio.to_thread(session.close))
        except asyncio.CancelledError:
            pass

    episode_meta = {
        "done_reason": result.done_reason,
        "turns": stats.turns,
        "round_number": stats.turns,  # --log-multi-turn reads this key
        "tool_calls": stats.tool_calls,
        "tool_errors": stats.tool_errors,
        "parse_failures": stats.parse_failures,
        "images": stats.images,
        "steps_closed": stats.steps_closed,
        "segments": len(recorder.segments),
        "verifier_failed": 1.0 if result.grade.verifier_failed else 0.0,
        "attempt_id": session.attempt_id,
        "reward_objective": metadata.get("reward_objective"),
        "safe_success": getattr(session, "safe_success", None),
        "raw_capability_reward": getattr(session, "raw_capability_reward", None),
        "verifier_execution_id": getattr(session, "verifier_execution_id", None),
    }
    return _output(recorder.finalize(result.grade.reward, episode_meta, env_time=stats.env_time))


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--fleet-max-turns", type=int, default=32)
    parser.add_argument(
        "--fleet-max-tokens-per-turn",
        type=int,
        default=4096,
        help="per-turn generation cap; the context cap is --rollout-max-context-len",
    )
    parser.add_argument("--fleet-max-concurrent-envs", type=int, default=8)
    parser.add_argument(
        "--fleet-max-concurrent-prepares",
        type=int,
        default=3,
        help="env cold boots in flight; episodes queue here without burning their wall clock",
    )
    parser.add_argument("--fleet-episode-timeout-s", type=float, default=2400.0)
    parser.add_argument("--fleet-runtime-root", type=str, default=None)
    parser.add_argument("--fleet-partial-reward", action="store_true")
    parser.add_argument("--fleet-tool-output-max-chars", type=int, default=4000)
    parser.add_argument("--fleet-screenshot-max-dim", type=int, default=768)
    parser.add_argument("--fleet-call-tool-timeout-s", type=float, default=300.0)
    parser.add_argument("--fleet-grade-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--fleet-tito-model",
        type=str,
        default="default",
        help="miles TITO tokenizer family (--tito-model values): qwen35, kimi25, ...",
    )
    parser.add_argument(
        "--fleet-vision",
        action="store_true",
        help="screenshots ride into the engine payload and multimodal_train_inputs (VL models)",
    )


generate.add_arguments = _add_arguments
