"""Fleet v2 tasksets on miles: custom generate function.

Wire with:
    --custom-generate-function-path examples.fleet.rollout.generate
    --prompt-data <prepared>.jsonl --input-key input --metadata-key metadata
    --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
    --fleet-tito-model glm47   (per model family; miles --tito-model values)

Each dataset row's metadata carries the task identity (taskset_ref, task_key);
all behavior knobs are --fleet-* args registered via generate.add_arguments.

The loop is token-in-token-out: sampled token ids come back from /generate and
are appended verbatim (loss mask 1); observation messages are tokenized
incrementally by miles's TITO tokenizer (loss mask 0), whose per-family
subclasses own the boundary quirks (GLM's ambiguous <|user|>/<|observation|>
stop tokens, Qwen's missing newline) and the keep-thinking template kwargs.
History is never re-rendered.

Multi-step tasks: fleet_submit closes the current step. A `preserve` boundary
appends the next step's prompt to the same conversation; a `reset` boundary
ends the current training sequence and opens a fresh conversation, so one
episode returns several Samples (miles groups them by rollout_id). The
episode's terminal reward is broadcast to every segment.

Termination: fleet_submit on the final step, a boundary stop rule, the turn
budget, a full context, or the episode wall clock. Every path except abort
grades: without a submission answer-style verifiers see a null submission,
but state-capture verifiers still grade real work.

Reward is set on the Sample directly (miles skips its RM hook when reward is
already set). metadata carries per-episode metrics; `verifier_failed` spikes
mean grading infrastructure trouble, not policy regression.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer
from miles.utils.http_utils import post
from miles.utils.types import Sample

from examples.fleet.parser import parse_tool_call
from examples.fleet.session import SUBMIT_TOOL, FleetSession, GradeResult, SessionConfig, sweep_leaked_networks

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- state


_SWEPT = False
_ENV_SEMAPHORE: Optional[asyncio.Semaphore] = None
_PREPARE_SEMAPHORE: Optional[asyncio.Semaphore] = None
_TITO_CACHE: Dict[str, Any] = {}


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


def _tito_for(state, args):
    tito = _TITO_CACHE.get(args.fleet_tito_model)
    if tito is None:
        tito = get_tito_tokenizer(state.tokenizer, args.fleet_tito_model)
        _TITO_CACHE[args.fleet_tito_model] = tito
    return tito


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


@dataclass
class _EpisodeStats:
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    parse_failures: int = 0
    images: int = 0
    steps_closed: int = 0
    env_time: float = 0.0
    # Digest the next complete_step must present after a reset boundary;
    # lives here so the caller can still grade after an episode timeout.
    reset_ack: Optional[str] = None
    trajectory: Optional[List[Dict[str, Any]]] = field(default=None)


@dataclass
class _Segment:
    """One training sequence: a Sample plus the message list that produced
    its token prefix (the TITO tokenizer diffs against these messages)."""

    sample: Sample
    messages: List[Dict[str, Any]]
    prompt_len: int
    # vision: per-turn processor outputs (pixel_values etc.), concatenated
    # into sample.multimodal_train_inputs at finalize
    mm_chunks: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- segments


def _build_messages(instructions: str, args, step_ordinal: int, step_count: int) -> List[Dict[str, Any]]:
    step_note = ""
    if step_count > 1:
        step_note = (
            f"This task has {step_count} sequential steps; you are on step {step_ordinal}. "
            f"Call {SUBMIT_TOOL} when you finish the CURRENT step; the next step's "
            "instructions follow. Only the final step takes an answer.\n"
        )
    system = (
        "You are completing a task in a live environment by calling tools.\n"
        f"You have at most {args.fleet_max_turns} turns. End EVERY response with exactly one tool call.\n"
        f"{step_note}"
        f"When finished, call {SUBMIT_TOOL} with your final answer (or empty if the task is "
        "about environment state); this ends the episode and triggers grading.\n\n"
        "Task instructions:\n"
        f"{instructions}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instructions},
    ]


def _start_segment(tito, base_sample: Sample, session: FleetSession, instructions: str, args) -> _Segment:
    """A fresh training sequence: used for the first step and after every
    reset boundary."""
    sample = deepcopy(base_sample)
    sample.metadata = dict(base_sample.metadata or {})
    messages = _build_messages(instructions, args, session.step_ordinal, session.step_count)
    sample.prompt = list(messages)
    prompt_ids = tito.apply_chat_template(messages, add_generation_prompt=True, tools=session.tools, tokenize=True)
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.status = Sample.Status.PENDING
    return _Segment(sample=sample, messages=messages, prompt_len=len(prompt_ids))


def _record_assistant(segment: _Segment, text: str, tool_call: Optional[Dict[str, Any]], turn: int) -> None:
    """Store the sampled turn in the message list so the TITO tokenizer's
    dummy prefix renders the right scaffold (tool_calls make a following
    role-"tool" message legal)."""
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_call is not None:
        message["tool_calls"] = [
            {
                "id": f"call_{turn:06d}",
                "type": "function",
                "function": {"name": tool_call["name"], "arguments": tool_call.get("arguments") or {}},
            }
        ]
    segment.messages.append(message)


def _append_messages(segment: _Segment, tito, new_messages: List[Dict[str, Any]]) -> None:
    """Append observation messages to the segment's token sequence with loss
    mask 0, via the TITO tokenizer's family-specific merge (which may trim a
    trailing ambiguous boundary token off the sampled prefix)."""
    sample = segment.sample
    before = list(sample.tokens)
    merged = tito.merge_tokens(segment.messages, segment.messages + new_messages, before)
    shared = 0
    while shared < len(before) and shared < len(merged) and before[shared] == merged[shared]:
        shared += 1
    removed = len(before) - shared
    added = len(merged) - shared
    if removed:
        sample.loss_mask = sample.loss_mask[: len(sample.loss_mask) - removed]
        sample.rollout_log_probs = sample.rollout_log_probs[: len(sample.rollout_log_probs) - removed]
    sample.loss_mask = sample.loss_mask + [0] * added
    sample.rollout_log_probs = sample.rollout_log_probs + [0.0] * added
    sample.tokens = merged
    sample.response_length = len(merged) - segment.prompt_len
    sample.response = tito.tokenizer.decode(merged[segment.prompt_len :])
    segment.messages.extend(new_messages)


# --------------------------------------------------------------------- vision


def _data_urls_to_pil(urls: List[str]) -> List[Any]:
    import base64
    import io

    from PIL import Image

    images = []
    for url in urls:
        payload = url.split(",", 1)[1]
        images.append(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"))
    return images


def _boundary_fix(tito, sample: Sample) -> None:
    """Apply the TITO family's junction rule to the sampled tail before an
    out-of-band (processor-based) append. Mirrors merge_tokens: Qwen inserts
    the newline its models stop before; GLM trims its ambiguous stop token."""
    tokens = sample.tokens
    im_end = getattr(tito, "_im_end_id", None)
    if im_end is not None and tokens and tokens[-1] == im_end:
        sample.tokens = tokens + [tito._newline_id]
        sample.response_length += 1
        sample.loss_mask = (sample.loss_mask or []) + [0]
        sample.rollout_log_probs = (sample.rollout_log_probs or []) + [0.0]
        return
    ambiguous = getattr(tito, "_ambiguous_boundary_ids", None)
    if ambiguous and tokens and tokens[-1] in ambiguous:
        sample.tokens = tokens[:-1]
        sample.response_length -= 1
        sample.loss_mask = sample.loss_mask[:-1]
        sample.rollout_log_probs = sample.rollout_log_probs[:-1]


def _append_multimodal(segment: _Segment, tito, state, message: Dict[str, Any], images: List[Any]) -> None:
    """Append an observation that carries images: render the message under
    the constant dummy prefix, expand image tokens with the PROCESSOR, trim
    the prefix by its tokenizer length (image expansion only happens after
    it), and append with loss mask 0. Accumulates engine images on the sample
    and the processor tensors on the segment for the finalize merge."""
    sample = segment.sample
    _boundary_fix(tito, sample)

    base = [_VISION_DUMMY_USER, {"role": "assistant", "content": ""}]
    dummy_text = tito.apply_chat_template(base, add_generation_prompt=False, tokenize=False)
    full_text = tito.apply_chat_template(base + [message], add_generation_prompt=True, tokenize=False)
    if not full_text.startswith(dummy_text):
        raise ValueError("chat template is not append-only over the vision dummy prefix")
    trim = len(state.tokenizer.encode(dummy_text, add_special_tokens=False))

    processor_output = state.processor(text=full_text, images=images)
    ids = list(processor_output["input_ids"][0])[trim:]
    chunk = {
        k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")
    }
    if chunk:
        segment.mm_chunks.append(chunk)

    sample.response += state.tokenizer.decode(ids)
    sample.response_length += len(ids)
    sample.tokens = sample.tokens + ids
    sample.loss_mask = (sample.loss_mask or []) + [0] * len(ids)
    sample.rollout_log_probs = (sample.rollout_log_probs or []) + [0.0] * len(ids)

    mm = sample.multimodal_inputs or {}
    mm["images"] = (mm.get("images") or []) + images
    sample.multimodal_inputs = mm
    segment.messages.append(message)


def _merge_mm_chunks(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Concatenate per-turn processor tensors (geo3k's merge)."""
    import torch

    values_by_key: Dict[str, List[Any]] = {}
    for chunk in chunks:
        for key, val in (chunk or {}).items():
            if val is not None:
                values_by_key.setdefault(key, []).append(val)
    merged = {
        key: torch.cat(vals, dim=0)
        for key, vals in values_by_key.items()
        if all(isinstance(v, torch.Tensor) for v in vals)
    }
    return merged or None


_VISION_DUMMY_USER = {"role": "user", "content": "dummy"}


# --------------------------------------------------------------------- loop


async def _episode_loop(
    args,
    state,
    session: FleetSession,
    sampling_params: Dict[str, Any],
    stats: _EpisodeStats,
    base_sample: Sample,
    segments: List[_Segment],
) -> Tuple[Optional[str], str]:
    """Run turns until a terminal condition. Returns (answer, done_reason);
    answer is non-None only for done_reason == "submitted". Grading happens
    in the caller, outside the episode wall clock. segments[-1] is always the
    live training sequence; reset boundaries append a new one."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tito = _tito_for(state, args)
    segment = segments[-1]

    def record(message: Dict[str, Any]) -> None:
        if stats.trajectory is not None:
            stats.trajectory.append(message)
            segment.sample.metadata["messages"] = list(stats.trajectory)

    while True:
        if state.aborted:
            return None, "aborted"

        params = dict(sampling_params)
        per_turn = args.fleet_max_tokens_per_turn
        params["max_new_tokens"] = min(per_turn, params.get("max_new_tokens") or per_turn)
        payload, halt_status = compute_request_payload(
            args, segment.sample.tokens, params, multimodal_inputs=segment.sample.multimodal_inputs
        )
        if payload is None:
            segment.sample.status = halt_status
            return None, "context_full"

        output = await post(url, payload, headers=compute_routing_headers(args, segment.sample))
        await update_sample_from_response(
            args, segment.sample, payload=payload, output=output, update_loss_mask=True
        )
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort":
            return None, "aborted"
        stats.turns += 1
        tool_call = parse_tool_call(output["text"])
        _record_assistant(segment, output["text"], tool_call, stats.turns)
        record(segment.messages[-1])
        if finish == "length":
            return None, "length"

        # -------- submission: grade, or close the step and continue --------
        if tool_call and tool_call["name"] == SUBMIT_TOOL:
            if not session.has_step_protocol or session.is_final_step:
                answer = (tool_call.get("arguments") or {}).get("answer")
                if answer is not None and not isinstance(answer, str):
                    answer = json.dumps(answer, default=str)
                return answer, "submitted"

            t0 = time.time()
            advance = await asyncio.to_thread(session.close_step, stats.reset_ack)
            stats.env_time += time.time() - t0
            stats.steps_closed += 1
            stats.reset_ack = advance.reset_ack
            if not advance.continues:
                return None, "boundary_stop"
            if stats.turns >= args.fleet_max_turns:
                return None, "max_turns"
            next_instructions = advance.next_prompt or session.current_instructions
            if advance.reset:
                # The conversation dies at the boundary; the world survives.
                # Finalize this training sequence and open a fresh one.
                if segment.sample.status == Sample.Status.PENDING:
                    segment.sample.status = Sample.Status.COMPLETED
                segment = _start_segment(tito, base_sample, session, next_instructions, args)
                segments.append(segment)
            else:
                body = (
                    f"Step {session.step_ordinal - 1} complete. "
                    f"Step {session.step_ordinal}/{session.step_count}:\n{next_instructions}"
                    f"\n[Turn {stats.turns}/{args.fleet_max_turns}]"
                )
                message = {"role": "user", "content": body}
                _append_messages(segment, tito, [message])
                record(message)
            continue

        # ---------------------- ordinary tool turn -------------------------
        turn_images: List[str] = []
        if tool_call:
            stats.tool_calls += 1
            t0 = time.time()
            outcome = await asyncio.to_thread(session.call_tool, tool_call["name"], tool_call.get("arguments") or {})
            stats.env_time += time.time() - t0
            if outcome.error:
                stats.tool_errors += 1
                body = f"Error: {outcome.error}"
            else:
                body = f"Tool result:\n{outcome.text}" if outcome.text else "Action executed."
                turn_images = outcome.images
                stats.images += len(turn_images)
        else:
            stats.parse_failures += 1
            body = "No tool call found. End your response with exactly one tool call."

        # The budget check runs AFTER the tool executes: the last call's side
        # effects land before grading.
        if stats.turns >= args.fleet_max_turns:
            return None, "max_turns"

        body += f"\n[Turn {stats.turns}/{args.fleet_max_turns}]"
        if tool_call:
            message = {
                "role": "tool",
                "tool_call_id": f"call_{stats.turns:06d}",
                "name": tool_call["name"],
                "content": body,
            }
        else:
            message = {"role": "user", "content": body}
        if turn_images:
            pils = _data_urls_to_pil(turn_images)
            message["content"] = [{"type": "text", "text": body}] + [{"type": "image"} for _ in pils]
            _append_multimodal(segment, tito, state, message, pils)
        else:
            _append_messages(segment, tito, [message])
        record(message)


# ------------------------------------------------------------------ generate


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    state = input.state
    assert not args.partial_rollout, "fleet episodes own live container state; partial rollout is not supported"

    base_sample = deepcopy(input.sample)
    _startup_once()

    metadata = dict(base_sample.metadata or {})
    taskset_ref = metadata.get("taskset_ref")
    task_key = metadata.get("task_key")
    if not taskset_ref or not task_key:
        raise ValueError("dataset row metadata must carry taskset_ref and task_key (see prepare_dataset.py)")

    session = FleetSession(taskset_ref, task_key, _session_config(args))
    stats = _EpisodeStats()
    if args.save_debug_trajectory_data is not None:
        stats.trajectory = []
    segments: List[_Segment] = []
    grade: Optional[GradeResult] = None
    done_reason = "unknown"

    try:
        async with _env_semaphore(args):
            if state.aborted:
                base_sample.status = Sample.Status.ABORTED
                return GenerateFnOutput(samples=base_sample)

            t0 = time.time()
            async with _prepare_semaphore(args):
                await asyncio.to_thread(session.open)
            stats.env_time += time.time() - t0

            tito = _tito_for(state, args)
            first = _start_segment(tito, base_sample, session, session.instructions, args)
            if stats.trajectory is not None:
                stats.trajectory.extend(first.messages)
            segments.append(first)

            answer: Optional[str] = None
            try:
                answer, done_reason = await asyncio.wait_for(
                    _episode_loop(args, state, session, input.sampling_params, stats, base_sample, segments),
                    timeout=args.fleet_episode_timeout_s,
                )
            except asyncio.TimeoutError:
                done_reason = "episode_timeout"

            if done_reason == "aborted" or state.aborted:
                for segment in segments:
                    segment.sample.status = Sample.Status.ABORTED
                samples = [s.sample for s in segments]
                return GenerateFnOutput(samples=samples[0] if len(samples) == 1 else samples)

            t0 = time.time()
            grade = await asyncio.to_thread(
                session.grade, answer, stats.reset_ack, done_reason == "submitted"
            )
            stats.env_time += time.time() - t0
    except asyncio.CancelledError:
        for segment in segments:
            segment.sample.status = Sample.Status.ABORTED
        raise
    except Exception as e:
        # An episode must never kill the run. Env prepare failures (an image
        # exceeding its own declared readiness budget on a loaded node killed
        # a 12h run on 2026-08-22), engine errors, or template drift become an
        # ABORTED write-off; the check_no_aborted filter rejects the group and
        # over-sampling replaces it.
        logger.warning("[%s] episode failed, writing off as ABORTED: %s", task_key, e)
        samples = [s.sample for s in segments] or [base_sample]
        for sample in samples:
            sample.status = Sample.Status.ABORTED
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["episode_error"] = str(e)[:300]
        return GenerateFnOutput(samples=samples[0] if len(samples) == 1 else samples)
    finally:
        # Shielded so a cancelled episode still reaps its containers.
        try:
            await asyncio.shield(asyncio.to_thread(session.close))
        except asyncio.CancelledError:
            pass

    episode_meta = {
        "done_reason": done_reason,
        "turns": stats.turns,
        "round_number": stats.turns,  # --log-multi-turn reads this key
        "tool_calls": stats.tool_calls,
        "tool_errors": stats.tool_errors,
        "parse_failures": stats.parse_failures,
        "images": stats.images,
        "steps_closed": stats.steps_closed,
        "segments": len(segments),
        "verifier_failed": 1.0 if grade.verifier_failed else 0.0,
        "attempt_id": session.attempt_id,
    }
    for index, segment in enumerate(segments):
        sample = segment.sample
        sample.reward = grade.reward
        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.COMPLETED
        if segment.mm_chunks:
            sample.multimodal_train_inputs = _merge_mm_chunks(segment.mm_chunks)
        sample.metadata.update(episode_meta)
        sample.metadata["segment_index"] = index
    # env wall-clock is an episode quantity; book it once, not per segment.
    segments[0].sample.non_generation_time = stats.env_time

    samples = [s.sample for s in segments]
    return GenerateFnOutput(samples=samples[0] if len(samples) == 1 else samples)


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
        help="miles TITO tokenizer family (--tito-model values): glm47, qwen3_5, kimi25, ...",
    )
    parser.add_argument(
        "--fleet-vision",
        action="store_true",
        help="screenshots ride into the engine payload and multimodal_train_inputs (VL models)",
    )


generate.add_arguments = _add_arguments
