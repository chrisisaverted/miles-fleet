"""Fleet agent loop: the message-level half of the rollout.

This is the harness. It drives one episode of one Fleet task as a
conversation: boot the environment, sample a turn, parse and execute the tool
call, feed the observation back, close steps, submit, grade. It works in
messages, tool schemas, and rewards; it never touches tokens, loss masks, or
Samples. Those live behind the TurnRunner interface, implemented by
examples.fleet.recording.

The cut follows miles's integration-shapes table (docs/user-guide/
environments): this module is the body a --custom-agent-function-path
integration would keep verbatim once miles's session-server recording carries
screenshots; recording.py is the part the session server would replace.

Environment interface (examples.fleet.session.FleetSession), in gym terms:
open() is reset, call_tool() is step, grade() is the episodic reward;
fleet_submit, boundary stop rules, and the turn/step budgets decide done.
Every session call runs through asyncio.to_thread because the SDK blocks on
docker.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from examples.fleet.parser import parse_tool_call
from examples.fleet.session import SUBMIT_TOOL, FleetSession, GradeResult


@dataclass(frozen=True)
class Turn:
    """One sampled model turn. finish: "ok", "length" (per-turn token cap),
    "context_full" (no room for another turn), "aborted" (engine abort)."""

    text: Optional[str]
    finish: str


class TurnRunner(Protocol):
    """What the agent loop needs from the policy side. The recording
    implementation (examples.fleet.recording.Recorder) speaks SGLang
    /generate with token ids and assembles training Samples; a session-server
    implementation would just relay messages."""

    @property
    def aborted(self) -> bool:
        """True when the run is shutting the rollout down (engine abort)."""
        ...

    def begin_segment(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> None:
        """Start a fresh conversation from a message prefix. Called once at
        episode start and again after every reset boundary."""
        ...

    async def sample(self) -> Turn:
        """Sample the next assistant turn of the live conversation."""
        ...

    def append_assistant(self, text: str, tool_call: Optional[Dict[str, Any]], turn: int) -> Dict[str, Any]:
        """Record the sampled turn as an assistant message; returns it."""
        ...

    def append_observation(self, message: Dict[str, Any], image_urls: List[str]) -> Dict[str, Any]:
        """Append a tool/user observation (with optional screenshots as data
        URLs) to the live conversation; returns the message as appended."""
        ...

    def note_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Debug-trajectory hook: stash the running message log."""
        ...


@dataclass(frozen=True)
class AgentConfig:
    max_turns: int
    episode_timeout_s: float


@dataclass
class EpisodeStats:
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


@dataclass(frozen=True)
class EpisodeResult:
    answer: Optional[str]
    done_reason: str
    grade: Optional[GradeResult]  # None means aborted: nothing to train on


# ----------------------------------------------------------------- messages


def build_messages(instructions: str, max_turns: int, step_ordinal: int, step_count: int) -> List[Dict[str, Any]]:
    step_note = ""
    if step_count > 1:
        step_note = (
            f"This task has {step_count} sequential steps; you are on step {step_ordinal}. "
            f"Call {SUBMIT_TOOL} when you finish the CURRENT step; the next step's "
            "instructions follow. Only the final step takes an answer.\n"
        )
    system = (
        "You are completing a task in a live environment by calling tools.\n"
        f"You have at most {max_turns} turns. End EVERY response with exactly one tool call.\n"
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


# --------------------------------------------------------------------- loop


async def _episode_loop(
    session: FleetSession,
    runner: TurnRunner,
    cfg: AgentConfig,
    stats: EpisodeStats,
) -> Tuple[Optional[str], str]:
    """Run turns until a terminal condition. Returns (answer, done_reason);
    answer is non-None only for done_reason == "submitted". Grading happens
    in the caller, outside the episode wall clock."""

    def record(message: Dict[str, Any]) -> None:
        if stats.trajectory is not None:
            stats.trajectory.append(message)
            runner.note_messages(stats.trajectory)

    while True:
        turn = await runner.sample()
        if turn.finish == "aborted":
            return None, "aborted"
        if turn.finish == "context_full":
            return None, "context_full"
        stats.turns += 1
        tool_call = parse_tool_call(turn.text)
        record(runner.append_assistant(turn.text, tool_call, stats.turns))
        if turn.finish == "length":
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
            if stats.turns >= cfg.max_turns:
                return None, "max_turns"
            next_instructions = advance.next_prompt or session.current_instructions
            if advance.reset:
                # The conversation dies at the boundary; the world survives.
                # The runner finalizes this training sequence and opens a
                # fresh one.
                runner.begin_segment(
                    build_messages(next_instructions, cfg.max_turns, session.step_ordinal, session.step_count),
                    session.tools,
                )
            else:
                body = (
                    f"Step {session.step_ordinal - 1} complete. "
                    f"Step {session.step_ordinal}/{session.step_count}:\n{next_instructions}"
                    f"\n[Turn {stats.turns}/{cfg.max_turns}]"
                )
                record(runner.append_observation({"role": "user", "content": body}, []))
            continue

        # ---------------------- ordinary tool turn -------------------------
        turn_images: List[str] = []
        if tool_call:
            stats.tool_calls += 1
            t0 = time.time()
            outcome = await asyncio.to_thread(session.call_tool, tool_call["name"], tool_call.get("arguments") or {})
            stats.env_time += time.time() - t0
            if outcome.fatal:
                # deadline expiry: the SDK call cannot be cancelled, so the
                # episode ends here and close() tears the container down
                raise RuntimeError(f"tool call deadline expired: {outcome.error}")
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
        if stats.turns >= cfg.max_turns:
            return None, "max_turns"

        body += f"\n[Turn {stats.turns}/{cfg.max_turns}]"
        if tool_call:
            message: Dict[str, Any] = {
                "role": "tool",
                "tool_call_id": f"call_{stats.turns:06d}",
                "name": tool_call["name"],
                "content": body,
            }
        else:
            message = {"role": "user", "content": body}
        record(runner.append_observation(message, turn_images))


# -------------------------------------------------------------------- agent


async def run_agent(
    session: FleetSession,
    runner: TurnRunner,
    cfg: AgentConfig,
    stats: EpisodeStats,
    prepare_gate: Optional[asyncio.Semaphore] = None,
) -> EpisodeResult:
    """One full episode: open the environment, run the loop under the episode
    wall clock, grade. Every terminal reason except abort grades: without a
    submission answer-style verifiers see a null submission, but state-capture
    verifiers still grade real work. The caller owns session.close()."""
    t0 = time.time()
    if prepare_gate is not None:
        async with prepare_gate:
            await asyncio.to_thread(session.open)
    else:
        await asyncio.to_thread(session.open)
    stats.env_time += time.time() - t0

    first = build_messages(session.instructions, cfg.max_turns, session.step_ordinal, session.step_count)
    runner.begin_segment(first, session.tools)
    if stats.trajectory is not None:
        stats.trajectory.extend(first)

    answer: Optional[str] = None
    done_reason = "unknown"
    try:
        answer, done_reason = await asyncio.wait_for(
            _episode_loop(session, runner, cfg, stats), timeout=cfg.episode_timeout_s
        )
    except asyncio.TimeoutError:
        done_reason = "episode_timeout"

    if done_reason == "aborted" or runner.aborted:
        return EpisodeResult(answer=None, done_reason="aborted", grade=None)

    t0 = time.time()
    grade = await asyncio.to_thread(session.grade, answer, stats.reset_ack, done_reason == "submitted")
    stats.env_time += time.time() - t0
    return EpisodeResult(answer=answer, done_reason=done_reason, grade=grade)
