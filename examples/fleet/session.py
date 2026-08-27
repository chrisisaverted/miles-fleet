"""FleetSession: one Fleet v2 platform attempt, driven synchronously.

The session speaks the fleet-runtime SDK on one side and plain Python types on
the other; it knows nothing about miles. The agent loop (examples.fleet.agent)
calls every method through asyncio.to_thread because the SDK blocks on docker.

Lifecycle:
    session = FleetSession(taskset_ref, task_key, config)
    session.open()          -> tools (OpenAI schemas incl. fleet_submit), instructions
    session.call_tool(...)  -> ToolOutcome(text, images, error)   [deadline-bounded]
    session.grade(answer)   -> GradeResult(reward, verifier_failed) [deadline-bounded]
    session.close()

Reward contract (float):
    report.ok                      -> 1.0
    verifier verdict failed        -> 0.0, or mean of per-verifier scores when
                                      config.partial_reward
    grading infrastructure failed  -> 0.0 AND verifier_failed=True; a spike in
                                      that metric is infra trouble, not policy
                                      regression

Timeouts: the SDK has no per-call deadlines, so a hung docker/MCP call would
pin a rollout worker thread forever. call_tool and grade run under internal
deadlines; on expiry the call's thread is abandoned (bounded leak: the episode
ends and close() tears the session down).

Environments are local docker containers on the node that hosts the rollout
manager. Killed runs leak per-attempt networks and the daemon's address pools
cap out (~31 networks), after which every prepare() fails; call
sweep_leaked_networks() once per process before the first episode.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from examples.fleet.content import MAX_TOOL_OUTPUT_CHARS, to_plain, tool_result_to_content, truncate_text

logger = logging.getLogger(__name__)

# Face name of the platform's submission tool. Present on every attempt
# surface; calling it closes the current step, so the rollout intercepts it
# and asks the session to grade (or advance) instead of forwarding it.
SUBMIT_TOOL = "fleet_submit"

# Some envs expose large tool surfaces; past this budget, schemas are dropped
# whole (never truncated mid-JSON: a truncated schema renders as garbage in
# any chat template, a dropped one is a visible gap).
TOOLS_JSON_MAX_CHARS = 48000

CALL_TOOL_TIMEOUT_S = 300
GRADE_TIMEOUT_S = 900  # grade_session seals + runs every verifier; minutes are normal

# Prepare retries spread over ~6 minutes of backoff: readiness failures
# cluster inside multi-minute node-load windows (measured: a sibling job's
# 32-minute SFS checkpoint save), so a 15s total backoff never outlives one.
PREPARE_ATTEMPTS = 6
PREPARE_BACKOFF_BASE_S = 15.0
PREPARE_BACKOFF_CAP_S = 120.0

# The SDK's default 180s readiness budget (shared sequentially across health
# endpoints + MCP discovery) is too small for ~15-service desktop envs under
# node load; no taskset in the registry declares its own budget and the SDK's
# only knob can merely lower it. The constant is bound into the docker
# driver's module globals at import and read at DockerEnvironment
# construction, so patching that namespace raises the effective default.
READINESS_BUDGET_S = 600.0
try:
    import fleet_runtime.drivers.environment.docker.environment as _fleet_docker_env

    _fleet_docker_env.DEFAULT_ENVIRONMENT_STARTUP_BUDGET_SEC = READINESS_BUDGET_S
except ImportError:  # SDK absent: nothing can run envs anyway
    pass


@dataclass(frozen=True)
class SessionConfig:
    runtime_root: Optional[str] = None
    partial_reward: bool = False
    tool_output_max_chars: int = MAX_TOOL_OUTPUT_CHARS
    screenshot_max_dim: Optional[int] = None
    vision: bool = False  # False: image blobs degrade to digest placeholders
    call_tool_timeout_s: float = CALL_TOOL_TIMEOUT_S
    grade_timeout_s: float = GRADE_TIMEOUT_S


@dataclass(frozen=True)
class ToolOutcome:
    text: str
    images: List[str] = field(default_factory=list)  # data URLs
    error: Optional[str] = None
    # A deadline expiry cannot cancel the underlying SDK call; its worker
    # thread stays alive until the operation returns. Continuing the episode
    # would stack one abandoned thread per hung call, so a timeout is
    # terminal: the caller aborts the episode and close() tears down the
    # container, which errors the hung call out and lets its thread exit.
    fatal: bool = False


@dataclass(frozen=True)
class GradeResult:
    reward: float
    verifier_failed: bool = False


@dataclass(frozen=True)
class StepAdvanceInfo:
    """Plain projection of the SDK's StepAdvance for the rollout loop."""

    continues: bool
    reset: bool
    next_prompt: Optional[str]
    # Digest the NEXT complete_step must present when reset is True.
    reset_ack: Optional[str]


def _prepare_with_retry(runtime, task, source, task_key: str):
    """prepare() with retries: transient docker slowness must not kill training.

    The SDK shells out to docker with fixed 30s subprocess timeouts; under
    node disk load (measured twice on 2026-08-20: a co-located run's image
    pulls) `docker network create` exceeds it and the exception would
    propagate through the rollout. Half-created resources from a failed
    attempt carry the fleet.runtime label and are reclaimed by
    sweep_leaked_networks().
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, PREPARE_ATTEMPTS + 1):
        try:
            return runtime.prepare(task, source=source)
        except Exception as e:
            last_err = e
            logger.warning("[%s] prepare attempt %d/%d failed: %s", task_key, attempt, PREPARE_ATTEMPTS, e)
            sweep_prepare_leftovers()
            if attempt < PREPARE_ATTEMPTS:
                time.sleep(min(PREPARE_BACKOFF_CAP_S, PREPARE_BACKOFF_BASE_S * 2 ** (attempt - 1)))
    raise RuntimeError(f"[{task_key}] prepare failed after {PREPARE_ATTEMPTS} attempts") from last_err


# resolve_source compiles YAML / loads the local store; once per
# (source, root) per process, not once per rollout.
_TASKSET_CACHE: Dict[Tuple[str, Optional[str]], Any] = {}
_TASKSET_CACHE_LOCK = threading.Lock()


def _resolve_taskset(source: str, root: Optional[str]):
    key = (source, root)
    with _TASKSET_CACHE_LOCK:
        cached = _TASKSET_CACHE.get(key)
    if cached is not None:
        return cached
    from fleet_runtime.cli.sources import resolve_source

    taskset, _ = resolve_source(source, root=root) if root else resolve_source(source)
    with _TASKSET_CACHE_LOCK:
        _TASKSET_CACHE.setdefault(key, taskset)
    return _TASKSET_CACHE[key]


def _image_locators_for(root_digest: str, runtime_root: Optional[str]) -> Dict[str, str]:
    """Read the flt client's image-location plan for this taskset root.

    (<runtime-root>/image-locations/sha256/<root-digest>.json). Maps config
    image IDs to registry locations for images the local docker cannot
    address; absent file means everything resolves locally.
    """
    import os
    from pathlib import Path

    base = Path(runtime_root or os.path.expanduser("~/.flt"))
    digest = root_digest.removeprefix("sha256:")
    plan_path = base / "image-locations" / "sha256" / f"{digest}.json"
    if not plan_path.is_file():
        return {}
    try:
        plan = json.loads(plan_path.read_text())
        return {
            image_id: entry["location"]
            for image_id, entry in (plan.get("image_sources") or {}).items()
            if entry.get("location")
        }
    except Exception as e:
        logger.warning("image-locations plan unreadable (%s): %s", plan_path, e)
        return {}


def sweep_prepare_leftovers() -> None:
    """Best-effort cleanup after a failed prepare attempt: remove non-running
    containers carrying the fleet.runtime label (a live episode's containers
    are running, so they are never matched), then the leaked networks."""
    try:
        for state in ("created", "exited"):
            out = subprocess.run(
                ["docker", "ps", "-aq", "--filter", "label=fleet.runtime=1", "--filter", f"status={state}"],
                capture_output=True, text=True, timeout=30,
            )
            for cid in out.stdout.split():
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=30)
    except Exception as e:
        logger.warning("container sweep skipped: %s", e)
    sweep_leaked_networks()


def sweep_leaked_networks() -> int:
    """Remove per-attempt docker networks left by killed runs."""
    try:
        out = subprocess.run(
            ["docker", "network", "ls", "--filter", "label=fleet.runtime=1", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ids = [line for line in out.stdout.split() if line]
        removed = 0
        for net_id in ids:
            rm = subprocess.run(["docker", "network", "rm", net_id], capture_output=True, timeout=30)
            removed += 1 if rm.returncode == 0 else 0
        if removed:
            logger.info("swept %d leaked fleet networks", removed)
        return removed
    except Exception as e:
        logger.warning("network sweep skipped: %s", e)
        return 0


def _with_deadline(fn: Callable[[], Any], timeout_s: float, label: str):
    """Run fn() with a hard deadline. On expiry the worker thread is
    abandoned (it cannot be killed); the episode errors and close() tears the
    session down, which bounds the leak to one thread per hung call."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"fleet-{label}")
    try:
        future = pool.submit(fn)
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"{label} exceeded {timeout_s:.0f}s") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _within_budget(tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Cap the schema list by dropping whole schemas past TOOLS_JSON_MAX_CHARS."""
    kept: List[Dict[str, Any]] = []
    used = 0
    dropped = 0
    for tool in tools:
        size = len(json.dumps(tool, default=str))
        if used + size > TOOLS_JSON_MAX_CHARS:
            dropped += 1
            continue
        kept.append(tool)
        used += size
    return kept, dropped


_SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": SUBMIT_TOOL,
        "description": "Submit your final answer and end the current step. "
        "Call with an empty answer if the task is about environment state.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "The final answer."}},
            "required": [],
        },
    },
}


class FleetSession:
    """One attempt against one Fleet task. Synchronous; call via asyncio.to_thread."""

    def __init__(self, taskset_ref: str, task_key: str, config: SessionConfig | None = None):
        if not taskset_ref or not task_key:
            raise ValueError("FleetSession requires taskset_ref and task_key")
        self.taskset_ref = taskset_ref
        self.task_key = task_key
        self.config = config or SessionConfig()

        self.tools: List[Dict[str, Any]] = []
        self.instructions: str = ""
        self.attempt_id: Optional[str] = None
        self.tools_dropped = 0

        self._task = None
        self._runtime = None
        self._session = None
        self._channel = None
        self._closed = False
        self._graded = False

    # ------------------------------------------------------------------ open

    def open(self) -> None:
        from fleet_runtime.local import LocalRuntime
        from fleet_runtime.session.session import AttemptSession

        taskset = _resolve_taskset(self.taskset_ref, self.config.runtime_root)
        compiled = taskset.select(self.task_key)[0]
        self._task = compiled.task

        locators = _image_locators_for(getattr(taskset, "root_digest", ""), self.config.runtime_root)
        kwargs: Dict[str, Any] = {"image_locators": locators} if locators else {}
        if self.config.runtime_root:
            kwargs["root"] = self.config.runtime_root
        self._runtime = LocalRuntime(**kwargs)
        self._runtime.load_blobs(tuple(compiled.blobs))
        self._session = AttemptSession(
            prepared=_prepare_with_retry(self._runtime, self._task, compiled.source, self.task_key)
        )
        self._channel = self._session.channel

        # close() can win a race against an abandoned open (episode timeout);
        # without this the session never closes and its containers pin the network.
        if self._closed:
            self._close_session()
            raise RuntimeError(f"[{self.task_key}] session closed during open")

        surface = [
            {
                "type": "function",
                "function": {
                    "name": entry.face_name,
                    "description": entry.description or "",
                    "parameters": to_plain(entry.schema) or {"type": "object", "properties": {}},
                },
            }
            for entry in self._channel.tool_surface()
        ]
        kept, dropped = _within_budget(surface)
        self.tools_dropped = dropped
        if dropped:
            logger.warning(
                "[%s] %d tool schemas dropped over the %d-char budget", self.task_key, dropped, TOOLS_JSON_MAX_CHARS
            )
        # The submit tool rides outside the budget: without it no episode can end.
        self.tools = kept + [_SUBMIT_SCHEMA]
        self.instructions = self._channel.instructions or ""
        self.attempt_id = getattr(self._session, "attempt_id", None)

    # ------------------------------------------------------------------ tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolOutcome:
        try:
            result = _with_deadline(
                lambda: self._channel.call_tool(name, arguments),
                self.config.call_tool_timeout_s,
                "call_tool",
            )
        except TimeoutError as e:
            return ToolOutcome(text="", error=str(e), fatal=True)
        except Exception as e:
            return ToolOutcome(text="", error=str(e))

        read_blob = self._channel.read_tool_result_blob if self.config.vision else None
        raw, images = tool_result_to_content(
            result,
            read_blob=read_blob,
            screenshot_max_dim=self.config.screenshot_max_dim,
        )
        text = truncate_text(raw, self.config.tool_output_max_chars)
        if str(getattr(result, "status", "")).endswith("error") or getattr(result, "error_code", None):
            error = f"{getattr(result, 'error_code', 'tool_error')}: {text}" if text else str(
                getattr(result, "error_code", "tool_error")
            )
            # Error results drop their images: the model should read the error.
            return ToolOutcome(text="", error=error)
        return ToolOutcome(text=text, images=images)

    # ------------------------------------------------------------------ steps

    @property
    def has_step_protocol(self) -> bool:
        """True when the task carries step evidence (several steps, or one
        non-bare step); such attempts must close every step via complete_step
        or the seal records step_protocol_incomplete."""
        return bool(getattr(self._task, "has_step_evidence", False))

    @property
    def step_ordinal(self) -> int:
        return self._session.step_ordinal

    @property
    def step_count(self) -> int:
        return self._session.step_count

    @property
    def is_final_step(self) -> bool:
        return self._session.step_ordinal == self._session.step_count

    @property
    def current_instructions(self) -> str:
        """Live: re-reads the current step after every advance."""
        return self._channel.instructions or ""

    def close_step(self, reset_ack: Optional[str]) -> StepAdvanceInfo:
        """Close the open step. Boundary transitions run inside the live env,
        so this is deadline-bounded like a tool call."""
        advance = _with_deadline(
            lambda: self._session.complete_step(reset_ack=reset_ack),
            self.config.call_tool_timeout_s,
            "complete_step",
        )
        next_context = str(getattr(advance, "next_context", ""))
        reset = next_context.endswith("reset") or next_context.endswith("RESET")
        next_step = getattr(advance, "next_step", None)
        return StepAdvanceInfo(
            continues=bool(advance.continues),
            reset=reset,
            next_prompt=getattr(next_step, "prompt", None) if next_step is not None else None,
            reset_ack=self._session.pending_reset_ack if reset else None,
        )

    # ------------------------------------------------------------------ grade

    def grade(self, answer: Optional[str], reset_ack: Optional[str] = None, close_final_step: bool = False) -> GradeResult:
        """Run grading exactly once; map the report to the float contract.

        For step-protocol tasks reached via submission, the final step is
        submitted then closed (submit-then-complete ordering is the SDK
        contract). Budget-exhausted episodes grade with the protocol
        incomplete; that seals as a failure the report maps to reward 0.0
        WITHOUT verifier_failed, because it is a policy outcome.
        """
        if self._graded:
            raise RuntimeError(f"[{self.task_key}] grade called twice")
        self._graded = True
        try:
            from fleet_runtime.harness.reference import grade_session

            if answer is not None:
                try:
                    self._channel.submit_answer(answer)
                except Exception as e:
                    logger.warning("[%s] submit_answer failed: %s", self.task_key, e)
            if close_final_step and self.has_step_protocol:
                try:
                    _with_deadline(
                        lambda: self._session.complete_step(reset_ack=reset_ack),
                        self.config.call_tool_timeout_s,
                        "complete_step",
                    )
                except Exception as e:
                    logger.warning("[%s] closing final step failed: %s", self.task_key, e)
            report = _with_deadline(
                lambda: grade_session(self._task, self._session), self.config.grade_timeout_s, "grade_session"
            )
        except Exception as e:
            logger.warning("[%s] grading infrastructure failed: %s", self.task_key, e)
            return GradeResult(reward=0.0, verifier_failed=True)

        failure = getattr(report, "failure", None)
        if failure is not None:
            # An abandoned step protocol is the model running out of budget
            # mid-task, not grading infrastructure trouble.
            if "step_protocol_incomplete" in str(failure):
                return GradeResult(reward=0.0, verifier_failed=False)
            return GradeResult(reward=0.0, verifier_failed=True)
        if report.ok:
            return GradeResult(reward=1.0)
        if self.config.partial_reward and getattr(report, "verifications", None):
            scores = []
            for v in report.verifications:
                try:
                    scores.append(float(v.score))  # SDK scores are strings
                except (TypeError, ValueError):
                    scores.append(0.0)
            return GradeResult(reward=sum(scores) / len(scores) if scores else 0.0)
        return GradeResult(reward=0.0)

    # ------------------------------------------------------------------ close

    def _close_session(self) -> None:
        session, self._session, self._channel = self._session, None, None
        if session is not None:
            try:
                session.close()
            except Exception as e:
                logger.warning("[%s] session close failed: %s", self.task_key, e)

    def close(self) -> None:
        self._closed = True
        self._close_session()
