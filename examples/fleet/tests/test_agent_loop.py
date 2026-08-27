"""The agent loop against a fake TurnRunner and a stubbed FleetSession.

Message-level only: no tokens, masks, or Samples anywhere. Pins the harness
sequencing that test_rollout_loop.py can only observe indirectly through
token assembly: what the loop asks the runner to do, in what order, with
which messages."""

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from examples.fleet.agent import AgentConfig, EpisodeStats, Turn, build_messages, run_agent
from examples.fleet.session import GradeResult, StepAdvanceInfo, ToolOutcome

from .test_rollout_loop import CALL, SUBMIT, SUBMIT_EMPTY, FakeFleetSession


class FakeRunner:
    """Scripted turns in; a flat event log out."""

    def __init__(self, turns: List[Any], aborted: bool = False):
        self._turns = list(turns)
        self.aborted = aborted
        self.events: List[tuple] = []
        self.noted: Optional[List[Dict[str, Any]]] = None

    def begin_segment(self, messages, tools):
        self.events.append(("begin_segment", messages, tools))

    async def sample(self) -> Turn:
        entry = self._turns.pop(0)
        if callable(entry):
            entry = await entry()
        if isinstance(entry, str):
            entry = Turn(text=entry, finish="ok")
        self.events.append(("sample",))
        return entry

    def append_assistant(self, text, tool_call, turn):
        message = {"role": "assistant", "content": text}
        self.events.append(("append_assistant", text, tool_call, turn))
        return message

    def append_observation(self, message, image_urls):
        self.events.append(("append_observation", message, image_urls))
        return message

    def note_messages(self, messages):
        self.noted = list(messages)


def make_cfg(**overrides) -> AgentConfig:
    base = dict(max_turns=8, episode_timeout_s=30.0)
    base.update(overrides)
    return AgentConfig(**base)


def run(session, runner, cfg=None, stats=None):
    stats = stats or EpisodeStats()
    result = asyncio.run(run_agent(session, runner, cfg or make_cfg(), stats))
    return result, stats


def events_of(runner, kind):
    return [e for e in runner.events if e[0] == kind]


# -------------------------------------------------------------------- basics


def test_open_then_first_segment_then_grade():
    session = FakeFleetSession()
    runner = FakeRunner([SUBMIT])
    result, stats = run(session, runner)
    assert session.opened
    # first segment carries the built system+user messages and the tool surface
    kind, messages, tools = runner.events[0]
    assert kind == "begin_segment"
    assert messages == build_messages("do the thing", 8, 1, 1)
    assert tools is session.tools
    assert result.done_reason == "submitted" and result.answer == "42"
    assert result.grade == GradeResult(reward=1.0)
    assert session.graded_with == ("42", None, True)
    assert not session.closed  # close is the caller's job


def test_tool_turn_sequencing():
    session = FakeFleetSession()
    runner = FakeRunner([CALL, SUBMIT])
    run(session, runner)
    kinds = [e[0] for e in runner.events]
    assert kinds == ["begin_segment", "sample", "append_assistant", "append_observation", "sample", "append_assistant"]
    (_, message, image_urls) = events_of(runner, "append_observation")[0]
    assert message["role"] == "tool" and message["name"] == "app__do"
    assert "Tool result:\nok" in message["content"] and "[Turn 1/8]" in message["content"]
    assert image_urls == []
    assert session.calls == [("app__do", {"x": 1})]


def test_image_urls_pass_through_untouched():
    session = FakeFleetSession(tool_outcomes=[ToolOutcome(text="shot", images=["data:image/png;base64,xx"])])
    runner = FakeRunner([CALL, SUBMIT])
    _, stats = run(session, runner)
    (_, _, image_urls) = events_of(runner, "append_observation")[0]
    assert image_urls == ["data:image/png;base64,xx"]
    assert stats.images == 1


def test_budget_check_runs_after_the_tool_executes():
    session = FakeFleetSession()
    runner = FakeRunner([CALL, CALL])
    result, stats = run(session, runner, cfg=make_cfg(max_turns=2))
    assert result.done_reason == "max_turns"
    assert len(session.calls) == 2  # the last call's side effects landed
    assert len(events_of(runner, "append_observation")) == 1  # turn 2's result is never appended
    assert session.graded_with == (None, None, False)


def test_parse_failure_appends_user_nudge():
    session = FakeFleetSession()
    runner = FakeRunner(["let me think...", SUBMIT])
    _, stats = run(session, runner)
    (_, message, _) = events_of(runner, "append_observation")[0]
    assert message["role"] == "user" and "No tool call found" in message["content"]
    assert stats.parse_failures == 1
    assert session.calls == []


def test_length_finish_records_the_turn_then_stops():
    session = FakeFleetSession(grade_result=GradeResult(reward=0.0))
    runner = FakeRunner([Turn(text="rambling", finish="length")])
    result, _ = run(session, runner)
    assert result.done_reason == "length"
    assert events_of(runner, "append_assistant")  # the sampled turn was recorded
    assert session.graded_with is not None


def test_context_full_stops_before_recording():
    session = FakeFleetSession()
    runner = FakeRunner([Turn(text=None, finish="context_full")])
    result, _ = run(session, runner)
    assert result.done_reason == "context_full"
    assert events_of(runner, "append_assistant") == []
    assert session.graded_with is not None


def test_abort_skips_grading():
    session = FakeFleetSession()
    runner = FakeRunner([Turn(text=None, finish="aborted")])
    result, _ = run(session, runner)
    assert result.done_reason == "aborted" and result.grade is None
    assert session.graded_with is None


def test_fatal_tool_outcome_raises():
    session = FakeFleetSession(tool_outcomes=[ToolOutcome(text="", error="call_tool exceeded 300s", fatal=True)])
    runner = FakeRunner([CALL])
    with pytest.raises(RuntimeError, match="deadline expired"):
        run(session, runner)


def test_episode_timeout_still_grades():
    async def hang():
        await asyncio.sleep(5)
        return Turn(text=CALL, finish="ok")

    session = FakeFleetSession(grade_result=GradeResult(reward=0.0))
    runner = FakeRunner([hang])
    result, _ = run(session, runner, cfg=make_cfg(episode_timeout_s=0.2))
    assert result.done_reason == "episode_timeout"
    assert session.graded_with == (None, None, False)


# --------------------------------------------------------------- multi-step


def test_preserve_boundary_appends_next_prompt():
    session = FakeFleetSession(
        step_count=2,
        has_steps=True,
        advances=[StepAdvanceInfo(continues=True, reset=False, next_prompt="now step 2", reset_ack=None)],
    )
    runner = FakeRunner([SUBMIT_EMPTY, SUBMIT])
    result, stats = run(session, runner)
    assert len(events_of(runner, "begin_segment")) == 1  # same conversation
    (_, message, _) = events_of(runner, "append_observation")[0]
    assert message["role"] == "user" and "Step 2/2:\nnow step 2" in message["content"]
    assert stats.steps_closed == 1
    assert result.done_reason == "submitted"


def test_reset_boundary_begins_fresh_segment_and_threads_ack():
    session = FakeFleetSession(
        step_count=2,
        has_steps=True,
        advances=[StepAdvanceInfo(continues=True, reset=True, next_prompt="fresh step 2", reset_ack="sha256:ack1")],
    )
    runner = FakeRunner([SUBMIT_EMPTY, SUBMIT])
    result, stats = run(session, runner)
    begins = events_of(runner, "begin_segment")
    assert len(begins) == 2
    assert begins[1][1] == build_messages("fresh step 2", 8, 2, 2)
    assert stats.reset_ack == "sha256:ack1"
    assert session.graded_with == ("42", "sha256:ack1", True)
    assert result.done_reason == "submitted"


def test_boundary_stop_ends_episode():
    session = FakeFleetSession(
        step_count=3,
        has_steps=True,
        advances=[StepAdvanceInfo(continues=False, reset=False, next_prompt=None, reset_ack=None)],
        grade_result=GradeResult(reward=0.0),
    )
    runner = FakeRunner([SUBMIT_EMPTY])
    result, _ = run(session, runner)
    assert result.done_reason == "boundary_stop"
    assert session.graded_with == (None, None, False)


# --------------------------------------------------------------- trajectory


def test_trajectory_records_prompt_and_every_message():
    session = FakeFleetSession()
    runner = FakeRunner([CALL, SUBMIT])
    stats = EpisodeStats(trajectory=[])
    run(session, runner, stats=stats)
    roles = [m["role"] for m in stats.trajectory]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert runner.noted == stats.trajectory  # runner saw the running log


def test_no_trajectory_means_no_notes():
    session = FakeFleetSession()
    runner = FakeRunner([CALL, SUBMIT])
    run(session, runner)
    assert runner.noted is None
