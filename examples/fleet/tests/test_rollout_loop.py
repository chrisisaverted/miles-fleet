"""generate() end to end against a stubbed FleetSession and a fake /generate.

Covers the loop mechanics the session tests cannot: token/loss-mask assembly,
budget-after-execute ordering, submit/step/reset handling, abort, timeout,
context caps, and metadata."""

import asyncio
import json
from argparse import Namespace
from typing import Any, Dict, List, Optional

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

import examples.fleet.rollout as rollout_mod
from examples.fleet.session import GradeResult, StepAdvanceInfo, ToolOutcome


# ----------------------------------------------------------------- fixtures


class StubTokenizer:
    """Append-only renderer; one token per character."""

    def _render(self, messages, tools=None, add_generation_prompt=False):
        parts = []
        if tools:
            names = ",".join(t["function"]["name"] for t in tools)
            parts.append(f"<tools>{names}</tools>")
        for m in messages:
            parts.append(f"<{m['role']}>{m.get('content') or ''}</>")
        if add_generation_prompt:
            parts.append("<gen>")
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, return_dict=False):
        text = self._render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeTito:
    """TITO stand-in over the stub tokenizer: plain concatenation merge."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, add_generation_prompt=False, tools=None, tokenize=False):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, tools=tools
        )

    def merge_tokens(self, old_messages, new_messages, pretokenized_token_ids, tools=None):
        appended = new_messages[len(old_messages):]
        incremental = self.tokenizer.encode(
            self.tokenizer._render(appended, add_generation_prompt=True)
        )
        return list(pretokenized_token_ids) + incremental


class FakeFleetSession:
    """Configured per test; the rollout module's FleetSession is patched to
    return this instance."""

    def __init__(
        self,
        *,
        step_count: int = 1,
        has_steps: bool = False,
        advances: Optional[List[StepAdvanceInfo]] = None,
        tool_outcomes: Optional[List[ToolOutcome]] = None,
        grade_result: GradeResult = GradeResult(reward=1.0),
        instructions: str = "do the thing",
    ):
        self.step_count = step_count
        self.step_ordinal = 1
        self.has_step_protocol = has_steps
        self.instructions = instructions
        self.current_instructions = instructions
        self.attempt_id = "attempt-1"
        self.tools = [{"type": "function", "function": {"name": "app__do", "parameters": {}}}]
        self._advances = list(advances or [])
        self._tool_outcomes = list(tool_outcomes or [])
        self._grade_result = grade_result
        self.opened = False
        self.closed = False
        self.calls: List[tuple] = []
        self.graded_with: Optional[tuple] = None

    @property
    def is_final_step(self):
        return self.step_ordinal == self.step_count

    def open(self):
        self.opened = True

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._tool_outcomes:
            return self._tool_outcomes.pop(0)
        return ToolOutcome(text="ok")

    def close_step(self, reset_ack):
        info = self._advances.pop(0)
        if info.continues:
            self.step_ordinal += 1
            self.current_instructions = info.next_prompt or ""
        return info

    def grade(self, answer, reset_ack=None, close_final_step=False):
        self.graded_with = (answer, reset_ack, close_final_step)
        return self._grade_result

    def close(self):
        self.closed = True


def make_args(**overrides) -> Namespace:
    base = dict(
        partial_rollout=False,
        save_debug_trajectory_data=None,
        sglang_router_ip="stub",
        sglang_router_port=0,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
        rollout_max_response_len=256,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fleet_max_turns=8,
        fleet_max_tokens_per_turn=64,
        fleet_max_concurrent_envs=4,
        fleet_episode_timeout_s=30.0,
        fleet_runtime_root=None,
        fleet_partial_reward=False,
        fleet_tool_output_max_chars=4000,
        fleet_screenshot_max_dim=0,
        fleet_call_tool_timeout_s=5.0,
        fleet_grade_timeout_s=5.0,
        fleet_tito_model="default",
        fleet_vision=False,
    )
    base.update(overrides)
    return Namespace(**base)


class StubState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = StubTokenizer()
        self.processor = None
        self.aborted = False


CALL = '<tool_call>{"name": "app__do", "arguments": {"x": 1}}</tool_call>'
SUBMIT = '<tool_call>{"name": "fleet_submit", "arguments": {"answer": "42"}}</tool_call>'
SUBMIT_EMPTY = '<tool_call>{"name": "fleet_submit", "arguments": {}}</tool_call>'


def make_post(script: List[Any], tokenizer: StubTokenizer):
    """Each script entry is assistant text (finish stop), or a dict with
    text/finish keys, or an awaitable factory."""

    async def fake_post(url, payload, headers=None):
        entry = script.pop(0)
        if callable(entry):
            entry = await entry()
        if isinstance(entry, str):
            entry = {"text": entry, "finish": "stop"}
        ids = tokenizer.encode(entry["text"])
        return {
            "text": entry["text"],
            "meta_info": {
                "finish_reason": {"type": entry.get("finish", "stop")},
                "output_token_logprobs": [(-0.1, tid, None) for tid in ids],
            },
        }

    return fake_post


def run_generate(monkeypatch, session: FakeFleetSession, script: List[Any], args=None, aborted=False):
    args = args or make_args()
    state = StubState(args)
    state.aborted = aborted
    monkeypatch.setattr(rollout_mod, "FleetSession", lambda *a, **k: session)
    monkeypatch.setattr(rollout_mod, "post", make_post(script, state.tokenizer))
    monkeypatch.setattr(rollout_mod, "_SWEPT", True)
    monkeypatch.setattr(rollout_mod, "_ENV_SEMAPHORE", None)
    monkeypatch.setattr(rollout_mod, "_tito_for", lambda state_, args_: FakeTito(state_.tokenizer))
    sample = Sample(prompt=[{"role": "user", "content": "row prompt"}], metadata={"taskset_ref": "ts", "task_key": "t1"})
    fn_input = GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 64}, evaluation=False)
    output = asyncio.run(rollout_mod.generate(fn_input))
    return output.samples


# -------------------------------------------------------------- happy paths


def test_tool_turn_then_submit(monkeypatch):
    session = FakeFleetSession()
    sample = run_generate(monkeypatch, session, [CALL, SUBMIT])
    assert isinstance(sample, Sample)
    assert sample.reward == 1.0
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata["done_reason"] == "submitted"
    assert sample.metadata["turns"] == 2 and sample.metadata["tool_calls"] == 1
    assert session.calls == [("app__do", {"x": 1})]
    assert session.graded_with == ("42", None, True)
    assert session.opened and session.closed

    # token/mask integrity: masks cover exactly the response region
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length
    assert len(sample.tokens) > sample.response_length
    gen_tokens = len(StubTokenizer().encode(CALL)) + len(StubTokenizer().encode(SUBMIT))
    assert sum(sample.loss_mask) == gen_tokens
    # observation region is masked out and contains the tool result
    assert "Tool result:\nok" in sample.response and "[Turn 1/8]" in sample.response


def test_prompt_carries_tools_and_instructions(monkeypatch):
    session = FakeFleetSession()
    sample = run_generate(monkeypatch, session, [SUBMIT])
    prompt_len = len(sample.tokens) - sample.response_length
    prompt_text = StubTokenizer().decode(sample.tokens[:prompt_len])
    assert "<tools>app__do</tools>" in prompt_text
    assert "do the thing" in prompt_text
    assert "fleet_submit" in prompt_text


def test_submit_non_string_answer_json_dumped(monkeypatch):
    session = FakeFleetSession()
    run_generate(monkeypatch, session, ['<tool_call>{"name": "fleet_submit", "arguments": {"answer": {"k": 1}}}</tool_call>'])
    answer, _, _ = session.graded_with
    assert json.loads(answer) == {"k": 1}


def test_submit_empty_answer_is_none(monkeypatch):
    session = FakeFleetSession()
    run_generate(monkeypatch, session, [SUBMIT_EMPTY])
    assert session.graded_with[0] is None


def test_budget_exhaustion_grades_after_last_call(monkeypatch):
    session = FakeFleetSession()
    sample = run_generate(monkeypatch, session, [CALL, CALL], args=make_args(fleet_max_turns=2))
    assert sample.metadata["done_reason"] == "max_turns"
    assert len(session.calls) == 2  # the last tool call still executed
    assert session.graded_with == (None, None, False)


def test_tool_error_and_parse_failure_bodies(monkeypatch):
    session = FakeFleetSession(tool_outcomes=[ToolOutcome(text="", error="invalid_args: boom")])
    sample = run_generate(monkeypatch, session, [CALL, "let me think...", SUBMIT])
    assert sample.metadata["tool_errors"] == 1
    assert sample.metadata["parse_failures"] == 1
    assert "Error: invalid_args: boom" in sample.response
    assert "No tool call found" in sample.response


# ------------------------------------------------------------ unhappy paths


def test_pre_open_abort(monkeypatch):
    session = FakeFleetSession()
    sample = run_generate(monkeypatch, session, [], aborted=True)
    assert sample.status == Sample.Status.ABORTED
    assert session.graded_with is None


def test_engine_abort_mid_episode(monkeypatch):
    session = FakeFleetSession()
    sample = run_generate(monkeypatch, session, [{"text": CALL, "finish": "abort"}])
    assert sample.status == Sample.Status.ABORTED
    assert session.graded_with is None
    assert session.closed


def test_length_finish_still_grades(monkeypatch):
    session = FakeFleetSession(grade_result=GradeResult(reward=0.0))
    sample = run_generate(monkeypatch, session, [{"text": "rambling" * 5, "finish": "length"}])
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.metadata["done_reason"] == "length"
    assert session.graded_with == (None, None, False)


def test_context_full_still_grades(monkeypatch):
    session = FakeFleetSession()
    sample = run_generate(monkeypatch, session, [], args=make_args(rollout_max_context_len=10))
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.metadata["done_reason"] == "context_full"
    assert session.graded_with is not None


def test_episode_timeout_still_grades(monkeypatch):
    async def hang():
        await asyncio.sleep(5)
        return {"text": CALL, "finish": "stop"}

    session = FakeFleetSession(grade_result=GradeResult(reward=0.0))
    sample = run_generate(monkeypatch, session, [hang], args=make_args(fleet_episode_timeout_s=0.2))
    assert sample.metadata["done_reason"] == "episode_timeout"
    assert session.graded_with is not None
    assert session.closed


def test_close_runs_when_grade_raises(monkeypatch):
    session = FakeFleetSession()

    def broken(answer, reset_ack=None, close_final_step=False):
        raise RuntimeError("boom")

    session.grade = broken
    with pytest.raises(RuntimeError, match="boom"):
        run_generate(monkeypatch, session, [SUBMIT])
    assert session.closed


def test_missing_identity_raises(monkeypatch):
    args = make_args()
    state = StubState(args)
    monkeypatch.setattr(rollout_mod, "_SWEPT", True)
    monkeypatch.setattr(rollout_mod, "_ENV_SEMAPHORE", None)
    sample = Sample(prompt=[], metadata={})
    fn_input = GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)
    with pytest.raises(ValueError, match="taskset_ref"):
        asyncio.run(rollout_mod.generate(fn_input))


# --------------------------------------------------------------- multi-step


def test_multistep_preserve_appends_next_prompt(monkeypatch):
    session = FakeFleetSession(
        step_count=2,
        has_steps=True,
        advances=[StepAdvanceInfo(continues=True, reset=False, next_prompt="now step 2", reset_ack=None)],
    )
    sample = run_generate(monkeypatch, session, [SUBMIT_EMPTY, SUBMIT])
    assert isinstance(sample, Sample)  # one training sequence
    assert "Step 2/2:\nnow step 2" in sample.response
    assert sample.metadata["steps_closed"] == 1
    assert session.graded_with == ("42", None, True)
    # the step-advance message is observation, not policy output
    assert sum(sample.loss_mask) == len(StubTokenizer().encode(SUBMIT_EMPTY)) + len(StubTokenizer().encode(SUBMIT))


def test_multistep_reset_splits_segments_and_broadcasts_reward(monkeypatch):
    session = FakeFleetSession(
        step_count=2,
        has_steps=True,
        advances=[StepAdvanceInfo(continues=True, reset=True, next_prompt="fresh step 2", reset_ack="sha256:ack1")],
        grade_result=GradeResult(reward=1.0),
    )
    samples = run_generate(monkeypatch, session, [SUBMIT_EMPTY, CALL, SUBMIT])
    assert isinstance(samples, list) and len(samples) == 2
    first, second = samples
    assert first.reward == 1.0 and second.reward == 1.0
    assert first.status == Sample.Status.COMPLETED and second.status == Sample.Status.COMPLETED
    # segment 2 is a fresh conversation seeded with step 2's prompt
    prompt_len = len(second.tokens) - second.response_length
    prompt_text = StubTokenizer().decode(second.tokens[:prompt_len])
    assert "fresh step 2" in prompt_text
    assert "<tools>app__do</tools>" in prompt_text
    # segment 1 ends at the boundary: its response has no step-2 content
    assert "fresh step 2" not in first.response
    # the ack from the reset boundary reaches the final grade call
    assert session.graded_with == ("42", "sha256:ack1", True)
    assert first.metadata["segment_index"] == 0 and second.metadata["segment_index"] == 1
    assert first.metadata["segments"] == 2


def test_multistep_boundary_stop_grades(monkeypatch):
    session = FakeFleetSession(
        step_count=3,
        has_steps=True,
        advances=[StepAdvanceInfo(continues=False, reset=False, next_prompt=None, reset_ack=None)],
        grade_result=GradeResult(reward=0.0),
    )
    sample = run_generate(monkeypatch, session, [SUBMIT_EMPTY])
    assert sample.metadata["done_reason"] == "boundary_stop"
    assert session.graded_with == (None, None, False)


def test_multistep_submit_on_final_step_grades(monkeypatch):
    session = FakeFleetSession(step_count=1, has_steps=True)
    sample = run_generate(monkeypatch, session, [SUBMIT])
    assert sample.metadata["done_reason"] == "submitted"
    assert session.graded_with == ("42", None, True)
