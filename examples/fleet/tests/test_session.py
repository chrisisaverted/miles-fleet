"""FleetSession against a stubbed SDK: reward mapping, deadlines, tool
outcomes, prepare retry, schema budget, and the step protocol."""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

import fleet_runtime.harness.reference as harness_reference

import examples.fleet.session as session_mod
from examples.fleet.session import FleetSession, SessionConfig, _within_budget, TOOLS_JSON_MAX_CHARS


# --------------------------------------------------------------- stub SDK


@dataclass
class FakeText:
    text: str


@dataclass
class FakeRef:
    digest: str = "sha256:abc"
    media_type: str = "image/png"
    size: int = 3


@dataclass
class FakeBlob:
    ref: FakeRef = field(default_factory=FakeRef)


@dataclass
class FakeResult:
    content: tuple = ()
    status: str = "ok"
    error_code: Optional[str] = None


class FakeChannel:
    def __init__(self, results: Optional[List[Any]] = None):
        self.results = results or []
        self.calls: List[tuple] = []
        self.submitted: List[str] = []
        self.blob_bytes = b"\x89PNG-fake"

    def call_tool(self, name, args):
        self.calls.append((name, args))
        item = self.results.pop(0) if self.results else FakeResult(content=(FakeText("ok"),))
        if callable(item):
            return item()
        if isinstance(item, Exception):
            raise item
        return item

    def read_tool_result_blob(self, ref):
        return self.blob_bytes

    def submit_answer(self, answer):
        self.submitted.append(answer)

    @property
    def instructions(self):
        return "do the thing"


@dataclass
class FakeVerification:
    score: str = "0.5"


@dataclass
class FakeReport:
    ok: bool = True
    failure: Optional[str] = None
    verifications: tuple = ()


class FakeAdvance:
    def __init__(self, continues=True, next_context="reset", prompt="step 2 prompt"):
        self.continues = continues
        self.next_context = next_context
        self.next_step = type("S", (), {"prompt": prompt})() if prompt else None


class FakeInnerSession:
    """Stands in for AttemptSession: the step cursor surface."""

    def __init__(self, step_count=1, advances=None):
        self.step_ordinal = 1
        self.step_count = step_count
        self.pending_reset_ack = None
        self.completed: List[Optional[str]] = []
        self._advances = advances or []

    def complete_step(self, *, reset_ack=None):
        self.completed.append(reset_ack)
        advance = self._advances.pop(0) if self._advances else FakeAdvance(continues=False, next_context="stop", prompt=None)
        if isinstance(advance, Exception):
            raise advance
        if str(advance.next_context).endswith("reset"):
            self.pending_reset_ack = f"sha256:ack{len(self.completed)}"
        else:
            self.pending_reset_ack = None
        if advance.continues:
            self.step_ordinal += 1
        return advance

    def close(self):
        pass


def make_session(monkeypatch, report=None, results=None, step_count=1, advances=None, has_steps=False, **cfg) -> FleetSession:
    session = FleetSession("ts", "t1", SessionConfig(**cfg))
    session._channel = FakeChannel(results=results)
    session._task = type("T", (), {"has_step_evidence": has_steps})()
    session._session = FakeInnerSession(step_count=step_count, advances=advances)
    report = report if report is not None else FakeReport(ok=True)
    grade = report if callable(report) else (lambda task, session: report)
    monkeypatch.setattr(harness_reference, "grade_session", grade)
    return session


# ------------------------------------------------------------ tool outcomes


def test_tool_result_text():
    s = make_session.__wrapped__ if hasattr(make_session, "__wrapped__") else None  # noqa: F841
    session = FleetSession("ts", "t1")
    session._channel = FakeChannel()
    out = session.call_tool("app__do", {"x": 1})
    assert out.error is None and out.text == "ok" and out.images == []
    assert session._channel.calls == [("app__do", {"x": 1})]


def test_tool_error_status_maps_to_error_and_drops_images():
    bad = FakeResult(content=(FakeText("boom detail"), FakeBlob()), status="error", error_code="invalid_args")
    session = FleetSession("ts", "t1", SessionConfig(vision=True))
    session._channel = FakeChannel(results=[bad])
    out = session.call_tool("app__do", {})
    assert out.error is not None and out.error.startswith("invalid_args")
    assert out.images == []


def test_tool_raises_maps_to_error():
    session = FleetSession("ts", "t1")
    session._channel = FakeChannel(results=[RuntimeError("container died")])
    out = session.call_tool("app__do", {})
    assert "container died" in out.error


def test_images_only_in_vision_mode():
    img = FakeResult(content=(FakeText("shot"), FakeBlob()))
    text_mode = FleetSession("ts", "t1", SessionConfig(vision=False))
    text_mode._channel = FakeChannel(results=[img])
    out = text_mode.call_tool("app__do", {})
    assert out.images == [] and "<blob sha256:abc" in out.text

    vision_mode = FleetSession("ts", "t1", SessionConfig(vision=True))
    vision_mode._channel = FakeChannel(results=[FakeResult(content=(FakeText("shot"), FakeBlob()))])
    out = vision_mode.call_tool("app__do", {})
    assert len(out.images) == 1 and out.images[0].startswith("data:image/png;base64,")
    assert "[screenshot]" in out.text


def test_call_tool_hang_hits_deadline():
    def hang():
        time.sleep(3)
        return FakeResult()

    session = FleetSession("ts", "t1", SessionConfig(call_tool_timeout_s=0.2))
    session._channel = FakeChannel(results=[hang])
    start = time.time()
    out = session.call_tool("app__do", {})
    assert time.time() - start < 2
    assert "exceeded" in out.error


# --------------------------------------------------------- reward mapping


def test_reward_ok(monkeypatch):
    session = make_session(monkeypatch, report=FakeReport(ok=True))
    result = session.grade("42")
    assert result.reward == 1.0 and result.verifier_failed is False
    assert session._channel.submitted == ["42"]


def test_reward_verdict_fail_is_not_infra(monkeypatch):
    session = make_session(monkeypatch, report=FakeReport(ok=False))
    result = session.grade(None)
    assert result.reward == 0.0 and result.verifier_failed is False


def test_reward_partial_mean_of_string_scores(monkeypatch):
    report = FakeReport(
        ok=False, verifications=(FakeVerification("1.0"), FakeVerification("0.0"), FakeVerification("junk"))
    )
    session = make_session(monkeypatch, report=report, partial_reward=True)
    assert session.grade(None).reward == pytest.approx(1.0 / 3.0)


def test_reward_infra_failure_flags_verifier(monkeypatch):
    session = make_session(monkeypatch, report=FakeReport(ok=False, failure="judge key rejected"))
    result = session.grade(None)
    assert result.reward == 0.0 and result.verifier_failed is True


def test_reward_incomplete_step_protocol_is_policy_not_infra(monkeypatch):
    session = make_session(monkeypatch, report=FakeReport(ok=False, failure="RUN/step_protocol_incomplete"))
    result = session.grade(None)
    assert result.reward == 0.0 and result.verifier_failed is False


def test_reward_grade_raises_is_infra(monkeypatch):
    def broken(task, session):
        raise RuntimeError("seal failed")

    session = make_session(monkeypatch, report=broken)
    result = session.grade(None)
    assert result.reward == 0.0 and result.verifier_failed is True


def test_grade_hang_hits_deadline_and_flags_verifier(monkeypatch):
    def slow(task, session):
        time.sleep(3)
        return FakeReport(ok=True)

    session = make_session(monkeypatch, report=slow, grade_timeout_s=0.2)
    result = session.grade("42")
    assert result.reward == 0.0 and result.verifier_failed is True


def test_grade_twice_raises(monkeypatch):
    session = make_session(monkeypatch)
    session.grade(None)
    with pytest.raises(RuntimeError, match="twice"):
        session.grade(None)


def test_submit_answer_failure_does_not_kill_grading(monkeypatch):
    session = make_session(monkeypatch, report=FakeReport(ok=True))

    def bad_submit(answer):
        raise RuntimeError("channel gone")

    session._channel.submit_answer = bad_submit
    assert session.grade("42").reward == 1.0


# ------------------------------------------------------------ step protocol


def test_close_step_preserve(monkeypatch):
    session = make_session(
        monkeypatch, step_count=3, has_steps=True, advances=[FakeAdvance(next_context="preserve", prompt="p2")]
    )
    info = session.close_step(None)
    assert info.continues is True and info.reset is False
    assert info.next_prompt == "p2" and info.reset_ack is None
    assert session._session.completed == [None]


def test_close_step_reset_carries_ack(monkeypatch):
    session = make_session(
        monkeypatch, step_count=3, has_steps=True, advances=[FakeAdvance(next_context="reset", prompt="p2")]
    )
    info = session.close_step(None)
    assert info.reset is True and info.reset_ack == "sha256:ack1"


def test_close_step_stop(monkeypatch):
    session = make_session(monkeypatch, step_count=3, has_steps=True, advances=[FakeAdvance(continues=False, next_context="stop", prompt=None)])
    info = session.close_step(None)
    assert info.continues is False


def test_grade_closes_final_step_for_step_protocol(monkeypatch):
    session = make_session(monkeypatch, step_count=2, has_steps=True, advances=[FakeAdvance(continues=False, next_context="not_applicable", prompt=None)])
    session.grade("42", reset_ack="sha256:prev", close_final_step=True)
    assert session._session.completed == ["sha256:prev"]
    assert session._channel.submitted == ["42"]


def test_grade_skips_complete_step_for_bare_tasks(monkeypatch):
    session = make_session(monkeypatch, has_steps=False)
    session.grade("42", close_final_step=True)
    assert session._session.completed == []


# ----------------------------------------------------------- schema budget


def test_tool_budget_drops_whole_schemas():
    small = {"type": "function", "function": {"name": "small", "parameters": {}}}
    huge = {"type": "function", "function": {"name": "huge", "parameters": {"d": "x" * TOOLS_JSON_MAX_CHARS}}}
    kept, dropped = _within_budget([small, huge, small])
    assert dropped == 1
    assert [t["function"]["name"] for t in kept] == ["small", "small"]


# ---------------------------------------------------------- prepare retry


def test_prepare_retries_transient_failures(monkeypatch):
    monkeypatch.setattr(session_mod, "PREPARE_BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(session_mod, "PREPARE_ATTEMPTS", 3)
    calls = {"n": 0}

    class FlakyRuntime:
        def prepare(self, task, source):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("docker network create timed out after 30 seconds")
            return "prepared"

    assert session_mod._prepare_with_retry(FlakyRuntime(), object(), "src", "t1") == "prepared"
    assert calls["n"] == 3


def test_prepare_gives_up_with_cause(monkeypatch):
    monkeypatch.setattr(session_mod, "PREPARE_BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(session_mod, "PREPARE_ATTEMPTS", 3)

    class DeadRuntime:
        def prepare(self, task, source):
            raise TimeoutError("still dead")

    with pytest.raises(RuntimeError, match="after 3 attempts") as exc:
        session_mod._prepare_with_retry(DeadRuntime(), object(), "src", "t1")
    assert isinstance(exc.value.__cause__, TimeoutError)


# ------------------------------------------------------------ constructor


def test_constructor_requires_identity():
    with pytest.raises(ValueError):
        FleetSession("", "t1")
    with pytest.raises(ValueError):
        FleetSession("ts", "")


def test_call_tool_timeout_is_fatal(monkeypatch):
    """PR review: a deadline expiry cannot cancel the SDK call, so the
    outcome must be terminal instead of an ordinary tool error."""
    import examples.fleet.session as S

    session = S.FleetSession.__new__(S.FleetSession)
    session.config = S.SessionConfig(call_tool_timeout_s=0.01)

    class SlowChannel:
        def call_tool(self, name, arguments):
            import time
            time.sleep(0.5)

    session._channel = SlowChannel()
    out = session.call_tool("t", {})
    assert out.fatal and "exceeded" in (out.error or "")


def test_prepare_retry_sweeps_between_attempts(monkeypatch):
    """PR review: each failed prepare attempt must clean its leftovers."""
    import examples.fleet.session as S

    sweeps = []
    monkeypatch.setattr(S, "sweep_prepare_leftovers", lambda: sweeps.append(1))
    monkeypatch.setattr(S, "PREPARE_ATTEMPTS", 3)
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)

    class FailingRuntime:
        def prepare(self, task, source):
            raise RuntimeError("docker slow")

    import pytest
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        S._prepare_with_retry(FailingRuntime(), task=None, source=None, task_key="t")
    assert len(sweeps) == 3
