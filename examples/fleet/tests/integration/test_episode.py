"""Scripted episodes against a real container (docker-marked).

Requires a Docker daemon and FLEET_FLT pointing at a v2 flt binary (the SDK
shells out to it to compile the YAML taskset). Skipped otherwise.

Drives FleetSession directly (no inference engine): correct submission ->
1.0, wrong -> 0.0, budget-style ungraded-submission path -> 0.0. Also asserts
nothing leaks: no fleet-labeled containers or networks survive close().

Run: pytest -m docker tests/integration/
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.docker

TASKSET = str(Path(__file__).parent / "taskset_hello.yaml")


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


requires_stack = pytest.mark.skipif(
    not (_docker_ok() and os.environ.get("FLEET_FLT")),
    reason="needs a Docker daemon and FLEET_FLT",
)


def make_session():
    from examples.fleet.session import FleetSession

    return FleetSession(TASKSET, "hello")


def _leaked() -> tuple[int, int]:
    nets = subprocess.run(
        ["docker", "network", "ls", "--filter", "label=fleet.runtime=1", "-q"], capture_output=True, text=True
    ).stdout.split()
    boxes = subprocess.run(
        ["docker", "ps", "--filter", "label=fleet.runtime=1", "-q"], capture_output=True, text=True
    ).stdout.split()
    return len(boxes), len(nets)


def _bash_face(session) -> str:
    return next(t["function"]["name"] for t in session.tools if "bash" in t["function"]["name"])


@requires_stack
def test_correct_answer_scores_one():
    session = make_session()
    session.open()
    try:
        assert session.instructions
        assert any(t["function"]["name"] == "fleet_submit" for t in session.tools)
        out = session.call_tool(_bash_face(session), {"command": "echo hello"})
        assert out.error is None and "hello" in out.text
        result = session.grade("hello", None, close_final_step=True)
        assert result.reward == 1.0 and result.verifier_failed is False
    finally:
        session.close()


@requires_stack
def test_wrong_answer_scores_zero():
    session = make_session()
    session.open()
    try:
        result = session.grade("goodbye", None, close_final_step=True)
        assert result.reward == 0.0 and result.verifier_failed is False
    finally:
        session.close()


@requires_stack
def test_no_submission_grades_and_nothing_leaks():
    before = _leaked()
    session = make_session()
    session.open()
    try:
        assert session.call_tool(_bash_face(session), {"command": "pwd"}).error is None
        assert session.call_tool(_bash_face(session), {"command": "ls"}).error is None
        # budget-exhaustion path: grade with no submission
        result = session.grade(None)
        assert result.reward == 0.0
    finally:
        session.close()
    after = _leaked()
    assert after <= before, f"leak: containers/networks went {before} -> {after}"
