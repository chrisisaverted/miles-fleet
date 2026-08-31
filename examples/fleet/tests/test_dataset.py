"""prepare_dataset: row schema, eligibility filters, split determinism."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from examples.fleet.authoritative import AUTHORITATIVE_BACKEND, SAFE_SUCCESS_OBJECTIVE
from examples.fleet.prepare_dataset import build_rows, seed_bytes, split_rows


# ------------------------------------------------------------- fake taskset


@dataclass
class FakeStep:
    prompt: str | None = "do it"


@dataclass
class FakeSeed:
    mounts: list = field(default_factory=list)

    def canonical_data(self):
        return {"mounts": self.mounts}


@dataclass
class FakeEnv:
    image_ref: str = "registry/app:1"
    seed: FakeSeed | None = None


@dataclass
class FakeTask:
    task_id: str = "t1"
    steps: tuple = (FakeStep(),)
    seats: tuple = ("seat1",)
    is_attempt_capable: bool = True
    environments: dict[str, Any] = field(default_factory=lambda: {"env": FakeEnv()})
    context_profile: str = "continuous"
    verifiers: tuple = ()


@dataclass
class FakeCompiled:
    task: FakeTask


@dataclass
class FakeTaskset:
    tasks: list
    name: str = "fake-ts"
    root_digest: str = "sha256:" + "0" * 64


def make_args(**overrides):
    base = dict(max_tasks=0, seed=42)
    base.update(overrides)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------- rows


def test_row_schema():
    ts = FakeTaskset(tasks=[FakeCompiled(FakeTask(task_id="a", steps=(FakeStep("read the file"),)))])
    rows, skipped = build_rows(ts, "ref://ts", make_args())
    assert skipped == []
    row = rows[0]
    assert row["input"] == [{"role": "user", "content": "read the file"}]
    assert row["label"] == "a"
    meta = row["metadata"]
    assert meta["task_key"] == "a" and meta["taskset_ref"] == "ref://ts"
    assert meta["data_source"] == "fake-ts"
    assert meta["steps"] == 1 and meta["context_profile"] == "continuous"


def test_multistep_tasks_are_kept():
    task = FakeTask(task_id="m", steps=(FakeStep("s1"), FakeStep("s2"), FakeStep("s3")))
    ts = FakeTaskset(tasks=[FakeCompiled(task)])
    rows, skipped = build_rows(ts, "ref://ts", make_args())
    assert skipped == []
    assert rows[0]["metadata"]["steps"] == 3
    assert rows[0]["input"][0]["content"] == "s1"  # step 1's prompt feeds the length filter


def test_multiseat_skipped():
    task = FakeTask(task_id="ms", seats=("a", "b"))
    rows, skipped = build_rows(FakeTaskset(tasks=[FakeCompiled(task)]), "r", make_args())
    assert rows == [] and skipped == [("ms", "seats=2")]


def test_not_attempt_capable_skipped():
    task = FakeTask(task_id="na", is_attempt_capable=False)
    rows, skipped = build_rows(FakeTaskset(tasks=[FakeCompiled(task)]), "r", make_args())
    assert skipped == [("na", "no bound prompt")]


def test_oversized_seed_skipped():
    seed = FakeSeed(mounts=[{"remote_blob": {"bytes": 42_000_000_000}}])
    task = FakeTask(task_id="big", environments={"env": FakeEnv(seed=seed)})
    rows, skipped = build_rows(FakeTaskset(tasks=[FakeCompiled(task)]), "r", make_args())
    assert rows == []
    assert "42.0GB" in skipped[0][1]


def test_seed_bytes_sums_mounts():
    seed = FakeSeed(mounts=[{"remote_blob": {"bytes": 3}}, {"remote_blob": {"bytes": 4}}, {}])
    task = FakeTask(environments={"a": FakeEnv(seed=seed), "b": FakeEnv()})
    assert seed_bytes(task) == 7


def test_max_tasks_subsets_deterministically():
    tasks = [FakeCompiled(FakeTask(task_id=f"t{i}")) for i in range(20)]
    ts = FakeTaskset(tasks=tasks)
    rows1, _ = build_rows(ts, "r", make_args(max_tasks=5))
    rows2, _ = build_rows(ts, "r", make_args(max_tasks=5))
    assert len(rows1) == 5
    assert [r["label"] for r in rows1] == [r["label"] for r in rows2]


def test_authoritative_rows_carry_exact_taskdump_v7_binding():
    task_key = "cyber-example__blackbox_ctf_v1"
    provenance = {
        "task_key": task_key,
        "task_version_id": "11111111-1111-4111-8111-111111111111",
        "environment_version_id": "22222222-2222-4222-8222-222222222222",
        "verifier_version_id": "33333333-3333-4333-8333-333333333333",
        "environment_key": "cysec1",
        "environment_version": "v1",
        "data_key": "seed",
        "data_version": "v2",
    }
    verifier = SimpleNamespace(evaluator=SimpleNamespace(params={"task": provenance}))
    task = FakeTask(task_id=task_key, verifiers=(verifier,))
    args = make_args(
        backend=AUTHORITATIVE_BACKEND,
        reward_objective=SAFE_SUCCESS_OBJECTIVE,
    )

    rows, skipped = build_rows(FakeTaskset(tasks=[FakeCompiled(task)]), "ref://sha256", args)

    assert skipped == []
    metadata = rows[0]["metadata"]
    assert metadata["fleet_backend"] == AUTHORITATIVE_BACKEND
    assert metadata["reward_objective"] == SAFE_SUCCESS_OBJECTIVE
    assert metadata["task_version_id"] == provenance["task_version_id"]
    assert metadata["environment_version_id"] == provenance["environment_version_id"]
    assert metadata["verifier_version_id"] == provenance["verifier_version_id"]
    assert metadata["task_prompt"] == "do it"


def test_authoritative_rows_reject_missing_immutable_binding():
    task = FakeTask(task_id="cyber-example__blackbox_ctf_v1")
    args = make_args(
        backend=AUTHORITATIVE_BACKEND,
        reward_objective=SAFE_SUCCESS_OBJECTIVE,
    )
    with pytest.raises(ValueError, match="exactly one TaskDump v7 binding"):
        build_rows(FakeTaskset(tasks=[FakeCompiled(task)]), "ref://sha256", args)


# ------------------------------------------------------------------ split


def test_split_deterministic_and_disjoint():
    rows = [{"label": f"t{i}", "metadata": {"task_key": f"t{i}"}} for i in range(10)]
    train1, eval1 = split_rows(rows, 0.2, seed=7)
    train2, eval2 = split_rows(rows, 0.2, seed=7)
    assert [r["label"] for r in train1] == [r["label"] for r in train2]
    assert [r["label"] for r in eval1] == [r["label"] for r in eval2]
    assert len(eval1) == 2 and len(train1) == 8
    assert {r["label"] for r in train1}.isdisjoint({r["label"] for r in eval1})


def test_split_zero_eval_fraction():
    rows = [{"label": f"t{i}"} for i in range(5)]
    train, evals = split_rows(rows, 0.0, seed=1)
    assert len(train) == 5 and evals == []


def test_split_minimum_one_eval_row():
    rows = [{"label": f"t{i}"} for i in range(3)]
    train, evals = split_rows(rows, 0.05, seed=1)
    assert len(evals) == 1 and len(train) == 2


def test_single_task_stays_in_train():
    """PR review: a one-task set must produce a training split (the launcher
    always requires train.jsonl); eval may be empty."""
    from examples.fleet.prepare_dataset import split_rows

    train, eval_rows = split_rows([{"label": "only"}], eval_fraction=0.2, seed=7)
    assert len(train) == 1 and eval_rows == []
