from pathlib import Path

import pytest

from examples.fleet.launch import run_fleet
from examples.fleet.launch.taskset_ref import parse_taskset_repository


QWEN36_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TASKSET_DIGEST = "0123456789abcdef" * 4


@pytest.mark.parametrize(
    ("taskset_ref", "expected"),
    [
        (
            f"registry-alpha.fleetai.me/fleet/cyber-tasks@sha256:{TASKSET_DIGEST}",
            ("fleet", "cyber-tasks"),
        ),
        (
            f"flt://registry-alpha.fleetai.me/fleet/cyber-tasks@sha256:{TASKSET_DIGEST}",
            ("fleet", "cyber-tasks"),
        ),
        ("registry-alpha.fleetai.me/library/ade-bench:latest", ("library", "ade-bench")),
        ("library/ade-bench", ("library", "ade-bench")),
    ],
)
def test_parse_taskset_repository(taskset_ref: str, expected: tuple[str, str]) -> None:
    assert parse_taskset_repository(taskset_ref) == expected


@pytest.mark.parametrize(
    "taskset_ref",
    [
        "fleet/cyber-tasks@sha256:",
        "fleet/cyber-tasks:",
        f"cyber-tasks@sha256:{TASKSET_DIGEST}",
        f"fleet/subdir/cyber-tasks@sha256:{TASKSET_DIGEST}",
        "fleet/cyber-tasks@md5:0123abcd",
        f"fleet/cyber-tasks@sha256:{TASKSET_DIGEST.upper()}",
    ],
)
def test_parse_taskset_repository_rejects_ambiguous_refs(taskset_ref: str) -> None:
    with pytest.raises(ValueError):
        parse_taskset_repository(taskset_ref)


def test_qwen36_recipe_pins_checkpoint_revision(monkeypatch, tmp_path: Path) -> None:
    commands: list[str] = []
    monkeypatch.setattr(run_fleet.U, "exec_command_cpu", commands.append)
    args = run_fleet.ScriptArgs(
        model_name="qwen3.6-27b",
        model_dir=str(tmp_path / "models"),
        data_dir=str(tmp_path / "data"),
    )

    run_fleet.prepare(args)

    expected_path = tmp_path / "models" / "Qwen3.6-27B" / QWEN36_REVISION
    assert run_fleet._hf_checkpoint_path(args) == expected_path
    assert args.recipe.tito_model == "qwen36"
    assert args.recipe.chat_template is None
    assert commands[-1] == (
        "hf download Qwen/Qwen3.6-27B " f"--revision {QWEN36_REVISION} --local-dir {expected_path}"
    )


def test_qwen36_training_reads_the_revision_pinned_checkpoint(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(run_fleet.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(run_fleet.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    args = run_fleet.ScriptArgs(
        model_name="qwen3.6-27b",
        model_dir=str(tmp_path / "models"),
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        dataset_dir=str(tmp_path / "dataset"),
        mode="debug_minimal",
    )

    run_fleet.execute(args)

    expected_path = tmp_path / "models" / "Qwen3.6-27B" / QWEN36_REVISION
    train_args = str(captured["train_args"])
    assert f"--hf-checkpoint {expected_path} " in train_args
    assert f"--ref-load {expected_path} " in train_args
    assert "--fleet-tito-model qwen36 " in train_args
    assert "qwen3.6_fixed.jinja" in train_args


def test_one_step_gate_requests_one_rollout_and_step_one_checkpoint(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(run_fleet.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(run_fleet.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    args = run_fleet.ScriptArgs(
        model_name="qwen3.6-27b",
        model_dir=str(tmp_path / "models"),
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        dataset_dir=str(tmp_path / "dataset"),
        mode="debug_one_step",
    )

    run_fleet.execute(args)

    train_args = str(captured["train_args"])
    assert "--num-rollout 1 " in train_args
    assert "--save-interval 1 " in train_args
    assert "--rollout-max-response-len 512 " in train_args


def test_fleet_fsdp_batch_shape_rejects_fewer_samples_than_ranks() -> None:
    args = run_fleet.ScriptArgs(
        num_nodes=1,
        num_gpus_per_node=8,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
    )

    with pytest.raises(ValueError, match=r"dp_size=8.*= 2"):
        run_fleet._validate_fsdp_batch_shape(args)


def test_fleet_fsdp_batch_shape_rejects_uneven_rank_shards() -> None:
    args = run_fleet.ScriptArgs(
        num_nodes=1,
        num_gpus_per_node=8,
        rollout_batch_size=1,
        n_samples_per_prompt=9,
    )

    with pytest.raises(ValueError, match=r"dp_size=8.*= 9"):
        run_fleet._validate_fsdp_batch_shape(args)


def test_fleet_fsdp_batch_shape_accepts_one_sample_per_rank() -> None:
    args = run_fleet.ScriptArgs(
        num_nodes=1,
        num_gpus_per_node=8,
        rollout_batch_size=1,
        n_samples_per_prompt=8,
    )

    run_fleet._validate_fsdp_batch_shape(args)


def test_qwen38_checkpoint_path_and_download_are_unchanged(monkeypatch, tmp_path: Path) -> None:
    commands: list[str] = []
    monkeypatch.setattr(run_fleet.U, "exec_command_cpu", commands.append)
    args = run_fleet.ScriptArgs(
        model_name="qwen3.8-27b",
        model_dir=str(tmp_path / "models"),
        data_dir=str(tmp_path / "data"),
    )

    run_fleet.prepare(args)

    expected_path = tmp_path / "models" / "Qwen3.8-27B"
    assert run_fleet._hf_checkpoint_path(args) == expected_path
    assert commands[-1] == f"hf download Qwen/Qwen3.8-27B --local-dir {expected_path}"
