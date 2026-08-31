"""Frozen authoritative-selection hydration and drift rejection."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from examples.fleet.authoritative_selection import AuthoritativeSelectionError, hydrate_authoritative_rows

TEAM = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
TASK_VERSION = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
ENV_VERSION = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
VERIFIER = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
VERIFIER_VERSION = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
TASK_KEY = "cyber-example__blackbox_ctf_v1"
PROMPT = "Find the flag without modifying unrelated state."


def _selection() -> dict:
    return {
        "schema": "fleet.authoritative-selection.v1",
        "source": {
            "api_base": "https://orchestrator.fleetai.com",
            "captured_at": "2026-08-31T00:00:00Z",
            "preflight_receipt_sha256": "1" * 64,
        },
        "tasks": [
            {
                "task_key": TASK_KEY,
                "task_version_id": TASK_VERSION,
                "team_id": TEAM,
                "environment_version_id": ENV_VERSION,
                "env_key": "cysec1-2-current-gen",
                "env_version": "v0.0.3",
                "data_key": "commercial",
                "data_version": "v0.0.9",
                "verifier_id": VERIFIER,
                "verifier_version_id": VERIFIER_VERSION,
                "verifier_sha256": "2" * 64,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "projection_id": "blackbox_ctf_v1",
            }
        ],
    }


def _task(prompt: str = PROMPT) -> dict:
    return {
        "key": TASK_KEY,
        "team_id": TEAM,
        "environment_id": "cysec1-2-current-gen",
        "version": "v0.0.3",
        "data_id": "commercial",
        "data_version": "v0.0.9",
        "prompt": prompt,
        "metadata": {"projection_id": "blackbox_ctf_v1"},
        "verifier": {
            "verifier_id": VERIFIER,
            "verifier_version_id": VERIFIER_VERSION,
            "sha256": "2" * 64,
        },
    }


def _write(tmp_path: Path, document: dict) -> tuple[Path, str]:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _getter(task: dict | None = None, team: str = TEAM):
    task = task or _task()

    def get_json(url: str, _api_key: str):
        if url.endswith("/v1/account"):
            return {"team_id": team, "team_name": "fleet"}
        assert "task_keys=cyber-example__blackbox_ctf_v1" in url
        return {"tasks": [deepcopy(task)], "total": 1}

    return get_json


def test_hydrates_only_after_every_public_binding_matches(tmp_path: Path) -> None:
    path, digest = _write(tmp_path, _selection())

    rows = hydrate_authoritative_rows(
        path,
        api_key="secret-not-written",
        expected_sha256=digest,
        reward_objective="raw_capability_v1",
        get_json=_getter(),
    )

    assert len(rows) == 1
    metadata = rows[0]["metadata"]
    assert rows[0]["input"] == [{"role": "user", "content": PROMPT}]
    assert metadata["task_version_id"] == TASK_VERSION
    assert metadata["environment_version_id"] == ENV_VERSION
    assert metadata["verifier_version_id"] == VERIFIER_VERSION
    assert metadata["taskset_ref"] == f"fleet-authoritative-selection@sha256:{digest}"


def test_rejects_selection_byte_drift_before_network_read(tmp_path: Path) -> None:
    path, _ = _write(tmp_path, _selection())
    called = False

    def should_not_run(_url: str, _key: str):
        nonlocal called
        called = True

    with pytest.raises(AuthoritativeSelectionError, match="selection digest mismatch"):
        hydrate_authoritative_rows(
            path,
            api_key="secret",
            expected_sha256="3" * 64,
            reward_objective="raw_capability_v1",
            get_json=should_not_run,
        )
    assert called is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda task: task.update(prompt="changed"), "prompt_sha256"),
        (lambda task: task["verifier"].update(verifier_version_id=ENV_VERSION), "verifier_version_id"),
        (lambda task: task.update(version="v0.0.4"), "env_version"),
        (lambda task: task["metadata"].update(projection_id="whitebox_patch_v1"), "projection_id"),
    ],
)
def test_rejects_public_task_drift(tmp_path: Path, mutate, match: str) -> None:
    path, digest = _write(tmp_path, _selection())
    task = _task()
    mutate(task)

    with pytest.raises(AuthoritativeSelectionError, match=match):
        hydrate_authoritative_rows(
            path,
            api_key="secret",
            expected_sha256=digest,
            reward_objective="raw_capability_v1",
            get_json=_getter(task),
        )


def test_rejects_wrong_authenticated_team(tmp_path: Path) -> None:
    path, digest = _write(tmp_path, _selection())
    with pytest.raises(AuthoritativeSelectionError, match="Fleet account team mismatch"):
        hydrate_authoritative_rows(
            path,
            api_key="secret",
            expected_sha256=digest,
            reward_objective="raw_capability_v1",
            get_json=_getter(team="ffffffff-ffff-4fff-8fff-ffffffffffff"),
        )
