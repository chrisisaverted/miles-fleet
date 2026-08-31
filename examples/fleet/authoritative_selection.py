"""Fail-closed hydration of immutable Fleet authoritative selections.

This is intentionally a different source contract from TaskDump v7.  It is
for the authoritative rollout backend, where Fleet provisions the exact task
version and executes its server-owned verifier.  The selection freezes every
identifier exposed by the task/version rows plus hashes of the public prompt
and verifier.  Hydration accepts the public API response only when all fields
still agree; the exact task-version UUID is revalidated again when the rollout
session is provisioned.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SELECTION_SCHEMA = "fleet.authoritative-selection.v1"
REPORT_ONLY_SUFFIX = "__blackbox_ctf_v1"
_HEX64 = frozenset("0123456789abcdef")


class AuthoritativeSelectionError(ValueError):
    """The frozen selection or live public readback contradicted its pins."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthoritativeSelectionError(f"{field} must be a non-empty string")
    return value


def _uuid(value: Any, field: str) -> str:
    try:
        return str(uuid.UUID(_text(value, field)))
    except ValueError as exc:
        raise AuthoritativeSelectionError(f"{field} must be a UUID") from exc


def _sha256(value: Any, field: str) -> str:
    value = _text(value, field)
    if len(value) != 64 or any(char not in _HEX64 for char in value):
        raise AuthoritativeSelectionError(f"{field} must be 64 lowercase hex characters")
    return value


def selection_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _default_get_json(url: str, api_key: str) -> Any:
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS base is validated below
        return json.load(response)


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthoritativeSelectionError(f"invalid selection JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"schema", "source", "tasks"}:
        raise AuthoritativeSelectionError("selection must contain exactly schema, source, and tasks")
    if document.get("schema") != SELECTION_SCHEMA:
        raise AuthoritativeSelectionError(f"selection schema must be {SELECTION_SCHEMA}")
    source_info = document.get("source")
    if not isinstance(source_info, dict) or set(source_info) != {
        "api_base",
        "captured_at",
        "preflight_receipt_sha256",
    }:
        raise AuthoritativeSelectionError(
            "selection source must contain exactly api_base, captured_at, and preflight_receipt_sha256"
        )
    api_base = _text(source_info.get("api_base"), "source.api_base").rstrip("/")
    if api_base != "https://orchestrator.fleetai.com":
        raise AuthoritativeSelectionError("source.api_base must be the production Fleet HTTPS authority")
    _text(source_info.get("captured_at"), "source.captured_at")
    _sha256(source_info.get("preflight_receipt_sha256"), "source.preflight_receipt_sha256")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise AuthoritativeSelectionError("selection requires at least one task")
    return document, hashlib.sha256(raw).hexdigest()


_TASK_FIELDS = {
    "task_key",
    "task_version_id",
    "team_id",
    "environment_version_id",
    "env_key",
    "env_version",
    "data_key",
    "data_version",
    "verifier_id",
    "verifier_version_id",
    "verifier_sha256",
    "prompt_sha256",
    "projection_id",
}


def _normalize_pin(raw: Any, index: int) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != _TASK_FIELDS:
        raise AuthoritativeSelectionError(f"tasks[{index}] must contain exactly {sorted(_TASK_FIELDS)}")
    pin = {field: _text(raw.get(field), f"tasks[{index}].{field}") for field in _TASK_FIELDS}
    for field in (
        "task_version_id",
        "team_id",
        "environment_version_id",
        "verifier_id",
        "verifier_version_id",
    ):
        pin[field] = _uuid(pin[field], f"tasks[{index}].{field}")
    pin["verifier_sha256"] = _sha256(pin["verifier_sha256"], f"tasks[{index}].verifier_sha256")
    pin["prompt_sha256"] = _sha256(pin["prompt_sha256"], f"tasks[{index}].prompt_sha256")
    if not pin["task_key"].endswith(REPORT_ONLY_SUFFIX):
        raise AuthoritativeSelectionError(f"tasks[{index}].task_key must end with {REPORT_ONLY_SUFFIX}")
    if pin["projection_id"] != "blackbox_ctf_v1":
        raise AuthoritativeSelectionError(f"tasks[{index}].projection_id must be blackbox_ctf_v1")
    return pin


def _mismatches(pin: Mapping[str, str], task: Mapping[str, Any]) -> list[str]:
    verifier = task.get("verifier") if isinstance(task.get("verifier"), Mapping) else {}
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    prompt = task.get("prompt")
    observed = {
        "task_key": task.get("key"),
        "team_id": task.get("team_id"),
        "env_key": task.get("environment_id"),
        "env_version": task.get("version"),
        "data_key": task.get("data_id"),
        "data_version": task.get("data_version"),
        "verifier_id": verifier.get("verifier_id"),
        "verifier_version_id": verifier.get("verifier_version_id"),
        "verifier_sha256": verifier.get("sha256"),
        "projection_id": metadata.get("projection_id"),
        "prompt_sha256": (hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None),
    }
    return [
        f"{field}: expected {pin[field]!r}, got {observed[field]!r}"
        for field in observed
        if observed[field] != pin[field]
    ]


def hydrate_authoritative_rows(
    path: str | Path,
    *,
    api_key: str,
    expected_sha256: str,
    reward_objective: str,
    get_json: Callable[[str, str], Any] = _default_get_json,
) -> list[dict[str, Any]]:
    """Read, hash-check, and hydrate an authoritative selection via Fleet.

    The public API intentionally does not expose the task-version UUID.  That
    UUID is frozen here and the exact-version provisioning route validates it
    before any agent step; every field the public API *does* expose is checked
    during hydration so current-pointer or content drift fails before training.
    """

    document, actual_sha256 = _load(path)
    expected_sha256 = _sha256(expected_sha256, "expected selection sha256")
    if actual_sha256 != expected_sha256:
        raise AuthoritativeSelectionError(
            f"selection digest mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    api_key = _text(api_key, "Fleet API key")
    base = document["source"]["api_base"].rstrip("/")
    account = get_json(f"{base}/v1/account", api_key)
    if not isinstance(account, Mapping):
        raise AuthoritativeSelectionError("Fleet account response must be an object")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_pin in enumerate(document["tasks"]):
        pin = _normalize_pin(raw_pin, index)
        identity = (pin["task_key"], pin["task_version_id"])
        if identity in seen:
            raise AuthoritativeSelectionError(f"duplicate frozen task identity {identity!r}")
        seen.add(identity)
        if account.get("team_id") != pin["team_id"]:
            raise AuthoritativeSelectionError(
                f"Fleet account team mismatch: expected {pin['team_id']}, got {account.get('team_id')!r}"
            )
        payload = get_json(f"{base}/v1/tasks?{urlencode({'task_keys': pin['task_key']})}", api_key)
        tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
        if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], Mapping):
            raise AuthoritativeSelectionError(
                f"public task readback for {pin['task_key']!r} must return exactly one task"
            )
        task = tasks[0]
        mismatches = _mismatches(pin, task)
        if mismatches:
            raise AuthoritativeSelectionError(
                f"public task readback contradicts frozen selection for {pin['task_key']!r}: " + "; ".join(mismatches)
            )
        prompt = task["prompt"]
        rows.append(
            {
                "input": [{"role": "user", "content": prompt}],
                "label": pin["task_key"],
                "metadata": {
                    "task_key": pin["task_key"],
                    "taskset_ref": f"fleet-authoritative-selection@sha256:{actual_sha256}",
                    "data_source": f"fleet-authoritative-selection@sha256:{actual_sha256}",
                    "steps": 1,
                    "context_profile": "continuous",
                    "fleet_backend": "fleet_authoritative_cyber_v1",
                    "reward_objective": reward_objective,
                    "task_prompt": prompt,
                    **{
                        field: pin[field]
                        for field in (
                            "task_version_id",
                            "environment_version_id",
                            "verifier_version_id",
                            "env_key",
                            "env_version",
                            "data_key",
                            "data_version",
                        )
                    },
                },
            }
        )
    return rows
