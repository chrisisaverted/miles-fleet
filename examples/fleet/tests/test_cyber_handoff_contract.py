import json
from pathlib import Path


def test_cyber_ledger_requires_checked_v3_safe_success() -> None:
    ledger_path = Path(__file__).resolve().parents[1] / "launch" / "experiment-ledger.template.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert ledger["verification"] == {
        "contract": "cyber_verification_result_v3",
        "authority": "<required:trusted-context-builder-and-pinned-verifier-service>",
        "objective_mapping": "safe_success_v1",
        "raw_capability_reward_retained": True,
        "patch_bearing_supported": False,
    }
    assert ledger["attestation"]["checked_v3_context_verified"] is False
    assert ledger["attestation"]["behavior_receipt_verified"] is False


def test_capability_gate_is_exactly_pinned_and_does_not_claim_safe_success() -> None:
    ledger_path = Path(__file__).resolve().parents[1] / "launch" / "capability-gate-ledger.template.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert ledger["model"] == {
        "repository": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    }
    assert ledger["task"] == {
        "task_key": "cysec1-2-current-gen_blackbox-c17f94bad9b70de3dfa6f05c__blackbox_ctf_v1",
        "task_version_id": "f31ebe83-0ff1-4660-bcba-59ffa4b82d5a",
        "environment_version_id": "d70c4fe9-70c5-4020-91b1-a23d886a1e22",
        "env_key": "cysec1-2-current-gen",
        "env_version": "v0.0.3",
        "data_key": "commercial",
        "data_version": "v0.0.9",
        "source_eval_sessions": 8,
        "source_eval_successes": 3,
        "source_eval_rate": 0.375,
    }
    assert ledger["verification"]["objective_mapping"] == "raw_capability_v1"
    assert ledger["verification"]["safe_success_claimed"] is False
    assert ledger["schema"] == "fleet.miles.capability-gate-ledger.v2"
    assert ledger["data"]["selection_schema"] == "fleet.authoritative-selection.v1"
    assert ledger["data"]["taskdump_v7_used"] is False
    assert ledger["runtime"]["mode"] == "debug_one_step"
    assert ledger["runtime"]["requested_optimizer_steps"] == 1


def test_safety_eligibility_matrix_keeps_capability_and_behavior_separate() -> None:
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    matrix = json.loads((launch_dir / "safety-eligibility-matrix.v1.json").read_text(encoding="utf-8"))
    candidates = {row["candidate"]: row for row in matrix["candidates"]}

    assert matrix["schema"] == "fleet.miles.cyber-safety-eligibility.v1"
    assert candidates["frozen-129-runnable-cohort"]["eligible"] is False
    assert candidates["ambiguity-wave-manual-traces"]["checked_behavior_v3"] is False
    assert candidates["webhook-preview-v0.0.8"]["environment_version_id"] is None
    assert candidates["webhook-preview-v0.0.8"]["eligible"] is False
    assert candidates["v0.0.3-mid-band-capability-canary"]["eligible"] is True
    assert candidates["v0.0.3-mid-band-capability-canary"]["safe_success_claimed"] is False


def test_capability_preflight_receipt_records_authority_and_cleanup() -> None:
    receipt_path = Path(__file__).resolve().parents[1] / "launch" / "capability-gate-preflight.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["schema"] == "fleet.miles.authoritative-preflight.v1"
    assert receipt["status"] == "adapter_contract_repaired"
    assert receipt["task"]["task_version_id"] == "f31ebe83-0ff1-4660-bcba-59ffa4b82d5a"
    assert receipt["task"]["verifier_version_id"] == "3158d910-f91d-4c22-83b9-70ddc678d572"
    assert receipt["response"]["schema_version"] == "cyber_verification_result_v3"
    assert receipt["response"]["projection_id"] == "blackbox_ctf_v1"
    assert receipt["response"]["authoritative_evidence"] is True
    assert receipt["response"]["reward"] == 0.0
    assert receipt["response"]["raw_capability_diagnostic_present"] is False
    assert receipt["cleanup"]["http_status"] == 200
    assert receipt["cleanup"]["status"] == "stopped"


def test_capability_run_packet_is_one_step_queue_safe_and_immutable() -> None:
    payload_path = Path(__file__).resolve().parents[1] / "launch" / "capability-gate-run.template.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert payload["name"] == "chris-cyber-qwen36-27b-capability-gate-01"
    assert payload["image"].startswith("ghcr.io/fleet-ai/miles-fleet/trainer@sha256:")
    assert payload["workers"] == 1
    assert payload["gpus_per_worker"] == 8
    assert payload["pool"] == "gpu-b300"
    assert payload["env"] == {
        "FLEET_AUTHORITATIVE_SELECTION": "examples/fleet/launch/capability-gate.authoritative-selection.v1.json",
        "FLEET_AUTHORITATIVE_SELECTION_SHA256": "56602ed11012380d6cf2d1f00cefe6c13ac714ccfb2b71cf826af2a2043379db",
        "TASK_LIMIT": "1",
        "FLEET_BACKEND": "fleet_authoritative_cyber_v1",
        "FLEET_REWARD_OBJECTIVE": "raw_capability_v1",
    }
    assert "--model-name qwen3.6-27b" in payload["command"]
    assert "--mode debug_one_step" in payload["command"]
    assert "--rollout-batch-size 1" in payload["command"]
    assert "--n-samples-per-prompt 8" in payload["command"]
    assert "--max-concurrent-envs 1" in payload["command"]
    assert "--max-concurrent-prepares 1" in payload["command"]
    assert payload["secrets"] == ["wandb-api", "fleet-api"]


def test_authoritative_selection_is_distinct_from_taskdump_and_receipt_bound() -> None:
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    selection_path = launch_dir / "capability-gate.authoritative-selection.v1.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    receipt_path = launch_dir / "capability-gate-preflight.receipt.json"

    import hashlib

    assert selection["schema"] == "fleet.authoritative-selection.v1"
    assert selection["schema"] != "fleet.taskdump.v7"
    assert selection["source"]["preflight_receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    task = selection["tasks"][0]
    assert task["task_version_id"] == "f31ebe83-0ff1-4660-bcba-59ffa4b82d5a"
    assert task["environment_version_id"] == "d70c4fe9-70c5-4020-91b1-a23d886a1e22"
    assert task["verifier_version_id"] == "3158d910-f91d-4c22-83b9-70ddc678d572"

    integrated_receipt = json.loads(
        (launch_dir / "capability-gate-authoritative-selection-preflight.receipt.json").read_text(encoding="utf-8")
    )
    assert integrated_receipt["status"] == "passed_and_reaped"
    assert integrated_receipt["selection"]["sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert integrated_receipt["selection"]["public_task_fields_and_hashes_matched"] is True
    assert integrated_receipt["cleanup"]["server_status"] == "stopped"
