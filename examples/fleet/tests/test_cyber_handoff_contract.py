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


def test_capability_run_packet_is_one_step_queue_safe_and_immutable() -> None:
    payload_path = Path(__file__).resolve().parents[1] / "launch" / "capability-gate-run.template.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert payload["name"] == "chris-cyber-qwen36-27b-capability-gate-01"
    assert payload["image"].startswith("ghcr.io/fleet-ai/miles-fleet/trainer@sha256:")
    assert payload["workers"] == 1
    assert payload["gpus_per_worker"] == 8
    assert payload["pool"] == "gpu-b300"
    assert payload["env"] == {
        "TASKSET_REF": "registry-alpha.fleetai.me/fleet/cysec1-2-current-gen-capability-canary@sha256:<required:64-hex-digest>",
        "TASK_LIMIT": "1",
        "FLEET_BACKEND": "fleet_authoritative_cyber_v1",
        "FLEET_REWARD_OBJECTIVE": "raw_capability_v1",
    }
    assert "--model-name qwen3.6-27b" in payload["command"]
    assert "--mode debug_one_step" in payload["command"]
    assert "--max-concurrent-envs 1" in payload["command"]
    assert "--max-concurrent-prepares 1" in payload["command"]
    assert payload["secrets"] == ["wandb-api", "fleet-api"]
