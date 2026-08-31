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
