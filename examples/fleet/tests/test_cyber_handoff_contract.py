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
