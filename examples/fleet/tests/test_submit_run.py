import json
from pathlib import Path

import pytest
import yaml

from examples.fleet.launch.submit_run import _load_payload, _render


def _payload(*, workers: int) -> dict:
    return {
        "name": "chris-cyber-qwen36-capability-gate",
        "owner": "chris",
        "submitted_by": "christopher@fleet.so",
        "image": "example.invalid/trainer@sha256:" + "a" * 64,
        "command": "echo ok",
        "workers": workers,
        "gpus_per_worker": 8,
        "secrets": ["wandb-api", "fleet-api"],
    }


def _secret_names(pod_spec: dict) -> list[str]:
    return [item["secretRef"]["name"] for item in pod_spec["containers"][0]["envFrom"]]


def test_single_node_render_keeps_requested_secrets() -> None:
    document = yaml.safe_load(_render(_payload(workers=1)))
    cluster = document["spec"]["rayClusterSpec"]

    assert cluster["workerGroupSpecs"] == []
    assert _secret_names(cluster["headGroupSpec"]["template"]["spec"]) == [
        "chris-cyber-qwen36-capability-gate-secrets",
        "wandb-api",
        "fleet-api",
    ]
    assert document["metadata"]["labels"]["owner"] == "chris"
    assert document["metadata"]["annotations"]["fleet.ai/submitted-by"] == "christopher@fleet.so"
    assert document["spec"]["submitterPodTemplate"]["metadata"]["labels"]["owner"] == "chris"
    assert cluster["headGroupSpec"]["template"]["metadata"]["labels"]["owner"] == "chris"


def test_multi_node_render_keeps_requested_secrets_on_every_gpu_pod() -> None:
    document = yaml.safe_load(_render(_payload(workers=2)))
    cluster = document["spec"]["rayClusterSpec"]

    assert _secret_names(cluster["headGroupSpec"]["template"]["spec"]) == [
        "chris-cyber-qwen36-capability-gate-secrets",
        "wandb-api",
        "fleet-api",
    ]
    assert _secret_names(cluster["workerGroupSpecs"][0]["template"]["spec"]) == [
        "chris-cyber-qwen36-capability-gate-secrets",
        "wandb-api",
        "fleet-api",
    ]
    assert cluster["workerGroupSpecs"][0]["template"]["metadata"]["labels"]["owner"] == "chris"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner", None),
        ("submitted_by", None),
        ("owner", "miles"),
        ("submitted_by", "service-account@example.com"),
    ],
)
def test_payload_rejects_missing_or_misattributed_human_owner(tmp_path: Path, field: str, value: str | None) -> None:
    payload = _payload(workers=1)
    if value is None:
        del payload[field]
    else:
        payload[field] = value
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit):
        _load_payload(str(path))


def test_capability_packet_renders_exact_queue_resources() -> None:
    payload_path = Path(__file__).resolve().parents[1] / "launch" / "capability-gate-run.template.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    document = yaml.safe_load(_render(payload))
    cluster = document["spec"]["rayClusterSpec"]
    head_spec = cluster["headGroupSpec"]["template"]["spec"]
    container = head_spec["containers"][0]

    assert document["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    assert document["spec"]["suspend"] is True
    assert document["spec"]["shutdownAfterJobFinishes"] is True
    assert document["spec"]["ttlSecondsAfterFinished"] == 0
    assert cluster["workerGroupSpecs"] == []
    assert head_spec["nodeSelector"]["node.kubernetes.io/instance-type"] == "gpu-b300-sxm"
    assert head_spec["priorityClassName"] == "fleet-train-high"
    assert container["resources"] == {
        "requests": {"cpu": "48", "memory": "1500Gi", "nvidia.com/gpu": 8},
        "limits": {"cpu": "64", "memory": "2400Gi", "nvidia.com/gpu": 8},
    }
    assert _secret_names(head_spec) == [
        "chris-cyber-qwen36-27b-capability-gate-01-secrets",
        "wandb-api",
        "fleet-api",
    ]


def test_image_builder_fetches_and_checks_exact_commit_without_mutating_latest() -> None:
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    script = (launch_dir / "build_image.sh").read_text(encoding="utf-8")
    template = (launch_dir / "build_job.yaml.tmpl").read_text(encoding="utf-8")

    assert 'EXPECTED_COMMIT="${3:-$(git -C "$REPO_DIR" rev-parse "$REF")}"' in script
    assert 'BUILD_JOB="${BUILD_JOB_PREFIX}-${SHA}"' in script
    assert "name: ${BUILD_JOB}" in template
    assert "refusing to replace existing build job" in script
    assert " delete job " not in script
    assert 'git fetch -q --depth 1 origin "${REF}"' in template
    assert 'test "$FETCHED_COMMIT" = "$EXPECTED_COMMIT"' in template
    assert "kueue.x-k8s.io/queue-name: default" in template
    assert 'docker push "ghcr.io/fleet-ai/miles-fleet/trainer:${SHA}"' in template
    assert "docker push ghcr.io/fleet-ai/miles-fleet/trainer:latest" not in template
    assert 'timeout 300 docker pull "ghcr.io/fleet-ai/miles-fleet/trainer:${CACHE_TAG}"' in template


def test_model_prestage_is_revision_pinned_and_never_replaces_a_job() -> None:
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    script = (launch_dir / "prestage.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--revision"' in script
    assert "--revision {args.revision}" in script
    assert "refusing to replace existing prestage job" in script
    assert '["create", "-f", "-"]' in script
    assert '["delete", "job"' not in script


def test_authoritative_payload_load_requires_image_and_selection_digests(tmp_path: Path) -> None:
    payload = _payload(workers=1)
    payload.update(
        image="ghcr.io/fleet-ai/miles-fleet/trainer@sha256:" + "a" * 64,
        env={
            "FLEET_BACKEND": "fleet_authoritative_cyber_v1",
            "FLEET_REWARD_OBJECTIVE": "raw_capability_v1",
            "FLEET_AUTHORITATIVE_SELECTION": "examples/fleet/launch/selection.json",
            "FLEET_AUTHORITATIVE_SELECTION_SHA256": "b" * 64,
        },
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    assert _load_payload(str(path)) == payload

    payload["image"] = "ghcr.io/fleet-ai/miles-fleet/trainer:latest"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        _load_payload(str(path))


def test_authoritative_payload_rejects_taskset_mixing(tmp_path: Path) -> None:
    payload = _payload(workers=1)
    payload.update(
        image="ghcr.io/fleet-ai/miles-fleet/trainer@sha256:" + "a" * 64,
        env={
            "FLEET_BACKEND": "fleet_authoritative_cyber_v1",
            "FLEET_REWARD_OBJECTIVE": "raw_capability_v1",
            "FLEET_AUTHORITATIVE_SELECTION": "examples/fleet/launch/selection.json",
            "FLEET_AUTHORITATIVE_SELECTION_SHA256": "b" * 64,
            "TASKSET_REF": "registry.example/taskset:mutable",
        },
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        _load_payload(str(path))
