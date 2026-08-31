import json
from pathlib import Path

import yaml

from examples.fleet.launch.submit_run import _render


def _payload(*, workers: int) -> dict:
    return {
        "name": "chris-cyber-qwen36-capability-gate",
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
    assert 'docker push "ghcr.io/fleet-ai/miles-fleet/trainer:${SHA}"' in template
    assert "docker push ghcr.io/fleet-ai/miles-fleet/trainer:latest" not in template
