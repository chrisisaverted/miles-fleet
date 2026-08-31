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
