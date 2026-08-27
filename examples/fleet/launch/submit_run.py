#!/usr/bin/env python3
"""Submit a Fleet training run from a run payload JSON.

Stand-in for POST /v1/runs: a run is a name, an image, a command, how many
GPUs, and env variables. The platform side (this script plus the RayJob
template) owns placement, env and secret injection, the SFS log/trace
location, queueing, and lifecycle; the code side lives in the image and is
whatever `command` invokes.

    ./submit_run.py examples/vision-qwen38-27b-2node.json
    ./submit_run.py my-run.json --dry-run

Payload:
    name             run and RayJob name (DNS-safe label)
    image            ghcr.io/fleet-ai/miles-fleet/trainer:<sha>
    command          what to run; no apostrophes (single-quoted entrypoint)
    workers          number of GPU pods; with gpus_per_worker=8 one pod fills
                     one node, so workers = machines
    gpus_per_worker  1..8
    env              free-form env vars injected into every GPU pod
    secrets          pre-created k8s secrets mounted as env (wandb-api is
                     always included; a per-run Fleet credentials secret is
                     created here because the token expires)
    pool             optional; defaults to gpu-b300, currently the only pool.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# A pool names a set of identical GPU machines and the cluster that hosts
# them. gpu-b300 is the Nebius production cluster (fleetai-training,
# 24 x 8 B300, 268GB per GPU, 2.7TB host RAM, InfiniBand between machines).
_POOLS = {
    "gpu-b300": dict(
        KUBE_CONTEXT="nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6", CPU_WORKLOAD="fleetai-training-ng-cpu",
        NODE_WORKLOAD="fleetai-training-ng-gpu", INSTANCE_TYPE="gpu-b300-sxm", MAIN_MEM="1500Gi", MAIN_MEM_LIM="2400Gi",
    ),
}


def _kubectl(pool: str) -> list:
    return ["kubectl", "--context", _POOLS[pool]["KUBE_CONTEXT"], "-n", "fleet-train-jobs"]


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_payload(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    for field in ("name", "image", "command", "workers", "gpus_per_worker"):
        if field not in payload:
            _fail(f"payload is missing '{field}'")
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,50}[a-z0-9])?", payload["name"]):
        _fail("name must be a short DNS-safe label (lowercase alphanumerics and dashes)")
    if not (isinstance(payload["workers"], int) and payload["workers"] >= 1):
        _fail("workers must be an integer >= 1")
    if not (isinstance(payload["gpus_per_worker"], int) and 1 <= payload["gpus_per_worker"] <= 8):
        _fail("gpus_per_worker must be 1..8")
    if "'" in payload["command"]:
        _fail("command must not contain apostrophes (the RayJob entrypoint is single-quoted)")
    if payload.get("pool", "gpu-b300") not in _POOLS:
        _fail(f"pool must be one of {sorted(_POOLS)}")
    env = payload.get("env", {})
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        _fail("env must be a flat map of string to string")
    return payload


def _env_lines(env: dict, indent: str) -> str:
    return "".join(f"\n{indent}- {{name: {k}, value: {json.dumps(v)}}}" for k, v in env.items())


def _render(payload: dict) -> str:
    env = dict(payload.get("env", {}))
    env.setdefault("RUN_ID", payload["name"])
    pool_vals = {k: v for k, v in _POOLS[payload.get("pool", "gpu-b300")].items() if k != "KUBE_CONTEXT"}
    values = {
        "JOB_NAME": payload["name"],
        "SECRET_NAME": f"{payload['name']}-secrets",
        "IMAGE": payload["image"],
        "COMMAND": payload["command"],
        "WORKER_REPLICAS": str(payload["workers"] - 1),
        "NUM_GPUS": str(payload["gpus_per_worker"]),
        "EXTRA_ENV": _env_lines(env, "                "),
        "WORKER_EXTRA_ENV": _env_lines(env, "                  ") or "\n                  []",
        **pool_vals,
    }
    template = (HERE / "rayjob.yaml.tmpl").read_text()
    rendered = re.sub(r"\$\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)
    unresolved = sorted(set(re.findall(r"\$\{(\w+)\}", rendered)))
    if unresolved:
        _fail(f"template variables left unresolved: {unresolved}")
    if payload["workers"] == 1:
        # Kueue never evaluates a workload containing a zero-count podset, so
        # a single-node run must omit the worker group entirely.
        import yaml

        doc = yaml.safe_load(rendered)
        doc["spec"]["rayClusterSpec"]["workerGroupSpecs"] = []
        rendered = yaml.safe_dump(doc, default_flow_style=False, sort_keys=False, width=10000)
    if payload.get("secrets"):
        extra = "".join(
            f"\n                - secretRef: {{name: {s}}}" for s in payload["secrets"] if s != "wandb-api"
        )
        rendered = rendered.replace(
            "                - secretRef: {name: wandb-api}",
            "                - secretRef: {name: wandb-api}" + extra,
        )
    return rendered


def _create_run_secret(name: str, kubectl: list) -> None:
    creds = Path.home() / ".config/fleet/credentials.json"
    if not creds.exists():
        _fail("~/.config/fleet/credentials.json not found; run `flt auth login registry-alpha.fleetai.me`")
    literals = [f"--from-literal=FLEET_CREDENTIALS_B64={base64.b64encode(creds.read_bytes()).decode()}"]
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if os.environ.get(key):
            literals.append(f"--from-literal={key}={os.environ[key]}")
    manifest = subprocess.run(
        kubectl + ["create", "secret", "generic", f"{name}-secrets", *literals, "--dry-run=client", "-o", "yaml"],
        check=True, capture_output=True, text=True,
    ).stdout
    subprocess.run(kubectl + ["apply", "-f", "-"], input=manifest, check=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("payload", help="path to a run payload JSON")
    parser.add_argument("--dry-run", action="store_true", help="print the rendered RayJob instead of applying")
    args = parser.parse_args()

    payload = _load_payload(args.payload)
    manifest = _render(payload)
    if args.dry_run:
        print(manifest)
        return
    kubectl = _kubectl(payload.get("pool", "gpu-b300"))
    _create_run_secret(payload["name"], kubectl)
    subprocess.run(kubectl + ["apply", "-f", "-"], input=manifest, check=True, text=True)
    name = payload["name"]
    ctx = _POOLS[payload.get("pool", "gpu-b300")]["KUBE_CONTEXT"]
    print(f"submitted: kubectl --context {ctx} -n fleet-train-jobs get rayjob {name} -w")
    print(f"logs:      kubectl --context {ctx} -n fleet-train-jobs logs -f job/{name}")
    print(f"sfs:       /mnt/sfs/miles-fleet/{name}/driver.log")


if __name__ == "__main__":
    main()
