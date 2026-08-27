#!/usr/bin/env python3
"""Submit a Fleet training run to rl1 from a RunPayload JSON.

The payload is the future runs-API contract (same shape as SkyRL-Fleet's
rl1 integration); this script is the API stand-in: it validates the payload,
creates the per-run credentials secret, renders rayjob.yaml.tmpl, and applies
it. `--dry-run` prints the manifest instead of applying.

    ./submit_run.py examples/vision-qwen38-27b.json
    ./submit_run.py my-run.json --dry-run

Payload fields:
    name             run and RayJob name; also the SFS and WandB group name
    image            ghcr.io/fleet-ai/miles-fleet/trainer:<sha>
    command          the training invocation, executed on the head after the
                     boot phase (taskset pull, dataset build, env-image
                     pre-pull, model prep). No apostrophes: the entrypoint is
                     single-quoted.
    workers          number of GPU pods (head included); >= 1
    gpus_per_worker  1..8
    env              the five boot/run knobs:
                       MODEL_NAME    recipe row in launch/run_fleet.py; also
                                     selects node pool and memory sizing
                       TASKSET_REF   registry-alpha taskset reference
                       MODE          normal | debug_minimal | rollout_only
                       TASK_LIMIT    task sample cap; "0" = whole taskset
                       RUN_ID        must equal name
    secrets          extra pre-created secrets to mount as env (wandb-api is
                     always included; the per-run Fleet credentials secret is
                     created by this script because the token expires)
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

KUBECTL = ["kubectl", "--context", "fleet-training-rl1-us-east-1", "-n", "fleet-train-jobs"]
HERE = Path(__file__).resolve().parent

REQUIRED_ENV = ("MODEL_NAME", "TASKSET_REF", "MODE", "TASK_LIMIT", "RUN_ID")

# The node pool and memory sizing are model facts, not run knobs: the 27B
# needs the B200s (179GB/GPU; its full-context train step is ~134GB/rank) and
# ~1.15TB host RAM for offload_train; GLM fits the H200 pool.
_MODEL_PLACEMENT = {
    "glm4.7-flash": dict(
        NODE_WORKLOAD="gpu-h200", INSTANCE_TYPE="p5en.48xlarge", MAIN_MEM="925Gi", MAIN_MEM_LIM="1300Gi"
    ),
    "qwen3.8-27b": dict(
        NODE_WORKLOAD="gpu-b200", INSTANCE_TYPE="p6-b200.48xlarge", MAIN_MEM="1500Gi", MAIN_MEM_LIM="1900Gi"
    ),
}


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_payload(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    for field in ("name", "image", "command", "workers", "gpus_per_worker", "env"):
        if field not in payload:
            _fail(f"payload is missing '{field}'")
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,50}[a-z0-9])?", payload["name"]):
        _fail("name must be a short DNS-safe label (lowercase alphanumerics and dashes)")
    if not (isinstance(payload["workers"], int) and payload["workers"] >= 1):
        _fail("workers must be an integer >= 1")
    if not (isinstance(payload["gpus_per_worker"], int) and 1 <= payload["gpus_per_worker"] <= 8):
        _fail("gpus_per_worker must be 1..8")
    missing = [k for k in REQUIRED_ENV if k not in payload["env"]]
    if missing:
        _fail(f"env is missing {missing}; the contract is exactly these knobs: {list(REQUIRED_ENV)}")
    if payload["env"]["MODEL_NAME"] not in _MODEL_PLACEMENT:
        _fail(f"unknown MODEL_NAME {payload['env']['MODEL_NAME']!r}; known: {sorted(_MODEL_PLACEMENT)}")
    if payload["env"]["MODE"] not in ("normal", "debug_minimal", "rollout_only"):
        _fail("MODE must be normal | debug_minimal | rollout_only")
    if payload["env"]["RUN_ID"] != payload["name"]:
        _fail("env.RUN_ID must equal name (it names the SFS dir and WandB group)")
    if "'" in payload["command"]:
        _fail("command must not contain apostrophes (the RayJob entrypoint is single-quoted)")
    for token in (
        f"--mode {payload['env']['MODE']}",
        f"--model-name {payload['env']['MODEL_NAME']}",
        f"--num-nodes {payload['workers']}",
    ):
        if token not in payload["command"]:
            _fail(f"command does not contain '{token}' declared in env")
    return payload


def _render(payload: dict) -> str:
    env = payload["env"]
    placement = _MODEL_PLACEMENT[env["MODEL_NAME"]]
    values = {
        "JOB_NAME": payload["name"],
        "SECRET_NAME": f"{payload['name']}-secrets",
        "IMAGE": payload["image"],
        "TASKSET_REMOTE_REF": env["TASKSET_REF"],
        "MODEL_NAME": env["MODEL_NAME"],
        "TASK_LIMIT": str(env["TASK_LIMIT"]),
        "COMMAND": payload["command"],
        "WORKER_REPLICAS": str(payload["workers"] - 1),
        "NUM_GPUS": str(payload["gpus_per_worker"]),
        **placement,
    }
    template = (HERE / "rayjob.yaml.tmpl").read_text()
    rendered = re.sub(r"\$\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)
    unresolved = sorted(set(re.findall(r"\$\{(\w+)\}", rendered)))
    if unresolved:
        _fail(f"template variables left unresolved: {unresolved}")
    if payload.get("secrets"):
        extra = "".join(
            f"\n                - secretRef: {{name: {s}}}" for s in payload["secrets"] if s != "wandb-api"
        )
        rendered = rendered.replace(
            "                - secretRef: {name: wandb-api}",
            "                - secretRef: {name: wandb-api}" + extra,
        )
    return rendered


def _create_run_secret(name: str) -> None:
    creds = Path.home() / ".config/fleet/credentials.json"
    if not creds.exists():
        _fail("~/.config/fleet/credentials.json not found; run `flt auth login registry-alpha.fleetai.me`")
    literals = [f"--from-literal=FLEET_CREDENTIALS_B64={base64.b64encode(creds.read_bytes()).decode()}"]
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if os.environ.get(key):
            literals.append(f"--from-literal={key}={os.environ[key]}")
    manifest = subprocess.run(
        KUBECTL + ["create", "secret", "generic", f"{name}-secrets", *literals, "--dry-run=client", "-o", "yaml"],
        check=True, capture_output=True, text=True,
    ).stdout
    subprocess.run(KUBECTL + ["apply", "-f", "-"], input=manifest, check=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("payload", help="path to a RunPayload JSON")
    parser.add_argument("--dry-run", action="store_true", help="print the rendered RayJob instead of applying")
    args = parser.parse_args()

    payload = _load_payload(args.payload)
    manifest = _render(payload)
    if args.dry_run:
        print(manifest)
        return
    _create_run_secret(payload["name"])
    subprocess.run(KUBECTL + ["apply", "-f", "-"], input=manifest, check=True, text=True)
    name = payload["name"]
    print(f"submitted: kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs get rayjob {name} -w")
    print(f"logs:      kubectl --context fleet-training-rl1-us-east-1 -n fleet-train-jobs logs -f job/{name}")
    print(f"sfs:       /mnt/sfs/miles-fleet/{name}/driver.log")


if __name__ == "__main__":
    main()
