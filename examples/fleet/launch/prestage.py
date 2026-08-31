#!/usr/bin/env python3
"""Pre-stage model weights onto the shared filesystem from a CPU node.

Downloads run inside a training job by default, which parks the job's GPUs
idle for the whole download (a 640GB checkpoint is 1-2 hours). This script
does the same download as a small job on the CPU pool instead, using the
same lock file as the training boot, so a training job submitted later (or
concurrently) finds the weights hot and skips straight to work.

    ./prestage.py zai-org/GLM-5.3-Flash-BF16
    ./prestage.py Qwen/Qwen3.6-27B --revision <40-hex-revision> \
      --image ghcr.io/fleet-ai/miles-fleet/trainer@sha256:<digest>

Idempotent: if the weights are already on /mnt/sfs the job exits in seconds.
"""

import argparse
import re
import subprocess

_KUBECTL = [
    "kubectl",
    "--context",
    "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
    "-n",
    "fleet-train-jobs",
]
_DEFAULT_IMAGE = "ghcr.io/fleet-ai/miles-fleet/trainer:de27a84a"
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_JOB_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

_MANIFEST = """
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: fleet-train-jobs
  labels: {{app: miles-fleet-prestage}}
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      nodeSelector: {{workload: fleetai-training-ng-cpu}}
      tolerations:
        - {{key: workload, operator: Exists, effect: NoSchedule}}
      imagePullSecrets:
        - name: ghcr-pull
      volumes:
        - name: sfs
          persistentVolumeClaim: {{claimName: sfs-shared}}
      containers:
        - name: prestage
          image: {image}
          command: ["bash", "-c"]
          args:
            - |
              set -euo pipefail
              MODEL_DIR=/mnt/sfs/miles-fleet/models
              mkdir -p "$MODEL_DIR"
              exec 9>"$MODEL_DIR/.prep.lock"
              flock -w 14400 9
              if [ -f "$MODEL_DIR/{path}/config.json" ]; then
                echo "already staged: $MODEL_DIR/{path}"
                exit 0
              fi
              hf download {repo}{revision_arg} --local-dir "$MODEL_DIR/{path}"
              echo "staged: $MODEL_DIR/{path}"
          volumeMounts:
            - {{name: sfs, mountPath: /mnt/sfs}}
          resources:
            requests: {{cpu: "4", memory: 8Gi}}
            limits: {{cpu: "8", memory: 16Gi}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="HF repo, e.g. zai-org/GLM-5.3-Flash-BF16")
    parser.add_argument("--image", default=_DEFAULT_IMAGE, help="image providing the hf CLI")
    parser.add_argument("--revision", help="exact 40-character Hugging Face commit")
    parser.add_argument("--job-name", help="explicit unique DNS-safe job name")
    args = parser.parse_args()

    if not _REPO_RE.fullmatch(args.repo):
        parser.error("repo must be an owner/name coordinate")
    if args.revision is not None and not _REVISION_RE.fullmatch(args.revision):
        parser.error("--revision must be exactly 40 lowercase hex characters")
    name = args.repo.split("/", 1)[1]
    suffix = f"-{args.revision[:8]}" if args.revision else ""
    default_job = "prestage-" + re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")[:40] + suffix
    job_name = args.job_name or default_job
    if not _JOB_RE.fullmatch(job_name):
        parser.error("--job-name must be a DNS-safe name no longer than 63 characters")
    path = f"{name}/{args.revision}" if args.revision else name
    revision_arg = f" --revision {args.revision}" if args.revision else ""
    manifest = _MANIFEST.format(
        job_name=job_name,
        image=args.image,
        repo=args.repo,
        path=path,
        revision_arg=revision_arg,
    )
    exists = subprocess.run(
        _KUBECTL + ["get", "job", job_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if exists.returncode == 0:
        raise SystemExit(f"refusing to replace existing prestage job: {job_name}")
    subprocess.run(_KUBECTL + ["create", "-f", "-"], input=manifest, check=True, text=True)
    print(f"watch: kubectl --context {_KUBECTL[2]} -n fleet-train-jobs logs -f job/{job_name}")


if __name__ == "__main__":
    main()
