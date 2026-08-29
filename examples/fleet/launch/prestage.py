#!/usr/bin/env python3
"""Pre-stage model weights onto the shared filesystem from a CPU node.

Downloads run inside a training job by default, which parks the job's GPUs
idle for the whole download (a 640GB checkpoint is 1-2 hours). This script
does the same download as a small job on the CPU pool instead, using the
same lock file as the training boot, so a training job submitted later (or
concurrently) finds the weights hot and skips straight to work.

    ./prestage.py zai-org/GLM-5.3-Flash-BF16
    ./prestage.py Qwen/Qwen3.8-27B --image ghcr.io/fleet-ai/miles-fleet/trainer:<sha>

Idempotent: if the weights are already on /mnt/sfs the job exits in seconds.
"""

import argparse
import re
import subprocess

_KUBECTL = [
    "kubectl", "--context", "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
    "-n", "fleet-train-jobs",
]
_DEFAULT_IMAGE = "ghcr.io/fleet-ai/miles-fleet/trainer:de27a84a"

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
              if [ -f "$MODEL_DIR/{name}/config.json" ]; then
                echo "already staged: $MODEL_DIR/{name}"
                exit 0
              fi
              hf download {repo} --local-dir "$MODEL_DIR/{name}"
              echo "staged: $MODEL_DIR/{name}"
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
    args = parser.parse_args()

    name = args.repo.split("/", 1)[1]
    job_name = "prestage-" + re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")[:40]
    manifest = _MANIFEST.format(job_name=job_name, image=args.image, repo=args.repo, name=name)
    subprocess.run(_KUBECTL + ["delete", "job", job_name, "--ignore-not-found"], check=True)
    subprocess.run(_KUBECTL + ["apply", "-f", "-"], input=manifest, check=True, text=True)
    print(f"watch: kubectl --context {_KUBECTL[2]} -n fleet-train-jobs logs -f job/{job_name}")


if __name__ == "__main__":
    main()
