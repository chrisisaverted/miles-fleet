"""Build miles train/eval JSONL from a Fleet v2 taskset.

Row schema (miles Dataset defaults: --input-key input, --label-key label,
--metadata-key metadata):
    input     one user message with step 1's instructions. The rollout
              rebuilds the real conversation (instructions + tool surface) at
              episode start; this column only feeds the loader's prompt-length
              filter.
    label     the task key (unused by the reward path; grading is the reward).
    metadata  task identity, delivered verbatim to the generate fn:
              task_key, taskset_ref, data_source, steps, context_profile.

Usage:
    python -m examples.fleet.prepare_dataset \
        --taskset-ref <flt-alias-or-digest-or-yaml> --output-dir data/my-taskset \
        [--eval-fraction 0.2] [--seed 42] [--max-tasks 0]

The taskset must already be in the local store (`flt pull <remote> <alias>`)
or be a YAML path; pulling from the registry is not this script's job.

Also writes images.txt: the env-image refs the SELECTED tasks need, one per
line, so a launch pre-pulls only those (evaluation-benchmark has 112 distinct
images; a 64-task subset needs a few).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

# Seed payloads stream into env containers via `docker cp` under the SDK's
# docker-op timeout (configurable since platform#470; runs export
# FLEET_DOCKER_TIMEOUT_S=600). 10GB at a loaded-disk ~20MB/s is ~500s, inside
# the fuse with margin. Anything bigger (evaluation-benchmark tops out at
# 42GB) needs platform-side volume seeding, not cp streaming.
MAX_SEED_BYTES = 10_000_000_000


def seed_bytes(task) -> int:
    total = 0
    for env in (getattr(task, "environments", None) or {}).values():
        seed = getattr(env, "seed", None)
        if seed is None:
            continue
        data = seed.canonical_data() if callable(getattr(seed, "canonical_data", None)) else None
        for mount in (data or {}).get("mounts", []) if isinstance(data, dict) else []:
            total += (mount.get("remote_blob") or {}).get("bytes", 0)
    return total


def eligible(task, max_seed_bytes: int = MAX_SEED_BYTES) -> tuple[bool, str]:
    if len(task.seats) != 1:
        return False, f"seats={len(task.seats)}"
    if not task.is_attempt_capable:
        return False, "no bound prompt"
    if max_seed_bytes:
        sb = seed_bytes(task)
        if sb > max_seed_bytes:
            return False, f"seed={sb / 1e9:.1f}GB > {max_seed_bytes / 1e9:.0f}GB cap"
    return True, ""


def build_rows(taskset, taskset_ref: str, args) -> tuple[list[dict], list[tuple[str, str]]]:
    rows, skipped = [], []
    for compiled in taskset.tasks:
        task = compiled.task
        ok, why = eligible(task)
        if not ok:
            skipped.append((task.task_id, why))
            continue
        instructions = task.steps[0].prompt or ""
        rows.append(
            {
                "input": [{"role": "user", "content": instructions}],
                "label": task.task_id,
                "metadata": {
                    "task_key": task.task_id,
                    "taskset_ref": taskset_ref,
                    "data_source": taskset.name or taskset_ref,
                    "steps": len(task.steps),
                    "context_profile": str(getattr(task, "context_profile", "")).rsplit(".", 1)[-1].lower(),
                },
            }
        )
    if args.max_tasks and len(rows) > args.max_tasks:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_tasks]
    return rows, skipped


def split_rows(rows: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic (seeded) shuffle, eval slice first. Returns (train, eval)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_fraction)) if eval_fraction > 0 else 0
    # the launcher always requires train.jsonl: training keeps at least one
    # task, so a one-task set trains rather than becoming eval-only
    n_eval = min(n_eval, len(rows) - 1)
    return rows[n_eval:], rows[:n_eval]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset-ref", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all; seeded random subset otherwise")
    args = parser.parse_args()

    from fleet_runtime.cli.sources import resolve_source

    # YAML-path refs get absolutized so the JSONL works from any cwd.
    if os.path.isfile(args.taskset_ref):
        args.taskset_ref = os.path.abspath(args.taskset_ref)

    taskset, _ = resolve_source(args.taskset_ref)
    rows, skipped = build_rows(taskset, args.taskset_ref, args)

    if skipped:
        print(f"skipped {len(skipped)} tasks:")
        for key, why in skipped[:20]:
            print(f"  {key}: {why}")
    if not rows:
        raise SystemExit("no eligible tasks")

    step_counts = collections.Counter(r["metadata"]["steps"] for r in rows)
    profiles = collections.Counter(r["metadata"]["context_profile"] for r in rows)
    print(f"step distribution: {dict(sorted(step_counts.items()))}")
    print(f"context profiles: {dict(profiles)}")

    train_rows, eval_rows = split_rows(rows, args.eval_fraction, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if train_rows:
        _write_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_rows)
    if eval_rows:
        _write_jsonl(os.path.join(args.output_dir, "eval.jsonl"), eval_rows)

    selected = {r["metadata"]["task_key"] for r in rows}
    needed = sorted(
        {
            str(env.image_ref)
            for compiled in taskset.tasks
            if compiled.task.task_id in selected
            for env in (getattr(compiled.task, "environments", None) or {}).values()
        }
    )
    with open(os.path.join(args.output_dir, "images.txt"), "w") as f:
        f.write("\n".join(needed) + ("\n" if needed else ""))
    print(f"images.txt: {len(needed)} env images for {len(selected)} tasks")
    print(
        f"taskset={taskset.name!r} root={taskset.root_digest[:24]} "
        f"train={len(train_rows)} eval={len(eval_rows)} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
