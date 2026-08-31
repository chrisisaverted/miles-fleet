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
import uuid
from collections.abc import Mapping

from examples.fleet.authoritative import (
    AUTHORITATIVE_BACKEND,
    RAW_CAPABILITY_OBJECTIVE,
    REPORT_ONLY_SUFFIX,
    SUPPORTED_OBJECTIVES,
)
from examples.fleet.authoritative_selection import hydrate_authoritative_rows

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


def _taskdump_binding(task) -> dict[str, str]:
    """Read exact Fleet row provenance sealed into a TaskDump v7 TaskSet.

    TaskDump stores this under the code evaluator's ``params.task``.  It is
    immutable Task content, unlike a mutable registry tag or API current row.
    """
    candidates = []
    for verifier in getattr(task, "verifiers", None) or ():
        evaluator = getattr(verifier, "evaluator", None)
        params = getattr(evaluator, "params", None)
        value = params.get("task") if isinstance(params, Mapping) else None
        if isinstance(value, Mapping) and value.get("task_version_id"):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(f"task {task.task_id!r} must carry exactly one TaskDump v7 binding; found {len(candidates)}")
    raw = candidates[0]
    required = {
        "task_key": "task_key",
        "task_version_id": "task_version_id",
        "environment_version_id": "environment_version_id",
        "verifier_version_id": "verifier_version_id",
        "environment_key": "env_key",
        "environment_version": "env_version",
        "data_key": "data_key",
        "data_version": "data_version",
    }
    binding: dict[str, str] = {}
    for source, target in required.items():
        value = raw.get(source)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"task {task.task_id!r} binding is missing {source}")
        binding[target] = value
    for field in ("task_version_id", "environment_version_id", "verifier_version_id"):
        binding[field] = str(uuid.UUID(binding[field]))
    if binding["task_key"] != task.task_id:
        raise ValueError(f"task identity mismatch: TaskSet has {task.task_id!r}, binding has {binding['task_key']!r}")
    if not binding["task_key"].endswith(REPORT_ONLY_SUFFIX):
        raise ValueError(f"authoritative cyber backend supports only {REPORT_ONLY_SUFFIX} projections")
    return binding


def build_rows(taskset, taskset_ref: str, args) -> tuple[list[dict], list[tuple[str, str]]]:
    rows, skipped = [], []
    backend = getattr(args, "backend", "local")
    objective = getattr(args, "reward_objective", RAW_CAPABILITY_OBJECTIVE)
    if backend == AUTHORITATIVE_BACKEND and objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported authoritative reward objective {objective!r}")
    for compiled in taskset.tasks:
        task = compiled.task
        ok, why = eligible(
            task,
            max_seed_bytes=0 if backend == AUTHORITATIVE_BACKEND else MAX_SEED_BYTES,
        )
        if not ok:
            skipped.append((task.task_id, why))
            continue
        instructions = task.steps[0].prompt or ""
        authoritative = _taskdump_binding(task) if backend == AUTHORITATIVE_BACKEND else {}
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
                    **(
                        {
                            "fleet_backend": AUTHORITATIVE_BACKEND,
                            "reward_objective": objective,
                            "task_prompt": instructions,
                            **authoritative,
                        }
                        if authoritative
                        else {}
                    ),
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--taskset-ref")
    source.add_argument(
        "--authoritative-selection",
        help="immutable fleet.authoritative-selection.v1 file; distinct from TaskDump v7",
    )
    parser.add_argument(
        "--authoritative-selection-sha256",
        help="required byte digest for --authoritative-selection",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all; seeded random subset otherwise")
    parser.add_argument(
        "--backend",
        choices=("local", AUTHORITATIVE_BACKEND),
        default="local",
        help="where environments and grading run",
    )
    parser.add_argument(
        "--reward-objective",
        choices=tuple(sorted(SUPPORTED_OBJECTIVES)),
        default=RAW_CAPABILITY_OBJECTIVE,
        help=(
            "named authoritative v3 objective; safe_success_v1 requires checked behavior evidence on every selected task"
        ),
    )
    args = parser.parse_args()

    taskset = None
    if args.authoritative_selection:
        if args.backend != AUTHORITATIVE_BACKEND:
            raise SystemExit("--authoritative-selection requires the authoritative Fleet backend")
        if not args.authoritative_selection_sha256:
            raise SystemExit("--authoritative-selection-sha256 is required")
        rows = hydrate_authoritative_rows(
            args.authoritative_selection,
            api_key=os.environ.get("FLEET_API_KEY", ""),
            expected_sha256=args.authoritative_selection_sha256,
            reward_objective=args.reward_objective,
        )
        skipped = []
        if args.max_tasks and len(rows) > args.max_tasks:
            random.Random(args.seed).shuffle(rows)
            rows = rows[: args.max_tasks]
    else:
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
    needed = (
        []
        if args.backend == AUTHORITATIVE_BACKEND
        else sorted(
            {
                str(env.image_ref)
                for compiled in taskset.tasks
                if compiled.task.task_id in selected
                for env in (getattr(compiled.task, "environments", None) or {}).values()
            }
        )
    )
    with open(os.path.join(args.output_dir, "images.txt"), "w") as f:
        f.write("\n".join(needed) + ("\n" if needed else ""))
    print(f"images.txt: {len(needed)} env images for {len(selected)} tasks")
    if taskset is None:
        source_summary = f"authoritative_selection=sha256:{args.authoritative_selection_sha256[:24]}"
    else:
        source_summary = f"taskset={taskset.name!r} root={taskset.root_digest[:24]}"
    print(f"{source_summary} train={len(train_rows)} eval={len(eval_rows)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
