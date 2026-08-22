"""Clean repro: gusto_mcp crashes to FATAL on every fos-hr boot.

Boots ONE fos-hr environment from evaluation-benchmark and prints two facts:
the tool surface (names), and gusto_mcp's supervisord lifecycle from the
container log. Expected on the broken image: 4x spawned/exited(status 1),
then FATAL, and no gusto tools on the surface.

Impact: 4 of 291 fos-hr tasks reference gusto; those run without gusto tools.

Run where docker + the flt store + the taskset live (e.g. the trainer pod):
    PYTHONPATH=/root/miles FLEET_FLT=/usr/local/bin/flt FLEET_DOCKER_TIMEOUT_S=600 \
      python repro_gusto_mcp.py taskset <fos-hr-task-key>
Set FLEET_DOCKER_PIDS_LIMIT=2048 to rule the SDK's 512-pid default cap in or
out as a cause.
"""

import subprocess
import sys

DEFAULT_REF = "evaluation-benchmark"
DEFAULT_KEY = "task_zhut1mawurbf_n_1781891669112_ryq6iakt7__synapse_increased_difficulty_candidate"


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REF
    key = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_KEY

    from fleet_runtime.cli.sources import resolve_source
    from fleet_runtime.local import LocalRuntime
    from fleet_runtime.session.session import AttemptSession

    from examples.fleet.session import _image_locators_for

    taskset, _ = resolve_source(ref)
    compiled = taskset.select(key)[0]
    locators = _image_locators_for(getattr(taskset, "root_digest", ""), None)
    runtime = LocalRuntime(image_locators=locators) if locators else LocalRuntime()
    runtime.load_blobs(tuple(compiled.blobs))
    session = AttemptSession(prepared=runtime.prepare(compiled.task, source=compiled.source))
    try:
        tools = sorted(entry.face_name for entry in session.channel.tool_surface())
        print(f"tool surface ({len(tools)}): {tools}")
        print(f"gusto tools present: {[t for t in tools if 'gusto' in t.lower()] or 'NO'}")
        for line in subprocess.run(
            ["docker", "ps", "--filter", "label=fleet.runtime=1", "--format", "{{.ID}}"],
            capture_output=True, text=True,
        ).stdout.split():
            logs = subprocess.run(["docker", "logs", line], capture_output=True, text=True, timeout=60).stdout
            if "gusto_mcp" in logs:
                print(f"gusto_mcp lifecycle (container {line}):")
                for entry in logs.splitlines():
                    if "gusto_mcp" in entry:
                        print(f"  {entry.strip()[:160]}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
