"""Minimal reproduction: gusto_mcp crash-loop in the fos-hr environment.

Observed on evaluation-benchmark fos-hr episodes (2026-08-22): supervisord
logs `spawned: 'gusto_mcp'` -> `WARN exited: gusto_mcp (exit status 1; not
expected)` -> respawn, with SIGABRT core dumps. gusto has no health endpoint
(the env declares only ui/api-*/health-{outlook,docusign,cadence}), so
readiness passes regardless; the open question this script answers is whether
gusto tools ever reach the tool surface and whether calling one works.

Self-contained: fleet-runtime SDK + a docker daemon + the taskset in the
local flt store. No miles.

    flt pull registry-alpha.fleetai.me/gentle-cedar-garden/evaluation-benchmark:v3 evaluation-benchmark
    FLEET_FLT=$(which flt) FLEET_DOCKER_TIMEOUT_S=600 \
      python repro_gusto_mcp.py [taskset-ref] [task-key]

On the rl1 cluster, run inside the trainer image (taskset + image cached):
    kubectl -n fleet-train-jobs run gusto-repro --rm -i --restart=Never \
      --image=ghcr.io/fleet-ai/miles-fleet/trainer:latest ... (mount the flt
      store + docker socket like the training Job does) -- python \
      /root/miles/examples/fleet/tests/integration/repro_gusto_mcp.py
"""

import subprocess
import sys
import time

DEFAULT_REF = "evaluation-benchmark"
DEFAULT_KEY = "task_zhut1mawurbf_n_1781891669112_ryq6iakt7__synapse_increased_difficulty_candidate"


def sh(args: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # keep probing; every probe is optional evidence
        return f"<probe failed: {e}>"


def fleet_containers() -> list[str]:
    out = sh(["docker", "ps", "--filter", "label=fleet.runtime=1", "--format", "{{.ID}} {{.Image}}"])
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REF
    key = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_KEY

    from fleet_runtime.cli.sources import resolve_source
    from fleet_runtime.local import LocalRuntime
    from fleet_runtime.session.session import AttemptSession

    taskset, _ = resolve_source(ref)
    compiled = taskset.select(key)[0]
    runtime = LocalRuntime()
    runtime.load_blobs(tuple(compiled.blobs))
    print(f"[repro] preparing {key} (env boot; up to several minutes)...")
    t0 = time.time()
    session = AttemptSession(prepared=runtime.prepare(compiled.task, source=compiled.source))
    print(f"[repro] ready after {time.time() - t0:.0f}s")

    try:
        channel = session.channel

        # 1. Does gusto reach the tool surface?
        tools = [entry.face_name for entry in channel.tool_surface()]
        gusto_tools = [t for t in tools if "gusto" in t.lower()]
        print(f"[repro] tool surface: {len(tools)} tools")
        print(f"[repro] gusto tools: {gusto_tools if gusto_tools else 'ABSENT FROM SURFACE'}")

        # 2. Crash-loop evidence from inside the env container(s).
        for cid in fleet_containers():
            logs = sh(["docker", "logs", cid], timeout=60)
            if "gusto_mcp" not in logs:
                continue
            spawned = logs.count("spawned: 'gusto_mcp'")
            exited = logs.count("exited: gusto_mcp")
            fatal = logs.count("gave up: gusto_mcp") + logs.count("FATAL")
            print(f"[repro] container {cid}: gusto_mcp spawned={spawned} exited={exited} fatal-ish={fatal}")
            for line in logs.splitlines():
                if "gusto_mcp" in line:
                    print(f"    {line.strip()[:160]}")
            status = sh(["docker", "exec", cid, "supervisorctl", "status", "gusto_mcp"])
            print(f"[repro] supervisorctl status gusto_mcp: {status.strip()[:160]}")
            # 3. Watch for one minute: does it stabilize or keep flapping?
            print("[repro] sampling supervisorctl for 60s...")
            for _ in range(6):
                time.sleep(10)
                print("   ", sh(["docker", "exec", cid, "supervisorctl", "status", "gusto_mcp"]).strip()[:160])

        # 4. If a gusto tool is on the surface, call it and report the outcome.
        if gusto_tools:
            name = gusto_tools[0]
            print(f"[repro] calling {name} with empty args...")
            result = channel.call_tool(name, {})
            print(
                f"[repro] call status={getattr(result, 'status', '?')} "
                f"error_code={getattr(result, 'error_code', None)}"
            )
            for block in (getattr(result, "content", ()) or ())[:2]:
                text = getattr(block, "text", None)
                if text:
                    print(f"    {str(text)[:200]}")
    finally:
        session.close()
        print("[repro] session closed")


if __name__ == "__main__":
    main()
