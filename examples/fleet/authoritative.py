"""Fleet-authoritative report-only cyber rollout transport and reward parsing.

Miles still owns policy sampling, token/log-prob capture, and optimization.  This
module only delegates the live environment and Verifier Contract v3 evidence to
Fleet.  A malformed or weakly-bound response raises: infrastructure ambiguity
must drop a sample, never become a policy reward of zero.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

REPORT_ONLY_SUFFIX = "__blackbox_ctf_v1"
AUTHORITATIVE_BACKEND = "fleet_authoritative_cyber_v1"
SAFE_SUCCESS_OBJECTIVE = "safe_success_v1"
RAW_CAPABILITY_OBJECTIVE = "raw_capability_v1"
SUPPORTED_OBJECTIVES = frozenset({SAFE_SUCCESS_OBJECTIVE, RAW_CAPABILITY_OBJECTIVE})

logger = logging.getLogger(__name__)


class AuthoritativeContractError(RuntimeError):
    """Fleet returned a response that cannot safely train against."""


def _fleet_sdk():
    """Load the owner SDK and reject the verifier-only compatibility facade.

    ``fleet-runtime`` and ``fleet-python`` both currently ship a top-level
    ``fleet`` package.  If the runtime wheel is installed last, its guarded
    verifier facade replaces the SDK's ``__init__`` and authoritative rollout
    provisioning cannot start.  Keep this check on the exact import path used
    by :meth:`AuthoritativeCyberSession.open` so image builds fail before a
    paid job can enter Miles's unbounded aborted-sample replacement loop.
    """
    try:
        import fleet
    except ImportError as exc:
        raise AuthoritativeContractError(
            "Fleet owner SDK is unavailable; install fleet-python after fleet-runtime"
        ) from exc
    if not callable(getattr(fleet, "Fleet", None)):
        raise AuthoritativeContractError(
            "Fleet owner SDK is shadowed by a non-client fleet package; " "install fleet-python after fleet-runtime"
        )
    return fleet


def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthoritativeContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AuthoritativeContractError(f"{label} must be finite and between 0 and 1")
    return number


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthoritativeContractError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class AuthoritativeReward:
    reward: float
    safe_success: bool | None
    raw_capability_reward: float
    verifier_execution_id: str
    evidence_run_id: str


def parse_authoritative_reward(
    body: Any,
    *,
    task_key: str,
    task_version_id: str,
    instance_id: str,
    evidence_run_id: str,
    objective: str,
) -> AuthoritativeReward:
    """Validate a checked v3 response and select one named training objective."""
    if objective not in SUPPORTED_OBJECTIVES:
        raise AuthoritativeContractError(f"unsupported reward objective {objective!r}")
    if not isinstance(body, dict):
        raise AuthoritativeContractError("authoritative reward response must be an object")

    expected_identity = {
        "task_key": task_key,
        "task_version_id": task_version_id,
        "instance_id": instance_id,
    }
    for field, expected in expected_identity.items():
        if body.get(field) != expected:
            raise AuthoritativeContractError(
                f"authoritative reward response mismatched {field}: expected {expected!r}, got {body.get(field)!r}"
            )

    execution_id = _required_text(body.get("verifier_execution_id"), "verifier_execution_id")
    structured = body.get("cyber_verification_result")
    if not isinstance(structured, dict):
        raise AuthoritativeContractError("cyber_verification_result is required")
    if structured.get("schema_version") != "cyber_verification_result_v3":
        raise AuthoritativeContractError("cyber_verification_result has the wrong schema")

    structured_reward = _unit_float(structured.get("reward"), "structured reward")
    response_reward = _unit_float(body.get("reward"), "response reward")
    if not math.isclose(structured_reward, response_reward, rel_tol=0.0, abs_tol=1e-12):
        raise AuthoritativeContractError("top-level and structured rewards disagree")

    bindings = structured.get("bindings")
    if not isinstance(bindings, dict):
        raise AuthoritativeContractError("structured bindings are required")
    required_bindings = {
        "task_version_id": task_version_id,
        "projection_id": "blackbox_ctf_v1",
    }
    for field, expected in required_bindings.items():
        if bindings.get(field) != expected:
            raise AuthoritativeContractError(
                f"structured binding {field} mismatched: expected {expected!r}, got {bindings.get(field)!r}"
            )

    diagnostics = structured.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise AuthoritativeContractError("structured diagnostics are required")
    raw_capability_value = diagnostics.get("raw_capability_reward")
    raw_capability = (
        structured_reward
        if raw_capability_value is None
        else _unit_float(raw_capability_value, "diagnostics.raw_capability_reward")
    )
    safe_success = diagnostics.get("safe_success")
    if safe_success is not None and not isinstance(safe_success, bool):
        raise AuthoritativeContractError("diagnostics.safe_success must be boolean or null")

    evidence = body.get("cyber_evidence")
    if not isinstance(evidence, dict):
        raise AuthoritativeContractError("authoritative cyber_evidence is required")
    if evidence.get("mode") != "authoritative" or evidence.get("status") != "authoritative":
        raise AuthoritativeContractError("cyber_evidence is not authoritative")
    if evidence.get("match") is not True:
        raise AuthoritativeContractError("cyber_evidence did not match the production grade")
    direct = evidence.get("direct_verifier")
    if not isinstance(direct, dict):
        raise AuthoritativeContractError("cyber_evidence.direct_verifier is required")
    if direct.get("status") != "authoritative" or direct.get("match") is not True:
        raise AuthoritativeContractError("direct verifier evidence is not authoritative")
    if direct.get("execution_id") != execution_id:
        raise AuthoritativeContractError("direct verifier execution identity mismatched")

    if objective == SAFE_SUCCESS_OBJECTIVE:
        if raw_capability_value is None:
            raise AuthoritativeContractError("safe_success_v1 requires diagnostics.raw_capability_reward")
        if not isinstance(safe_success, bool):
            raise AuthoritativeContractError("safe_success_v1 requires boolean diagnostics.safe_success")
        behavior = (structured.get("components") or {}).get("behavior")
        if not isinstance(behavior, dict) or behavior.get("safe_success") is not safe_success:
            raise AuthoritativeContractError("behavior and diagnostic safe_success disagree")
        selected_reward = 1.0 if safe_success else 0.0
    else:
        selected_reward = raw_capability

    return AuthoritativeReward(
        reward=selected_reward,
        safe_success=safe_success,
        raw_capability_reward=raw_capability,
        verifier_execution_id=execution_id,
        evidence_run_id=evidence_run_id,
    )


class MCPError(RuntimeError):
    """Transport or protocol failure talking to a managed instance MCP endpoint."""


class MCPClient:
    """Small synchronous streamable-HTTP MCP client for tools/list and tools/call."""

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(self, url: str, *, auth_token: str, timeout: float):
        self.url = url
        self.auth_token = auth_token
        self.timeout = timeout
        self._id = 0
        self._session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.auth_token}",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            payload["params"] = params
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self._session_id = session_id
                raw = response.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            raise MCPError(f"{method} -> HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise MCPError(f"{method} -> {type(exc).__name__}: {exc}") from exc

        message = self._parse_body(raw)
        if "error" in message:
            raise MCPError(f"{method} -> rpc error: {json.dumps(message['error'])[:300]}")
        result = message.get("result", {})
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _parse_body(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("{"):
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        last: dict[str, Any] | None = None
        for line in raw.splitlines():
            if line.strip().startswith("data:"):
                candidate = line.strip()[5:].strip()
                if candidate and candidate != "[DONE]":
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        last = value
        if last is None:
            raise MCPError(f"unparseable MCP response: {raw[:200]}")
        return last

    def initialize(self) -> None:
        self._rpc(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "miles-fleet", "version": "0.1.0"},
            },
        )
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=self.timeout).read()
        except Exception:
            pass

    def openai_tools(self) -> list[dict[str, Any]]:
        tools = self._rpc("tools/list").get("tools") or []
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": (tool.get("description") or "")[:1024],
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(item, default=str)[:2000])
        return "\n".join(part for part in parts if part), result.get("isError") is True


class AuthoritativeCyberSession:
    """One report-only cyber attempt provisioned and graded by Fleet."""

    def __init__(
        self,
        *,
        task_key: str,
        task_version_id: str,
        instructions: str,
        objective: str,
        config: Any,
        expected_binding: dict[str, str],
    ):
        if not task_key.endswith(REPORT_ONLY_SUFFIX):
            raise ValueError("authoritative cyber sessions require a report-only projection")
        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"unsupported authoritative objective {objective!r}")
        self.task_key = task_key
        self.task_version_id = str(uuid.UUID(task_version_id))
        self.instructions = _required_text(instructions, "task instructions")
        self.objective = objective
        self.config = config
        self.expected_binding = dict(expected_binding)
        self.tools: list[dict[str, Any]] = []
        self.tools_dropped = 0
        self.attempt_id: str | None = None
        self.raw_capability_reward: float | None = None
        self.safe_success: bool | None = None
        self.verifier_execution_id: str | None = None
        self._fleet = None
        self._env = None
        self._mcp: MCPClient | None = None
        self._instance_id: str | None = None
        self._evidence_run_id: str | None = None
        self._closed = False
        self._instance_reaped = False
        self._close_lock = threading.Lock()
        self._graded = False

    def open(self) -> None:
        fleet = _fleet_sdk()

        from examples.fleet.session import _SUBMIT_SCHEMA, _within_budget

        token = secrets.token_hex(32)
        self._fleet = fleet.Fleet(timeout=self.config.call_tool_timeout_s)
        response = self._fleet.client.request(
            "POST",
            f"/v1/rollout-rewards/{quote(self.task_key, safe='')}/versions/{quote(self.task_version_id, safe='')}/instances",
            json={"mcp_auth_token": token},
            timeout=self.config.call_tool_timeout_s,
        )
        body = response.json()
        if not isinstance(body, dict):
            raise AuthoritativeContractError("instance response must be an object")
        for field, expected in {
            "task_key": self.task_key,
            "task_version_id": self.task_version_id,
        }.items():
            if body.get(field) != expected:
                raise AuthoritativeContractError(f"instance response mismatched {field}")
        instance_id = _required_text(body.get("instance_id"), "instance_id")
        self._evidence_run_id = str(uuid.UUID(_required_text(body.get("evidence_run_id"), "evidence_run_id")))
        self.attempt_id = self._evidence_run_id
        with self._close_lock:
            self._instance_id = instance_id
            closed_during_create = self._closed

        # Episode timeout closes the session from another thread.  If that won
        # while provisioning was in flight, reap the just-created managed
        # instance instead of letting an abandoned paid environment escape.
        if closed_during_create:
            self.close()
            raise RuntimeError(f"[{self.task_key}] session closed during open")

        # Reuse the authenticated owner client used for provisioning.  This
        # preserves its base URL, timeout, retry policy, and team credentials.
        env = self._fleet.instance(self._instance_id)
        env.instance.load()
        with self._close_lock:
            if self._closed:
                closed_during_load = True
            else:
                self._env = env
                closed_during_load = False
        if closed_during_load:
            self.close()
            raise RuntimeError(f"[{self.task_key}] session closed during open")
        actual_binding = {
            "env_key": str(getattr(env, "env_key", "") or ""),
            "env_version": str(getattr(env, "version", "") or ""),
            "data_key": str(getattr(env, "data_key", "") or ""),
            "data_version": str(getattr(env, "data_version", "") or ""),
        }
        for field, expected in self.expected_binding.items():
            if field in actual_binding and actual_binding[field] != expected:
                raise AuthoritativeContractError(
                    f"live instance mismatched immutable TaskSet {field}: expected {expected!r}, got {actual_binding[field]!r}"
                )

        self._mcp = MCPClient(
            env.mcp.url,
            auth_token=token,
            timeout=self.config.call_tool_timeout_s,
        )
        self._mcp.initialize()
        kept, dropped = _within_budget(self._mcp.openai_tools())
        self.tools = kept + [_SUBMIT_SCHEMA]
        self.tools_dropped = dropped

    def call_tool(self, name: str, arguments: dict[str, Any]):
        from examples.fleet.session import ToolOutcome, _with_deadline

        assert self._mcp is not None
        try:
            text, is_error = _with_deadline(
                lambda: self._mcp.call_tool(name, arguments),
                self.config.call_tool_timeout_s,
                "call_tool",
            )
        except TimeoutError as exc:
            return ToolOutcome(text="", error=str(exc), fatal=True)
        except Exception as exc:
            return ToolOutcome(text="", error=str(exc))
        if is_error:
            return ToolOutcome(text="", error=text or "tool_error")
        return ToolOutcome(text=text)

    @property
    def has_step_protocol(self) -> bool:
        return False

    @property
    def step_ordinal(self) -> int:
        return 1

    @property
    def step_count(self) -> int:
        return 1

    @property
    def is_final_step(self) -> bool:
        return True

    @property
    def current_instructions(self) -> str:
        return self.instructions

    def grade(self, answer: str | None, reset_ack=None, close_final_step: bool = False):
        del reset_ack, close_final_step
        from examples.fleet.session import GradeResult

        if self._graded:
            raise RuntimeError(f"[{self.task_key}] grade called twice")
        self._graded = True
        if not self._fleet or not self._instance_id or not self._evidence_run_id:
            raise AuthoritativeContractError("authoritative session is not open")
        payload: dict[str, Any] = {
            "instance_id": self._instance_id,
            "scoring_mode": "partial" if self.config.partial_reward else "binary",
            "multi_app_aggregation_mode": "fractional",
        }
        if answer is not None:
            payload["final_answer"] = answer
        response = self._fleet.client.request(
            "POST",
            f"/v1/rollout-rewards/{quote(self.task_key, safe='')}/versions/{quote(self.task_version_id, safe='')}",
            json=payload,
            timeout=self.config.grade_timeout_s,
        )
        parsed = parse_authoritative_reward(
            response.json(),
            task_key=self.task_key,
            task_version_id=self.task_version_id,
            instance_id=self._instance_id,
            evidence_run_id=self._evidence_run_id,
            objective=self.objective,
        )
        self.raw_capability_reward = parsed.raw_capability_reward
        self.safe_success = parsed.safe_success
        self.verifier_execution_id = parsed.verifier_execution_id
        return GradeResult(reward=parsed.reward)

    def close(self) -> None:
        with self._close_lock:
            self._closed = True
            if self._instance_reaped:
                return
            env, self._env = self._env, None
            fleet_owner = self._fleet
            instance_id = self._instance_id
            if env is None and (fleet_owner is None or not instance_id):
                # Provisioning has not produced an instance yet.  open() will
                # observe _closed as soon as it receives one and call us again.
                return
            self._instance_reaped = True
        if env is not None:
            try:
                env.close()
                return
            except Exception as exc:
                logger.warning("[%s] managed environment close failed: %s", self.task_key, exc)

        # Provisioning can succeed before Fleet.instance() does.  The owning
        # Fleet client can still reap that instance by its returned identity.
        if fleet_owner is not None and instance_id:
            try:
                fleet_owner.close(instance_id)
            except Exception as exc:
                logger.warning("[%s] managed instance fallback close failed: %s", self.task_key, exc)
