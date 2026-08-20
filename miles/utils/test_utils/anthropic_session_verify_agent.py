"""Single-turn Anthropic Messages agent for per-model session verification."""

from __future__ import annotations

import json
import logging

import httpx

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import generate as _base_generate
from miles.utils.test_utils.session_verify_agent import (
    INITIAL_SYSTEM_PROMPT,
    INITIAL_USER_PROMPT,
    TOOLS,
    _verify_tito_samples,
)
from miles.utils.test_utils.session_verify_agent import generate as _session_verify_generate

logger = logging.getLogger(__name__)

_ANTHROPIC_TOOLS = [
    {
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "input_schema": tool["function"]["parameters"],
    }
    for tool in TOOLS
]


def _build_payload(request_kwargs: dict, metadata: dict) -> dict:
    payload = {
        "model": metadata["anthropic_model"],
        "max_tokens": request_kwargs["max_tokens"],
        "system": INITIAL_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": INITIAL_USER_PROMPT}],
        "tools": _ANTHROPIC_TOOLS,
        "tool_choice": {"type": "any"},
        "stream": False,
    }
    for key in ("temperature", "top_p", "top_k"):
        if request_kwargs.get(key) is not None:
            payload[key] = request_kwargs[key]
    if request_kwargs.get("stop") is not None:
        stop = request_kwargs["stop"]
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    return payload


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Call the native Messages endpoint once and verify its canonical record."""
    payload = _build_payload(request_kwargs, metadata)
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{base_url}/v1/messages", json=payload)
        assert response.status_code == 200, f"Anthropic tool turn failed ({response.status_code}): {response.text}"
        body = response.json()
        tool_uses = _assert_anthropic_response(body)

        session_response = await client.get(base_url)
        assert session_response.status_code == 200, session_response.text
        _assert_canonical_record(session_response.json(), tool_uses)

    return {
        "endpoint": "anthropic",
        "driver_events": ["anthropic_tool_use"],
        "request_count": 1,
        "tool_use_count": len(tool_uses),
    }


def _assert_anthropic_response(body: dict) -> list[dict]:
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "tool_use"
    tool_uses = [block for block in body["content"] if block["type"] == "tool_use"]
    assert tool_uses, f"Anthropic response contained no tool_use block: {body!r}"
    assert all(block["name"] == "get_weather" for block in tool_uses)
    assert all(isinstance(block["input"], dict) for block in tool_uses)
    return tool_uses


def _assert_canonical_record(snapshot: dict, tool_uses: list[dict]) -> None:
    [record] = snapshot["records"]
    assert record["path"] == "/v1/chat/completions"
    assert record["request"]["input_ids"]
    assert [message["role"] for message in record["request"]["messages"]] == [
        "system",
        "user",
    ]

    tool_calls = record["response"]["choices"][0]["message"]["tool_calls"]
    calls_by_id = {tool_call["id"]: tool_call for tool_call in tool_calls}
    assert set(calls_by_id) == {tool_use["id"] for tool_use in tool_uses}
    for tool_use in tool_uses:
        tool_call = calls_by_id[tool_use["id"]]
        assert tool_call["function"]["name"] == tool_use["name"]
        assert json.loads(tool_call["function"]["arguments"]) == tool_use["input"]


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run the Anthropic agent, check hard TITO mismatches, and write metrics."""
    input.sample.metadata["anthropic_model"] = input.args.hf_checkpoint
    output = await _base_generate(input)

    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    events_per_sample = [sample.metadata.get("driver_events", []) for sample in samples]
    for i, sample in enumerate(samples):
        if sample.metadata.get("endpoint") != "anthropic":
            raise AssertionError(f"Anthropic per-model e2e: sample {i} did not retain agent metadata")
        if "anthropic_tool_use" not in events_per_sample[i]:
            raise AssertionError(f"Anthropic per-model e2e: sample {i} did not produce a tool_use")

    _verify_tito_samples(samples, events_per_sample, allowed_roles=[])
    logger.info("Anthropic endpoint verified: samples=%d", len(samples))
    return output


generate.add_arguments = _session_verify_generate.add_arguments
