from __future__ import annotations

import json

import httpx


_SYSTEM_PROMPT = "You are a weather assistant. Use the provided tool when asked about weather."
_USER_PROMPT = "Use get_weather to check the weather in Beijing."
_FINAL_PROMPT = "Summarize the result in one short sentence without calling another tool."
_TOOL_RESULT = '{"temperature_celsius": 22, "condition": "sunny"}'
_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
]


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    messages = [{"role": "user", "content": _USER_PROMPT}]
    payload = {
        "model": metadata["anthropic_model"],
        "max_tokens": request_kwargs["max_tokens"],
        "temperature": request_kwargs.get("temperature", 0),
        "system": _SYSTEM_PROMPT,
        "messages": messages,
        "tools": _TOOLS,
        "tool_choice": {"type": "any"},
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        first = await _post_messages(client, base_url, payload, label="Anthropic tool turn")
        assert first["type"] == "message"
        assert first["role"] == "assistant"
        assert first["stop_reason"] == "tool_use"
        tool_uses = [block for block in first["content"] if block["type"] == "tool_use"]
        assert len(tool_uses) == 1
        tool_use = tool_uses[0]
        assert tool_use["name"] == "get_weather"
        assert isinstance(tool_use["input"], dict)

        first_snapshot = await _get_session(client, base_url)
        _assert_first_turn(first_snapshot, tool_use)

        messages.extend(
            [
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": _TOOL_RESULT,
                        },
                        {"type": "text", "text": _FINAL_PROMPT},
                    ],
                },
            ]
        )
        second_payload = {**payload, "messages": messages, "tool_choice": {"type": "none"}}
        second = await _post_messages(client, base_url, second_payload, label="Anthropic final turn")
        assert second["type"] == "message"
        assert second["role"] == "assistant"
        assert any(block["type"] == "text" and block["text"] for block in second["content"])
        assert all(block["type"] != "tool_use" for block in second["content"])

        second_snapshot = await _get_session(client, base_url)
        _assert_second_turn(first_snapshot, second_snapshot, tool_use)

    return {
        "endpoint": "anthropic",
        "record_count": 2,
        "stable_prefix_checked": True,
        "tool_use_count": 1,
    }


async def _post_messages(client: httpx.AsyncClient, base_url: str, payload: dict, *, label: str) -> dict:
    response = await client.post(f"{base_url}/v1/messages", json=payload)
    assert response.status_code == 200, f"{label} failed ({response.status_code}): {response.text}"
    return response.json()


async def _get_session(client: httpx.AsyncClient, base_url: str) -> dict:
    response = await client.get(base_url)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_first_turn(snapshot: dict, tool_use: dict) -> None:
    [record] = snapshot["records"]
    assert record["path"] == "/v1/chat/completions"
    assert record["request"]["input_ids"]
    assert [message["role"] for message in record["request"]["messages"]] == ["system", "user"]
    [tool_call] = record["response"]["choices"][0]["message"]["tool_calls"]
    assert tool_call["id"] == tool_use["id"]
    assert tool_call["function"]["name"] == tool_use["name"]
    assert json.loads(tool_call["function"]["arguments"]) == tool_use["input"]


def _assert_second_turn(first_snapshot: dict, second_snapshot: dict, tool_use: dict) -> None:
    first_record, second_record = second_snapshot["records"]
    assert first_record == first_snapshot["records"][0]
    assert second_record["path"] == "/v1/chat/completions"
    second_input_ids = second_record["request"]["input_ids"]
    assert second_input_ids

    messages = second_record["request"]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "user"]
    [tool_call] = messages[2]["tool_calls"]
    assert tool_call["id"] == tool_use["id"]
    assert messages[3]["tool_call_id"] == tool_use["id"]
    assert messages[3]["content"] == _TOOL_RESULT

    accumulated = first_snapshot["metadata"]["accumulated_token_ids"]
    max_trim_tokens = first_snapshot["metadata"]["max_trim_tokens"]
    stable_prefix_len = len(accumulated) - max_trim_tokens
    assert stable_prefix_len > 0
    assert second_input_ids[:stable_prefix_len] == accumulated[:stable_prefix_len]
