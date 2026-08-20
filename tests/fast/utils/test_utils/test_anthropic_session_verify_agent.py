import asyncio
import json
from types import SimpleNamespace

from miles.rollout.base_types import GenerateFnOutput
from miles.utils.test_utils import anthropic_session_verify_agent
from miles.utils.types import Sample


class _Response:
    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


def test_run_agent_calls_messages_and_checks_canonical_record(monkeypatch):
    posted = []
    tool_use = {
        "type": "tool_use",
        "id": "call_weather",
        "name": "get_weather",
        "input": {"location": "Beijing"},
    }
    anthropic_body = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should use the weather tool."},
            tool_use,
        ],
        "stop_reason": "tool_use",
    }
    snapshot = {
        "records": [
            {
                "path": "/v1/chat/completions",
                "request": {
                    "input_ids": [1, 2, 3],
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "user"},
                    ],
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"location": "Beijing"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ]
    }

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            assert timeout == 180

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, json):
            posted.append((url, json))
            return _Response(anthropic_body)

        async def get(self, url):
            assert url == "http://session"
            return _Response(snapshot)

    monkeypatch.setattr(
        anthropic_session_verify_agent.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    result = asyncio.run(
        anthropic_session_verify_agent.run_agent(
            "http://session",
            prompt=None,
            request_kwargs={
                "max_tokens": 128,
                "temperature": 0.2,
                "stop": "<stop>",
            },
            metadata={"anthropic_model": "/models/test"},
        )
    )

    [(url, payload)] = posted
    assert url == "http://session/v1/messages"
    assert payload["model"] == "/models/test"
    assert payload["stream"] is False
    assert payload["tool_choice"] == {"type": "any"}
    assert payload["stop_sequences"] == ["<stop>"]
    assert payload["tools"][0]["input_schema"]["required"] == ["location"]
    assert result == {
        "endpoint": "anthropic",
        "driver_events": ["anthropic_tool_use"],
        "request_count": 1,
        "tool_use_count": 1,
    }


def test_generate_injects_model_and_writes_tito_metrics(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    returned_sample = Sample(
        metadata={
            "endpoint": "anthropic",
            "driver_events": ["anthropic_tool_use"],
            "tool_use_count": 1,
            "tito_session_mismatch": [],
        }
    )

    async def fake_base_generate(input):
        assert input.sample.metadata["anthropic_model"] == "/models/test"
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(
        anthropic_session_verify_agent,
        "_base_generate",
        fake_base_generate,
    )
    input_sample = Sample()
    input_value = SimpleNamespace(
        sample=input_sample,
        args=SimpleNamespace(hf_checkpoint="/models/test"),
    )

    output = asyncio.run(anthropic_session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    assert input_sample.metadata["anthropic_model"] == "/models/test"
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["driver_events"] == ["anthropic_tool_use"]
    assert metric["had_assistant_mismatch"] is False
    assert metric["total_mismatches"] == 0
