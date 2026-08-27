"""Token assembly against the REAL GLM-4.7-Flash tokenizer (network: downloads
it from HF). Pins the properties the rollout depends on, for the exact model
we train:

- the GLM47 TITO tokenizer appends tool/user observations incrementally,
- the ambiguous trailing boundary token (<|user|>/<|observation|>) the model
  stops on is trimmed at the junction, and our mask arithmetic follows,
- the native template renders tools= and keeps thinking with the registered
  clear_thinking=False kwarg.

Run: pytest -m network tests/test_glm_template.py
"""

import pytest

from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer

from examples.fleet.agent import build_messages
from examples.fleet.recording import _append_messages, _record_assistant, _Segment
from miles.utils.types import Sample

pytestmark = pytest.mark.network

MODEL = "zai-org/GLM-4.7-Flash"


@pytest.fixture(scope="module")
def tito():
    from transformers import AutoTokenizer

    return get_tito_tokenizer(AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True), "glm47")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "app__query",
            "description": "Query the app database.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]


def make_segment(tito):
    messages = build_messages("count the rows", 32, 1, 1)
    sample = Sample(prompt=list(messages), metadata={})
    prompt_ids = tito.apply_chat_template(messages, add_generation_prompt=True, tools=TOOLS, tokenize=True)
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    return _Segment(sample=sample, messages=messages, prompt_len=len(prompt_ids))


def simulate_generation(tito, segment, text, tool_call=None, stop_token="<|observation|>"):
    """Append sampled ids the way update_sample_from_response does: the raw
    text's tokens plus the boundary stop token the model actually emits."""
    ids = tito.tokenizer.encode(text, add_special_tokens=False)
    ids = ids + [tito.tokenizer.convert_tokens_to_ids(stop_token)]
    s = segment.sample
    s.tokens = s.tokens + ids
    s.response_length += len(ids)
    s.response += text
    s.loss_mask = (s.loss_mask or []) + [1] * len(ids)
    s.rollout_log_probs = (s.rollout_log_probs or []) + [-0.1] * len(ids)
    _record_assistant(segment, text, tool_call, turn=1)
    return len(ids)


def test_prompt_render_carries_tools(tito):
    segment = make_segment(tito)
    prompt_text = tito.tokenizer.decode(segment.sample.tokens)
    assert "app__query" in prompt_text
    assert "count the rows" in prompt_text
    assert prompt_text.endswith("<think>")  # generation opener present


def test_tool_observation_appends_and_trims_boundary(tito):
    segment = make_segment(tito)
    call = {"name": "app__query", "arguments": {"q": "select 1"}}
    gen = simulate_generation(
        tito, segment, "thinking</think>I will query.<tool_call>app__query</tool_call>", call
    )
    obs_id = tito.tokenizer.convert_tokens_to_ids("<|observation|>")
    assert segment.sample.tokens[-1] == obs_id

    before_len = len(segment.sample.tokens)
    _append_messages(
        segment,
        tito,
        [{"role": "tool", "tool_call_id": "call_000001", "name": "app__query", "content": "Tool result:\n6 7\n[Turn 1/32]"}],
    )
    s = segment.sample
    # the ambiguous stop token was trimmed, the observation appended
    assert len(s.tokens) > before_len
    assert len(s.loss_mask) == s.response_length == len(s.tokens) - segment.prompt_len
    assert len(s.rollout_log_probs) == s.response_length
    # the sampled boundary token overlaps the observation's opening token, so
    # the positional diff keeps it attributed to the model (mask 1, real
    # logprob); nothing else in the observation carries loss
    assert sum(s.loss_mask) == gen
    text = tito.tokenizer.decode(s.tokens)
    assert "6 7" in text
    assert text.endswith("<think>")  # ready for the next generation
    # no duplicated boundary marker at the junction
    assert "<|observation|><|observation|>" not in text


def test_mismatched_boundary_token_is_trimmed(tito):
    """Model stopped with <|observation|> but the next message is role-user
    (parse failure): the stale boundary token is trimmed and its mask/logprob
    entries go with it."""
    segment = make_segment(tito)
    gen = simulate_generation(tito, segment, "no call here</think>oops", None, stop_token="<|observation|>")
    _append_messages(segment, tito, [{"role": "user", "content": "No tool call found."}])
    s = segment.sample
    assert sum(s.loss_mask) == gen - 1
    assert len(s.loss_mask) == s.response_length == len(s.tokens) - segment.prompt_len
    text = tito.tokenizer.decode(s.tokens)
    assert "<|observation|>" not in text[-400:] or "<|observation|><|user|>" not in text


def test_user_observation_appends(tito):
    segment = make_segment(tito)
    simulate_generation(tito, segment, "rambling with no call</think>oops", None, stop_token="<|user|>")
    _append_messages(
        segment,
        tito,
        [{"role": "user", "content": "No tool call found. End your response with exactly one tool call."}],
    )
    s = segment.sample
    assert len(s.loss_mask) == s.response_length
    text = tito.tokenizer.decode(s.tokens)
    assert "No tool call found" in text
    assert "<|user|><|user|>" not in text
    assert text.endswith("<think>")


def test_multi_turn_assembly_stays_consistent(tito):
    segment = make_segment(tito)
    call = {"name": "app__query", "arguments": {"q": "a"}}
    simulate_generation(tito, segment, "t1</think>c1<tool_call>app__query</tool_call>", call)
    _append_messages(segment, tito, [{"role": "tool", "tool_call_id": "c1", "name": "app__query", "content": "r1"}])
    simulate_generation(tito, segment, "t2</think>c2<tool_call>app__query</tool_call>", call)
    _append_messages(segment, tito, [{"role": "tool", "tool_call_id": "c2", "name": "app__query", "content": "r2"}])
    s = segment.sample
    assert len(s.loss_mask) == s.response_length == len(s.tokens) - segment.prompt_len
    text = tito.tokenizer.decode(s.tokens)
    assert "r1" in text and "r2" in text and "t1" in text and "t2" in text  # thinking preserved in tokens
