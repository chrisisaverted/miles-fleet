"""Every tool-call grammar the parser claims, happy and broken."""

from examples.fleet.parser import parse_tool_call


def test_qwen_json_tag():
    action = 'I will search.\n<tool_call>{"name": "app__search", "arguments": {"q": "x"}}</tool_call>'
    assert parse_tool_call(action) == {"name": "app__search", "arguments": {"q": "x"}}


def test_qwen_tag_without_closing():
    action = '<tool_call>{"name": "a__b", "arguments": {}}'
    call = parse_tool_call(action)
    assert call and call["name"] == "a__b"


def test_xml_function_grammar():
    action = "<tool_call>\n<function=app__get>\n<parameter=id>\n7\n</parameter>\n</function>\n</tool_call>"
    call = parse_tool_call(action)
    assert call and call["name"] == "app__get"
    assert call["arguments"]["id"] == 7


def test_kimi_special_tokens():
    action = (
        "<|tool_call_begin|>functions.app__list:1<|tool_call_argument_begin|>"
        '{"limit": 5}<|tool_call_end|>'
    )
    call = parse_tool_call(action)
    assert call and call["name"] == "app__list"
    assert call["arguments"] == {"limit": 5}


def test_bare_function_tag():
    action = "<function=app__run>\n<parameter=cmd>ls</parameter>\n</function>"
    call = parse_tool_call(action)
    assert call and call["name"] == "app__run"


def test_missing_brace_repair():
    action = '<tool_call>{"name": "a__b", "arguments": {"x": 1}</tool_call>'
    call = parse_tool_call(action)
    assert call and call["arguments"] == {"x": 1}


def test_no_tool_call():
    assert parse_tool_call("I think the answer is 42.") is None


def test_first_call_wins_deterministically():
    action = (
        '<tool_call>{"name": "first", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "second", "arguments": {}}</tool_call>'
    )
    assert parse_tool_call(action)["name"] == "first"


def test_empty_arguments():
    call = parse_tool_call('<tool_call>{"name": "a__b"}</tool_call>')
    assert call and (call.get("arguments") in ({}, None))


def test_submit_with_answer():
    call = parse_tool_call('<tool_call>{"name": "fleet_submit", "arguments": {"answer": "42"}}</tool_call>')
    assert call["name"] == "fleet_submit"
    assert call["arguments"]["answer"] == "42"


def test_glm_arg_key_value_grammar():
    """GLM-4.6 family: name line + arg_key/arg_value pairs, exactly as the
    vendor chat template instructs the model to emit."""
    action = (
        "<think>ok</think>Doing it.\n"
        "<tool_call>app__click\n"
        "<arg_key>x</arg_key>\n"
        "<arg_value>10</arg_value>\n"
        "<arg_key>label</arg_key>\n"
        "<arg_value>submit</arg_value>\n"
        "</tool_call>"
    )
    call = parse_tool_call(action)
    assert call == {"name": "app__click", "arguments": {"x": 10, "label": "submit"}}


def test_glm_grammar_missing_closing_tag():
    action = "<tool_call>app__do\n<arg_key>q</arg_key>\n<arg_value>{\"a\": 1}</arg_value>"
    call = parse_tool_call(action)
    assert call and call["name"] == "app__do"
    assert call["arguments"] == {"q": {"a": 1}}


def test_glm_grammar_no_args():
    """A bare name with no arg pairs is NOT claimed by the GLM parser (it
    requires at least one <arg_key>), so garbage payloads still parse-fail."""
    assert parse_tool_call("<tool_call>just some prose</tool_call>") is None


def test_glm_grammar_zero_arg_call():
    call = parse_tool_call("<tool_call>app__list\n</tool_call>")
    assert call == {"name": "app__list", "arguments": {}}
