"""Vision observation encoding against the REAL Qwen3.8-27B tokenizer +
processor (network: downloads them from HF). Pins the properties the
fleet-vision path depends on for the model we plan to train:

- the dummy-prefix trim is exact under the real chat template,
- image content entries expand into real image-pad tokens via the processor,
- the qwen35 TITO boundary rule (insert the newline the model stops before)
  keeps mask arithmetic aligned.

Run: pytest -m network tests/test_qwen38_vision.py
"""

import pytest

from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer
from miles.utils.types import Sample

from examples.fleet.recording import _append_multimodal, _boundary_fix, _Segment

pytestmark = pytest.mark.network

MODEL = "Qwen/Qwen3.8-27B"


def _fixed_template() -> str:
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    path, _ = resolve_fixed_chat_template("qwen35")
    assert path is not None
    with open(path) as f:
        return f.read()


@pytest.fixture(scope="module")
def stack():
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # production swaps in the family's fixed template via --chat-template-path
    # (run_fleet.py); mirror that here
    tokenizer.chat_template = _fixed_template()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tito = get_tito_tokenizer(tokenizer, "qwen35")

    class State:
        pass

    state = State()
    state.tokenizer = tokenizer
    state.processor = processor
    return tito, state


def _pil(w=64, h=64):
    from PIL import Image

    return Image.new("RGB", (w, h), (0, 128, 255))


def make_segment(tito):
    messages = [
        {"role": "system", "content": "You complete tasks."},
        {"role": "user", "content": "click the button"},
    ]
    ids = tito.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    sample = Sample(prompt=list(messages), metadata={})
    sample.tokens = list(ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    return _Segment(sample=sample, messages=messages, prompt_len=len(ids))


def simulate_generation(tito, state, segment, text):
    ids = state.tokenizer.encode(text, add_special_tokens=False)
    im_end = state.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = ids + [im_end]
    s = segment.sample
    s.tokens = s.tokens + ids
    s.response_length += len(ids)
    s.loss_mask = (s.loss_mask or []) + [1] * len(ids)
    s.rollout_log_probs = (s.rollout_log_probs or []) + [-0.1] * len(ids)
    segment.messages.append({"role": "assistant", "content": text})
    return len(ids)


def test_multimodal_observation_appends_aligned(stack):
    tito, state = stack
    segment = make_segment(tito)
    gen = simulate_generation(tito, state, segment, "clicking now <tool_call>...</tool_call>")

    image = _pil()
    message = {
        "role": "tool",
        "tool_call_id": "call_000001",
        "name": "browser__computer",
        "content": [{"type": "text", "text": "Tool result:\n[screenshot]\n[Turn 1/32]"}, {"type": "image"}],
    }
    _append_multimodal(segment, tito, state, message, [image])

    s = segment.sample
    assert len(s.loss_mask) == s.response_length == len(s.tokens) - segment.prompt_len
    assert len(s.rollout_log_probs) == s.response_length
    # sampled tokens keep mask 1 (plus the boundary newline at mask 0)
    assert sum(s.loss_mask) == gen
    # the processor expanded real image tokens into the sequence
    text = state.tokenizer.decode(s.tokens[segment.prompt_len:])
    assert "<|image_pad|>" in text or "image_pad" in text
    assert "Tool result:" in text
    # processor tensors captured for the train-input merge
    assert segment.mm_chunks and "pixel_values" in segment.mm_chunks[0]
    # engine-side images accumulated
    assert len(s.multimodal_inputs["images"]) == 1


def test_text_observation_append_requires_fixed_template():
    """The TITO suffix render ([dummy system, dummy assistant, tool result])
    has no user turn; Qwen3.5's stock template raises on it ("No user query
    found in messages"), so every text-observation append aborts the episode.
    The fixed template drops that raise — this is why run_fleet.py passes
    --chat-template-path (observed live: qwen-eval-gate 2026-08-23, all GUI
    episodes written off as ABORTED). Tokenizer-only on purpose: the stack
    fixture needs torch for the processor, this property doesn't."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.chat_template = _fixed_template()
    tito = get_tito_tokenizer(tokenizer, "qwen35")
    old = [
        {"role": "system", "content": "You complete tasks."},
        {"role": "user", "content": "click the button"},
        {"role": "assistant", "content": "clicking"},
    ]
    new = old + [{"role": "tool", "content": "clicked"}]

    stock = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    stock_tito = get_tito_tokenizer(stock, "qwen35")
    with pytest.raises(ValueError, match="No user query found"):
        stock_tito.merge_tokens(old, new, stock_tito.apply_chat_template(old, add_generation_prompt=True, tokenize=True))

    prefix = tito.apply_chat_template(old, add_generation_prompt=True, tokenize=True)
    merged = tito.merge_tokens(old, new, prefix)
    tail = tokenizer.decode(merged[len(prefix) - tito.max_trim_tokens :])
    assert "<tool_response>" in tail and "clicked" in tail


def test_boundary_fix_inserts_qwen_newline(stack):
    tito, state = stack
    segment = make_segment(tito)
    simulate_generation(tito, state, segment, "short turn")
    before = len(segment.sample.tokens)
    _boundary_fix(tito, segment.sample)
    assert len(segment.sample.tokens) == before + 1
    assert segment.sample.tokens[-1] == tito._newline_id
    assert segment.sample.loss_mask[-1] == 0
