"""Pins the glm4v attention-mask off-by-one and our Dockerfile patch for it.

Upstream context (sglang PR #36833, unmerged): with SGLANG_MM_AVOID_RETOKENIZE
(the default), glm4v.py keeps the client's pretokenized input_ids verbatim but
computes ret.attention_mask over decode(input_ids) re-tokenized. Any sampled
adjacent token pair that is not the canonical BPE encoding of its string (e.g.
two single-"\\n" tokens, which re-tokenize as one "\\n\\n" token) makes the
retokenized side one token shorter. get_rope_index_glm4v then runs
`ids[mask == 1]` (mrope_rope_index.py:522 on branch sglang-miles-glm53next)
and raises IndexError. Observed on miles-glm53-96k-probe 2026-08-29 at
8323/8324 and 36643/36644 tokens.

Our fix (applied as a Dockerfile RUN over the overlaid glm4v.py): pass
attention_mask=None at glm4v.py:613. The callee then builds
torch.ones_like(input_ids). This is behavior-preserving for upstream's own
path: base_processor.py calls processor(text=[one_text], padding=True), a
batch of one, so ret.attention_mask is always all ones; whenever the lengths
agree, an all-ones mask over the ids is bitwise the same thing.

The _collapse/_expand helpers below are verbatim copies from sglang @
21572b4470442fe49ea4ff36e64b1118b465c3b0 (glm4v.py / base_processor.py); they
are the two sides of the seam our recording.py output crosses.
"""

import re
from pathlib import Path

import pytest
import torch

MODEL = "zai-org/GLM-5.3-Flash-BF16"
IMG = 154854  # <|image|> == hf_config.image_token_id

LAUNCH = Path(__file__).resolve().parent.parent / "launch"

# ---------------------------------------------------------------------------
# CPU: the Dockerfile must keep carrying the patch until upstream merges it.
# ---------------------------------------------------------------------------


def test_dockerfile_carries_glm4v_mask_patch():
    dockerfile = (LAUNCH / "Dockerfile").read_text()
    # the anchor the in-image patch asserts on, and the replacement it writes
    assert 'attention_mask=getattr(ret, "attention_mask", None),' in dockerfile
    assert "attention_mask=None," in dockerfile
    assert "glm4v.py" in dockerfile
    # the patch must run as its own heredoc RUN, not inside a && chain
    # (BuildKit parses an embedded heredoc's end as a new instruction)
    assert re.search(r"^RUN python3 - <<'PATCH'$", dockerfile, re.M)


# ---------------------------------------------------------------------------
# network: the drift itself, with the real GLM-5.3 tokenizer.
# ---------------------------------------------------------------------------

pytestmark_network = pytest.mark.network


def _collapse_glm5_next_image_tokens(input_ids, image_token_id):
    # verbatim from glm4v.py @ 21572b44 (PR #36833)
    return [
        current
        for index, current in enumerate(input_ids)
        if current != image_token_id
        or index == 0
        or input_ids[index - 1] != image_token_id
    ]


def _expand_input_ids(original_ids, counts, placeholder_token_id):
    # verbatim from base_processor.py @ 21572b44 (AVOID_RETOKENIZE path)
    num_placeholders = sum(1 for t in original_ids if t == placeholder_token_id)
    assert num_placeholders == len(counts)
    rebuilt = []
    next_image_idx = 0
    for token_id in original_ids:
        if token_id == placeholder_token_id:
            rebuilt.extend([placeholder_token_id] * counts[next_image_idx])
            next_image_idx += 1
        else:
            rebuilt.append(token_id)
    return rebuilt


def _expand_image_text(text, counts):
    """What the HF processor does to the text before tokenizing."""
    parts = text.split("<|image|>")
    assert len(parts) - 1 == len(counts)
    out = [parts[0]]
    for n, part in zip(counts, parts[1:]):
        out.append("<|image|>" * n)
        out.append(part)
    return "".join(out)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def _episode(tok, noncanonical):
    """One screenshot turn; optionally inject a sampled two-token "\\n"+"\\n"."""
    msgs = [
        {"role": "system", "content": "You are a browser agent."},
        {"role": "user", "content": "Book a flight."},
    ]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    tokens = tok.encode(prompt, add_special_tokens=False)
    a_ids = tok.encode("<think>click</think>done", add_special_tokens=False)
    if noncanonical:
        nl = tok.encode("\n", add_special_tokens=False)
        assert len(nl) == 1
        a_ids = a_ids + nl + nl + tok.encode("done", add_special_tokens=False)
    tokens = tokens + a_ids
    obs = {
        "role": "tool",
        "content": [{"type": "image"}, {"type": "text", "text": "Screenshot."}],
    }
    full = tok.apply_chat_template(
        msgs + [obs], add_generation_prompt=True, tokenize=False
    )
    # keep only the observation suffix, as recording.py does
    base = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    suffix = full[len(base):]
    span = 250
    tokens = tokens + tok.encode(
        _expand_image_text(suffix, [span]), add_special_tokens=False
    )
    return tokens, [span]


def _engine_side(tok, tokens, counts):
    """glm4v.py's two sequences: verbatim-rebuilt ids vs retokenized text."""
    collapsed = _collapse_glm5_next_image_tokens(tokens, IMG)
    decoded = tok.decode(collapsed)
    retok = tok.encode(_expand_image_text(decoded, counts), add_special_tokens=False)
    rebuilt = _expand_input_ids(collapsed, counts, IMG)
    return rebuilt, retok


@pytest.mark.network
def test_canonical_stream_has_no_drift(tok):
    tokens, counts = _episode(tok, noncanonical=False)
    rebuilt, retok = _engine_side(tok, tokens, counts)
    assert len(rebuilt) == len(retok)
    # when lengths agree, an all-ones mask (attention_mask=None) selects
    # every id: the fix cannot change upstream's behavior on this path
    ids = torch.tensor(rebuilt)
    assert torch.equal(ids[torch.ones_like(ids) == 1], ids)


@pytest.mark.network
def test_noncanonical_pair_drifts_and_crashes_with_retok_mask(tok):
    tokens, counts = _episode(tok, noncanonical=True)
    rebuilt, retok = _engine_side(tok, tokens, counts)
    assert len(rebuilt) - len(retok) == 1  # the off-by-one
    ids = torch.tensor(rebuilt)
    retok_mask = torch.ones(len(retok), dtype=torch.long)
    # unpatched glm4v: mask over retokenized text, ids verbatim ->
    # exactly the probe's IndexError at mrope_rope_index.py:522
    with pytest.raises(IndexError):
        _ = ids[retok_mask == 1]
    # patched (attention_mask=None -> ones_like(ids)): identity selection
    assert torch.equal(ids[torch.ones_like(ids) == 1], ids)
