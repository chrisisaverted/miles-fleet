"""Fleet rollout recording: the token-level half of the rollout.

Implements the TurnRunner interface the agent loop (examples.fleet.agent)
drives. This side owns everything miles trains on: /generate payloads against
SGLang, token-in-token-out assembly, loss masks, multimodal tensors, and the
Sample objects returned to the trainer. If miles's session-server recording
ever carries screenshots, this module is what it replaces.

The assembly is token-in-token-out: sampled token ids come back from
/generate and are appended verbatim (loss mask 1); observation messages are
tokenized incrementally by miles's TITO tokenizer (loss mask 0), whose
per-family subclasses own the boundary quirks (e.g. Qwen's missing newline
at the assistant/observation junction) and the keep-thinking template
kwargs. History is never re-rendered.

One episode can hold several training sequences (_Segment): a reset boundary
ends the current conversation and opens a fresh one, so finalize() returns
several Samples (miles groups them by rollout_id) with the episode's terminal
reward broadcast to every one.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer
from miles.utils.http_utils import post
from miles.utils.types import Sample

from examples.fleet.agent import Turn

logger = logging.getLogger(__name__)


_TITO_CACHE: Dict[str, Any] = {}


def _tito_for(state, args):
    tito = _TITO_CACHE.get(args.fleet_tito_model)
    if tito is None:
        tito = get_tito_tokenizer(state.tokenizer, args.fleet_tito_model)
        _TITO_CACHE[args.fleet_tito_model] = tito
    return tito


@dataclass
class _Segment:
    """One training sequence: a Sample plus the message list that produced
    its token prefix (the TITO tokenizer diffs against these messages)."""

    sample: Sample
    messages: List[Dict[str, Any]]
    prompt_len: int
    # vision: per-turn processor outputs (pixel_values etc.), concatenated
    # into sample.multimodal_train_inputs at finalize
    mm_chunks: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- segments


def _start_segment(tito, base_sample: Sample, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> _Segment:
    """A fresh training sequence: used for the first step and after every
    reset boundary."""
    sample = deepcopy(base_sample)
    sample.metadata = dict(base_sample.metadata or {})
    sample.prompt = list(messages)
    prompt_ids = tito.apply_chat_template(messages, add_generation_prompt=True, tools=tools, tokenize=True)
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.status = Sample.Status.PENDING
    return _Segment(sample=sample, messages=list(messages), prompt_len=len(prompt_ids))


def _record_assistant(segment: _Segment, text: str, tool_call: Optional[Dict[str, Any]], turn: int) -> None:
    """Store the sampled turn in the message list so the TITO tokenizer's
    dummy prefix renders the right scaffold (tool_calls make a following
    role-"tool" message legal)."""
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_call is not None:
        message["tool_calls"] = [
            {
                "id": f"call_{turn:06d}",
                "type": "function",
                "function": {"name": tool_call["name"], "arguments": tool_call.get("arguments") or {}},
            }
        ]
    segment.messages.append(message)


def _append_messages(segment: _Segment, tito, new_messages: List[Dict[str, Any]]) -> None:
    """Append observation messages to the segment's token sequence with loss
    mask 0, via the TITO tokenizer's family-specific merge (which may trim a
    trailing ambiguous boundary token off the sampled prefix)."""
    sample = segment.sample
    before = list(sample.tokens)
    merged = tito.merge_tokens(segment.messages, segment.messages + new_messages, before)
    shared = 0
    while shared < len(before) and shared < len(merged) and before[shared] == merged[shared]:
        shared += 1
    removed = len(before) - shared
    added = len(merged) - shared
    if removed:
        sample.loss_mask = sample.loss_mask[: len(sample.loss_mask) - removed]
        sample.rollout_log_probs = sample.rollout_log_probs[: len(sample.rollout_log_probs) - removed]
    sample.loss_mask = sample.loss_mask + [0] * added
    sample.rollout_log_probs = sample.rollout_log_probs + [0.0] * added
    sample.tokens = merged
    sample.response_length = len(merged) - segment.prompt_len
    sample.response = tito.tokenizer.decode(merged[segment.prompt_len :])
    segment.messages.extend(new_messages)


# --------------------------------------------------------------------- vision


def _data_urls_to_pil(urls: List[str]) -> List[Any]:
    import base64
    import io

    from PIL import Image

    images = []
    for url in urls:
        payload = url.split(",", 1)[1]
        images.append(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"))
    return images


def _boundary_fix(tito, sample: Sample) -> None:
    """Apply the TITO family's junction rule to the sampled tail before an
    out-of-band (processor-based) append. Mirrors merge_tokens: Qwen inserts
    the newline its models stop before; families whose stop token is
    ambiguous at the junction trim it instead."""
    tokens = sample.tokens
    im_end = getattr(tito, "_im_end_id", None)
    if im_end is not None and tokens and tokens[-1] == im_end:
        sample.tokens = tokens + [tito._newline_id]
        sample.response_length += 1
        sample.loss_mask = (sample.loss_mask or []) + [0]
        sample.rollout_log_probs = (sample.rollout_log_probs or []) + [0.0]
        return
    ambiguous = getattr(tito, "_ambiguous_boundary_ids", None)
    if ambiguous and tokens and tokens[-1] in ambiguous:
        sample.tokens = tokens[:-1]
        sample.response_length -= 1
        sample.loss_mask = sample.loss_mask[:-1]
        sample.rollout_log_probs = sample.rollout_log_probs[:-1]


def _append_multimodal(segment: _Segment, tito, state, message: Dict[str, Any], images: List[Any]) -> None:
    """Append an observation that carries images: render the message under
    the constant dummy prefix, expand image tokens with the PROCESSOR, trim
    the prefix by its tokenizer length (image expansion only happens after
    it), and append with loss mask 0. Accumulates engine images on the sample
    and the processor tensors on the segment for the finalize merge."""
    sample = segment.sample
    _boundary_fix(tito, sample)

    base = [_VISION_DUMMY_USER, {"role": "assistant", "content": ""}]
    dummy_text = tito.apply_chat_template(base, add_generation_prompt=False, tokenize=False)
    full_text = tito.apply_chat_template(base + [message], add_generation_prompt=True, tokenize=False)
    if not full_text.startswith(dummy_text):
        raise ValueError("chat template is not append-only over the vision dummy prefix")
    trim = len(state.tokenizer.encode(dummy_text, add_special_tokens=False))

    processor_output = state.processor(text=full_text, images=images)
    ids = list(processor_output["input_ids"][0])[trim:]
    chunk = {
        k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")
    }
    if chunk:
        segment.mm_chunks.append(chunk)

    sample.response += state.tokenizer.decode(ids)
    sample.response_length += len(ids)
    sample.tokens = sample.tokens + ids
    sample.loss_mask = (sample.loss_mask or []) + [0] * len(ids)
    sample.rollout_log_probs = (sample.rollout_log_probs or []) + [0.0] * len(ids)

    mm = sample.multimodal_inputs or {}
    mm["images"] = (mm.get("images") or []) + images
    sample.multimodal_inputs = mm
    segment.messages.append(message)


def _merge_mm_chunks(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Concatenate per-turn processor tensors (geo3k's merge)."""
    import torch

    values_by_key: Dict[str, List[Any]] = {}
    for chunk in chunks:
        for key, val in (chunk or {}).items():
            if val is not None:
                values_by_key.setdefault(key, []).append(val)
    merged = {
        key: torch.cat(vals, dim=0)
        for key, vals in values_by_key.items()
        if all(isinstance(v, torch.Tensor) for v in vals)
    }
    return merged or None


_VISION_DUMMY_USER = {"role": "user", "content": "dummy"}


# ------------------------------------------------------------------ recorder


class Recorder:
    """TurnRunner implementation over SGLang /generate and miles Samples."""

    def __init__(self, args, state, base_sample: Sample, sampling_params: Dict[str, Any]):
        self.args = args
        self.state = state
        self.base_sample = base_sample
        self.sampling_params = sampling_params
        self.tito = _tito_for(state, args)
        self.url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        self.segments: List[_Segment] = []

    @property
    def aborted(self) -> bool:
        return self.state.aborted

    # ------------------------------------------------------------- protocol

    def begin_segment(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> None:
        if self.segments and self.segments[-1].sample.status == Sample.Status.PENDING:
            self.segments[-1].sample.status = Sample.Status.COMPLETED
        self.segments.append(_start_segment(self.tito, self.base_sample, messages, tools))

    async def sample(self) -> Turn:
        if self.state.aborted:
            return Turn(text=None, finish="aborted")
        segment = self.segments[-1]
        params = dict(self.sampling_params)
        per_turn = self.args.fleet_max_tokens_per_turn
        params["max_new_tokens"] = min(per_turn, params.get("max_new_tokens") or per_turn)
        payload, halt_status = compute_request_payload(
            self.args, segment.sample.tokens, params, multimodal_inputs=segment.sample.multimodal_inputs
        )
        if payload is None:
            segment.sample.status = halt_status
            return Turn(text=None, finish="context_full")

        output = await post(self.url, payload, headers=compute_routing_headers(self.args, segment.sample))
        await update_sample_from_response(
            self.args, segment.sample, payload=payload, output=output, update_loss_mask=True
        )
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort":
            return Turn(text=None, finish="aborted")
        return Turn(text=output["text"], finish="length" if finish == "length" else "ok")

    def append_assistant(self, text: str, tool_call: Optional[Dict[str, Any]], turn: int) -> Dict[str, Any]:
        segment = self.segments[-1]
        _record_assistant(segment, text, tool_call, turn)
        return segment.messages[-1]

    def append_observation(self, message: Dict[str, Any], image_urls: List[str]) -> Dict[str, Any]:
        segment = self.segments[-1]
        if image_urls:
            pils = _data_urls_to_pil(image_urls)
            message["content"] = [{"type": "text", "text": message["content"]}] + [
                {"type": "image"} for _ in pils
            ]
            _append_multimodal(segment, self.tito, self.state, message, pils)
        else:
            _append_messages(segment, self.tito, [message])
        return message

    def note_messages(self, messages: List[Dict[str, Any]]) -> None:
        self.segments[-1].sample.metadata["messages"] = list(messages)

    # ------------------------------------------------------------- finalize

    def write_off(self, error: Optional[str] = None) -> List[Sample]:
        """Mark everything ABORTED (nothing to train on); with an error, tag
        it so check_no_aborted's rejects are diagnosable. Falls back to the
        base sample when the episode died before its first segment."""
        samples = [s.sample for s in self.segments] or [self.base_sample]
        for sample in samples:
            sample.status = Sample.Status.ABORTED
            if error is not None:
                sample.metadata = dict(sample.metadata or {})
                sample.metadata["episode_error"] = error[:300]
        return samples

    def finalize(self, reward: float, episode_meta: Dict[str, Any], env_time: float) -> List[Sample]:
        """Broadcast the episode's terminal reward to every segment, merge
        the per-turn multimodal tensors, and stamp the episode metadata."""
        for index, segment in enumerate(self.segments):
            sample = segment.sample
            sample.reward = reward
            if sample.status == Sample.Status.PENDING:
                sample.status = Sample.Status.COMPLETED
            if segment.mm_chunks:
                sample.multimodal_train_inputs = _merge_mm_chunks(segment.mm_chunks)
            sample.metadata.update(episode_meta)
            sample.metadata["segment_index"] = index
        # env wall-clock is an episode quantity; book it once, not per segment.
        self.segments[0].sample.non_generation_time = env_time
        return [s.sample for s in self.segments]
