"""Pure projections from fleet_runtime tool results to OpenAI message content.

Boundary note: everything model-specific lives in the model's own chat
template/processor on the trainer side. These functions only translate the
SDK's native shapes (TextBlock / JsonBlock / BlobRefBlock) into model-neutral
OpenAI content; they never see or care which model is training.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, List, Optional, Tuple

# Per-tool-result char cap in observations. Big dbt/SQL outputs at 16K chars
# (~4K tokens each) fill a 32K context in ~25 turns and end episodes with
# stop=length long before the turn budget (measured on Fi runs).
MAX_TOOL_OUTPUT_CHARS = 16000


def truncate_text(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Head+tail truncation: build tools (dbt, pytest, compilers) print the
    error at the END of a long log, so head-only truncation hides exactly what
    the agent needs to iterate on."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    elided = len(text) - max_chars
    return text[:head] + f"\n\n[TRUNCATED — {elided} chars elided.]\n\n" + text[-tail:]


def downscale_image(data: bytes, media_type: str, max_dim: int) -> Tuple[bytes, str]:
    """Shrink an image so its longest side is <= max_dim; re-encode as PNG.

    Optional payload optimization, not correctness: the model's own processor
    resizes to its pixel budget anyway. This cuts the data-URL that travels
    through the conversation, the /render HTTP body, and Ray's object store
    (a raw 1080p PNG is a 2-4MB data URL, every turn it stays in context).
    """
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_dim:
        return data, media_type
    img.thumbnail((max_dim, max_dim))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue(), "image/png"


def tool_result_to_content(
    result: Any,
    read_blob: Optional[Callable[[Any], bytes]] = None,
    screenshot_max_dim: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Flatten a fleet_runtime ToolResult into (text, image_data_urls).

    ToolResult.content is a tuple of ContentBlocks: TextBlock(text=...),
    JsonBlock(value=...), BlobRefBlock(ref=BlobRef). Image blobs (media_type
    image/*) are fetched via read_blob (channel.read_tool_result_blob) and
    returned as data URLs, with a [screenshot] marker holding their slot in
    the text. Non-image or unreadable blobs degrade to a digest placeholder,
    mirroring the SDK's own grading projection — degrading beats raising
    because the episode already paid for the turn.
    """
    parts: List[str] = []
    images: List[str] = []
    for block in getattr(result, "content", ()) or ():
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        value = getattr(block, "value", None)
        if value is not None:
            try:
                # JsonBlock values are mappingproxy-wrapped; unwrap so dumps
                # renders JSON instead of a stringified repr.
                if hasattr(value, "items"):
                    value = dict(value)
                parts.append(json.dumps(value, default=str))
            except Exception:
                parts.append(str(value))
            continue
        ref = getattr(block, "ref", None)
        if ref is not None:
            media = str(getattr(ref, "media_type", "") or "")
            if media.startswith("image/") and read_blob is not None:
                try:
                    data = read_blob(ref)
                    if screenshot_max_dim:
                        data, media = downscale_image(data, media, screenshot_max_dim)
                    images.append(f"data:{media};base64," + base64.b64encode(data).decode("ascii"))
                    parts.append("[screenshot]")
                except Exception as e:
                    parts.append(f"<blob {getattr(ref, 'digest', '?')} unreadable: {e}>")
                continue
            parts.append(f"<blob {getattr(ref, 'digest', '?')} ({getattr(ref, 'size', '?')} bytes)>")
            continue
        parts.append(str(block))
    return "\n".join(p for p in parts if p), images


def to_plain(obj: Any) -> Any:
    """Deep-convert Contract mappings (mappingproxy) into plain JSON-able types.

    Tool schemas from tool_surface() are immutable nested mappings; json.dumps
    refuses mappingproxy, and a shallow dict() only fixes the top level.
    """
    if hasattr(obj, "items"):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
