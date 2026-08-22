"""Content projection: SDK blocks -> OpenAI message content."""

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from examples.fleet.content import tool_result_to_content, truncate_text


@dataclass
class FakeText:
    text: str


@dataclass
class FakeJson:
    value: Dict[str, Any]


@dataclass
class FakeRef:
    digest: str = "sha256:abc"
    media_type: str = "image/png"
    size: int = 3


@dataclass
class FakeBlob:
    ref: FakeRef = field(default_factory=FakeRef)


@dataclass
class FakeResult:
    content: tuple
    status: str = "ok"
    error_code: Optional[str] = None


def _png(w, h):
    from PIL import Image

    img = Image.new("RGB", (w, h), (200, 30, 30))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_text_and_json_flatten():
    text, images = tool_result_to_content(FakeResult(content=(FakeText("hi"), FakeJson({"a": 1}))))
    assert text.splitlines()[0] == "hi"
    assert json.loads(text.splitlines()[1]) == {"a": 1}
    assert images == []


def test_image_blob_to_data_url_with_marker():
    png = _png(64, 32)
    text, images = tool_result_to_content(
        FakeResult(content=(FakeText("took screenshot"), FakeBlob())), read_blob=lambda r: png
    )
    assert "[screenshot]" in text
    assert len(images) == 1
    assert images[0].startswith("data:image/png;base64,")
    assert base64.b64decode(images[0].split(",", 1)[1]) == png


def test_downscale_respects_max_dim():
    from PIL import Image

    _, images = tool_result_to_content(
        FakeResult(content=(FakeBlob(),)), read_blob=lambda r: _png(2000, 1000), screenshot_max_dim=512
    )
    img = Image.open(io.BytesIO(base64.b64decode(images[0].split(",", 1)[1])))
    assert max(img.size) <= 512


def test_no_downscale_by_default():
    png = _png(2000, 1000)
    _, images = tool_result_to_content(FakeResult(content=(FakeBlob(),)), read_blob=lambda r: png)
    assert base64.b64decode(images[0].split(",", 1)[1]) == png  # untouched


def test_non_image_blob_placeholder():
    blob = FakeBlob(ref=FakeRef(media_type="application/zip", size=99))
    text, images = tool_result_to_content(FakeResult(content=(blob,)), read_blob=lambda r: b"x")
    assert images == []
    assert "sha256:abc" in text and "99" in text


def test_unreadable_blob_degrades():
    def boom(ref):
        raise RuntimeError("blob store gone")

    text, images = tool_result_to_content(FakeResult(content=(FakeBlob(),)), read_blob=boom)
    assert images == []
    assert "unreadable" in text


def test_no_read_blob_means_placeholder():
    text, images = tool_result_to_content(FakeResult(content=(FakeBlob(),)))
    assert images == [] and "sha256:abc" in text


def test_truncation_keeps_head_and_tail():
    text = "START " + ("x" * 50000) + " THE_ERROR_LINE"
    out = truncate_text(text, 1000)
    assert len(out) < 1200
    assert out.startswith("START")
    assert out.endswith("THE_ERROR_LINE")
    assert "TRUNCATED" in out


def test_truncation_noop_under_cap():
    assert truncate_text("short", 1000) == "short"


def test_empty_content():
    text, images = tool_result_to_content(FakeResult(content=()))
    assert text == "" and images == []
