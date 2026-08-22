"""Test bootstrap: repo root on sys.path, sglang stubbed.

The CPU suite runs without GPUs or sglang. miles.utils.chat_template_utils
(pulled in transitively by miles.rollout.base_types) imports one pydantic
model from sglang at module level; stub exactly that surface so the rest of
miles imports cleanly on a laptop.
"""

import sys
import types
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if "sglang" not in sys.modules:
    from pydantic import BaseModel

    class _Function(BaseModel):
        name: str
        description: str | None = None
        parameters: dict | None = None

    class _Tool(BaseModel):
        type: str = "function"
        function: _Function

    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol.Tool = _Tool

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    encoding_dsv4 = types.ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")

    sys.modules["sglang.srt.entrypoints.openai.protocol"] = protocol
    sys.modules["sglang.srt.entrypoints.openai.encoding_dsv4"] = encoding_dsv4
    sys.modules["sglang.srt.entrypoints.openai"].protocol = protocol
    sys.modules["sglang.srt.entrypoints.openai"].encoding_dsv4 = encoding_dsv4
