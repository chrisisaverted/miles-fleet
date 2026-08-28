from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tests.ci.ci_register import register_cpu_ci

from miles_plugins.models.glm5_next.image_processing import Glm5NextImageProcessor, smart_resize
from miles_plugins.models.glm5_next.vision import Glm5NextVisionModel

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])


def test_glm5_next_image_processor_uses_dynamic_padded_grid():
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)

    output = Glm5NextImageProcessor()(images=Image.fromarray(pixels), return_tensors="pt")

    assert output.image_grid_thw.tolist() == [[1, 8, 10]]
    assert output.pixel_values.shape == (80, 1176)
    assert output.pixel_values.isfinite().all()


def test_glm5_next_smart_resize_rejects_impossible_token_budget():
    with pytest.raises(ValueError, match="max_image_tokens=0 is too small"):
        smart_resize(
            num_frames=2,
            height=64,
            width=64,
            temporal_factor=2,
            factor=28,
            max_image_tokens=0,
        )


def test_glm5_next_visual_tower_emits_one_embedding_per_merged_patch():
    config = SimpleNamespace(
        attention_bias=True,
        attention_dropout=0.0,
        depth=1,
        hidden_act="silu",
        hidden_size=8,
        in_channels=3,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        patch_size=2,
        projection_intermediate_size=16,
        rms_norm_eps=1e-5,
        spatial_merge_size=2,
        swiglu_limit=10.0,
        temporal_patch_size=2,
        _attn_implementation="sdpa",
    )
    visual = Glm5NextVisionModel(config)
    pixel_values = torch.randn(4, config.in_channels * config.temporal_patch_size * config.patch_size**2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    output = visual(pixel_values, image_grid_thw)

    assert output.shape == (1, config.out_hidden_size)
    assert output.isfinite().all()
