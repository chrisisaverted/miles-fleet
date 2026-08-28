from argparse import Namespace

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import _should_sync_frozen_mm_tower


@pytest.mark.parametrize(
    ("provider", "offload_rollout", "levels", "expected"),
    [
        ("miles_plugins.models.inkling.model.inkling_mm_model_provider", False, [], True),
        ("miles_plugins.models.glm5_next.vision.glm5_next_vlm_model_provider", True, ["weight"], True),
        ("miles_plugins.models.glm5_next.vision.glm5_next_vlm_model_provider", True, ["kv_cache"], False),
        ("miles_plugins.models.glm5_next.vision.glm5_next_vlm_model_provider", False, ["weight"], False),
        ("miles_plugins.models.glm5_next.model_provider", True, ["weight"], False),
    ],
)
def test_should_sync_frozen_mm_tower(provider, offload_rollout, levels, expected):
    args = Namespace(
        custom_model_provider_path=provider,
        offload_rollout=offload_rollout,
        offload_rollout_level=levels,
    )

    assert _should_sync_frozen_mm_tower(args) is expected
