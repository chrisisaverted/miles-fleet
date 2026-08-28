"""Training processor loader for GLM-5.3 on Transformers 5.12.1."""

import json
from pathlib import Path


def is_glm5_next_checkpoint(name_or_path: str) -> bool:
    """Return whether a local checkpoint declares ``model_type=glm5_next``."""
    config_path = Path(name_or_path) / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file).get("model_type") == "glm5_next"


def load_glm5_next_processor(name_or_path: str, **tokenizer_kwargs):
    """Build the official image processor without upgrading Transformers."""
    # Keep the vision dependencies lazy: ``load_processor`` calls the lightweight
    # checkpoint predicate for text-only model families too.
    from transformers.models.glm46v.processing_glm46v import Glm46VProcessor
    from transformers.models.glm46v.video_processing_glm46v import Glm46VVideoProcessor

    from miles.utils.processing_utils import load_tokenizer
    from miles_plugins.models.glm5_next.image_processing import Glm5NextImageProcessor

    checkpoint = Path(name_or_path)
    processor_config_path = checkpoint / "processor_config.json"
    if processor_config_path.is_file():
        with processor_config_path.open(encoding="utf-8") as config_file:
            processor_config = json.load(config_file)
        image_config = dict(processor_config["image_processor"])
    else:
        image_config = {}
    image_config.pop("image_processor_type", None)
    image_processor = Glm5NextImageProcessor(**image_config)
    tokenizer_kwargs.setdefault("trust_remote_code", True)
    tokenizer = load_tokenizer(name_or_path, **tokenizer_kwargs)
    return Glm46VProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        # ProcessorMixin 5.12 requires every declared component even for an
        # image-only call. Video support is intentionally outside this smoke.
        video_processor=Glm46VVideoProcessor(),
        chat_template=getattr(tokenizer, "chat_template", None),
    )
