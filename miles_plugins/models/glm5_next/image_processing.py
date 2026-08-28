"""GLM-5.3 image preprocessing compatible with the pinned Transformers 5.12.1.

The implementation follows ``Glm5NextImageProcessor`` from Transformers.  It
is kept in the model plugin because the GLM-5.3 Miles/SGLang image deliberately
stays on Transformers 5.12.1, which predates that processor.
"""

import math

import torch
from torchvision.transforms.v2 import functional as tvF
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ImageInput, PILImageResampling, SizeDict
from transformers.processing_utils import ImagesKwargs, Unpack
from transformers.utils import TensorType


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    min_image_tokens: int
    max_image_tokens: int


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
) -> tuple[int, int]:
    """Return an aligned canvas within the configured visual-token budget."""
    pixels_per_token = temporal_factor * factor**2
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token

    def _align(value: int) -> int:
        return math.ceil(value / factor) * factor

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = _align(height)
    aligned_width = _align(width)
    aligned_pixels = aligned_frames * aligned_height * aligned_width

    if aligned_pixels < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = _align(max(1, math.ceil(height * scale)))
        aligned_width = _align(max(1, math.ceil(width * scale)))
        aligned_pixels = aligned_frames * aligned_height * aligned_width

    if aligned_pixels <= max_pixels:
        return aligned_height, aligned_width

    minimum_pixels = aligned_frames * factor**2
    if max_pixels < minimum_pixels:
        raise ValueError(
            f"max_image_tokens={max_image_tokens} is too small: at least "
            f"{minimum_pixels // pixels_per_token} token(s) are required"
        )

    low, high = 1, height
    best_height = best_width = factor
    while low <= high:
        content_height = (low + high) // 2
        content_width = max(1, math.floor(width * content_height / height))
        candidate_height = _align(content_height)
        candidate_width = _align(content_width)
        if aligned_frames * candidate_height * candidate_width <= max_pixels:
            best_height, best_width = candidate_height, candidate_width
            low = content_height + 1
        else:
            high = content_height - 1
    return best_height, best_width


class Glm5NextImageProcessor(TorchvisionBackend):
    """Image-only GLM-5.3 processor matching the upstream dynamic resize."""

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]
    patch_expand_factor = 1
    min_image_tokens = 16
    max_image_tokens = 8000

    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        images: torch.Tensor,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Resize without upscaling normal images, then pad to the GLM canvas."""
        height, width = images.shape[-2:]
        target_height, target_width = smart_resize(
            height=height,
            width=width,
            num_frames=temporal_factor,
            factor=factor,
            temporal_factor=temporal_factor,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
        )

        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_factor * height * width >= pixels_per_token * min_image_tokens:
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        if (content_height, content_width) != (height, width):
            images = super().resize(
                images,
                SizeDict(height=content_height, width=content_width),
                resample=resample,
            )
        return tvF.pad(images, [0, 0, target_width - content_width, target_height - content_height], fill=0)

    @staticmethod
    def patchify(
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        """Flatten images in the patch order consumed by the visual tower."""
        batch_size, channel, resized_height, resized_width = images.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = images.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        flattened = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )
        return flattened, grid_h, grid_w

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        del size
        grouped, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_groups = {}
        for shape, stacked_images in grouped.items():
            if do_resize:
                stacked_images = self.resize(
                    images=stacked_images,
                    resample=resample,
                    factor=patch_size * merge_size * patch_expand_factor,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )
            resized_groups[shape] = stacked_images
        resized_images = reorder_images(resized_groups, grouped_index)

        grouped, grouped_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_groups = {}
        grid_groups = {}
        for shape, stacked_images in grouped.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches, grid_h, grid_w = self.patchify(
                stacked_images,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_groups[shape] = patches
            grid_groups[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

        processed_images = reorder_images(processed_groups, grouped_index)
        processed_grids = reorder_images(grid_groups, grouped_index)
        pixel_values = processed_images[0] if len(processed_images) == 1 else torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: dict | None = None,
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=patch_size * merge_size,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)
