"""Frozen online DINOv2-L feature extraction for RAIN multi-scale training."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.dinov2 import load_dinov2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDINOv2LargeMultiScale(nn.Module):
    """Extract normalized patch tokens from DINOv2-L blocks 11/17/23.

    Inputs are RGB float tensors in [0, 1]. Outputs are three patch-only
    tensors shaped (B, 256, 1024).
    """

    scale_layers = (11, 17, 23)

    def __init__(self, input_size: int = 224) -> None:
        super().__init__()
        if input_size != 224:
            raise ValueError(
                f"RAIN multi-scale features require 224px input, got {input_size}"
            )
        self.input_size = int(input_size)
        self.backbone = load_dinov2()
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        images = (images - self.mean) / self.std
        outputs = self.backbone.get_intermediate_layers(
            images,
            n=list(self.scale_layers),
            reshape=False,
            norm=True,
        )
        return list(outputs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self
