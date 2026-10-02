"""The DINOv2 source revision used by feature extraction and online inference."""

import torch


DINO_REVISION = "7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"
DINO_REPOSITORY = f"facebookresearch/dinov2:{DINO_REVISION}"


def load_dinov2(model_name: str = "dinov2_vitl14_reg"):
    """Load the recorded public backbone using the configured Torch Hub cache."""
    return torch.hub.load(
        DINO_REPOSITORY,
        model_name,
        verbose=False,
        trust_repo=True,
        skip_validation=True,
    )
