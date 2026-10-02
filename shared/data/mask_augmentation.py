"""Training-time mask corruption matched to VLM + SAM2 failure modes.

The augmentation operates on full-resolution binary masks before they are
pooled into DINO patch masks.  A mask pair is ``(third_view, wrist_view)``.
Every operation preserves at least one originally visible view for each role;
the policy is never trained to act from two synthetic empty masks.
"""

from dataclasses import dataclass
import random
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np


MASK_AUGMENTATION_RECIPES = ("none", "vlm_sam2_v1")

MASK_AUGMENTATION_MODE_IDS = {
    "clean": 0,
    "wrist_dropout": 1,
    "dilate": 2,
    "erode": 3,
    "shift": 4,
    "view_dropout": 5,
}


@dataclass(frozen=True)
class MaskAugmentationOutcome:
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]]
    mode: str
    changed: bool


def validate_mask_augmentation_recipe(recipe: str) -> str:
    normalized = str(recipe or "none").strip().lower()
    if normalized not in MASK_AUGMENTATION_RECIPES:
        choices = ", ".join(MASK_AUGMENTATION_RECIPES)
        raise ValueError(
            f"Unsupported mask augmentation recipe {recipe!r}; expected one of: {choices}"
        )
    return normalized


def _binary_copy(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional, got shape={array.shape}")
    return (array > 0).astype(np.uint8, copy=True)


def _visible(mask: np.ndarray, visibility_grid: Optional[int] = None) -> bool:
    if not np.any(mask):
        return False
    if visibility_grid is None:
        return True
    pooled = cv2.resize(
        mask.astype(np.float32),
        (visibility_grid, visibility_grid),
        interpolation=cv2.INTER_AREA,
    )
    return bool(np.any(pooled > 0.5))


def _restore_if_all_missing(
    original: Tuple[np.ndarray, np.ndarray],
    augmented: Tuple[np.ndarray, np.ndarray],
    visibility_grid: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Never manufacture a two-view absence when GT had a visible mask."""
    if not (
        _visible(original[0], visibility_grid)
        or _visible(original[1], visibility_grid)
    ):
        return augmented
    if _visible(augmented[0], visibility_grid) or _visible(
        augmented[1], visibility_grid
    ):
        return augmented

    # Restore the view with more visible pixels.  This also keeps tiny objects
    # from being erased by an aggressive erosion or an out-of-frame shift.
    restore_view = int(original[1].sum() > original[0].sum())
    restored = [augmented[0], augmented[1]]
    restored[restore_view] = original[restore_view].copy()
    return restored[0], restored[1]


def _morph(mask: np.ndarray, operation: str, rng) -> np.ndarray:
    if not _visible(mask):
        return mask.copy()
    if operation == "dilate":
        radius = rng.randint(4, 12)
        op = cv2.MORPH_DILATE
    elif operation == "erode":
        radius = rng.randint(2, 8)
        op = cv2.MORPH_ERODE
    else:
        raise ValueError(f"Unknown morphology operation: {operation}")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.morphologyEx(mask, op, kernel).astype(np.uint8, copy=False)


def _shift(mask: np.ndarray, rng) -> np.ndarray:
    if not _visible(mask):
        return mask.copy()
    magnitude_x = rng.randint(4, 16)
    magnitude_y = rng.randint(4, 16)
    dx = magnitude_x if rng.random() < 0.5 else -magnitude_x
    dy = magnitude_y if rng.random() < 0.5 else -magnitude_y
    height, width = mask.shape
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.uint8, copy=False)


def _drop_view_when_redundant(
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    view_index: int,
    visibility_grid: Optional[int],
) -> bool:
    """Drop one camera stream role-by-role only when the other view is visible."""
    changed = False
    other_index = 1 - view_index
    for role, pair in list(pairs.items()):
        views = [pair[0], pair[1]]
        if _visible(views[view_index], visibility_grid) and _visible(
            views[other_index], visibility_grid
        ):
            views[view_index] = np.zeros_like(views[view_index])
            pairs[role] = views[0], views[1]
            changed = True
    return changed


def augment_multiview_masks(
    mask_pairs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    recipe: str,
    rng: Optional[random.Random] = None,
    visibility_grid: Optional[int] = None,
) -> MaskAugmentationOutcome:
    """Apply one exclusive corruption mode to all mask roles in a sample.

    ``vlm_sam2_v1`` samples 50% clean, 40% visibility-aware wrist dropout,
    and 10% SAM2-like geometry/noise.  The final 10% is split uniformly over
    dilation, erosion, shift, and redundant-view dropout.
    """
    recipe = validate_mask_augmentation_recipe(recipe)
    rng = rng or random
    original = {
        role: (_binary_copy(pair[0]), _binary_copy(pair[1]))
        for role, pair in mask_pairs.items()
    }
    augmented = {
        role: (pair[0].copy(), pair[1].copy()) for role, pair in original.items()
    }

    draw = rng.random()
    if recipe == "none" or draw < 0.50:
        return MaskAugmentationOutcome(augmented, "clean", False)

    if draw < 0.90:
        changed = _drop_view_when_redundant(
            augmented, view_index=1, visibility_grid=visibility_grid
        )
        return MaskAugmentationOutcome(augmented, "wrist_dropout", changed)

    mode = rng.choice(("dilate", "erode", "shift", "view_dropout"))
    if mode == "view_dropout":
        # Wrist loss is already heavily represented above.  This branch also
        # exposes the policy to a missing third-view mask when wrist is valid.
        eligible_views = []
        for view_index in (0, 1):
            other_index = 1 - view_index
            if any(
                _visible(pair[view_index], visibility_grid)
                and _visible(pair[other_index], visibility_grid)
                for pair in augmented.values()
            ):
                eligible_views.append(view_index)
        changed = False
        if eligible_views:
            changed = _drop_view_when_redundant(
                augmented,
                view_index=rng.choice(eligible_views),
                visibility_grid=visibility_grid,
            )
    else:
        for role, pair in list(augmented.items()):
            views = []
            for mask in pair:
                if mode in ("dilate", "erode"):
                    views.append(_morph(mask, mode, rng))
                else:
                    views.append(_shift(mask, rng))
            augmented[role] = _restore_if_all_missing(
                original[role], (views[0], views[1]), visibility_grid
            )
        changed = any(
            not np.array_equal(augmented[role][view], original[role][view])
            for role in original
            for view in (0, 1)
        )

    for role in original:
        augmented[role] = _restore_if_all_missing(
            original[role], augmented[role], visibility_grid
        )
    return MaskAugmentationOutcome(augmented, mode, changed)


def mask_augmentation_mode_id(mode: str) -> int:
    return MASK_AUGMENTATION_MODE_IDS[mode]
