"""Utilities for CLIP text-model selection and feature-cache naming."""

from pathlib import Path


DEFAULT_CLIP_TEXT_MODEL = "openai/clip-vit-large-patch14"
_CLIP_TEXT_DIMS = {
    "openai/clip-vit-large-patch14": 768,
    "openai/clip-vit-base-patch32": 512,
}
_CLIP_TEXT_ALIASES = {
    "large": "openai/clip-vit-large-patch14",
    "base": "openai/clip-vit-base-patch32",
    "clip-large": "openai/clip-vit-large-patch14",
    "clip-base": "openai/clip-vit-base-patch32",
}


def normalize_clip_model_name(model_name: str) -> str:
    name = str(model_name or DEFAULT_CLIP_TEXT_MODEL).strip()
    lowered = name.lower()
    if lowered in _CLIP_TEXT_ALIASES:
        return _CLIP_TEXT_ALIASES[lowered]
    return name


def get_clip_text_feature_dim(model_name: str) -> int:
    norm = normalize_clip_model_name(model_name)
    if norm not in _CLIP_TEXT_DIMS:
        supported = ", ".join(sorted(_CLIP_TEXT_DIMS))
        raise ValueError(
            f"Unsupported CLIP text model {model_name!r}; supported: {supported}"
        )
    return int(_CLIP_TEXT_DIMS[norm])


def get_text_cache_path(packed_dir: str | Path, model_name: str) -> Path:
    packed = Path(packed_dir)
    norm = normalize_clip_model_name(model_name)
    if norm == DEFAULT_CLIP_TEXT_MODEL:
        return packed / "text_features.npz"
    suffix = norm.split("/")[-1].replace("-", "_")
    return packed / f"text_features_{suffix}.npz"
