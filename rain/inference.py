"""Load RAIN checkpoints and predict from cached or online features."""
from dataclasses import fields
import json
from pathlib import Path

import torch

from rain.configs.config import TrainingConfig, ThirdEncoderConfig, WristEncoderConfig, DiTConfig, ProgressConfig, DataConfig
from .model import PoolingModel


def load_config(path):
    data = json.loads(Path(path).read_text())
    nested = {"third_encoder": ThirdEncoderConfig, "wrist_encoder": WristEncoderConfig, "dit": DiTConfig, "progress": ProgressConfig, "data": DataConfig}
    for key, cls in nested.items():
        values = data.get(key, {})
        valid = {f.name for f in fields(cls)}
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"Unknown {key} config fields: {sorted(unknown)}")
        data[key] = cls(**values)
    return TrainingConfig(**data)


def load_policy(action_checkpoint, transition_checkpoint, config_path, device="cpu"):
    config = load_config(config_path)
    config.online_dino = False
    # CLIP features are checkpoint tensors. Suppress only their redundant download
    # while constructing the model, then restore and verify every action tensor.
    import rain.models.model as implementation
    import rain.models.progress_heads as progress
    old_main, old_progress = implementation._compute_clip_action_type_features, progress._compute_clip_action_type_features
    def checkpoint_placeholder(*args, **kwargs):
        return torch.zeros(7, config.dit.text_dim)
    try:
        implementation._compute_clip_action_type_features = checkpoint_placeholder
        progress._compute_clip_action_type_features = checkpoint_placeholder
        model = PoolingModel(config)
    finally:
        implementation._compute_clip_action_type_features = old_main
        progress._compute_clip_action_type_features = old_progress
    model.load_action_checkpoint(action_checkpoint)
    model.load_progress_checkpoint(transition_checkpoint)
    return model.to(device).eval()
