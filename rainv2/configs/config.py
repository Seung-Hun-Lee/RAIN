"""RAIN model configuration.

Two-stage training: action (TCE + PlanDiT) then progress (frozen TCE + PlanDiT).
Config field names are kept stable for checkpoint/config compatibility.
"""

from dataclasses import dataclass, field


@dataclass
class ThirdEncoderConfig:
    """Third-person branch config inside the TCE."""

    dino_dim: int = 1024
    hidden_dim: int = 448
    num_heads: int = 8
    num_blocks: int = 1
    mlp_ratio: float = 4.0
    film_embed_dim: int = 32
    use_target_place: bool = True
    use_transformer_blocks: bool = True
    zero_view_masks: bool = False


@dataclass
class WristEncoderConfig:
    """Wrist-view branch config inside the TCE."""

    dino_dim: int = 1024
    hidden_dim: int = 448
    num_heads: int = 8
    num_blocks: int = 1
    mlp_ratio: float = 4.0
    film_embed_dim: int = 32
    conditioning_mode: str = "corr"  # "corr" | "mask" | "none"
    use_corr_modulation: bool = True
    use_goal_mask: bool = False
    use_target_place: bool = False
    use_transformer_blocks: bool = True
    zero_view_masks: bool = False


@dataclass
class DiTConfig:
    """PlanDiT: flow-matching decoder with optional plan tokens."""

    hidden_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_action_tokens: int = 16
    action_dim: int = 7
    use_plan: bool = True
    num_plan_tokens: int = 8
    plan_dim: int = 3
    num_inference_steps: int = 4
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    noise_s: float = 0.999
    goal_xyz_consistency_weight: float = 0.1
    plan_action_causal: bool = True
    text_dim: int = 768
    state_dim: int = 8
    vision_input_dim: int = 448
    num_scales: int = 1


@dataclass
class ProgressConfig:
    """Progress head config (gated fusion branch)."""

    hidden_dim: int = 448
    text_dim: int = 768
    clip_model_name: str = "openai/clip-vit-large-patch14"
    smooth_l1_beta: float = 0.1
    lambda_dist: float = 0.3
    lambda_align: float = 0.3
    lambda_tc: float = 1.2
    task_comp_fixed_temp: float = 5.0
    release_task_comp_weight: float = 1.5
    tc_mask_dist_thresh: float = 0.8


@dataclass
class DataConfig:
    """Data loading config."""

    episodes_json: str = ""
    tc_event_manifest: str = ""
    frames_dir: str = ""
    parquet_dir: str = ""
    packed_features_dir: str = ""
    images_dir: str = ""
    clip_model_name: str = "openai/clip-vit-large-patch14"
    text_feature_dim: int = 768
    num_action_steps: int = 16
    num_plan_waypoints: int = 8
    distance_alpha: float = 10.0
    max_post_subtask_frames: int = 15
    text_dropout_prob: float = 0.0
    mask_augmentation: str = "none"
    reverse_text_dropout_prob: float = 1.0
    reverse_max_mask_overlap: float = 0.05
    xyz_source: str = "action_delta"
    use_retarget_aug: bool = True
    retarget_version: str = "legacy"
    retarget_rollout_mode: str = "tangent_replay"
    retarget_text_dropout_prob: float = 0.8
    use_goal_consistency_pair_batch: bool = False
    retarget_template_path: str = ""
    consistency_group_size: int = 4
    tail_exclude_ratio: float = 0.05
    train_ratio: float = 0.9
    seed: int = 42


@dataclass
class TrainingConfig:
    """Top-level training config for RAIN model."""

    third_encoder: ThirdEncoderConfig = field(default_factory=ThirdEncoderConfig)
    wrist_encoder: WristEncoderConfig = field(default_factory=WristEncoderConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    data: DataConfig = field(default_factory=DataConfig)

    condition_mode: str = "action_type"  # "text" | "action_type"
    cross_attn_mode: str = "bidirectional"  # "bidirectional" | "third_to_wrist" | "wrist_to_third"
    use_reverse_aug: bool = False
    online_dino: bool = False
    dino_input_size: int = 224

    # Training
    output_dir: str = "outputs"
    experiment_name: str = ""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_epochs: int = 500
    max_steps: int = 0
    batch_size_per_gpu: int = 64
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    num_workers: int = 4
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1

    # Progress stage
    early_stop_patience: int = 50
    early_stop_metric: str = "task_comp_accuracy"
