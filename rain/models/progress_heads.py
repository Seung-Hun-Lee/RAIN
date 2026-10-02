"""Progress Head modules for RAIN.

Each view (third, wrist) has its own SingleViewProgressHead that predicts
distance, alignment, and task_completion independently.

Third view: global = mean(patches), obj = mask_pool(patches, mask)
Wrist view: global = mean(patches), obj = corr_weighted_pool(patches, corr_map)

Action-type embeddings use normalized CLIP text features.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rain.configs.config import ProgressConfig
from shared.clip_utils import (
    DEFAULT_CLIP_TEXT_MODEL,
    get_clip_text_feature_dim,
    normalize_clip_model_name,
)

ACTION_TYPE_MAP = {
    "grasp": 0, "release": 1, "push": 2,
    "turn_on": 3, "close": 4, "open": 5,
    "approach": 6,
}

ACTION_TYPE_CLIP_PROMPTS = [
    "grasp", "release", "push", "turn on", "close", "open", "approach",
]


def _compute_clip_action_type_features(
    model_name: str = DEFAULT_CLIP_TEXT_MODEL,
    feature_dim: Optional[int] = None,
) -> torch.Tensor:
    """Pre-compute L2-normalized CLIP features for action type prompts.

    Returns (7, D) float tensor on CPU.
    """
    model_name = normalize_clip_model_name(model_name)
    feature_dim = (
        get_clip_text_feature_dim(model_name)
        if feature_dim is None else int(feature_dim)
    )
    skip_init = str(
        os.environ.get("RAIN_SKIP_CLIP_ACTION_TYPE_INIT", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if skip_init:
        return torch.zeros(
            len(ACTION_TYPE_CLIP_PROMPTS), feature_dim, dtype=torch.float32
        )

    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_name)
    clip = CLIPModel.from_pretrained(model_name).eval()
    inputs = tok(
        ACTION_TYPE_CLIP_PROMPTS, padding=True, truncation=True,
        max_length=77, return_tensors="pt",
    )
    with torch.no_grad():
        out = clip.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
    del clip, tok
    return out.float()  # (7, D)


class SingleViewProgressHead(nn.Module):
    """Progress head for a single view.

    Input: [view_global(D), view_obj_feat(D), text_emb(D), act_emb(64)] = 3D + 64.
    Outputs: distance (sigmoid), alignment (sigmoid), task_completion (dual-path).
    """

    def __init__(self, config: ProgressConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim
        unified_in = D * 3 + 64

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        # CLIP action type features
        clip_feats = _compute_clip_action_type_features(
            config.clip_model_name, config.text_dim
        )
        self.register_buffer("action_type_clip_features", clip_feats)
        self.action_clip_proj = nn.Linear(config.text_dim, 64)

        # Distance head
        self.distance_head = nn.Sequential(
            nn.LayerNorm(unified_in, elementwise_affine=False),
            nn.Linear(unified_in, D),
            nn.SiLU(),
            nn.Linear(D, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, 1),
            nn.Sigmoid(),
        )

        # Alignment head
        self.alignment_head = nn.Sequential(
            nn.LayerNorm(unified_in, elementwise_affine=False),
            nn.Linear(unified_in, D),
            nn.SiLU(),
            nn.Linear(D, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, 1),
            nn.Sigmoid(),
        )

        # Task completion (dual-path)
        # Path A: cross-modal matching
        self.task_comp_visual_proj = nn.Sequential(
            nn.LayerNorm(D, elementwise_affine=False),
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )
        self.task_comp_text_proj = nn.Sequential(
            nn.Linear(config.text_dim, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )
        # Path B: MLP
        self.task_comp_mlp = nn.Sequential(
            nn.LayerNorm(unified_in, elementwise_affine=False),
            nn.Linear(unified_in, D),
            nn.SiLU(),
            nn.Linear(D, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self,
        view_global: torch.Tensor,       # (B, D) mean-pooled patches
        view_obj_feat: torch.Tensor,      # (B, D) object-focused features
        text_feat: torch.Tensor,          # (B, text_dim)
        action_type: Optional[torch.Tensor],  # (B,) long or None
    ) -> Dict[str, torch.Tensor]:
        """Compute per-view progress predictions."""
        B = view_global.shape[0]
        device = view_global.device

        # Text embedding
        text_emb = self.text_proj(text_feat)  # (B, D)

        # Action type embedding
        act_emb = (
            self.action_clip_proj(self.action_type_clip_features[action_type])
            if action_type is not None
            else torch.zeros(B, 64, device=device)
        )

        # Unified input
        unified = torch.cat([view_global, view_obj_feat, text_emb, act_emb], dim=-1)

        result: Dict[str, torch.Tensor] = {}

        # Distance
        result["pred_distance"] = self.distance_head(unified).squeeze(-1)

        # Alignment
        result["pred_alignment"] = self.alignment_head(unified).squeeze(-1)

        # Task completion (dual-path)
        vis_norm = F.normalize(self.task_comp_visual_proj(view_obj_feat), dim=-1)
        txt_norm = F.normalize(self.task_comp_text_proj(text_feat), dim=-1)
        logit_A = (vis_norm * txt_norm).sum(-1) * self.config.task_comp_fixed_temp
        logit_B = self.task_comp_mlp(unified).squeeze(-1)
        task_comp_logit = logit_A + logit_B
        result["task_comp_logit"] = task_comp_logit
        result["task_comp_prob"] = torch.sigmoid(task_comp_logit)

        return result

    @staticmethod
    def compute_loss(
        preds: Dict[str, torch.Tensor],
        gt_distance: torch.Tensor,
        gt_alignment: torch.Tensor,
        gt_task_completion: torch.Tensor,
        config: ProgressConfig,
        action_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted progress loss for one view."""
        device = next(iter(preds.values())).device
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=device)

        # Distance
        loss_dist = F.smooth_l1_loss(
            preds["pred_distance"], gt_distance, beta=config.smooth_l1_beta)
        losses["loss_dist"] = loss_dist.detach()
        total = total + config.lambda_dist * loss_dist

        # Alignment
        loss_align = F.smooth_l1_loss(
            preds["pred_alignment"], gt_alignment, beta=config.smooth_l1_beta)
        losses["loss_align"] = loss_align.detach()
        total = total + config.lambda_align * loss_align

        # Task completion
        tc_loss_per_sample = F.binary_cross_entropy_with_logits(
            preds["task_comp_logit"], gt_task_completion, reduction="none")

        if action_type is not None:
            is_release = (action_type == ACTION_TYPE_MAP["release"]).float()
            is_completed = (gt_task_completion > 0.5).float()
            tc_weight = torch.ones_like(tc_loss_per_sample)
            tc_weight = tc_weight + (config.release_task_comp_weight - 1.0) * is_release * is_completed
        else:
            tc_weight = torch.ones_like(tc_loss_per_sample)

        # TC loss masking: skip ambiguous action frames (high dist but TC=0)
        if config.tc_mask_dist_thresh > 0:
            tc_mask = (gt_distance <= config.tc_mask_dist_thresh) | (gt_task_completion > 0.5)
            loss_tc = (tc_loss_per_sample * tc_weight * tc_mask.float()).sum() / tc_mask.float().sum().clamp(min=1)
        else:
            loss_tc = (tc_loss_per_sample * tc_weight).mean()

        losses["loss_task_comp"] = loss_tc.detach()
        total = total + config.lambda_tc * loss_tc

        losses["progress_loss"] = total
        return losses


class GatedFusionBranch(nn.Module):
    """Learned gated fusion of third and wrist view features.

    Gate network produces per-sample weights (B, 2) via sigmoid:
      gate=0 -> trust third view, gate=1 -> trust wrist view.

    Fused features are fed to a SingleViewProgressHead for prediction.
    """

    def __init__(self, config: ProgressConfig) -> None:
        super().__init__()
        D = config.hidden_dim

        # Gate: [third_global, third_obj, wrist_global, wrist_obj] -> (B, 2)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(4 * D, elementwise_affine=False),
            nn.Linear(4 * D, D),
            nn.SiLU(),
            nn.Linear(D, 2),
            nn.Sigmoid(),
        )

        # Prediction head on fused features
        self.head = SingleViewProgressHead(config)

    def forward(
        self,
        third_global: torch.Tensor,   # (B, D)
        third_obj: torch.Tensor,       # (B, D)
        wrist_global: torch.Tensor,    # (B, D)
        wrist_obj: torch.Tensor,       # (B, D)
        text_feat: torch.Tensor,       # (B, 768)
        action_type: Optional[torch.Tensor],  # (B,) long or None
    ) -> tuple:
        """Returns (preds_dict, gate_tensor).

        gate_tensor: (B, 2), containing [global_gate, obj_gate].
        """
        gate_input = torch.cat([third_global, third_obj, wrist_global, wrist_obj], dim=-1)
        gate = self.gate_net(gate_input)  # (B, 2)
        global_gate = gate[:, 0:1]  # (B, 1)
        obj_gate = gate[:, 1:2]     # (B, 1)

        fused_global = (1 - global_gate) * third_global + global_gate * wrist_global
        fused_obj = (1 - obj_gate) * third_obj + obj_gate * wrist_obj

        preds = self.head(fused_global, fused_obj, text_feat, action_type)
        return preds, gate


def extract_third_view_features(
    third_patches: torch.Tensor,   # (B, 256, D)
    goal_mask: torch.Tensor,        # (B, 256) float
) -> tuple:
    """Extract global and object features for third view.

    global = mean(patches)
    obj = mask_pool(patches, mask), the mean of mask-region patches
    """
    view_global = third_patches.mean(dim=1)  # (B, D)

    obj_mask = (goal_mask > 0.5).float()  # (B, 256)
    obj_count = obj_mask.sum(-1, keepdim=True).clamp(1.0)
    view_obj_feat = (third_patches * obj_mask.unsqueeze(-1)).sum(1) / obj_count  # (B, D)

    return view_global, view_obj_feat


def extract_wrist_view_features(
    wrist_patches: torch.Tensor,   # (B, 256, D)
    goal_mask: torch.Tensor,        # (B, 256) float
) -> tuple:
    """Extract global and object features for wrist view.

    global = mean(patches)
    obj = mask_pool(patches, mask), the mean of mask-region patches
    """
    view_global = wrist_patches.mean(dim=1)  # (B, D)

    obj_mask = (goal_mask > 0.5).float()  # (B, 256)
    obj_count = obj_mask.sum(-1, keepdim=True).clamp(1.0)
    view_obj_feat = (wrist_patches * obj_mask.unsqueeze(-1)).sum(1) / obj_count  # (B, D)

    return view_global, view_obj_feat
