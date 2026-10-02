"""Six attention-free pooling heads on frozen TCE features.

Pooling precedes the shared visual projection.  A missing region mask uses the
whole-view mean; it does not suppress that view or force a fusion weight.
"""
import torch
from torch import nn
from torch.nn import functional as F

from rainv2.models.progress_heads import ACTION_TYPE_MAP


VARIANTS = (
    "global_gated", "global_concat", "region_gated", "region_concat",
    "global_region_gated", "global_region_concat",
)


def tc_loss(logits, gt_distance, gt_tc, action_type, config):
    """TC-only loss with sample eligibility and completed-release weighting."""
    assert config.lambda_dist == 0 and config.lambda_align == 0
    per_sample = F.binary_cross_entropy_with_logits(logits, gt_tc, reduction="none")
    weight = torch.ones_like(per_sample)
    if action_type is not None:
        is_release = (action_type == ACTION_TYPE_MAP["release"]).float()
        is_completed = (gt_tc > 0.5).float()
        weight = weight + (config.release_task_comp_weight - 1.0) * is_release * is_completed
    if config.tc_mask_dist_thresh > 0:
        eligible = (gt_distance <= config.tc_mask_dist_thresh) | (gt_tc > 0.5)
        raw = (per_sample * weight * eligible.float()).sum() / eligible.float().sum().clamp(min=1)
    else:
        raw = (per_sample * weight).mean()
    total = torch.tensor(0.0, device=logits.device) + config.lambda_tc * raw
    return {"loss_task_comp": raw.detach(), "progress_loss": total}


def region_mean(tokens, mask):
    """Mean of mask-region tokens, or all tokens when that view has no region.

    Region tokens have mask values >0.5. The data pipeline and frozen encoder
    receive the unmodified mask.
    """
    target = mask > 0.5
    target = target | ~target.any(-1, keepdim=True)
    weights = target.to(tokens.dtype)
    return (tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True)


class PoolingTCHead(nn.Module):
    def __init__(self, variant="region_gated"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown pooling architecture: {variant}")
        self.variant = variant
        self.view_dim = 512 if variant.startswith("global_region_") else 256
        self.fused_dim = self.view_dim if variant.endswith("_gated") else 2 * self.view_dim

        # Initialize TC before the optional gate so equal-shaped TC modules
        # receive identical initial values for the same seed across variants.
        self.input_proj = nn.Linear(1024, 256)
        self.action_proj = nn.Linear(768, 64)
        self.tc = nn.Sequential(
            nn.LayerNorm(self.fused_dim + 64, elementwise_affine=False),
            nn.Linear(self.fused_dim + 64, 256), nn.SiLU(), nn.Linear(256, 1),
        )
        if variant.endswith("_gated"):
            self.gate = nn.Sequential(
                nn.LayerNorm(2 * self.view_dim, elementwise_affine=False),
                nn.Linear(2 * self.view_dim, 256), nn.SiLU(),
                nn.Linear(256, 1), nn.Sigmoid(),
            )

    def pool_view(self, tokens, mask):
        """Return one view vector; no operation mixes tokens before pooling."""
        if self.variant.startswith("global_region_"):
            global_feature = self.input_proj(tokens.mean(1))
            region_feature = self.input_proj(region_mean(tokens, mask))
            return torch.cat((global_feature, region_feature), dim=-1)
        if self.variant.startswith("global_"):
            return self.input_proj(tokens.mean(1))
        return self.input_proj(region_mean(tokens, mask))

    def forward(self, third, wrist, third_mask, wrist_mask, action_type_feature):
        if third.ndim != 3 or third.shape != wrist.shape or third.shape[-1] != 1024:
            raise ValueError("Expected two matching [B,N,1024] encoded views")
        batch, length, _ = third.shape
        if length == 0:
            raise ValueError("Encoded views must contain at least one spatial token")
        if third_mask.shape != (batch, length) or wrist_mask.shape != (batch, length):
            raise ValueError("Expected one [B,N] region mask per view")
        if action_type_feature.shape != (batch, 768):
            raise ValueError("Expected the unchanged [B,768] CLIP action-type feature")

        third_pooled = self.pool_view(third, third_mask)
        wrist_pooled = self.pool_view(wrist, wrist_mask)
        concatenated = torch.cat((third_pooled, wrist_pooled), dim=-1)
        if self.variant.endswith("_gated"):
            wrist_weight = self.gate(concatenated)
            fused = (1.0 - wrist_weight) * third_pooled + wrist_weight * wrist_pooled
            fusion_weights = torch.cat((1.0 - wrist_weight, wrist_weight), dim=-1)
        else:
            fused = concatenated
            fusion_weights = None  # Direct concatenation has no view weights.

        action_feature = self.action_proj(action_type_feature)
        logit = self.tc(torch.cat((fused, action_feature), dim=-1)).squeeze(-1)
        return {"task_comp_logit": logit, "task_comp_prob": torch.sigmoid(logit)}, fusion_weights
