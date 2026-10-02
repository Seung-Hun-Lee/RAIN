"""RAIN action model with a pooling Transition Head on frozen TCE features."""
from typing import Dict, Optional
import os

import torch

from rain.models.model import RAINModel
from .transition_head import PoolingTCHead, tc_loss
from .checkpoints import load_exact_action, load_exact_transition


class PoolingModel(RAINModel):
    def __init__(self, config):
        super().__init__(config)
        self.progress_architecture = os.environ.get("RAIN_POOLING_HEAD_VARIANT", "region_gated")
        if self.condition_mode != "action_type":
            raise ValueError("This model requires action-type conditioning.")
        # Initialize the pooling head independently of the base progress head.
        # Forking the RNG preserves its state after the base constructor.
        with torch.random.fork_rng(devices=[]):
            self.fusion_branch = PoolingTCHead(self.progress_architecture)

    def load_action_checkpoint(self, path):
        load_exact_action(self, path)

    def load_progress_checkpoint(self, path):
        load_exact_transition(self, path, self.progress_architecture)

    def forward_progress(
        self,
        dino_third: Optional[torch.Tensor] = None,
        dino_wrist: Optional[torch.Tensor] = None,
        image_third: Optional[torch.Tensor] = None,
        image_wrist: Optional[torch.Tensor] = None,
        goal_mask_third: Optional[torch.Tensor] = None,
        goal_mask_wrist: Optional[torch.Tensor] = None,
        target_place_mask: Optional[torch.Tensor] = None,
        target_place_mask_wrist: Optional[torch.Tensor] = None,
        text_feat: Optional[torch.Tensor] = None,
        action_type: Optional[torch.Tensor] = None,
        gt_distance: Optional[torch.Tensor] = None,
        gt_alignment: Optional[torch.Tensor] = None,
        gt_task_completion: Optional[torch.Tensor] = None,
        text_feat_full: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        compact_offline_progress = (
            self.online_dino is None
            and int(getattr(self.config.dit, "num_scales", 1)) > 1
            and dino_third is not None
            and dino_wrist is not None
            and dino_third.ndim == 3
            and dino_wrist.ndim == 3
        )
        if compact_offline_progress:
            scales = [(dino_third, dino_wrist)]
            patch_only = True
        else:
            scales, patch_only = self._resolve_vision_scales(
                dino_third, dino_wrist, image_third, image_wrist
            )
        B = scales[0][0].shape[0]
        device = scales[0][0].device
        patch_offset = 0 if patch_only else 1
        num_patches_third = scales[0][0].shape[1] - patch_offset
        num_patches_wrist = scales[0][1].shape[1] - patch_offset
        mask_third = (
            goal_mask_third if goal_mask_third is not None
            else torch.zeros(B, num_patches_third, device=device)
        )
        mask_wrist = (
            goal_mask_wrist if goal_mask_wrist is not None
            else torch.zeros(B, num_patches_wrist, device=device)
        )
        with torch.no_grad():
            third_patches, wrist_patches, corr_map, _ = self._encode_views_split(
                scales[-1][0], scales[-1][1], mask_third,
                goal_mask_wrist=mask_wrist,
                target_place_mask=target_place_mask,
                target_place_mask_wrist=target_place_mask_wrist,
                patch_only=patch_only,
            )
            third_patches = third_patches.detach()
            wrist_patches = wrist_patches.detach()
            corr_map = corr_map.detach()
        progress_text = text_feat_full if text_feat_full is not None else text_feat
        cond_text = self._build_condition_text_feat(progress_text, action_type, B, device)
        progress_preds, _ = self.fusion_branch(
            third_patches, wrist_patches, mask_third, mask_wrist, cond_text,
        )
        losses: Dict[str, torch.Tensor] = {}
        if self.training and gt_distance is not None:
            progress_losses = tc_loss(
                progress_preds["task_comp_logit"], gt_distance,
                gt_task_completion, action_type, self.config.progress,
            )
            for k, v in progress_losses.items():
                losses[f"fusion_{k}"] = v
            losses["progress_loss"] = progress_losses["progress_loss"]
            losses["loss"] = progress_losses["progress_loss"]
        else:
            losses["loss"] = torch.tensor(0.0, device=device)
        return losses

    @torch.no_grad()
    def predict(
        self,
        dino_third: Optional[torch.Tensor] = None,
        dino_wrist: Optional[torch.Tensor] = None,
        image_third: Optional[torch.Tensor] = None,
        image_wrist: Optional[torch.Tensor] = None,
        goal_mask_third: Optional[torch.Tensor] = None,
        goal_mask_wrist: Optional[torch.Tensor] = None,
        target_place_mask: Optional[torch.Tensor] = None,
        target_place_mask_wrist: Optional[torch.Tensor] = None,
        text_feat: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        action_type: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        sampling_generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        compact_progress_only = (
            state is None and self.online_dino is None
            and int(getattr(self.config.dit, "num_scales", 1)) > 1
            and dino_third is not None and dino_wrist is not None
            and dino_third.ndim == 3 and dino_wrist.ndim == 3
        )
        if compact_progress_only:
            scales = [(dino_third, dino_wrist)]
            patch_only = True
        else:
            scales, patch_only = self._resolve_vision_scales(
                dino_third, dino_wrist, image_third, image_wrist
            )
        B = scales[0][0].shape[0]
        device = scales[0][0].device
        patch_offset = 0 if patch_only else 1
        num_patches_third = scales[0][0].shape[1] - patch_offset
        num_patches_wrist = scales[0][1].shape[1] - patch_offset
        mask_third = (
            goal_mask_third if goal_mask_third is not None
            else torch.zeros(B, num_patches_third, device=device)
        )
        mask_wrist = (
            goal_mask_wrist if goal_mask_wrist is not None
            else torch.zeros(B, num_patches_wrist, device=device)
        )
        encoded_scales = self._encode_vision_scales(
            scales, patch_only, mask_third, mask_wrist,
            target_place_mask, target_place_mask_wrist,
        )
        third_patches, wrist_patches, corr_map, view_features = encoded_scales[-1]
        scale_features = [encoded[3] for encoded in encoded_scales]
        cond_text_feat = self._build_condition_text_feat(text_feat, action_type, B, device)
        result: Dict[str, torch.Tensor] = {}
        if state is not None:
            result = self.dit.predict(
                cond_text_feat, state, view_features, num_steps=num_steps,
                scale_features=scale_features if len(scale_features) > 1 else None,
                sampling_generator=sampling_generator,
            )
        progress_preds, fusion_gate = self.fusion_branch(
            third_patches, wrist_patches, mask_third, mask_wrist, cond_text_feat,
        )
        result["task_comp_prob"] = progress_preds["task_comp_prob"]
        result["fusion_gate"] = fusion_gate
        result["corr_map"] = corr_map
        return result
