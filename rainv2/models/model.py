"""RAIN: Unified action + progress model (two-stage training).

Stage 1 (action): Train encoder + DiT for action generation.
Stage 2 (progress): Freeze encoder + DiT, train progress head.
Inference: Both action and progress predicted in a single forward pass.

Flat structure (no self.base) so checkpoint keys stay flat and Stage 1
checkpoints are directly loadable without key remapping.
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from rainv2.configs.config import TrainingConfig
from rainv2.models.vision_encoder import TargetAdaptiveCrossViewEncoder
from rainv2.models.multiscale_vision import FrozenDINOv2LargeMultiScale
from shared.plan_dit import PlanDiT
from rainv2.models.progress_heads import (
    ACTION_TYPE_MAP,
    GatedFusionBranch,
    SingleViewProgressHead,
    _compute_clip_action_type_features,
    extract_third_view_features,
    extract_wrist_view_features,
)
from shared.clip_utils import DEFAULT_CLIP_TEXT_MODEL

logger = logging.getLogger(__name__)


class RAINModel(nn.Module):
    """RAIN: Unified action generation + progress prediction.

    Architecture (flat):
      dual_view_encoder -> Target-adaptive Cross-view Encoder (TCE)
      dit               -> PlanDiT action (+ plan) generation via flow matching
      fusion_branch  -> progress prediction

    Training stages:
      "action"   -> encoder + DiT trainable, progress head frozen
      "progress" -> encoder + DiT frozen, progress head trainable
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self._training_stage: str = "action"  # "action" or "progress"

        self.condition_mode = str(config.condition_mode).strip().lower()
        if self.condition_mode not in {"text", "action_type"}:
            raise ValueError(
                f"Unsupported condition_mode={config.condition_mode!r}; "
                "expected one of: text, action_type"
            )
        self._warned_missing_action_type = False
        self._condition_text_dim = config.dit.text_dim

        if self.condition_mode == "action_type":
            clip_model_name = getattr(
                config.progress, "clip_model_name", DEFAULT_CLIP_TEXT_MODEL
            )
            clip_feats = _compute_clip_action_type_features(
                clip_model_name, config.dit.text_dim
            )
            self.register_buffer("action_type_clip_features", clip_feats)

        # `dual_view_encoder` is the TCE namespace in checkpoint keys.
        self.dual_view_encoder = TargetAdaptiveCrossViewEncoder(
            config.third_encoder,
            config.wrist_encoder,
            cross_attn_mode=str(config.cross_attn_mode).strip().lower(),
        )

        self.online_dino = None
        if bool(getattr(config, "online_dino", False)):
            if int(getattr(config.dit, "num_scales", 1)) != 3:
                raise ValueError("Online RAIN recipe requires exactly three DINO scales")
            if int(config.third_encoder.dino_dim) != 1024:
                raise ValueError("Online RAIN recipe requires DINOv2-L (feature dim 1024)")
            self.online_dino = FrozenDINOv2LargeMultiScale(
                input_size=int(getattr(config, "dino_input_size", 224))
            )

        # `dit` is the PlanDiT namespace in checkpoint keys.
        config.dit.vision_input_dim = config.third_encoder.hidden_dim
        self.dit = PlanDiT(config.dit)

        # Progress head: gated fusion branch
        self.fusion_branch = GatedFusionBranch(config.progress)

        # Log model info
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("RAINModel: %.2fM trainable / %.2fM total params",
                     trainable / 1e6, total / 1e6)

    # ------------------------------------------------------------------
    # Training stage management
    # ------------------------------------------------------------------

    def set_training_stage(self, stage: str) -> None:
        """Set training stage and freeze/unfreeze accordingly.

        "action":   encoder + DiT trainable, progress head frozen
        "progress": encoder + DiT frozen, progress head trainable
        """
        if stage not in {"action", "progress"}:
            raise ValueError(f"Unknown stage: {stage!r}")
        self._training_stage = stage

        if stage == "action":
            for p in self.dual_view_encoder.parameters():
                p.requires_grad = True
            for p in self.dit.parameters():
                p.requires_grad = True
            self._set_progress_head_grad(False)
        else:  # progress
            for p in self.dual_view_encoder.parameters():
                p.requires_grad = False
            for p in self.dit.parameters():
                p.requires_grad = False
            self._set_progress_head_grad(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("Stage=%s: %.2fM trainable / %.2fM total params",
                     stage, trainable / 1e6, total / 1e6)

    def _set_progress_head_grad(self, requires_grad: bool) -> None:
        """Set requires_grad for the progress fusion branch."""
        for p in self.fusion_branch.parameters():
            p.requires_grad = requires_grad

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_action_checkpoint(self, path: str) -> None:
        """Load encoder + DiT weights from a Stage 1 checkpoint.

        Expects keys: third_encoder.*, wrist_encoder.*, dit.*,
        action_type_clip_features (buffer).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)

        # Filter to action-related keys only
        action_prefixes = (
            "dual_view_encoder.",
            "dit.",
            "action_type_clip_features",
        )
        action_state = {k: v for k, v in state.items() if k.startswith(action_prefixes)}

        at_key = "action_type_clip_features"
        if at_key in action_state and hasattr(self, at_key):
            ckpt_feat = action_state[at_key]
            model_feat = getattr(self, at_key)
            if tuple(ckpt_feat.shape) != tuple(model_feat.shape):
                if (
                    ckpt_feat.ndim == 2
                    and model_feat.ndim == 2
                    and ckpt_feat.shape[1] == model_feat.shape[1]
                ):
                    patched = model_feat.detach().cpu().clone()
                    n = min(int(ckpt_feat.shape[0]), int(model_feat.shape[0]))
                    patched[:n] = ckpt_feat[:n]
                    action_state[at_key] = patched
                    logger.warning(
                        "load_action_checkpoint: adapted %s shape %s -> %s (copied %d rows)",
                        at_key,
                        tuple(ckpt_feat.shape),
                        tuple(model_feat.shape),
                        n,
                    )
                else:
                    logger.warning(
                        "load_action_checkpoint: dropped %s due incompatible shape %s vs %s",
                        at_key,
                        tuple(ckpt_feat.shape),
                        tuple(model_feat.shape),
                    )
                    action_state.pop(at_key, None)

        # Stage-1 checkpoints may contain a frozen progress head. Drop those
        # keys so progress training starts from a freshly initialized head.
        action_state = {
            k: v for k, v in action_state.items() if not k.startswith("fusion_branch.")
        }

        missing, unexpected = self.load_state_dict(action_state, strict=False)
        # Expected missing: progress-head keys
        progress_missing = [
            k for k in missing
            if k.startswith("fusion_branch.")
        ]
        real_missing = [k for k in missing if k not in progress_missing]
        if real_missing:
            logger.warning("load_action_checkpoint: missing non-progress keys: %s", real_missing)
        if unexpected:
            logger.warning("load_action_checkpoint: unexpected keys: %s", unexpected)
        logger.info("Loaded action checkpoint from %s (%d keys, %d progress-head keys skipped)",
                     path, len(action_state), len(progress_missing))

    def load_progress_checkpoint(self, path: str) -> None:
        """Load progress-head weights from a Stage 2 checkpoint.

        Stage 2 saves full state_dict (frozen encoder+DiT + trained progress head).
        Load only progress-head keys to avoid overwriting action weights.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)

        # Filter to progress-head keys only
        progress_state = {
            k: v
            for k, v in state.items()
            if k.startswith("fusion_branch.")
        }

        model_state = self.state_dict()
        for key, val in list(progress_state.items()):
            if key not in model_state:
                continue
            tgt = model_state[key]
            if tuple(val.shape) == tuple(tgt.shape):
                continue
            if (
                key.endswith("action_type_clip_features")
                and val.ndim == 2
                and tgt.ndim == 2
                and val.shape[1] == tgt.shape[1]
            ):
                patched = tgt.detach().cpu().clone()
                n = min(int(val.shape[0]), int(tgt.shape[0]))
                patched[:n] = val[:n]
                progress_state[key] = patched
                logger.warning(
                    "load_progress_checkpoint: adapted %s shape %s -> %s (copied %d rows)",
                    key,
                    tuple(val.shape),
                    tuple(tgt.shape),
                    n,
                )
            else:
                logger.warning(
                    "load_progress_checkpoint: dropped %s due incompatible shape %s vs %s",
                    key,
                    tuple(val.shape),
                    tuple(tgt.shape),
                )
                progress_state.pop(key, None)

        if not progress_state:
            logger.warning(
                "load_progress_checkpoint: no fusion_branch keys in %s", path
            )
            return

        missing, unexpected = self.load_state_dict(progress_state, strict=False)
        if missing:
            logger.info("load_progress_checkpoint: missing keys after load: %s", missing)
        if unexpected:
            logger.info("load_progress_checkpoint: unexpected keys after load: %s", unexpected)
        logger.info("Loaded progress checkpoint from %s (%d progress-head keys)",
                     path, len(progress_state))

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def _build_condition_text_feat(
        self,
        text_feat: Optional[torch.Tensor],
        action_type: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build text-conditioning feature from either text or action_type mode."""
        if self.condition_mode == "text":
            if text_feat is None:
                return torch.zeros(batch_size, self._condition_text_dim, device=device)
            return text_feat

        if action_type is None:
            if not self._warned_missing_action_type:
                logger.warning(
                    "condition_mode=action_type but action_type is missing; "
                    "falling back to zero conditioning."
                )
                self._warned_missing_action_type = True
            return torch.zeros(batch_size, self._condition_text_dim, device=device)

        action_type = action_type.long().to(device)
        action_type = action_type.clamp(min=0, max=len(ACTION_TYPE_MAP) - 1)
        return self.action_type_clip_features[action_type]

    def _progress_action_type(self, action_type: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Select which conditioning branch progress head can use."""
        if self.condition_mode == "action_type":
            return action_type
        return None

    # ------------------------------------------------------------------
    # View encoding (split variant returns patches separately for progress)
    # ------------------------------------------------------------------

    def _encode_views_split(
        self,
        dino_third: torch.Tensor,
        dino_wrist: torch.Tensor,
        goal_mask_third: torch.Tensor,
        goal_mask_wrist: Optional[torch.Tensor] = None,
        target_place_mask: Optional[torch.Tensor] = None,
        target_place_mask_wrist: Optional[torch.Tensor] = None,
        patch_only: bool = False,
    ) -> tuple:
        """Encode both views, returning split patches + corr_map.

        Returns:
            third_patches: (B, 256, D)
            wrist_patches: (B, 256, D)
            corr_map: (B, 256)
            view_features: (B, 512, D) concatenated
        """
        if not patch_only:
            dino_third = dino_third[:, 1:]
            dino_wrist = dino_wrist[:, 1:]
        if dino_third.shape[1] != goal_mask_third.shape[1]:
            raise ValueError(
                "Third-view feature/mask patch mismatch: "
                f"{dino_third.shape[1]} vs {goal_mask_third.shape[1]}"
            )
        if goal_mask_wrist is not None and dino_wrist.shape[1] != goal_mask_wrist.shape[1]:
            raise ValueError(
                "Wrist-view feature/mask patch mismatch: "
                f"{dino_wrist.shape[1]} vs {goal_mask_wrist.shape[1]}"
            )
        third_patches, wrist_patches, corr_map = self.dual_view_encoder(
            dino_third, dino_wrist,
            goal_mask_third,
            goal_mask_wrist=goal_mask_wrist,
            place_mask=target_place_mask,
            place_mask_wrist=target_place_mask_wrist,
        )
        view_features = torch.cat([third_patches, wrist_patches], dim=1)  # (B, 512, D)
        return third_patches, wrist_patches, corr_map, view_features

    def _resolve_vision_scales(
        self,
        dino_third: Optional[torch.Tensor],
        dino_wrist: Optional[torch.Tensor],
        image_third: Optional[torch.Tensor],
        image_wrist: Optional[torch.Tensor],
    ) -> tuple:
        """Return [(third, wrist), ...] and whether tensors are patch-only."""
        if self.online_dino is None:
            if dino_third is None or dino_wrist is None:
                raise ValueError("Packed-DINO mode requires dino_third and dino_wrist")
            num_scales = int(getattr(self.config.dit, "num_scales", 1))
            if num_scales > 1:
                if dino_third.ndim != 4 or dino_wrist.ndim != 4:
                    raise ValueError(
                        "Offline multi-scale mode requires (B,S,N,D) feature tensors"
                    )
                if dino_third.shape[1] != num_scales or dino_wrist.shape[1] != num_scales:
                    raise ValueError(
                        f"Expected {num_scales} cached scales, got "
                        f"{dino_third.shape[1]} and {dino_wrist.shape[1]}"
                    )
                return [
                    (dino_third[:, scale], dino_wrist[:, scale])
                    for scale in range(num_scales)
                ], True
            if dino_third.ndim != 3 or dino_wrist.ndim != 3:
                raise ValueError("Single-scale mode requires (B,N,D) feature tensors")
            # A selected intermediate-layer cache contains patch tokens only;
            # older final-layer caches include one CLS token. Both encode the
            # same square spatial grid. Never discard a real patch as CLS.
            tokens = int(dino_third.shape[1])
            if int(dino_wrist.shape[1]) != tokens:
                raise ValueError("Single-scale views must use the same token layout")
            patch_only = math.isqrt(tokens) ** 2 == tokens
            if not patch_only and math.isqrt(tokens - 1) ** 2 != tokens - 1:
                raise ValueError(f"Invalid single-scale DINO token count: {tokens}")
            return [(dino_third, dino_wrist)], patch_only

        if image_third is None or image_wrist is None:
            raise ValueError("Online multi-scale mode requires image_third and image_wrist")
        batch_size = image_third.shape[0]
        both_views = torch.cat([image_third, image_wrist], dim=0)
        both_scales = self.online_dino(both_views)
        scales = [
            (features[:batch_size], features[batch_size:])
            for features in both_scales
        ]
        return scales, True

    def _encode_vision_scales(
        self,
        scales: list,
        patch_only: bool,
        goal_mask_third: torch.Tensor,
        goal_mask_wrist: torch.Tensor,
        target_place_mask: Optional[torch.Tensor],
        target_place_mask_wrist: Optional[torch.Tensor],
    ) -> list:
        return [
            self._encode_views_split(
                third,
                wrist,
                goal_mask_third,
                goal_mask_wrist=goal_mask_wrist,
                target_place_mask=target_place_mask,
                target_place_mask_wrist=target_place_mask_wrist,
                patch_only=patch_only,
            )
            for third, wrist in scales
        ]

    # ------------------------------------------------------------------
    # Forward dispatching
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Dispatch to stage-specific forward pass."""
        if self._training_stage == "action":
            return self.forward_action(**kwargs)
        else:
            return self.forward_progress(**kwargs)

    # ------------------------------------------------------------------
    # Stage 1: Action training (encoder + DiT)
    # ------------------------------------------------------------------

    def forward_action(
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
        gt_action: Optional[torch.Tensor] = None,
        gt_plan: Optional[torch.Tensor] = None,
        action_loss_mask: Optional[torch.Tensor] = None,
        plan_loss_mask: Optional[torch.Tensor] = None,
        goal_group_index: Optional[torch.Tensor] = None,
        goal_ref_index: Optional[torch.Tensor] = None,
        is_post_subtask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass for action stage (encoder + DiT)."""
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
            scales,
            patch_only,
            mask_third,
            mask_wrist,
            target_place_mask,
            target_place_mask_wrist,
        )
        view_features = encoded_scales[-1][3]
        scale_features = [encoded[3] for encoded in encoded_scales]

        cond_text_feat = self._build_condition_text_feat(text_feat, action_type, B, device)

        # Build loss masks: disable DiT losses on post-subtask frames
        if action_loss_mask is None and is_post_subtask is not None:
            action_loss_mask = (~is_post_subtask.bool()).float()
        if plan_loss_mask is None and is_post_subtask is not None:
            plan_loss_mask = (~is_post_subtask.bool()).float()

        losses: Dict[str, torch.Tensor] = {}
        if gt_action is not None:
            dit_out = self.dit(
                cond_text_feat, state, gt_action, view_features,
                gt_plan=gt_plan,
                action_loss_mask=action_loss_mask,
                plan_loss_mask=plan_loss_mask,
                goal_group_index=goal_group_index,
                goal_ref_index=goal_ref_index,
                view_mask=None,
                scale_features=scale_features if len(scale_features) > 1 else None,
            )
            losses.update(dit_out)

        total = losses.get("dit_loss", torch.tensor(0.0, device=device))
        losses["loss"] = total

        return losses

    # ------------------------------------------------------------------
    # Stage 2: Progress training (frozen encoder, train progress head)
    # ------------------------------------------------------------------

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
        """Training forward pass for progress stage (frozen encoder + progress head)."""
        # Offline stage-2 loads only the final (layer-23) packed scale. The
        # action model retains num_scales=3; this compact 3-D tensor avoids
        # reading the two intermediate scales used only by the action decoder.
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

        # Encode with frozen encoder (no grad)
        with torch.no_grad():
            third_patches, wrist_patches, corr_map, _ = self._encode_views_split(
                scales[-1][0], scales[-1][1],
                mask_third,
                goal_mask_wrist=mask_wrist,
                target_place_mask=target_place_mask,
                target_place_mask_wrist=target_place_mask_wrist,
                patch_only=patch_only,
            )
            third_patches = third_patches.detach()
            wrist_patches = wrist_patches.detach()
            corr_map = corr_map.detach()

        # Build conditioning
        progress_text = text_feat_full if text_feat_full is not None else text_feat
        cond_text = self._build_condition_text_feat(progress_text, action_type, B, device)
        progress_action_type = self._progress_action_type(action_type)

        # Extract features
        third_global, third_obj = extract_third_view_features(third_patches, mask_third)
        wrist_global, wrist_obj = extract_wrist_view_features(wrist_patches, mask_wrist)

        # Fusion branch prediction
        progress_preds, _ = self.fusion_branch(
            third_global, third_obj, wrist_global, wrist_obj,
            cond_text, progress_action_type,
        )

        # Compute loss
        losses: Dict[str, torch.Tensor] = {}
        if self.training and gt_distance is not None:
            progress_losses = SingleViewProgressHead.compute_loss(
                progress_preds, gt_distance, gt_alignment, gt_task_completion,
                self.config.progress, action_type,
            )
            for k, v in progress_losses.items():
                losses[f"fusion_{k}"] = v
            losses["progress_loss"] = progress_losses["progress_loss"]
            losses["loss"] = progress_losses["progress_loss"]
        else:
            losses["loss"] = torch.tensor(0.0, device=device)

        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
        """Inference: generate action + predict progress simultaneously."""
        # Stage-2 validation calls predict() without robot state and, for an
        # offline multiscale checkpoint, intentionally loads only the final
        # packed DINO scale.  Keep that progress-only path compact.  Online
        # policy inference has state and must continue to supply/use every
        # configured scale for the action decoder.
        compact_progress_only = (
            state is None
            and self.online_dino is None
            and int(getattr(self.config.dit, "num_scales", 1)) > 1
            and dino_third is not None
            and dino_wrist is not None
            and dino_third.ndim == 3
            and dino_wrist.ndim == 3
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
            scales,
            patch_only,
            mask_third,
            mask_wrist,
            target_place_mask,
            target_place_mask_wrist,
        )
        third_patches, wrist_patches, corr_map, view_features = encoded_scales[-1]
        scale_features = [encoded[3] for encoded in encoded_scales]

        cond_text_feat = self._build_condition_text_feat(text_feat, action_type, B, device)

        # Skip action generation when state is unavailable (progress-only eval).
        result: Dict[str, torch.Tensor] = {}
        if state is not None:
            result = self.dit.predict(
                cond_text_feat, state, view_features,
                num_steps=num_steps,
                scale_features=scale_features if len(scale_features) > 1 else None,
                sampling_generator=sampling_generator,
            )

        # Progress prediction
        progress_action_type = self._progress_action_type(action_type)
        third_global, third_obj = extract_third_view_features(third_patches, mask_third)
        wrist_global, wrist_obj = extract_wrist_view_features(wrist_patches, mask_wrist)
        progress_preds, fusion_gate = self.fusion_branch(
            third_global, third_obj, wrist_global, wrist_obj,
            cond_text_feat, progress_action_type,
        )

        result["task_comp_prob"] = progress_preds["task_comp_prob"]
        result["pred_distance"] = progress_preds["pred_distance"]
        result["pred_alignment"] = progress_preds["pred_alignment"]
        result["fusion_gate"] = fusion_gate
        result["corr_map"] = corr_map

        return result
