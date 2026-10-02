"""PlanDiT: plan-guided diffusion transformer with flow matching.

Configured by DiTConfig.use_plan:
  - use_plan=False: [text(1) | state(1) | action(16)] = 18 tokens, no blockwise mask
  - use_plan=True:  [text(1) | state(1) | plan(8) | action(16)] = 26 tokens, blockwise mask
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from shared.components import (
    AdaLayerNorm,
    FeedForward,
    StateEncoder,
    TextProjector,
    TimestepEmbedding,
)


# ---------------------------------------------------------------------------
# DiT Blocks
# ---------------------------------------------------------------------------

class DiTSelfAttnBlock(nn.Module):
    """Self-attention block with AdaLN and optional blockwise masking."""

    def __init__(self, hidden_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(hidden_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = FeedForward(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.adaln(x, condition)
        x = x + self.self_attn(h, h, h, attn_mask=attn_mask)[0]
        x = x + self.ffn(x)
        return x


class DiTCrossAttnBlock(nn.Module):
    """Cross-attention block with AdaLN: Q from tokens, KV from vision."""

    def __init__(self, hidden_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_kv = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = FeedForward(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                kv: torch.Tensor,
                kv_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.adaln(x, condition)
        kv_normed = self.norm_kv(kv)
        x = x + self.cross_attn(
            h, kv_normed, kv_normed, key_padding_mask=kv_key_padding_mask)[0]
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# PlanDiT
# ---------------------------------------------------------------------------

class PlanDiT(nn.Module):
    """Plan-guided DiT decoder: action-only or plan+action.

    Token layout:
        Without plan: [text(1) | state(1) | action(16)] = 18
        With plan:    [text(1) | state(1) | plan(8) | action(16)] = 26

    KV: projected view features.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # Condition encoders
        self.time_embed = TimestepEmbedding(D)
        self.text_proj = TextProjector(config.text_dim, D)
        self.state_encoder = StateEncoder(config.state_dim, D)

        # Vision projection (TCE hidden dim -> DiT hidden dim). Multi-scale
        # inputs use an independent projection per DINO layer.
        vision_dim = getattr(config, 'dino_dim', getattr(config, 'vision_input_dim', 1024))
        self.num_scales = int(getattr(config, "num_scales", 1))
        if self.num_scales > 1:
            self.view_projs = nn.ModuleList(
                [nn.Linear(vision_dim, D) for _ in range(self.num_scales)]
            )
        else:
            self.view_proj = nn.Linear(vision_dim, D)

        # Action encoder
        self.action_proj = nn.Sequential(nn.Linear(config.action_dim, D), nn.SiLU())
        self.action_time_fuse = nn.Sequential(
            nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.action_pos = nn.Parameter(torch.randn(1, config.num_action_tokens, D) * 0.02)

        # Plan encoder (if use_plan)
        if config.use_plan:
            self.plan_encoder = nn.Sequential(
                nn.Linear(config.plan_dim, D), nn.SiLU(), nn.Linear(D, D),
            )
            self.plan_pos = nn.Parameter(torch.randn(1, config.num_plan_tokens, D) * 0.02)

        # DiT blocks (interleaved cross/self)
        self.blocks_self = nn.ModuleList()
        self.blocks_cross = nn.ModuleList()
        for i in range(config.num_blocks):
            if i % 2 == 0:
                self.blocks_cross.append(
                    DiTCrossAttnBlock(D, config.num_heads, config.mlp_ratio, config.dropout)
                )
            else:
                self.blocks_self.append(
                    DiTSelfAttnBlock(D, config.num_heads, config.mlp_ratio, config.dropout)
                )

        # Modulated output layer
        self.norm_out = nn.LayerNorm(D, elementwise_affine=False)
        self.proj_out_mod = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        self.proj_out = nn.Linear(D, D)

        # Output heads
        self.action_head = nn.Linear(D, config.action_dim)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

        if config.use_plan:
            self.plan_head = nn.Linear(D, config.plan_dim)
            nn.init.zeros_(self.plan_head.weight)
            nn.init.zeros_(self.plan_head.bias)

        # Blockwise attention mask (cached)
        self._attn_mask: Optional[torch.Tensor] = None

        # Timestep sampling distribution
        self._beta_dist = Beta(config.beta_alpha, config.beta_beta)

    def _get_attn_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        """Build blockwise mask: plan cannot see action tokens."""
        if not self.config.use_plan:
            return None
        if hasattr(self.config, "plan_action_causal") and not self.config.plan_action_causal:
            return None

        if self._attn_mask is None or self._attn_mask.device != device:
            n_t = 1
            n_s = 1
            n_p = self.config.num_plan_tokens
            n_a = self.config.num_action_tokens
            total = n_t + n_s + n_p + n_a
            mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            plan_start = n_t + n_s
            action_start = plan_start + n_p
            mask[plan_start:action_start, action_start:] = True
            self._attn_mask = mask
        return self._attn_mask

    def _build_condition(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_embed(t)

    def _encode_tokens(
        self,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        t_embed: torch.Tensor,
        text_feat: torch.Tensor,
        noisy_plan: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode and concatenate token sequence."""
        text_tok = self.text_proj(text_feat).unsqueeze(1)   # (B, 1, D)
        state_tok = self.state_encoder(state).unsqueeze(1)  # (B, 1, D)

        a_proj = self.action_proj(noisy_action)
        t_exp = t_embed.unsqueeze(1).expand(-1, self.config.num_action_tokens, -1)
        action_tok = self.action_time_fuse(torch.cat([a_proj, t_exp], dim=-1))
        action_tok = action_tok + self.action_pos

        if self.config.use_plan and noisy_plan is not None:
            plan_tok = self.plan_encoder(noisy_plan) + self.plan_pos
            return torch.cat([text_tok, state_tok, plan_tok, action_tok], dim=1)
        else:
            return torch.cat([text_tok, state_tok, action_tok], dim=1)

    def _build_kv(
        self,
        view_features: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project view features for KV.

        Args:
            view_mask: (B, N_view) bool, True = masked (dropout).

        Returns:
            kv: (B, N, D)
            kv_key_padding_mask: (B, N) bool or None (True = masked)
        """
        kv = self.view_proj(view_features)
        return kv, view_mask

    def _build_kv_multiscale(
        self,
        scale_features: list,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list, Optional[torch.Tensor]]:
        if len(scale_features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} vision scales, got {len(scale_features)}"
            )
        kv_scales = [
            projection(features)
            for projection, features in zip(self.view_projs, scale_features)
        ]
        return kv_scales, view_mask

    def _decode_output(
        self, x: torch.Tensor, condition: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Modulated output layer -> plan + action heads."""
        mod = self.proj_out_mod(condition)
        scale, shift = mod.chunk(2, dim=-1)
        x = self.norm_out(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out(x)

        if self.config.use_plan:
            n_t = 1
            n_s = 1
            n_p = self.config.num_plan_tokens
            plan_start = n_t + n_s
            action_start = plan_start + n_p
            plan_out = self.plan_head(x[:, plan_start:action_start])
            action_out = self.action_head(x[:, action_start:])
            return plan_out, action_out
        else:
            action_start = 2  # skip text(1) + state(1)
            action_out = self.action_head(x[:, action_start:])
            return None, action_out

    def _run_dit(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        kv: Optional[torch.Tensor],
        kv_key_padding_mask: Optional[torch.Tensor] = None,
        kv_scales: Optional[list] = None,
    ) -> torch.Tensor:
        attn_mask = self._get_attn_mask(tokens.device)
        cross_idx = 0
        self_idx = 0
        for i in range(self.config.num_blocks):
            if i % 2 == 0:
                if kv_scales is not None:
                    block_kv = kv_scales[cross_idx % self.num_scales]
                else:
                    if kv is None:
                        raise ValueError("Cross-attention requires kv or kv_scales")
                    block_kv = kv
                tokens = self.blocks_cross[cross_idx](
                    tokens,
                    condition,
                    block_kv,
                    kv_key_padding_mask=kv_key_padding_mask,
                )
                cross_idx += 1
            else:
                tokens = self.blocks_self[self_idx](tokens, condition, attn_mask)
                self_idx += 1
        return tokens

    def _goal_xyz_consistency_loss(
        self,
        pred_plan_x1: torch.Tensor,
        state: torch.Tensor,
        goal_group_index: torch.Tensor,
        goal_ref_index: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match predicted final goal xyz to a per-group reference sample."""
        device = pred_plan_x1.device
        group_idx = goal_group_index.long().to(device)
        valid = group_idx >= 0
        if sample_mask is not None:
            valid = valid & sample_mask.bool().to(device)
        if not bool(valid.any()):
            return torch.zeros((), device=device, dtype=pred_plan_x1.dtype)

        pred_goal_xyz = state[:, :3].float() + pred_plan_x1[:, -1, :3].float()
        if goal_ref_index is not None:
            ref = goal_ref_index.long().to(device)
            B = int(pred_goal_xyz.shape[0])
            valid = valid & (ref >= 0) & (ref < B)
            if sample_mask is not None:
                sm = sample_mask.bool().to(device)
                valid = valid & sm[ref]
            idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return torch.zeros((), device=device, dtype=pred_plan_x1.dtype)
            ref_idx = ref[idx]
            non_self = idx != ref_idx
            if not bool(non_self.any()):
                return torch.zeros((), device=device, dtype=pred_plan_x1.dtype)
            src = pred_goal_xyz[idx[non_self]]
            tgt = pred_goal_xyz[ref_idx[non_self]]
            return F.smooth_l1_loss(src, tgt, reduction="none").mean()

        losses = []
        for gid in torch.unique(group_idx[valid]):
            idx = (group_idx == gid) & valid
            if int(idx.sum().item()) < 2:
                continue
            pts = pred_goal_xyz[idx]
            n = int(pts.shape[0])
            pi, pj = torch.triu_indices(n, n, offset=1, device=device)
            diffs = pts[pi] - pts[pj]
            l = F.smooth_l1_loss(diffs, torch.zeros_like(diffs), reduction="none").mean()
            losses.append(l)
        if len(losses) == 0:
            return torch.zeros((), device=device, dtype=pred_plan_x1.dtype)
        return torch.stack(losses).mean()

    def forward(
        self,
        text_feat: torch.Tensor,
        state: torch.Tensor,
        gt_action: torch.Tensor,
        view_features: torch.Tensor,
        gt_plan: Optional[torch.Tensor] = None,
        action_loss_mask: Optional[torch.Tensor] = None,
        plan_loss_mask: Optional[torch.Tensor] = None,
        goal_group_index: Optional[torch.Tensor] = None,
        goal_ref_index: Optional[torch.Tensor] = None,
        action_dim_mask: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        scale_features: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward: flow matching loss."""
        B = state.shape[0]
        device = state.device

        # Sample timestep
        t = self._beta_dist.sample((B,)).float()
        t = t * self.config.noise_s
        t = t.to(device)

        # Flow matching noise
        noise_action = torch.randn_like(gt_action)
        noisy_action = (1.0 - t.view(B, 1, 1)) * noise_action + t.view(B, 1, 1) * gt_action
        vel_action = gt_action - noise_action

        noisy_plan = None
        vel_plan = None
        if self.config.use_plan and gt_plan is not None:
            noise_plan = torch.randn_like(gt_plan)
            noisy_plan = (1.0 - t.view(B, 1, 1)) * noise_plan + t.view(B, 1, 1) * gt_plan
            vel_plan = gt_plan - noise_plan

        condition = self._build_condition(t)
        tokens = self._encode_tokens(noisy_action, state, condition, text_feat, noisy_plan)
        if scale_features is not None:
            kv_scales, kv_mask = self._build_kv_multiscale(
                scale_features, view_mask=view_mask
            )
            tokens = self._run_dit(
                tokens,
                condition,
                kv=None,
                kv_key_padding_mask=kv_mask,
                kv_scales=kv_scales,
            )
        else:
            kv, kv_mask = self._build_kv(view_features, view_mask=view_mask)
            tokens = self._run_dit(tokens, condition, kv, kv_key_padding_mask=kv_mask)
        pred_vel_plan, pred_vel_action = self._decode_output(tokens, condition)

        # Reconstruct x1 estimate
        x1_hat_action = noisy_action + (1.0 - t.view(B, 1, 1)) * pred_vel_action
        x1_hat_plan = None
        if self.config.use_plan and noisy_plan is not None and pred_vel_plan is not None:
            x1_hat_plan = noisy_plan + (1.0 - t.view(B, 1, 1)) * pred_vel_plan

        # Action loss
        action_sq_err = (pred_vel_action - vel_action).pow(2)
        if action_dim_mask is not None:
            action_sq_err = action_sq_err * action_dim_mask.unsqueeze(1)
            valid_count = action_dim_mask.unsqueeze(1).expand_as(action_sq_err).sum(dim=(1, 2)).clamp_min(1.0)
            action_per_sample = action_sq_err.sum(dim=(1, 2)) / valid_count
        else:
            action_per_sample = action_sq_err.mean(dim=(1, 2))
        if action_loss_mask is not None:
            m = action_loss_mask.float().to(device)
            action_loss = (action_per_sample * m).sum() / m.sum().clamp_min(1.0)
        else:
            action_loss = action_per_sample.mean()

        result: Dict[str, torch.Tensor] = {"action_loss": action_loss}

        # Plan loss
        if self.config.use_plan and vel_plan is not None and pred_vel_plan is not None:
            plan_per_sample = (pred_vel_plan - vel_plan).pow(2).mean(dim=(1, 2))
            if plan_loss_mask is not None:
                m = plan_loss_mask.float().to(device)
                plan_loss = (plan_per_sample * m).sum() / m.sum().clamp_min(1.0)
            else:
                plan_loss = plan_per_sample.mean()
            result["plan_loss"] = plan_loss
            result["dit_loss"] = plan_loss + action_loss

            goal_weight = float(getattr(self.config, "goal_xyz_consistency_weight", 0.0))
            if goal_weight > 0.0 and x1_hat_plan is not None and goal_group_index is not None:
                goal_mask = plan_loss_mask.bool() if plan_loss_mask is not None else None
                goal_xyz_consistency_loss = self._goal_xyz_consistency_loss(
                    pred_plan_x1=x1_hat_plan,
                    state=state,
                    goal_group_index=goal_group_index,
                    goal_ref_index=goal_ref_index,
                    sample_mask=goal_mask,
                )
                result["goal_xyz_consistency_loss"] = goal_xyz_consistency_loss
                result["dit_loss"] = result["dit_loss"] + goal_weight * goal_xyz_consistency_loss
        else:
            result["dit_loss"] = action_loss

        return result

    @torch.no_grad()
    def predict(
        self,
        text_feat: torch.Tensor,
        state: torch.Tensor,
        view_features: torch.Tensor,
        num_steps: Optional[int] = None,
        view_mask: Optional[torch.Tensor] = None,
        scale_features: Optional[list] = None,
        sampling_generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate action (and plan) via flow matching Euler integration.

        ``sampling_generator`` controls only this request's initial action and
        plan noise.  A caller that needs rollout determinism should pass a
        request-local generator instead of mutating torch's global RNG.
        """
        B = state.shape[0]
        device = state.device
        num_steps = num_steps or self.config.num_inference_steps

        if scale_features is not None:
            kv_scales, kv_mask = self._build_kv_multiscale(
                scale_features, view_mask=view_mask
            )
            kv = None
        else:
            kv, kv_mask = self._build_kv(view_features, view_mask=view_mask)
            kv_scales = None

        action = torch.randn(
            B,
            self.config.num_action_tokens,
            self.config.action_dim,
            device=device,
            generator=sampling_generator,
        )
        plan = None
        if self.config.use_plan:
            plan = torch.randn(
                B,
                self.config.num_plan_tokens,
                self.config.plan_dim,
                device=device,
                generator=sampling_generator,
            )

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps
            t_tensor = torch.full((B,), t_val, device=device)

            condition = self._build_condition(t_tensor)
            tokens = self._encode_tokens(action, state, condition, text_feat, plan)
            tokens = self._run_dit(
                tokens,
                condition,
                kv,
                kv_key_padding_mask=kv_mask,
                kv_scales=kv_scales,
            )
            vel_plan, vel_action = self._decode_output(tokens, condition)

            action = action + vel_action * dt
            if plan is not None and vel_plan is not None:
                plan = plan + vel_plan * dt

        result = {"action": action}
        if plan is not None:
            result["plan"] = plan
        return result
