"""Vision encoders for RAIN.

TargetAdaptiveCrossViewEncoder (TCE) orchestrates:
  ThirdConditioner  -> project + TarLN mask modulation
  WristConditioner  -> project + wrist conditioning
  EncoderBlock[]    -> cross-attn + self-attn + FFN (per view)

cross_attn_mode controls which views receive cross-attention:
  "bidirectional":   both views cross-attend to each other
  "third_to_wrist":  only wrist cross-attends to third
  "wrist_to_third":  only third cross-attends to wrist
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rainv2.configs.config import ThirdEncoderConfig, WristEncoderConfig


# ---------------------------------------------------------------------------
# Target-adaptive normalization modules
# ---------------------------------------------------------------------------


class TargetAdaptiveLayerNorm(nn.Module):
    """Target-adaptive Layer Normalization (TarLN).

    mask -> Embedding -> reshape 2D -> Conv2d(1x1) -> (gamma, beta).
    Applied as either:
      - (1 + gamma) * LN(feat) + beta for identity-init modulation
      - gamma * LN(feat) + beta for additive zero-init modulation
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int = 32,
        identity_scale: bool = True,
    ) -> None:
        super().__init__()
        self.identity_scale = bool(identity_scale)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mask_embed = nn.Embedding(2, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim * 2, hidden_dim * 2, kernel_size=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feat: torch.Tensor, mask_binary: torch.Tensor) -> torch.Tensor:
        B, P, D = feat.shape
        G = int(math.sqrt(P))

        m = self.mask_embed(mask_binary)                          # (B, P, E)
        m = m.permute(0, 2, 1).reshape(B, -1, G, G)              # (B, E, G, G)
        mod = self.conv(m)                                         # (B, D*2, G, G)
        mod = mod.reshape(B, D * 2, P).permute(0, 2, 1)           # (B, P, D*2)
        gamma, beta = mod.chunk(2, dim=-1)

        norm_feat = self.norm(feat)
        if self.identity_scale:
            return (1.0 + gamma) * norm_feat + beta
        return gamma * norm_feat + beta


TarLN = TargetAdaptiveLayerNorm


class SpatialCorrFiLM(nn.Module):
    """SPADE-style spatial FiLM from correspondence map via 1x1 conv.

    corr_map (B, 256) -> reshape (B, 1, 16, 16)
    -> Conv2d(1, E, 1x1) -> SiLU -> Conv2d(E, D*2, 1x1) -> (gamma, beta).
    Applied as: (1 + gamma) * LN(feat) + beta.
    Zero-init last conv so initial behaviour = LN(feat).
    """

    def __init__(self, hidden_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.conv = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(embed_dim, hidden_dim * 2, kernel_size=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feat: torch.Tensor, corr_map: torch.Tensor) -> torch.Tensor:
        B, P, D = feat.shape
        G = int(math.sqrt(P))

        c = corr_map.reshape(B, 1, G, G)
        mod = self.conv(c)
        mod = mod.reshape(B, D * 2, P).permute(0, 2, 1)
        gamma, beta = mod.chunk(2, dim=-1)

        return (1.0 + gamma) * self.norm(feat) + beta


# ---------------------------------------------------------------------------
# Correspondence map
# ---------------------------------------------------------------------------


def compute_correspondence_map(
    dino_third_raw: torch.Tensor,   # (B, 256, 1024) spatial patches
    dino_wrist_raw: torch.Tensor,   # (B, 256, 1024) spatial patches
    mask_binary: torch.Tensor,      # (B, 256) long
) -> torch.Tensor:
    """Compute per-wrist-patch correspondence scores from raw DINO features.

    For each wrist patch, find the max cosine similarity to any masked
    third-view patch. Non-object third patches are masked out (-inf).
    Negative similarities are clamped to 0.
    """
    third_patches = dino_third_raw
    wrist_patches = dino_wrist_raw

    third_norm = F.normalize(third_patches, dim=-1)
    wrist_norm = F.normalize(wrist_patches, dim=-1)

    # Cosine similarity: (B, 256_wrist, 256_third)
    sim = torch.bmm(wrist_norm, third_norm.transpose(1, 2))

    # Mask out non-object third patches
    obj_mask = mask_binary.bool().unsqueeze(1)  # (B, 1, 256)
    sim = sim.masked_fill(~obj_mask, float("-inf"))

    # Max across masked third patches
    corr_map = sim.max(dim=-1).values  # (B, 256)

    # Handle all-zero mask: replace -inf with 0
    corr_map = corr_map.clamp(min=0.0)

    return corr_map


# ---------------------------------------------------------------------------
# Unified encoder block
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Module):
    """Unified block: optional cross-attn + self-attn + FFN.

    When use_cross_attn=True, creates cross-attention layers. The cross-attn
    is applied before self-attn when cross_kv is provided.

    Flow: [cross-attn(x, cross_kv)] -> self-attn(x) -> FFN(x)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn

        if use_cross_attn:
            self.cross_ln_q = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_ln_kv = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
            )

        self.self_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )

        ffn_dim = int(hidden_dim * mlp_ratio)
        self.ffn_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cross_kv: torch.Tensor = None) -> torch.Tensor:
        if self.use_cross_attn and cross_kv is not None:
            q = self.cross_ln_q(x)
            kv = self.cross_ln_kv(cross_kv)
            x = x + self.cross_attn(q, kv, kv)[0]

        h = self.self_ln(x)
        x = x + self.self_attn(h, h, h)[0]

        x = x + self.ffn(self.ffn_ln(x))
        return x


# ---------------------------------------------------------------------------
# Conditioners
# ---------------------------------------------------------------------------


class ThirdConditioner(nn.Module):
    """Project DINO patches + apply TarLN + optional target-place TarLN."""

    def __init__(self, config: ThirdEncoderConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        self.dino_proj = nn.Sequential(
            nn.LayerNorm(config.dino_dim, elementwise_affine=False),
            nn.Linear(config.dino_dim, D),
            nn.LayerNorm(D, elementwise_affine=False),
        )
        self.mask_film = TargetAdaptiveLayerNorm(
            D, config.film_embed_dim, identity_scale=True
        )

        self.use_target_place = getattr(config, "use_target_place", False)
        self.zero_view_masks = bool(getattr(config, "zero_view_masks", False))
        if self.use_target_place:
            self.place_film = TargetAdaptiveLayerNorm(
                D, config.film_embed_dim, identity_scale=False
            )

    def forward(
        self,
        dino_third: torch.Tensor,
        goal_mask: torch.Tensor,
        place_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cond_third: (B, 256, D) TarLN-conditioned patches
            mask_binary: (B, 256) long
        """
        mask_binary = (goal_mask > 0.5).long()
        if self.zero_view_masks:
            mask_binary = torch.zeros_like(mask_binary)
        patches = self.dino_proj(dino_third)  # (B, 256, D)
        cond = self.mask_film(patches, mask_binary)

        # Additive target place embedding (zero-init → starts as +0)
        if self.use_target_place:
            if self.zero_view_masks:
                place_binary = torch.zeros_like(mask_binary)
            elif place_mask is not None:
                place_binary = (place_mask > 0.5).long()
            else:
                place_binary = None
            if place_binary is not None:
                place_mod = self.place_film(cond, place_binary)
                cond = cond + place_mod

        return cond, mask_binary


class WristConditioner(nn.Module):
    """Project DINO patches + apply corr/mask/none wrist conditioning."""

    def __init__(self, config: WristEncoderConfig) -> None:
        super().__init__()
        mode = str(getattr(config, "conditioning_mode", "") or "").strip().lower()
        if not mode:
            mode = "corr" if bool(getattr(config, "use_corr_modulation", True)) else "none"
        if mode not in {"corr", "mask", "none"}:
            raise ValueError(
                f"Unsupported wrist conditioning_mode={mode!r}; expected one of: corr, mask, none"
            )
        self.conditioning_mode = mode
        D = config.hidden_dim
        self.dino_proj = nn.Sequential(
            nn.LayerNorm(config.dino_dim, elementwise_affine=False),
            nn.Linear(config.dino_dim, D),
            nn.LayerNorm(D, elementwise_affine=False),
        )
        self.corr_film = SpatialCorrFiLM(D, config.film_embed_dim) if mode == "corr" else None
        self.mask_film = (
            TargetAdaptiveLayerNorm(D, config.film_embed_dim, identity_scale=True)
            if mode == "mask" else None
        )
        self.use_target_place = bool(getattr(config, "use_target_place", False))
        self.zero_view_masks = bool(getattr(config, "zero_view_masks", False))
        if mode == "mask" and self.use_target_place:
            self.place_film = TargetAdaptiveLayerNorm(
                D, config.film_embed_dim, identity_scale=False
            )

    def forward(
        self,
        dino_wrist: torch.Tensor,
        dino_third_raw: torch.Tensor,
        mask_binary: torch.Tensor,
        goal_mask_wrist: Optional[torch.Tensor] = None,
        place_mask_wrist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cond_wrist: (B, 256, D) conditioned patches
            corr_map: (B, 256) correspondence scores, or zeros for non-corr modes
        """
        patches = self.dino_proj(dino_wrist)  # (B, 256, D)
        if self.conditioning_mode == "corr":
            corr_map = compute_correspondence_map(dino_third_raw, dino_wrist, mask_binary)
            cond = self.corr_film(patches, corr_map)
        else:
            corr_map = torch.zeros(
                patches.shape[0], patches.shape[1],
                device=patches.device, dtype=patches.dtype,
            )

        if self.conditioning_mode == "mask":
            if self.zero_view_masks or goal_mask_wrist is None:
                goal_mask_wrist = torch.zeros_like(corr_map)
            wrist_mask_binary = (goal_mask_wrist > 0.5).long()
            cond = self.mask_film(patches, wrist_mask_binary)
            if self.use_target_place:
                if self.zero_view_masks:
                    place_binary = torch.zeros_like(wrist_mask_binary)
                elif place_mask_wrist is not None:
                    place_binary = (place_mask_wrist > 0.5).long()
                else:
                    place_binary = None
                if place_binary is not None:
                    place_mod = self.place_film(cond, place_binary)
                    cond = cond + place_mod
        elif self.conditioning_mode == "none":
            cond = patches
        return cond, corr_map


# ---------------------------------------------------------------------------
# Target-adaptive Cross-view Encoder
# ---------------------------------------------------------------------------

_VALID_CROSS_ATTN_MODES = {"bidirectional", "third_to_wrist", "wrist_to_third"}


class TargetAdaptiveCrossViewEncoder(nn.Module):
    """Target-adaptive Cross-view Encoder (TCE).

    Orchestrates both view conditioners and cross-view encoder blocks.

    cross_attn_mode controls which views get cross-attention blocks:
      bidirectional:   third blocks have cross-attn, wrist blocks have cross-attn
      third_to_wrist:  third blocks plain,           wrist blocks have cross-attn
      wrist_to_third:  third blocks have cross-attn, wrist blocks plain
    """

    def __init__(
        self,
        third_cfg: ThirdEncoderConfig,
        wrist_cfg: WristEncoderConfig,
        cross_attn_mode: str = "bidirectional",
    ) -> None:
        super().__init__()

        cross_attn_mode = str(cross_attn_mode).strip().lower()
        if cross_attn_mode not in _VALID_CROSS_ATTN_MODES:
            raise ValueError(
                f"Unsupported cross_attn_mode={cross_attn_mode!r}; "
                f"expected one of: {_VALID_CROSS_ATTN_MODES}"
            )
        self.cross_attn_mode = cross_attn_mode

        use_third_cross = cross_attn_mode in ("bidirectional", "wrist_to_third")
        use_wrist_cross = cross_attn_mode in ("bidirectional", "third_to_wrist")
        self.use_third_cross = use_third_cross
        self.use_wrist_cross = use_wrist_cross
        third_use_blocks = bool(getattr(third_cfg, "use_transformer_blocks", True))
        wrist_use_blocks = bool(getattr(wrist_cfg, "use_transformer_blocks", True))
        if third_use_blocks != wrist_use_blocks:
            raise ValueError(
                "TargetAdaptiveCrossViewEncoder requires third/wrist "
                "use_transformer_blocks to match; "
                f"got third={third_use_blocks}, wrist={wrist_use_blocks}."
            )
        self.use_transformer_blocks = third_use_blocks

        if (
            (use_third_cross or use_wrist_cross)
            and int(third_cfg.hidden_dim) != int(wrist_cfg.hidden_dim)
        ):
            raise ValueError(
                "Cross-attention requires matching hidden_dim for third and wrist encoders, "
                f"got {third_cfg.hidden_dim} vs {wrist_cfg.hidden_dim}."
            )

        self.third_cond = ThirdConditioner(third_cfg)
        self.wrist_cond = WristConditioner(wrist_cfg)

        D = int(third_cfg.hidden_dim)
        num_heads = int(third_cfg.num_heads)
        mlp_ratio = float(third_cfg.mlp_ratio)
        num_blocks = int(third_cfg.num_blocks)

        if self.use_transformer_blocks:
            self.third_blocks = nn.ModuleList([
                EncoderBlock(D, num_heads, mlp_ratio, use_cross_attn=use_third_cross)
                for _ in range(num_blocks)
            ])
            self.wrist_blocks = nn.ModuleList([
                EncoderBlock(D, num_heads, mlp_ratio, use_cross_attn=use_wrist_cross)
                for _ in range(num_blocks)
            ])
        else:
            self.third_blocks = nn.ModuleList()
            self.wrist_blocks = nn.ModuleList()

    def forward(
        self,
        dino_third: torch.Tensor,   # (B, 256, 1024) spatial patches
        dino_wrist: torch.Tensor,   # (B, 256, 1024) spatial patches
        goal_mask: torch.Tensor,    # (B, 256) float
        goal_mask_wrist: Optional[torch.Tensor] = None,  # (B, 256) float
        place_mask: Optional[torch.Tensor] = None,  # (B, 256) float
        place_mask_wrist: Optional[torch.Tensor] = None,  # (B, 256) float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            third_out: (B, 256, D)
            wrist_out: (B, 256, D)
            corr_map:  (B, 256)
        """
        cond_third, mask_binary = self.third_cond(dino_third, goal_mask, place_mask)
        cond_wrist, corr_map = self.wrist_cond(
            dino_wrist,
            dino_third,
            mask_binary,
            goal_mask_wrist=goal_mask_wrist,
            place_mask_wrist=place_mask_wrist,
        )

        if not self.use_transformer_blocks:
            return cond_third, cond_wrist, corr_map

        third_out, wrist_out = cond_third, cond_wrist
        for t_blk, w_blk in zip(self.third_blocks, self.wrist_blocks):
            third_out = t_blk(
                third_out,
                cross_kv=wrist_out if self.use_third_cross else None,
            )
            wrist_out = w_blk(
                wrist_out,
                cross_kv=third_out if self.use_wrist_cross else None,
            )

        return third_out, wrist_out, corr_map
