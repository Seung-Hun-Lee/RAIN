"""Shared model components.

FrozenDINOv2, timestep embeddings, attention blocks, state/text encoders.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.dinov2 import load_dinov2

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Frozen DINOv2 backbone
# ---------------------------------------------------------------------------

class FrozenDINOv2(nn.Module):
    """Frozen DINOv2 ViT-L/14 backbone.

    Input:  (B, 3, H, W) float [0, 1]
    Output: (B, 1 + patches, 1024) = CLS + patch tokens.

    Uses the recorded public DINOv2 source and the configured Torch Hub cache.
    """

    def __init__(self, input_size: int = 448, dino_model: str = "dinov2_vitl14_reg"):
        super().__init__()
        self.input_size = int(input_size)
        self.dino_model = str(dino_model)
        self.backbone = load_dinov2(self.dino_model)
        self.register_buffer(
            "_pixel_mean",
            torch.tensor(DINO_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_pixel_std",
            torch.tensor(DINO_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        x = (x - self._pixel_mean) / self._pixel_std
        feat = self.backbone.forward_features(x)
        cls_tok = feat["x_norm_clstoken"].unsqueeze(1)
        pat_tok = feat["x_norm_patchtokens"]
        return torch.cat([cls_tok, pat_tok], dim=1)

    def train(self, mode: bool = True):
        super().train(False)
        return self


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive Layer Normalization (timestep-modulated)
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, hidden_dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)
        if params.dim() == 2:
            params = params.unsqueeze(1)
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# GEGLU Feed-Forward
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult * 2)  # *2 for GEGLU split
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim // 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# State / Text Encoders
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int = 7, hidden_dim: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


class TextProjector(nn.Module):
    def __init__(self, text_dim: int = 768, hidden_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_dim)

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(text_feat)


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(256),
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)
