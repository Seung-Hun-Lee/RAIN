"""Packed DINO/image loaders and cached CLIP text features."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from shared.clip_utils import (
    DEFAULT_CLIP_TEXT_MODEL,
    get_clip_text_feature_dim,
    normalize_clip_model_name,
)

logger = logging.getLogger(__name__)


class PackedDINO:
    def __init__(self, packed_dir: str):
        p = Path(packed_dir)
        with open(p / "dino_index.json") as f:
            self.index = json.load(f)
        N = len(self.index)
        npy_path = p / "dino_packed.npy"
        meta_path = p / "dino_meta.json"
        hidden_dim = 1024
        num_tokens = None
        num_scales = 1
        multi_scale = False
        patch_only = False
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("completed") is False:
                    raise RuntimeError(f"Incomplete feature extraction: {packed_dir}")
                hidden_dim = int(meta.get("hidden_dim", hidden_dim))
                if meta.get("num_tokens") is not None:
                    num_tokens = int(meta["num_tokens"])
                num_scales = int(meta.get("num_scales", 1))
                multi_scale = bool(meta.get("multi_scale", False))
                patch_only = bool(meta.get("patch_only", multi_scale))
            except (ValueError, OSError):
                pass

        if num_tokens is None and not multi_scale:
            file_bytes = npy_path.stat().st_size
            num_tokens = file_bytes // (N * hidden_dim * 2)
        if num_tokens is None:
            raise ValueError(f"Missing num_tokens in multi-scale metadata: {meta_path}")

        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.num_scales = int(num_scales)
        self.patch_only = patch_only
        self.multi_scale = bool(multi_scale and self.num_scales > 1)
        if self.multi_scale:
            self.data = None
            self.scale_data = []
            for scale_index in range(self.num_scales):
                scale_path = p / f"dino_packed_s{scale_index + 1}.npy"
                if not scale_path.is_file():
                    raise FileNotFoundError(scale_path)
                self.scale_data.append(
                    np.memmap(
                        str(scale_path),
                        dtype="float16",
                        mode="r",
                        shape=(N, self.num_tokens, self.hidden_dim),
                    )
                )
            logger.info(
                "PackedDINO multi-scale: %d entries, %d scales, %d tokens, hidden_dim=%d",
                N,
                self.num_scales,
                self.num_tokens,
                self.hidden_dim,
            )
        else:
            self.scale_data = None
            self.data = np.memmap(
                str(npy_path),
                dtype="float16",
                mode="r",
                shape=(N, self.num_tokens, self.hidden_dim),
            )
            logger.info(
                "PackedDINO: %d entries, %d tokens/entry, hidden_dim=%d",
                N, self.num_tokens, self.hidden_dim,
            )

    def get(self, eidx: int, frame: int, view: str) -> Optional[torch.Tensor]:
        idx = self.index.get(f"{eidx:04d}_{frame:04d}_{view}")
        if idx is None:
            return None
        if self.multi_scale:
            return torch.from_numpy(self.scale_data[-1][idx].copy())
        return torch.from_numpy(self.data[idx].copy())

    def get_multiscale(
        self, eidx: int, frame: int, view: str
    ) -> Optional[torch.Tensor]:
        if not self.multi_scale:
            raise RuntimeError("get_multiscale called on a single-scale cache")
        idx = self.index.get(f"{eidx:04d}_{frame:04d}_{view}")
        if idx is None:
            return None
        return torch.stack(
            [torch.from_numpy(scale[idx].copy()) for scale in self.scale_data],
            dim=0,
        )


class PackedImages:
    """Read pre-resized RGB observations from a uint8 memmap.

    The index contract intentionally matches :class:`PackedDINO`, so online
    frozen-DINO training can replace feature I/O without changing episode or
    frame selection.
    """

    def __init__(self, packed_dir: str):
        p = Path(packed_dir)
        with open(p / "images_index.json") as f:
            self.index = json.load(f)
        with open(p / "images_meta.json") as f:
            meta = json.load(f)

        self.input_size = int(meta["input_size"])
        num_entries = int(meta.get("num_entries", len(self.index)))
        if num_entries != len(self.index):
            raise ValueError(
                f"PackedImages index/meta mismatch: {len(self.index)} vs {num_entries}"
            )
        self.data = np.memmap(
            str(p / "images_packed.npy"),
            dtype="uint8",
            mode="r",
            shape=(num_entries, self.input_size, self.input_size, 3),
        )
        logger.info(
            "PackedImages: %d entries, %dx%d RGB",
            num_entries,
            self.input_size,
            self.input_size,
        )

    def get(self, eidx: int, frame: int, view: str) -> Optional[torch.Tensor]:
        idx = self.index.get(f"{eidx:04d}_{frame:04d}_{view}")
        if idx is None:
            return None
        image = torch.from_numpy(self.data[idx].copy()).permute(2, 0, 1)
        return image.float().div_(255.0)


class TextFeatureCache:
    def __init__(
        self,
        cache_path: Optional[str] = None,
        model_name: str = DEFAULT_CLIP_TEXT_MODEL,
        feature_dim: Optional[int] = None,
    ):
        self.features: Dict[str, np.ndarray] = {}
        self.model_name = normalize_clip_model_name(model_name)
        self.feature_dim = int(
            get_clip_text_feature_dim(self.model_name)
            if feature_dim is None else feature_dim
        )
        if cache_path and Path(cache_path).exists():
            npz = np.load(cache_path)
            for k in npz.files:
                arr = np.asarray(npz[k], dtype=np.float32)
                if arr.ndim == 1 and int(arr.shape[0]) == self.feature_dim:
                    self.features[k] = arr

    def get(self, desc: str) -> np.ndarray:
        return self.features.get(desc, np.zeros(self.feature_dim, dtype=np.float32))

    def has_features(self) -> bool:
        return len(self.features) > 0

    def precompute(self, descriptions: List[str], save: bool = True, cache_path: str = "") -> None:
        unique = list(set(descriptions))
        try:
            from transformers import CLIPModel, CLIPTokenizer
            tok = CLIPTokenizer.from_pretrained(self.model_name)
            model = CLIPModel.from_pretrained(
                self.model_name, attn_implementation="sdpa"
            ).eval()
            for i in range(0, len(unique), 64):
                batch = unique[i:i+64]
                inputs = tok(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
                with torch.no_grad():
                    out = model.get_text_features(**inputs)
                    out = out / out.norm(dim=-1, keepdim=True)
                for j, d in enumerate(batch):
                    self.features[d] = out[j].numpy().astype(np.float32)
            del model, tok
            if save and cache_path:
                np.savez_compressed(cache_path, **self.features)
        except Exception as e:
            logger.warning("CLIP failed: %s", e)
            for d in unique:
                self.features[d] = np.zeros(self.feature_dim, dtype=np.float32)
