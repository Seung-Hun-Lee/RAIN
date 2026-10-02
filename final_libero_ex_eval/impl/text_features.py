"""CLIP text-feature cache for LIBERO-Analogy evaluation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from final_libero_ex_eval.impl.clip_utils import (
    DEFAULT_CLIP_TEXT_MODEL,
    get_clip_text_feature_dim,
    normalize_clip_model_name,
)

logger = logging.getLogger(__name__)


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
                self.model_name, attn_implementation="eager"
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
