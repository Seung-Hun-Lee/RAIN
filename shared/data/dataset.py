"""RAIN unified dataset.

Subtask-aware sampling with 7D state, progress labels, plan waypoints,
action chunks, and optional reverse augmentation. Progress labels can use
TC event annotations or confident gripper-state classification.
"""

import hashlib
import json
import logging
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_TYPE_MAP = {
    "grasp": 0, "release": 1, "push": 2,
    "turn_on": 3, "close": 4, "open": 5,
    "approach": 6,
}

SAMPLE_MODE_NORMAL = 0
SAMPLE_MODE_REVERSE = 1
SAMPLE_MODE_RETARGET = 2

DINO_PATCH_GRID = 16
DINO_NUM_PATCHES = DINO_PATCH_GRID * DINO_PATCH_GRID  # 256


def dino_patch_grid_from_tokens(n_tokens: int, *, patch_only: bool = False) -> int:
    """Compute the square DINO patch grid from a cached token count."""
    n_patches = int(n_tokens) if patch_only else int(n_tokens) - 1
    grid = int(math.isqrt(n_patches))
    if grid * grid != n_patches:
        token_kind = "patch-only" if patch_only else "CLS+patch"
        raise ValueError(
            f"Invalid {token_kind} DINO token count: {n_tokens} "
            f"does not describe a square patch grid"
        )
    return grid

# Per-axis controller gain: EEF delta → action command
# Empirically measured across 1693 episodes (std <15%)
# Physical meaning: 1/(OSC_scale × PD_tracking_ratio) ≈ 1/(0.05 × 0.25) ≈ 80
CONTROLLER_XYZ_GAIN = np.array([81.24, 76.36, 79.60], dtype=np.float32)
CONTROLLER_XYZ_SCALE_MEAN = float(np.mean(1.0 / CONTROLLER_XYZ_GAIN))  # ~0.0127

# Task Completion thresholds
CONTACT_DIST_THRESH = 0.02  # 2cm

# Confident GT criteria for grasp/release
CONF_COMPLETED_POS = 0.20
CONF_NOT_POS = 0.80
STABILITY_WINDOW = 3
STABILITY_THRESH = 0.10
MIN_RANGE = 0.006
POST_FRAMES_RANGE = 15

# Rotation-based thresholds for turn_on
TURNON_ROT_LOW = 0.3    # below → tc=0
TURNON_ROT_HIGH = 0.95  # above → tc=1
# Uncertain tail for push/open/close
NON_GRASP_UNCERTAIN_TAIL = 5  # last N frames before end_frame → uncertain (skip)
# Tail exclusion: last X% of action segment → ambiguous transition zone (skip)
TAIL_EXCLUDE_RATIO = 0.05


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _euler_to_rotmat(euler: np.ndarray) -> np.ndarray:
    r, p, y = euler
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that aligns unit vector *a* onto unit vector *b* (Rodrigues)."""
    a = a / max(np.linalg.norm(a), 1e-8)
    b = b / max(np.linalg.norm(b), 1e-8)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def compute_distance_score(obs_xyz, target_xyz, alpha=10.0):
    d = np.linalg.norm(obs_xyz - target_xyz)
    return float(np.exp(-alpha * d))


def compute_alignment_score(obs_rotmat, target_rotmat):
    R_diff = obs_rotmat.T @ target_rotmat
    cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float((1.0 + cos_angle) / 2.0)


def classify_gripper_confident(
    action_type: str, gripper_traj: np.ndarray, obs_idx: int,
    g_min: float, g_max: float,
) -> Optional[float]:
    """Confident grasp/release classification from gripper state and stability."""
    g_range = g_max - g_min
    if g_range < MIN_RANGE:
        return None
    current = gripper_traj[obs_idx]
    if np.isnan(current):
        return None
    norm_pos = (current - g_min) / g_range

    ws = max(0, obs_idx - STABILITY_WINDOW)
    we = min(len(gripper_traj), obs_idx + STABILITY_WINDOW + 1)
    local = gripper_traj[ws:we]
    local_valid = local[~np.isnan(local)]
    norm_std = (float(np.std(local_valid)) / g_range) if len(local_valid) > 2 else 999.0

    derivs = []
    for i in range(max(1, obs_idx - STABILITY_WINDOW), min(len(gripper_traj), obs_idx + 1)):
        if not np.isnan(gripper_traj[i]) and not np.isnan(gripper_traj[i - 1]):
            derivs.append(gripper_traj[i] - gripper_traj[i - 1])
    norm_deriv = (np.mean(derivs) / g_range) if derivs else 0.0

    valid_to_obs = gripper_traj[:obs_idx + 1]
    valid_to_obs = valid_to_obs[~np.isnan(valid_to_obs)]
    norm_cumchange = ((valid_to_obs[-1] - valid_to_obs[0]) / g_range
                      if len(valid_to_obs) > 1 else 0.0)

    is_stable = norm_std < STABILITY_THRESH
    is_changing = abs(norm_deriv) > 0.05

    if action_type == "grasp":
        if norm_pos < CONF_COMPLETED_POS and is_stable and not is_changing:
            if norm_cumchange < -0.3:
                return 1.0
        if norm_pos > CONF_NOT_POS and is_stable and not is_changing:
            if abs(norm_cumchange) < 0.3:
                return 0.0
    elif action_type == "release":
        if norm_pos > (1.0 - CONF_COMPLETED_POS) and is_stable and not is_changing:
            if norm_cumchange > 0.3:
                return 1.0
        if norm_pos < (1.0 - CONF_NOT_POS) and is_stable and not is_changing:
            if abs(norm_cumchange) < 0.3:
                return 0.0
    return None


def compute_task_completion_score(obs_xyz, target_xyz):
    return 1.0 if np.linalg.norm(obs_xyz - target_xyz) < CONTACT_DIST_THRESH else 0.0


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def decode_rle_mask(rle: Dict[str, Any]) -> np.ndarray:
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + c] = 1
        pos += c
    return flat.reshape((h, w), order="F")


def _decode_rle_mask_start_one(rle: Dict[str, Any]) -> np.ndarray:
    """Decode uncompressed RLE assuming the first run is ones."""
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 0:
            flat[pos:pos + c] = 1
        pos += c
    return flat.reshape((h, w), order="F")


def _mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def decode_rle_mask_with_bbox(rle: Dict[str, Any], bbox_hint: Optional[List[int]]) -> np.ndarray:
    """Decode legacy RLE robustly by selecting parity matching stored bbox.

    Canonical data uses zero-run-first encoding. Older dumps may reverse the
    run parity, so the stored bounding box identifies the matching decoding.
    """
    m0 = decode_rle_mask(rle)
    if bbox_hint is None:
        return m0
    if _mask_bbox(m0) == list(bbox_hint):
        return m0
    m1 = _decode_rle_mask_start_one(rle)
    if _mask_bbox(m1) == list(bbox_hint):
        return m1
    return m0


def _get_wrist_mask_rle(mask_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(mask_data, dict):
        return None
    for key in ("mask_visible_wrist", "mask_visible_robot0_eye_in_hand"):
        rle = mask_data.get(key)
        if isinstance(rle, dict) and "counts" in rle:
            return rle
    return None


def _get_wrist_bbox(mask_data: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if not isinstance(mask_data, dict):
        return None
    bbox = mask_data.get("bbox_wrist")
    if bbox is not None:
        return bbox
    bbox = mask_data.get("bbox_robot0_eye_in_hand")
    if bbox is not None:
        return bbox
    return None


def downsample_mask_to_patches(mask: np.ndarray, grid: int = DINO_PATCH_GRID) -> np.ndarray:
    if not mask.any():
        return np.zeros(grid * grid, dtype=np.float32)
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mode = str(os.environ.get("RAIN_MASK_DOWNSAMPLE_MODE", "avg")).strip().lower()
    if mode == "avg":
        pooled = F.adaptive_avg_pool2d(t, (grid, grid))
    elif mode == "bilinear":
        pooled = F.interpolate(t, size=(grid, grid), mode="bilinear", align_corners=False)
    else:
        raise ValueError(
            f"Unsupported RAIN_MASK_DOWNSAMPLE_MODE={mode!r}; expected one of: avg, bilinear"
        )
    return pooled.squeeze().flatten().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Packed feature loaders
# ---------------------------------------------------------------------------

from shared.data.packed_features import PackedDINO, PackedImages, TextFeatureCache
from shared.data.mask_augmentation import (
    augment_multiview_masks,
    mask_augmentation_mode_id,
    validate_mask_augmentation_recipe,
)
from shared.clip_utils import (
    DEFAULT_CLIP_TEXT_MODEL,
    get_text_cache_path,
)
from shared.data.episode_io import load_episode_records, normalize_episode_record


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [s[key] for s in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, bool, np.bool_)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values
    return collated


# ---------------------------------------------------------------------------
# Paired batch sampler (goal-xyz consistency)
# ---------------------------------------------------------------------------

class PairConsistencyBatchSampler(Sampler[List[int]]):
    """Batch sampler: half anchors + half paired samples from same target group.

    Each yielded batch has layout:
      [anchor_0, ..., anchor_{H-1}, pair_0, ..., pair_{H-1}]
    where pair_i is sampled from the same (episode, target subtask) group.
    """

    def __init__(
        self,
        dataset: "RAINDataset",
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 1 or (batch_size % 2) != 0:
            raise ValueError(f"paired batch requires even batch_size>=2, got {batch_size}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"invalid rank {rank} for num_replicas {num_replicas}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.half = int(batch_size // 2)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._anchor_indices = list(getattr(dataset, "goal_pair_batch_anchor_indices", []))
        if len(self._anchor_indices) == 0:
            self._anchor_indices = list(getattr(dataset, "goal_consistency_anchor_indices", []))
        if len(self._anchor_indices) == 0:
            raise ValueError("no goal-consistency anchors available; check dataset groups")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _distributed_anchor_indices(self) -> List[int]:
        indices = list(self._anchor_indices)
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)

        if self.drop_last:
            total_size = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = int(np.ceil(len(indices) / self.num_replicas) * self.num_replicas)
            if len(indices) < total_size:
                pad = total_size - len(indices)
                indices.extend(indices[:pad])

        return indices[self.rank:total_size:self.num_replicas]

    def __iter__(self):
        rank_indices = self._distributed_anchor_indices()
        usable = (len(rank_indices) // self.half) * self.half
        rank_indices = rank_indices[:usable]
        rng = random.Random(self.seed + 100003 * self.epoch + self.rank)

        for start in range(0, usable, self.half):
            anchors = rank_indices[start:start + self.half]
            pairs = [self.dataset.sample_goal_consistency_partner(i, rng) for i in anchors]
            yield anchors + pairs

    def __len__(self) -> int:
        rank_indices = self._distributed_anchor_indices()
        return len(rank_indices) // self.half


class TripletPostConsistencyBatchSampler(Sampler[List[int]]):
    """Batch sampler: [same-target triplet] + [same-episode post] blocks.

    Each block has 4 samples:
      [x0, x1, x2, post]
    where x0/x1/x2 are sampled (with replacement) from one
    (episode, target_subtask) group, and post is sampled from any post frame
    in the same episode.
    """

    def __init__(
        self,
        dataset: "RAINDataset",
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 4 or (batch_size % 4) != 0:
            raise ValueError(f"triplet-post batch requires batch_size multiple of 4, got {batch_size}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"invalid rank {rank} for num_replicas {num_replicas}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.blocks_per_batch = int(batch_size // 4)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._anchor_indices = list(getattr(dataset, "goal_pair_batch_anchor_indices", []))
        if len(self._anchor_indices) == 0:
            self._anchor_indices = list(range(len(dataset)))

        self._valid_keys = []
        groups = getattr(dataset, "goal_consistency_groups", {})
        post_by_ep = getattr(dataset, "episode_post_indices", {})
        for key, members in groups.items():
            if len(members) <= 0:
                continue
            eidx = int(key[0])
            if len(post_by_ep.get(eidx, [])) <= 0:
                continue
            self._valid_keys.append(key)

        if len(self._valid_keys) == 0:
            raise ValueError("no valid (target triplet + post) groups available")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _distributed_anchor_indices(self) -> List[int]:
        indices = list(self._anchor_indices)
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)

        # Keep all samples across ranks; pad to world-size for even partition.
        total_size = int(np.ceil(len(indices) / self.num_replicas) * self.num_replicas)
        if len(indices) < total_size:
            pad = total_size - len(indices)
            indices.extend(indices[:pad])
        return indices[self.rank:total_size:self.num_replicas]

    def _pick_key(self, anchor_idx: int, rng: random.Random):
        key = None
        group_of_sample = getattr(self.dataset, "goal_consistency_group_of_sample", [])
        if 0 <= int(anchor_idx) < len(group_of_sample):
            key = group_of_sample[int(anchor_idx)]
        if key is None or key not in self.dataset.goal_consistency_groups:
            key = self._valid_keys[rng.randrange(len(self._valid_keys))]
        eidx = int(key[0])
        if len(self.dataset.episode_post_indices.get(eidx, [])) <= 0:
            key = self._valid_keys[rng.randrange(len(self._valid_keys))]
        return key

    def __iter__(self):
        rank_indices = self._distributed_anchor_indices()
        rng = random.Random(self.seed + 100003 * self.epoch + self.rank)

        # Do not discard tail: pad with replacement to keep high data efficiency.
        usable = int(np.ceil(len(rank_indices) / self.blocks_per_batch) * self.blocks_per_batch)
        if len(rank_indices) < usable and len(rank_indices) > 0:
            pad = usable - len(rank_indices)
            rank_indices.extend(rank_indices[:pad])

        for start in range(0, usable, self.blocks_per_batch):
            anchors = rank_indices[start:start + self.blocks_per_batch]
            batch: List[int] = []
            for aidx in anchors:
                key = self._pick_key(aidx, rng)
                eidx = int(key[0])
                tri_pool = self.dataset.goal_consistency_groups[key]
                post_pool = self.dataset.episode_post_indices[eidx]
                if len(tri_pool) >= 3:
                    tri = rng.sample(tri_pool, 3)
                    t0, t1, t2 = int(tri[0]), int(tri[1]), int(tri[2])
                elif len(tri_pool) == 2:
                    a = int(tri_pool[0])
                    b = int(tri_pool[1])
                    c = int(tri_pool[rng.randrange(2)])
                    t0, t1, t2 = a, b, c
                else:
                    t = int(tri_pool[0])
                    t0, t1, t2 = t, t, t
                p = int(post_pool[rng.randrange(len(post_pool))])
                batch.extend([t0, t1, t2, p])
            yield batch

    def __len__(self) -> int:
        rank_indices = self._distributed_anchor_indices()
        usable = int(np.ceil(len(rank_indices) / self.blocks_per_batch) * self.blocks_per_batch)
        return usable // self.blocks_per_batch


class GoalConsistencyBatchSampler(Sampler[List[int]]):
    """Full-coverage sampler: group same-target frames into fixed-size blocks.

    Each epoch uses every local sample once, with minimal padding for distributed
    sampling. Contiguous ``group_size`` slots share the same goal-consistency
    group whenever possible.

    Samples without a valid goal-consistency group are still included once per
    epoch, but are packed into generic blocks where consistency loss is simply
    skipped by the training loop.
    """

    def __init__(
        self,
        dataset: "RAINDataset",
        batch_size: int,
        group_size: int = 4,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size < group_size or (batch_size % group_size) != 0:
            raise ValueError(
                f"GoalConsistencyBatchSampler requires batch_size ({batch_size}) "
                f"divisible by group_size ({group_size})"
            )
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"invalid rank {rank} for num_replicas {num_replicas}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.groups_per_batch = int(batch_size // group_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self._all_indices = list(getattr(dataset, "goal_pair_batch_anchor_indices", []))
        if len(self._all_indices) == 0:
            self._all_indices = list(range(len(dataset)))
        self._gid_of_sample = list(getattr(dataset, "goal_consistency_group_id_of_sample", []))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _distributed_sample_indices(self) -> List[int]:
        indices = list(self._all_indices)
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(indices)

        total_size = int(np.ceil(len(indices) / self.num_replicas) * self.num_replicas)
        if len(indices) < total_size:
            pad = total_size - len(indices)
            indices.extend(indices[:pad])
        return indices[self.rank:total_size:self.num_replicas]

    def _sample_gid(self, sample_idx: int) -> int:
        if 0 <= int(sample_idx) < len(self._gid_of_sample):
            return int(self._gid_of_sample[int(sample_idx)])
        return -1

    def _build_blocks(self, rank_indices: List[int], rng: random.Random) -> List[List[int]]:
        grouped: Dict[int, List[int]] = {}
        mixed_pool: List[int] = []

        for sample_idx in rank_indices:
            gid = self._sample_gid(int(sample_idx))
            if gid >= 0:
                grouped.setdefault(gid, []).append(int(sample_idx))
            else:
                mixed_pool.append(int(sample_idx))

        blocks: List[List[int]] = []

        # Same-goal blocks: without replacement within the epoch, with only
        # full blocks kept valid for consistency loss. Group remainders are
        # merged into a mixed pool so full-epoch coverage is preserved without
        # inflating the number of steps by padding every group separately.
        for gid in sorted(grouped.keys()):
            members = list(grouped[gid])
            rng.shuffle(members)
            n_full = len(members) // self.group_size
            for bi in range(n_full):
                i0 = bi * self.group_size
                blocks.append(members[i0:i0 + self.group_size])
            rem = members[n_full * self.group_size:]
            if rem:
                mixed_pool.extend(int(v) for v in rem)

        # Samples with no valid group still participate in the epoch once.
        rng.shuffle(mixed_pool)
        for i0 in range(0, len(mixed_pool), self.group_size):
            block = list(mixed_pool[i0:i0 + self.group_size])
            if len(block) < self.group_size:
                if self.drop_last:
                    break
                pool = mixed_pool if len(mixed_pool) > 0 else rank_indices
                while len(block) < self.group_size:
                    block.append(int(pool[rng.randrange(len(pool))]))
            if len(block) == self.group_size:
                blocks.append(block)

        rng.shuffle(blocks)
        return blocks

    def _num_blocks(self, rank_indices: List[int]) -> int:
        grouped_counts: Dict[int, int] = {}
        mixed_count = 0
        for sample_idx in rank_indices:
            gid = self._sample_gid(int(sample_idx))
            if gid >= 0:
                grouped_counts[gid] = grouped_counts.get(gid, 0) + 1
            else:
                mixed_count += 1
        num_blocks = 0
        for count in grouped_counts.values():
            num_blocks += count // self.group_size
            mixed_count += count % self.group_size
        if mixed_count > 0:
            if self.drop_last:
                num_blocks += int(mixed_count // self.group_size)
            else:
                num_blocks += int(math.ceil(mixed_count / self.group_size))
        return num_blocks

    def __iter__(self):
        rank_indices = self._distributed_sample_indices()
        rng = random.Random(self.seed + 100003 * self.epoch + self.rank)
        blocks = self._build_blocks(rank_indices, rng)
        if len(blocks) == 0:
            return

        rem = len(blocks) % self.groups_per_batch
        if rem != 0:
            if self.drop_last:
                blocks = blocks[:len(blocks) - rem]
            else:
                pad = self.groups_per_batch - rem
                for _ in range(pad):
                    blocks.append(list(blocks[rng.randrange(len(blocks))]))

        for start in range(0, len(blocks), self.groups_per_batch):
            chunk = blocks[start:start + self.groups_per_batch]
            if len(chunk) < self.groups_per_batch:
                if self.drop_last:
                    continue
                while len(chunk) < self.groups_per_batch:
                    chunk.append(list(blocks[rng.randrange(len(blocks))]))
            batch: List[int] = []
            for block in chunk:
                batch.extend(int(c) for c in block)
            yield batch

    def __len__(self) -> int:
        rank_indices = self._distributed_sample_indices()
        num_blocks = self._num_blocks(rank_indices)
        if self.drop_last:
            return num_blocks // self.groups_per_batch
        return int(math.ceil(num_blocks / self.groups_per_batch))


# ---------------------------------------------------------------------------
# Plan waypoint sampling
# ---------------------------------------------------------------------------

def arc_length_sample(traj: np.ndarray, n_waypoints: int) -> np.ndarray:
    """Sample n_waypoints from trajectory by arc length.

    Args:
        traj: (T, 3) xyz trajectory
        n_waypoints: number of waypoints to sample

    Returns:
        (n_waypoints, 3) sampled waypoints
    """
    if len(traj) < 2:
        return np.zeros((n_waypoints, 3), dtype=np.float32)

    # Compute cumulative arc length
    diffs = np.diff(traj, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-6:
        return np.tile(traj[0], (n_waypoints, 1)).astype(np.float32)

    # Sample at uniform arc length intervals
    target_lengths = np.linspace(0, total_length, n_waypoints)
    indices = np.searchsorted(cum_lengths, target_lengths, side="right") - 1
    indices = np.clip(indices, 0, len(traj) - 1)

    return traj[indices].astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RAINDataset(Dataset):
    """Unified RAIN dataset: subtask-aware with configurable features.

    Sample index: (episode_idx, subtask_idx, obs_frame, is_reverse)

    Returns:
        dino_third, dino_wrist: (257, 1024) packed DINO features
        goal_mask_third, goal_mask_wrist: (256,) object patch masks
        text_feat: (768,) CLIP text features
        state: (7,) xyz + cumRPY + gripper
        gt_action: (16, 7) action chunk
        gt_plan: (8, 3) plan waypoints (relative to current EEF)
        gt_distance, gt_alignment, gt_task_completion: scalar GT
        action_type: int (0-5)
        is_reverse: bool
    """

    def __init__(
        self,
        episodes_json: str,
        tc_event_manifest: str,
        frames_dir: str,
        parquet_dir: str,
        packed_features_dir: str,
        images_dir: str = "",
        num_action_steps: int = 16,
        num_plan_waypoints: int = 8,
        distance_alpha: float = 10.0,
        max_post_subtask_frames: int = 15,
        use_progress: bool = True,
        use_dit: bool = True,
        use_plan: bool = False,
        use_reverse_aug: bool = False,
        use_retarget_aug: bool = False,
        retarget_version: str = "legacy",
        retarget_rollout_mode: str = "tangent_replay",
        retarget_text_dropout_prob: float = 0.8,
        retarget_template_path: str = "",
        text_dropout_prob: float = 0.0,
        reverse_text_dropout_prob: float = 1.0,
        reverse_max_mask_overlap: float = 0.05,
        xyz_source: str = "action_delta",
        post_only_tc: bool = False,
        post_only_tc_types: Optional[List[str]] = None,
        tail_exclude_ratio: float = 0.0,
        use_target_place: bool = False,
        use_wrist_goal_mask: bool = False,
        use_wrist_target_place: bool = False,
        clip_model_name: str = DEFAULT_CLIP_TEXT_MODEL,
        text_feature_dim: int = 768,
        mask_augmentation: str = "none",
        episode_ids: Optional[List[int]] = None,
        is_training: bool = True,
    ):
        super().__init__()
        self.frames_dir = Path(frames_dir)
        self.num_action_steps = num_action_steps
        self.num_plan_waypoints = num_plan_waypoints
        self.distance_alpha = distance_alpha
        self.max_post_subtask_frames = max_post_subtask_frames
        self.is_training = is_training
        self.use_progress = use_progress
        self.use_dit = use_dit
        self.use_plan = use_plan
        self.use_reverse_aug = use_reverse_aug
        self.use_retarget_aug = use_retarget_aug
        self.retarget_version = str(retarget_version or "legacy").strip().lower()
        self.retarget_rollout_mode = str(
            retarget_rollout_mode or "tangent_replay"
        ).strip().lower()
        self.use_target_place = use_target_place
        self.use_wrist_goal_mask = bool(use_wrist_goal_mask)
        self.use_wrist_target_place = bool(use_target_place and use_wrist_target_place)
        self.clip_model_name = str(clip_model_name)
        self.text_feature_dim = int(text_feature_dim)
        self.mask_augmentation = validate_mask_augmentation_recipe(mask_augmentation)
        self.packed_images = None
        if images_dir:
            image_root = Path(images_dir)
            required = (
                image_root / "images_packed.npy",
                image_root / "images_index.json",
                image_root / "images_meta.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Online DINO image cache is incomplete: " + ", ".join(missing)
                )
            self.packed_images = PackedImages(str(image_root))
        self.post_only_tc = post_only_tc
        self.post_only_tc_types = set(post_only_tc_types) if post_only_tc_types else None
        self.tail_exclude_ratio = float(tail_exclude_ratio)
        self.retarget_text_dropout_prob = float(retarget_text_dropout_prob)
        self.text_dropout_prob = float(text_dropout_prob)
        self.reverse_text_dropout_prob = float(reverse_text_dropout_prob)
        self.reverse_max_mask_overlap = float(reverse_max_mask_overlap)
        self.xyz_source = str(xyz_source)
        self.tc_event_manifest = str(tc_event_manifest)
        if not 0.0 <= self.text_dropout_prob <= 1.0:
            raise ValueError(f"text_dropout_prob must be in [0,1], got {self.text_dropout_prob}")
        if not 0.0 <= self.reverse_text_dropout_prob <= 1.0:
            raise ValueError(
                f"reverse_text_dropout_prob must be in [0,1], got {self.reverse_text_dropout_prob}")
        if not 0.0 <= self.reverse_max_mask_overlap <= 1.0:
            raise ValueError(
                f"reverse_max_mask_overlap must be in [0,1], got {self.reverse_max_mask_overlap}")
        if self.retarget_version not in ("legacy", "stateful_v1"):
            raise ValueError(
                f"retarget_version must be one of ('legacy', 'stateful_v1'), got {self.retarget_version}"
            )
        if self.retarget_rollout_mode not in ("tangent_replay", "linear_interp_v1"):
            raise ValueError(
                "retarget_rollout_mode must be one of "
                f"('tangent_replay', 'linear_interp_v1'), got {self.retarget_rollout_mode}"
            )
        if self.xyz_source != "action_delta":
            raise ValueError(
                f"xyz_source must be 'action_delta', got {self.xyz_source}"
            )

        # Load episodes (with pickle cache for speed)
        pkl_path = Path(episodes_json).with_suffix(".pkl")
        json_mtime = os.path.getmtime(episodes_json)

        loaded_from_pkl = False
        if pkl_path.exists() and os.path.getmtime(str(pkl_path)) >= json_mtime:
            logger.info("Loading episodes from pickle cache: %s", pkl_path)
            with open(pkl_path, "rb") as f:
                self.episodes: Dict[int, Dict] = pickle.load(f)
            loaded_from_pkl = True
        else:
            logger.info("Loading episodes from JSON: %s", episodes_json)
            self.episodes = load_episode_records(episodes_json)
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(self.episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Saved pickle cache: %s", pkl_path)
            except Exception as e:
                logger.warning("Could not save pickle cache: %s", e)

        # Normalize cached episodes in memory so stale pickles cannot supply
        # incorrect primary/target object IDs for release/push subtasks.
        normalized_episodes: Dict[int, Dict] = {}
        cache_changed = False
        for raw_eidx, raw_episode in self.episodes.items():
            norm_episode = normalize_episode_record(raw_episode)
            norm_eidx = int(norm_episode.get("episode_index", raw_eidx))
            normalized_episodes[norm_eidx] = norm_episode
            if not cache_changed and norm_episode != raw_episode:
                cache_changed = True
        self.episodes = normalized_episodes

        if loaded_from_pkl and cache_changed:
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(self.episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Refreshed normalized pickle cache: %s", pkl_path)
            except Exception as e:
                logger.warning("Could not refresh pickle cache: %s", e)

        self.tc_events: Dict[Tuple[int, int], Dict[str, int]] = {}
        if self.use_progress:
            if not self.tc_event_manifest:
                raise ValueError("Progress training requires tc_event_manifest")
            if self.post_only_tc:
                raise ValueError(
                    "TC v3 supplies the completion labels; set post_only_tc=False"
                )
            if self.tail_exclude_ratio != 0.0:
                raise ValueError(
                    "TC v3 supplies an explicit IGNORE interval; set tail_exclude_ratio=0"
                )
            if self.use_reverse_aug or self.use_retarget_aug:
                raise ValueError(
                    "TC v3 labels are defined only for normal samples; "
                    "disable reverse/retarget progress augmentation"
                )
            self._load_tc_event_manifest(self.tc_event_manifest)

        # Load actions + real-world EEF + absolute RPY from parquet
        self.episode_actions, self.episode_eef, self.episode_abs_rpy, self.episode_full_states = (
            self._load_actions_and_eef(parquet_dir)
        )
        self.episode_action_xyz: Dict[int, np.ndarray] = {}
        for eidx, actions in self.episode_actions.items():
            cum_xyz = np.zeros((len(actions) + 1, 3), dtype=np.float32)
            cum_xyz[1:] = np.cumsum(actions[:, :3], axis=0)
            self.episode_action_xyz[eidx] = cum_xyz
        self.episode_motion_profiles: Dict[int, Dict[str, float]] = {}
        self._precompute_episode_motion_profiles()

        # Packed features
        packed = Path(packed_features_dir) if packed_features_dir else None
        # PackedDINO validates the concrete single- or multi-scale payload.
        # Gate only on the shared index here: multi-scale caches intentionally
        # contain dino_packed_s{1,2,3}.npy instead of dino_packed.npy.
        self.packed_dino = (
            PackedDINO(str(packed))
            if packed and (packed / "dino_index.json").is_file()
            else None
        )
        assert self.packed_dino is not None, "PackedDINO required"
        self.dino_num_tokens = int(self.packed_dino.num_tokens)
        self.dino_hidden_dim = int(self.packed_dino.hidden_dim)

        # Retarget trajectory templates
        self.retarget_templates = None
        if retarget_template_path and Path(retarget_template_path).is_file():
            self.retarget_templates = dict(np.load(retarget_template_path, allow_pickle=False))
            logger.info("Loaded trajectory templates from %s (%d keys)",
                        retarget_template_path, len(self.retarget_templates))

        # Text features
        text_cache = str(get_text_cache_path(packed, self.clip_model_name)) if packed else None
        self.text_cache = TextFeatureCache(
            text_cache,
            model_name=self.clip_model_name,
            feature_dim=self.text_feature_dim,
        )
        if not self.text_cache.has_features():
            descs = set()
            for ep in self.episodes.values():
                for seg in ep.get("subtask_segments", []):
                    d = seg.get("description", "")
                    if d:
                        descs.add(d)
            self.text_cache.precompute(list(descs), save=True,
                                       cache_path=text_cache or "")

        # Precompute gripper ranges for confident GT
        self.segment_gripper_ranges: Dict[Tuple[int, int], Tuple[float, float, np.ndarray]] = {}
        self._precompute_gripper_ranges()

        # Precompute rotation progress for turn_on segments.
        self.segment_rotation_progress: Dict[Tuple[int, int], np.ndarray] = {}
        self._precompute_rotation_progress()

        # Build sample index
        self.samples = self._build_samples(episode_ids)
        self.goal_consistency_groups: Dict[Tuple[int, int], List[int]] = {}
        self.goal_consistency_group_of_sample: List[Optional[Tuple[int, int]]] = []
        self.goal_consistency_group_id_of_sample: List[int] = []
        self.goal_consistency_anchor_indices: List[int] = []
        self.goal_pair_batch_anchor_indices: List[int] = []
        self.episode_post_indices: Dict[int, List[int]] = {}
        self._build_goal_consistency_groups()
        logger.info("XYZ source: %s", self.xyz_source)
        logger.info("Built %d samples (%s)", len(self.samples),
                     "train" if is_training else "val")

    def _load_tc_event_manifest(self, manifest_path: str) -> None:
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"TC event manifest not found: {path}")
        with path.open() as handle:
            manifest = json.load(handle)
        if manifest.get("version") != "tc_event_v3":
            raise ValueError(
                f"Expected tc_event_v3 manifest, got {manifest.get('version')!r}"
            )

        manifest_episodes = {
            int(row["episode_index"]): row for row in manifest.get("episodes", [])
        }
        expected_episode_ids = set(self.episodes)
        manifest_episode_ids = set(manifest_episodes)
        if expected_episode_ids != manifest_episode_ids:
            missing = sorted(expected_episode_ids - manifest_episode_ids)
            extra = sorted(manifest_episode_ids - expected_episode_ids)
            raise ValueError(
                "TC manifest/episodes mismatch: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        for eidx, episode in self.episodes.items():
            manifest_episode = manifest_episodes[eidx]
            rows_by_subtask = {
                int(row["subtask_id"]): row
                for row in manifest_episode.get("segments", [])
            }
            segments = episode.get("subtask_segments", [])
            if len(rows_by_subtask) != len(segments):
                raise ValueError(
                    f"TC segment count mismatch for episode {eidx}: "
                    f"manifest={len(rows_by_subtask)}, episodes={len(segments)}"
                )
            n_frames = int(episode.get("num_frames", 0))
            for si, segment in enumerate(segments):
                subtask_id = int(segment["subtask_id"])
                row = rows_by_subtask.get(subtask_id)
                if row is None:
                    raise ValueError(
                        f"Missing TC event: episode={eidx}, subtask={subtask_id}"
                    )
                expected = (
                    str(segment["action_type"]),
                    int(segment["start_frame"]),
                    int(segment["end_frame"]),
                )
                observed = (
                    str(row["action_type"]),
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                )
                if observed != expected:
                    raise ValueError(
                        f"TC segment mismatch for episode={eidx}, subtask={subtask_id}: "
                        f"manifest={observed}, episodes={expected}"
                    )
                guard_start = int(row["guard_start"])
                event_frame = int(row["event_frame"])
                start_frame = int(segment["start_frame"])
                if not (start_frame <= guard_start <= event_frame < n_frames):
                    raise ValueError(
                        f"Invalid TC interval for episode={eidx}, subtask={subtask_id}: "
                        f"start={start_frame}, guard={guard_start}, "
                        f"event={event_frame}, frames={n_frames}"
                    )
                self.tc_events[(eidx, si)] = {
                    "guard_start": guard_start,
                    "event_frame": event_frame,
                }
        logger.info(
            "Loaded TC v3 labels: %d episodes, %d segments from %s",
            len(manifest_episodes),
            len(self.tc_events),
            path,
        )

    def _tc_event_label(self, eidx: int, si: int, frame: int) -> Optional[float]:
        event = self.tc_events[(int(eidx), int(si))]
        if int(frame) < event["guard_start"]:
            return 0.0
        if int(frame) < event["event_frame"]:
            return None
        return 1.0

    @staticmethod
    def _load_actions_and_eef(parquet_dir: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Load actions, real-world EEF positions, absolute RPY, and full states from parquet.

        Returns:
            (episode_actions, episode_eef, episode_abs_rpy, episode_states) where
            episode_eef[eidx] = (N, 3) parquet observation.state[:3]
            episode_abs_rpy[eidx] = (N, 3) parquet observation.state[3:6] (absolute euler angles)
            episode_states[eidx] = (N, 8) full observation.state from parquet
        """
        import pandas as pd
        actions: Dict[int, np.ndarray] = {}
        eef: Dict[int, np.ndarray] = {}
        abs_rpy: Dict[int, np.ndarray] = {}
        full_states: Dict[int, np.ndarray] = {}
        for chunk in sorted(Path(parquet_dir).glob("chunk-*")):
            for pf in sorted(chunk.glob("*.parquet")):
                df = pd.read_parquet(pf, columns=[
                    "episode_index", "frame_index", "action", "observation.state"])
                for eidx in df["episode_index"].unique():
                    sub = df[df["episode_index"] == eidx].sort_values("frame_index")
                    e = int(eidx)
                    actions[e] = np.stack(sub["action"].values).astype(np.float32)
                    states = np.stack(sub["observation.state"].values).astype(np.float32)
                    eef[e] = states[:, :3]
                    abs_rpy[e] = states[:, 3:6]
                    full_states[e] = states
        return actions, eef, abs_rpy, full_states

    def _get_frame_action_pos(self, eidx: int, frame: int) -> Optional[np.ndarray]:
        traj = self.episode_action_xyz.get(int(eidx))
        if traj is None or len(traj) == 0:
            return None
        idx = int(np.clip(int(frame), 0, len(traj) - 1))
        prof = self.episode_motion_profiles.get(int(eidx), {})
        scale = float(np.clip(prof.get("cmd_to_world_scale", 1.0), 1e-3, 1.0))
        return (traj[idx] * scale).astype(np.float32)

    @staticmethod
    def _sample_mode_tuple(
        eidx: int,
        si: int,
        obs_frame: int,
        mode: int,
        target_si: int = -1,
    ) -> Tuple[int, int, int, int, int]:
        return (int(eidx), int(si), int(obs_frame), int(mode), int(target_si))

    @staticmethod
    def _parse_sample_tuple(sample: Tuple[Any, ...]) -> Tuple[int, int, int, int, int]:
        # Backward compatibility for old 4-tuple format.
        if len(sample) == 4:
            eidx, si, obs, is_reverse = sample
            mode = SAMPLE_MODE_REVERSE if bool(is_reverse) else SAMPLE_MODE_NORMAL
            target_si = int(si - 1) if mode == SAMPLE_MODE_REVERSE else -1
            return int(eidx), int(si), int(obs), int(mode), int(target_si)
        if len(sample) >= 5:
            eidx, si, obs, mode, target_si = sample[:5]
            return int(eidx), int(si), int(obs), int(mode), int(target_si)
        raise ValueError(f"invalid sample tuple: {sample}")

    def _precompute_episode_motion_profiles(self) -> None:
        # parquet action xyz are controller commands (normalized), not metric deltas.
        # Build a stable pseudo-world scale per episode for geometry labels.
        nominal_world_step = 0.005
        for eidx, ep in self.episodes.items():
            actions = self.episode_actions.get(eidx)
            if actions is None or len(actions) == 0:
                continue

            cmd_xyz = np.linalg.norm(actions[:, :3], axis=1)
            cmd_rpy = np.linalg.norm(actions[:, 3:6], axis=1)
            med_cmd_xyz = float(np.median(cmd_xyz[cmd_xyz > 1e-6])) if np.any(cmd_xyz > 1e-6) else 0.12
            med_cmd_rpy = float(np.median(cmd_rpy[cmd_rpy > 1e-6])) if np.any(cmd_rpy > 1e-6) else 0.05

            gripper_vals: List[float] = []
            frames = ep.get("frames", {})
            for f in range(max(0, len(actions) - 1)):
                fd0 = frames.get(str(f), {})
                g = fd0.get("gripper_state")
                if g is not None:
                    gv = float(g)
                    if np.isfinite(gv):
                        gripper_vals.append(gv)

            med_cmd_xyz_safe = float(max(med_cmd_xyz, 1e-4))
            cmd_to_world_scale = float(
                np.clip(nominal_world_step / med_cmd_xyz_safe, 1e-3, 0.2)
            )
            med_world_step = med_cmd_xyz_safe * cmd_to_world_scale
            # Convert pseudo-world delta back to action command scale.
            xyz_gain = float(np.clip(1.0 / cmd_to_world_scale, 0.01, 1000.0))

            if gripper_vals:
                g_min = float(np.min(gripper_vals))
                g_max = float(np.max(gripper_vals))
            else:
                g_min, g_max = 0.0, 1.0

            # Real-world xyz_gain from parquet EEF
            eef = self.episode_eef.get(eidx)
            if eef is not None and len(eef) > 1:
                eef_deltas = np.linalg.norm(eef[1:] - eef[:-1], axis=1)
                med_eef_step = float(np.median(eef_deltas[eef_deltas > 1e-6])) if np.any(eef_deltas > 1e-6) else 0.005
                real_xyz_gain = float(np.clip(1.0 / max(med_eef_step, 1e-6), 0.01, 1000.0))
                # Workspace bounds from real-world EEF
                ws_margin = 0.03
                workspace_min = eef.min(axis=0) - ws_margin
                workspace_max = eef.max(axis=0) + ws_margin
            else:
                real_xyz_gain = xyz_gain
                positions = self.episode_action_xyz[eidx] * cmd_to_world_scale
                ws_margin = 0.03
                workspace_min = positions.min(axis=0) - ws_margin
                workspace_max = positions.max(axis=0) + ws_margin

            self.episode_motion_profiles[eidx] = {
                "median_cmd_xyz": float(np.clip(med_cmd_xyz, 0.05, 0.45)),
                "median_cmd_rpy": float(np.clip(med_cmd_rpy, 0.02, 0.18)),
                "median_world_step": float(np.clip(med_world_step, 0.002, 0.03)),
                "cmd_to_world_scale": cmd_to_world_scale,
                "xyz_gain": xyz_gain,
                "real_xyz_gain": real_xyz_gain,
                "gripper_action_gain": self._estimate_gripper_action_gain(
                    eidx=eidx,
                    actions=actions,
                    g_min=g_min,
                    g_max=g_max,
                ),
                "gripper_min": g_min,
                "gripper_max": g_max,
                "workspace_min": workspace_min.astype(np.float32),
                "workspace_max": workspace_max.astype(np.float32),
            }

    def _estimate_gripper_action_gain(
        self,
        eidx: int,
        actions: np.ndarray,
        g_min: float,
        g_max: float,
    ) -> float:
        """Estimate a simple state-delta -> gripper-action gain.

        Retarget baselines operate in state space, while stage-1 supervision uses
        controller-action space. XYZ uses a calibrated gain. This estimates a
        per-episode linear gain from observed gripper state deltas and recorded
        gripper commands.
        """
        full_states = self.episode_full_states.get(eidx)
        if full_states is not None and len(full_states) >= 2 and full_states.shape[1] > 6:
            n = min(len(actions), len(full_states) - 1)
            g0 = full_states[:n, 6].astype(np.float32)
            g1 = full_states[1:n + 1, 6].astype(np.float32)
            dg = g1 - g0
            ga = actions[:n, 6].astype(np.float32)
            valid = np.isfinite(dg) & np.isfinite(ga) & (np.abs(dg) > 1e-5) & (np.abs(ga) > 1e-4)
            if np.any(valid):
                ratios = np.abs(ga[valid]) / np.maximum(np.abs(dg[valid]), 1e-6)
                ratios = ratios[np.isfinite(ratios)]
                if len(ratios) > 0:
                    return float(np.clip(np.median(ratios), 1.0, 1000.0))

        g_range = max(float(g_max - g_min), 1e-4)
        # Fallback: map a 10% range change to roughly unit command.
        return float(np.clip(10.0 / g_range, 1.0, 1000.0))

    def _precompute_gripper_ranges(self):
        for eidx, ep in self.episodes.items():
            if eidx not in self.episode_actions:
                continue
            n_actions = len(self.episode_actions[eidx])
            for si, seg in enumerate(ep.get("subtask_segments", [])):
                action_type = seg.get("action_type", "")
                if action_type not in ("grasp", "release"):
                    continue
                start = seg["start_frame"]
                end = seg["end_frame"]
                if end >= n_actions:
                    continue
                range_end = min(end + POST_FRAMES_RANGE, n_actions - 1)
                vals = []
                for f in range(start, range_end + 1):
                    fd = ep["frames"].get(str(f), {})
                    g = fd.get("gripper_state", None)
                    if g is None:
                        vals.append(np.nan)
                    else:
                        vals.append(float(g))
                traj = np.array(vals, dtype=np.float32)
                valid = traj[~np.isnan(traj)]
                if len(valid) >= 5:
                    self.segment_gripper_ranges[(eidx, si)] = (
                        float(np.min(valid)), float(np.max(valid)), traj)

    def _precompute_rotation_progress(self):
        """Precompute cumulative yaw progress for turn_on segments."""
        n_computed = 0
        n_skipped = 0
        for eidx, ep in self.episodes.items():
            if eidx not in self.episode_actions:
                continue
            actions = self.episode_actions[eidx]
            n_actions = len(actions)
            for si, seg in enumerate(ep.get("subtask_segments", [])):
                if seg.get("action_type") != "turn_on":
                    continue
                start = seg["start_frame"]
                end = seg["end_frame"]
                if end >= n_actions:
                    continue
                rpy = actions[start:end + 1, 3:6]  # (L, 3)
                yaw_deltas = rpy[:, 2]
                cum_yaw = np.cumsum(yaw_deltas)
                total_yaw = cum_yaw[-1] if len(cum_yaw) > 0 else 0.0
                if abs(total_yaw) < 0.1:
                    n_skipped += 1
                    continue  # degenerate: no meaningful rotation
                progress = cum_yaw / total_yaw
                self.segment_rotation_progress[(eidx, si)] = progress.astype(np.float32)
                n_computed += 1
        logger.info("Rotation progress: %d turn_on segments computed, %d skipped (degenerate)",
                     n_computed, n_skipped)

    def _get_assess_goal_frame(self, seg: Dict) -> int:
        phases = seg.get("phases", [])
        end = seg["end_frame"]
        if phases and phases[0].get("phase") == "move":
            return min(phases[0]["end_frame"], end)
        acf = seg.get("action_change_frame")
        return min(acf, end) if acf is not None else end

    def _get_plan_goal_frame(self, seg: Dict, obs_frame: int) -> int:
        assess_goal = self._get_assess_goal_frame(seg)
        if assess_goal - obs_frame < 4:
            return obs_frame + 4
        return assess_goal

    def _get_rotmat_at_frame(self, eidx: int, frame: int) -> np.ndarray:
        abs_rpy = self.episode_abs_rpy.get(eidx)
        if abs_rpy is not None and frame < len(abs_rpy):
            rpy = abs_rpy[frame].copy()
        else:
            rpy = np.zeros(3, dtype=np.float32)
        return _euler_to_rotmat(rpy)

    def _get_frame_world_pos(self, ep: Dict[str, Any], frame: int) -> Optional[np.ndarray]:
        """Get real-world EEF position at frame from parquet observation.state[:3]."""
        eidx = int(ep.get("episode_index", -1))
        eef = self.episode_eef.get(eidx)
        if eef is not None and len(eef) > 0:
            fidx = int(np.clip(frame, 0, len(eef) - 1))
            return eef[fidx].copy().astype(np.float32)
        # Fallback to pseudo-world (should not happen with parquet data)
        pos = self._get_frame_action_pos(eidx, frame)
        if pos is None:
            return None
        return pos.astype(np.float32)

    def _get_frame_abs_rpy(self, eidx: int, frame: int) -> np.ndarray:
        abs_rpy = self.episode_abs_rpy.get(eidx)
        if abs_rpy is None or int(frame) >= len(abs_rpy):
            return np.zeros(3, dtype=np.float32)
        return abs_rpy[int(frame)].copy()

    def _get_segment_trajectory(
        self, eidx: int, seg: Dict[str, Any],
    ) -> List[Tuple[int, np.ndarray]]:
        """Extract pseudo-world EEF trajectory for a segment.

        Returns: [(frame_idx, xyz), ...] sorted by frame_idx.
        """
        ep = self.episodes.get(eidx)
        if ep is None:
            return []
        start = int(seg.get("start_frame", 0))
        end = int(seg.get("end_frame", start))
        result: List[Tuple[int, np.ndarray]] = []
        for f in range(start, end + 1):
            pos = self._get_frame_world_pos(ep, f)
            if pos is not None:
                result.append((f, pos.astype(np.float32)))
        return result

    def _map_gripper_cmd(self, action_type: str) -> float:
        at = str(action_type).lower()
        if at == "grasp":
            return 1.0
        if at == "release":
            return -1.0
        return 0.0

    def _infer_holding_state(
        self,
        eidx: int,
        si: int,
        obs_frame: int,
        action_type: str,
        obs_gripper: float,
    ) -> bool:
        at = str(action_type).lower()
        if at == "release":
            return True
        if at == "grasp":
            return False
        prof = self.episode_motion_profiles.get(eidx, {})
        g_min = float(prof.get("gripper_min", 0.0))
        g_max = float(prof.get("gripper_max", 1.0))
        if g_max - g_min < 1e-6:
            return bool(obs_gripper < 0.5)
        norm_pos = (float(obs_gripper) - g_min) / (g_max - g_min)
        return bool(norm_pos < 0.45)

    @staticmethod
    def _segment_manip_object_id(seg: Dict[str, Any]) -> str:
        return str(seg.get("manip_object_id") or seg.get("primary_object_id", ""))

    @staticmethod
    def _segment_target_object_id(seg: Dict[str, Any]) -> str:
        return str(seg.get("target_object_id") or seg.get("primary_object_id", ""))

    def _segment_retarget_mask_object_id(self, seg: Dict[str, Any]) -> str:
        at = str(seg.get("action_type", "")).lower()
        if at == "release":
            return self._segment_manip_object_id(seg) or str(seg.get("primary_object_id", ""))
        return str(seg.get("primary_object_id", ""))

    def _infer_retarget_source_state_v1(
        self,
        eidx: int,
        seg: Dict[str, Any],
        si: int,
        obs_frame: int,
        obs_gripper: float,
    ) -> Tuple[bool, str, str]:
        """Infer source holding state/object for the stateful retarget selector.

        Segment phase progress and normalized gripper state identify late-grasp
        frames as holding the object and post-release frames as empty-hand.
        """
        del si  # The selector interface includes the unused subtask index.
        at = str(seg.get("action_type", "")).lower()
        manip_obj_id = self._segment_manip_object_id(seg)
        mask_obj_id = self._segment_retarget_mask_object_id(seg) or manip_obj_id
        norm_g = self._normalize_gripper_state(eidx, obs_gripper)
        assess_goal = int(self._get_assess_goal_frame(seg))

        if at == "grasp":
            holding = bool(obs_frame >= assess_goal and norm_g < 0.60)
            held_obj_id = manip_obj_id if holding else ""
            return holding, held_obj_id, mask_obj_id
        if at == "release":
            holding = bool(obs_frame <= assess_goal and norm_g < 0.60)
            held_obj_id = manip_obj_id if holding else ""
            return holding, held_obj_id, mask_obj_id

        holding = bool(norm_g < 0.45)
        return holding, "", mask_obj_id

    def _find_release_target_for_object(
        self,
        segs: List[Dict[str, Any]],
        source_si: int,
        held_obj_id: str,
    ) -> str:
        """Find the recorded release target for the currently held object."""
        if not held_obj_id:
            return ""
        for tj in range(max(int(source_si), 0), len(segs)):
            tseg = segs[tj]
            if str(tseg.get("action_type", "")).lower() != "release":
                continue
            if self._segment_manip_object_id(tseg) != held_obj_id:
                continue
            return self._segment_target_object_id(tseg)
        return ""

    def _normalize_gripper_state(self, eidx: int, gripper_state: float) -> float:
        prof = self.episode_motion_profiles.get(eidx, {})
        g_min = float(prof.get("gripper_min", 0.0))
        g_max = float(prof.get("gripper_max", 1.0))
        if g_max - g_min < 1e-6:
            return 0.5
        return float(np.clip((float(gripper_state) - g_min) / (g_max - g_min), 0.0, 1.0))

    def _interaction_task_completion(
        self,
        eidx: int,
        obs_xyz: np.ndarray,
        target_xyz: np.ndarray,
        obs_gripper: float,
        target_action_type: str,
    ) -> float:
        """Action-type aware task_completion label for interaction augmentation."""
        if compute_task_completion_score(obs_xyz, target_xyz) <= 0.5:
            return 0.0

        at = str(target_action_type).lower()
        norm_g = self._normalize_gripper_state(eidx, obs_gripper)
        if at == "grasp":
            return 1.0 if norm_g < 0.40 else 0.0
        if at == "release":
            return 1.0 if norm_g > 0.60 else 0.0
        return 1.0

    def _segment_target_frame(self, seg: Dict[str, Any]) -> int:
        at = str(seg.get("action_type", "")).lower()
        if at in ("grasp", "release"):
            return int(self._get_assess_goal_frame(seg))
        return int(seg.get("end_frame", seg.get("start_frame", 0)))

    def _segment_target_pose(
        self,
        ep: Dict[str, Any],
        eidx: int,
        seg: Dict[str, Any],
        fallback_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        frame = self._segment_target_frame(seg)
        xyz = self._get_frame_world_pos(ep, frame)
        if xyz is None:
            xyz = np.asarray(fallback_xyz, dtype=np.float32)
        rpy = self._get_frame_abs_rpy(eidx, frame)
        return xyz.astype(np.float32), rpy.astype(np.float32), int(frame)

    def _segment_target_state(
        self,
        ep: Dict[str, Any],
        eidx: int,
        seg: Dict[str, Any],
        fallback_xyz: np.ndarray,
        fallback_rpy: np.ndarray,
        fallback_gripper: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Fetch target interaction state at the segment terminal interaction frame."""
        xyz, rpy, frame = self._segment_target_pose(ep, eidx, seg, fallback_xyz=fallback_xyz)
        gripper = float(fallback_gripper)
        full_states = self.episode_full_states.get(eidx)
        if full_states is not None and 0 <= frame < len(full_states):
            tgt_state = full_states[frame]
            if len(tgt_state) >= 6:
                xyz = tgt_state[:3].astype(np.float32)
                rpy = tgt_state[3:6].astype(np.float32)
            if len(tgt_state) > 6:
                gripper = float(tgt_state[6])
        else:
            fd = ep.get("frames", {}).get(str(frame), {})
            if "gripper_state" in fd:
                try:
                    gripper = float(fd["gripper_state"])
                except (TypeError, ValueError):
                    gripper = float(fallback_gripper)
            else:
                gripper = float(fallback_gripper)
            if xyz is None:
                xyz = np.asarray(fallback_xyz, dtype=np.float32)
            if rpy is None:
                rpy = np.asarray(fallback_rpy, dtype=np.float32)
        return xyz.astype(np.float32), rpy.astype(np.float32), float(gripper), int(frame)

    def _gripper_delta_to_action(self, eidx: int, delta: np.ndarray) -> np.ndarray:
        """Map gripper-state deltas to controller-style gripper commands."""
        prof = self.episode_motion_profiles.get(eidx, {})
        gain = float(prof.get("gripper_action_gain", 1.0))
        return np.clip(delta.astype(np.float32) * gain, -1.0, 1.0).astype(np.float32)

    def _infer_segment_gripper_cmd(
        self,
        eidx: int,
        seg: Dict[str, Any],
        default_cmd: float = -1.0,
    ) -> float:
        at = str(seg.get("action_type", "")).lower()
        if at in ("grasp", "release"):
            return self._map_gripper_cmd(at)

        actions = self.episode_actions.get(eidx)
        if actions is None or len(actions) == 0:
            return float(default_cmd)

        s = int(seg.get("start_frame", 0))
        e = int(seg.get("end_frame", s))
        if e < s:
            s, e = e, s
        s = int(np.clip(s, 0, len(actions) - 1))
        e = int(np.clip(e, s, len(actions) - 1))

        # End-of-segment commands reflect terminal manipulation intent.
        window = actions[max(s, e - 4): e + 1, 6]
        if len(window) == 0:
            window = actions[s:e + 1, 6]
        if len(window) == 0:
            return float(default_cmd)

        v = float(np.median(window))
        if abs(v) < 1e-5:
            return float(default_cmd)
        return 1.0 if v > 0 else -1.0

    def _select_retarget_target_segment(
        self,
        ep: Dict[str, Any],
        eidx: int,
        si: int,
        obs_frame: int,
    ) -> Optional[int]:
        if self.retarget_version == "stateful_v1":
            return self._select_retarget_target_segment_stateful_v1(
                ep=ep, eidx=eidx, si=si, obs_frame=obs_frame,
            )
        return self._select_retarget_target_segment_legacy(
            ep=ep, eidx=eidx, si=si, obs_frame=obs_frame,
        )

    def _select_retarget_target_segment_legacy(
        self,
        ep: Dict[str, Any],
        eidx: int,
        si: int,
        obs_frame: int,
    ) -> Optional[int]:
        """Select retarget target: interaction-aware selection.

        Filters: different object, post-interaction exclusion, mask overlap,
        distance floor 3cm, and holding-state compatibility (gripper consistency).
        """
        segs = ep.get("subtask_segments", [])
        if si < 0 or si >= len(segs):
            return None
        cur_seg = segs[si]
        cur_obj = str(cur_seg.get("primary_object_id", ""))
        if cur_obj == "":
            return None

        obs_xyz = self._get_frame_world_pos(ep, obs_frame)
        if obs_xyz is None:
            obs_xyz = self._get_frame_world_pos(ep, int(cur_seg.get("end_frame", obs_frame)))
        if obs_xyz is None:
            return None

        cur_start = int(cur_seg.get("start_frame", obs_frame))
        cur_end = int(cur_seg.get("end_frame", cur_start))
        cur_is_post = obs_frame > cur_end
        cur_mask = self._decode_object_mask(
            ep, obs_frame, cur_obj, cur_is_post, cur_end, cur_start
        )

        # Infer current holding state for gripper compatibility filter
        src_action_type = str(cur_seg.get("action_type", ""))
        obs_fd = ep["frames"].get(str(obs_frame), {})
        obs_gripper = float(obs_fd.get("gripper_state", 0.0))
        holding = self._infer_holding_state(
            eidx=eidx, si=si, obs_frame=obs_frame,
            action_type=src_action_type, obs_gripper=obs_gripper,
        )

        # Post-interaction object exclusion
        already_interacted = set()
        for tk, tseg_k in enumerate(segs):
            tk_obj = str(tseg_k.get("primary_object_id", ""))
            tk_end = int(tseg_k.get("end_frame", 0))
            if tk_end < obs_frame and tk_obj:
                already_interacted.add(tk_obj)

        candidates: List[Tuple[float, int]] = []
        for tj, tseg in enumerate(segs):
            if tj == si:
                continue
            obj_id = str(tseg.get("primary_object_id", ""))
            if obj_id == "" or obj_id == cur_obj:
                continue
            if obj_id in already_interacted:
                continue

            # Strict interaction compatibility: holding state vs target action type
            target_at = str(tseg.get("action_type", "")).lower()
            if holding and target_at not in ("release", "open", "close"):
                continue   # holding → only release/open/close allowed
            if not holding and target_at not in ("grasp", "push", "turn_on"):
                continue   # not holding → only grasp/push/turn_on allowed

            t_xyz, _, t_frame = self._segment_target_pose(ep, eidx, tseg, fallback_xyz=obs_xyz)
            t_start = int(tseg.get("start_frame", t_frame))
            t_end = int(tseg.get("end_frame", t_start))
            tgt_mask = self._decode_object_mask(
                ep, obs_frame, obj_id, obs_frame > t_end, t_end, t_start
            )
            if tgt_mask is None:
                tgt_mask = self._decode_object_mask(
                    ep, t_frame, obj_id, False, t_end, t_start
                )
            if tgt_mask is None:
                continue

            if cur_mask is not None:
                overlap = self._mask_overlap_ratio(tgt_mask, cur_mask)
                if overlap > self.reverse_max_mask_overlap:
                    continue

            dist = float(np.linalg.norm(t_xyz - obs_xyz))
            if dist < 0.03:
                continue
            candidates.append((dist, tj))

        candidates.sort(key=lambda x: x[0])
        return int(candidates[0][1]) if candidates else None

    def _select_retarget_target_segment_stateful_v1(
        self,
        ep: Dict[str, Any],
        eidx: int,
        si: int,
        obs_frame: int,
    ) -> Optional[int]:
        """State-aware target selection for retarget augmentation.

        Rules:
        - only future subtasks are considered
        - empty-hand states can retarget to future grasp/close
        - holding states can retarget to future release/close
        - release retargets are only allowed when the target place differs from
          the held object's original completion target
        """
        segs = ep.get("subtask_segments", [])
        if si < 0 or si >= len(segs):
            return None
        cur_seg = segs[si]

        obs_xyz = self._get_frame_world_pos(ep, obs_frame)
        if obs_xyz is None:
            obs_xyz = self._get_frame_world_pos(ep, int(cur_seg.get("end_frame", obs_frame)))
        if obs_xyz is None:
            return None

        cur_start = int(cur_seg.get("start_frame", obs_frame))
        cur_end = int(cur_seg.get("end_frame", cur_start))
        obs_fd = ep["frames"].get(str(obs_frame), {})
        obs_gripper = float(obs_fd.get("gripper_state", 0.0))
        holding, held_obj_id, cur_mask_obj = self._infer_retarget_source_state_v1(
            eidx=eidx,
            seg=cur_seg,
            si=si,
            obs_frame=obs_frame,
            obs_gripper=obs_gripper,
        )
        source_release_target_id = self._find_release_target_for_object(
            segs=segs,
            source_si=si,
            held_obj_id=held_obj_id,
        )

        cur_mask = None
        if cur_mask_obj:
            cur_mask = self._decode_object_mask(
                ep, obs_frame, cur_mask_obj, obs_frame > cur_end, cur_end, cur_start
            )
            if cur_mask is None:
                cur_mask = self._decode_object_mask(
                    ep, cur_start, cur_mask_obj, False, cur_end, cur_start
                )

        candidates: List[Tuple[float, int]] = []
        for tj in range(si + 1, len(segs)):
            tseg = segs[tj]
            target_at = str(tseg.get("action_type", "")).lower()
            if target_at == "grasp":
                if holding:
                    continue
            elif target_at == "close":
                pass
            elif target_at == "release":
                if not holding or not held_obj_id:
                    continue
                target_place_id = self._segment_target_object_id(tseg)
                if not target_place_id:
                    continue
                if source_release_target_id and target_place_id == source_release_target_id:
                    continue
            else:
                continue

            obj_id = str(tseg.get("primary_object_id", ""))
            if obj_id == "":
                continue

            t_xyz, _, t_frame = self._segment_target_pose(ep, eidx, tseg, fallback_xyz=obs_xyz)
            t_start = int(tseg.get("start_frame", t_frame))
            t_end = int(tseg.get("end_frame", t_start))
            tgt_mask = self._decode_object_mask(
                ep, obs_frame, obj_id, False, t_end, t_start
            )
            if tgt_mask is None:
                tgt_mask = self._decode_object_mask(
                    ep, t_frame, obj_id, False, t_end, t_start
                )
            if tgt_mask is None:
                continue

            if cur_mask is not None:
                overlap = self._mask_overlap_ratio(tgt_mask, cur_mask)
                if overlap > self.reverse_max_mask_overlap:
                    continue

            dist = float(np.linalg.norm(t_xyz - obs_xyz))
            if dist < 0.03:
                continue
            candidates.append((dist, tj))

        candidates.sort(key=lambda x: x[0])
        return int(candidates[0][1]) if candidates else None

    def _build_samples(
        self, episode_ids: Optional[List[int]],
    ) -> List[Tuple[int, int, int, int, int]]:
        """Build sample list: (episode_idx, subtask_idx, obs_frame, mode, target_si).

        For progress mode: uses confident GT filtering for grasp/release.
        Includes post-subtask frames.
        Optionally generates reverse and interaction-target augmentation samples.
        """
        samples: List[Tuple[int, int, int, int, int]] = []
        n_uncertain = 0
        n_tail_excluded = 0
        n_total = 0
        n_rev_missing_mask = 0
        n_rev_overlap_skip = 0
        n_retarget = 0
        n_retarget_skip = 0
        tail_ratio = self.tail_exclude_ratio

        for eidx, ep in self.episodes.items():
            if episode_ids is not None and eidx not in episode_ids:
                continue
            if eidx not in self.episode_actions:
                continue
            n_actions = len(self.episode_actions[eidx])
            segments = ep.get("subtask_segments", [])

            for si, seg in enumerate(segments):
                start = seg["start_frame"]
                end = seg["end_frame"]
                length = end - start + 1
                if length < 8 or seg.get("action_type") not in ACTION_TYPE_MAP:
                    continue
                if end >= n_actions:
                    continue
                action_type = seg["action_type"]

                # Tail exclusion boundary
                if tail_ratio > 0:
                    tail_start = end - max(int(math.ceil(tail_ratio * length)), 0)
                else:
                    tail_start = end  # no exclusion

                # Post-subtask boundary
                # Allow post frames to extend into the next segment's range.
                # A grasp's post frames overlap with release's action frames,
                # but they represent different subtasks (different si, mask, etc.).
                post_anchor = (
                    max(end, self.tc_events[(eidx, si)]["event_frame"])
                    if self.use_progress
                    else end
                )
                post_end = min(
                    post_anchor + 1 + self.max_post_subtask_frames,
                    n_actions,
                )

                if self.use_progress:
                    for obs in range(start, post_end):
                        n_total += 1
                        label = self._tc_event_label(eidx, si, obs)
                        if label is None:
                            n_uncertain += 1
                            continue
                        samples.append(self._sample_mode_tuple(
                            eidx, si, obs, SAMPLE_MODE_NORMAL, -1
                        ))
                    continue

                # Build normal samples
                if action_type in ("grasp", "release"):
                    key = (eidx, si)
                    if key not in self.segment_gripper_ranges:
                        # Need gripper ranges for progress filtering;
                        # skip segment entirely only when progress filtering is active.
                        if self.use_progress:
                            continue
                    if self.use_progress:
                        g_min, g_max, grip_traj = self.segment_gripper_ranges[key]
                    for obs in range(start, post_end):
                        n_total += 1
                        is_post_frame = obs > end

                        # Uncertain filtering: only when use_progress=True
                        if self.use_progress and not is_post_frame:
                            # Tail exclusion: last tail_ratio% of segment
                            if tail_ratio > 0 and obs > tail_start:
                                n_tail_excluded += 1
                                continue
                            obs_idx = obs - start
                            if obs_idx < 0 or obs_idx >= len(grip_traj):
                                n_uncertain += 1
                                continue
                            label = classify_gripper_confident(
                                action_type, grip_traj, obs_idx, g_min, g_max)
                            if label is None:
                                n_uncertain += 1
                                continue

                        samples.append(self._sample_mode_tuple(
                            eidx, si, obs, SAMPLE_MODE_NORMAL, -1))
                        if self.use_retarget_aug:
                            retarget_si = self._select_retarget_target_segment(ep, eidx, si, obs)
                            if retarget_si is None:
                                n_retarget_skip += 1
                            else:
                                samples.append(self._sample_mode_tuple(
                                    eidx, si, obs, SAMPLE_MODE_RETARGET, int(retarget_si)))
                                n_retarget += 1
                else:  # push/turn_on/close/open
                    for obs in range(start, post_end):
                        n_total += 1
                        is_post_frame = obs > end

                        # Uncertain filtering: only when use_progress=True
                        if self.use_progress and not is_post_frame:
                            # Tail exclusion: last tail_ratio% of segment
                            if tail_ratio > 0 and obs > tail_start:
                                n_tail_excluded += 1
                                continue
                            if action_type == "turn_on":
                                # Rotation-based filtering
                                key = (eidx, si)
                                if key not in self.segment_rotation_progress:
                                    n_uncertain += 1
                                    continue
                                rot_prog = self.segment_rotation_progress[key]
                                obs_idx = obs - start
                                if obs_idx >= len(rot_prog):
                                    n_uncertain += 1
                                    continue
                                p = rot_prog[obs_idx]
                                if TURNON_ROT_LOW <= p <= TURNON_ROT_HIGH:
                                    n_uncertain += 1
                                    continue  # uncertain zone
                            else:
                                # push/open/close: skip last N frames before end_frame
                                if end - NON_GRASP_UNCERTAIN_TAIL < obs <= end:
                                    n_uncertain += 1
                                    continue

                        samples.append(self._sample_mode_tuple(
                            eidx, si, obs, SAMPLE_MODE_NORMAL, -1))
                        if self.use_retarget_aug:
                            retarget_si = self._select_retarget_target_segment(ep, eidx, si, obs)
                            if retarget_si is None:
                                n_retarget_skip += 1
                            else:
                                samples.append(self._sample_mode_tuple(
                                    eidx, si, obs, SAMPLE_MODE_RETARGET, int(retarget_si)))
                                n_retarget += 1

                # Reverse augmentation: release-from-grasp pairs
                if self.use_reverse_aug and action_type == "release" and si > 0:
                    prev_seg = segments[si - 1]
                    if prev_seg.get("action_type") == "grasp":
                        # Find first grip frame
                        first_grip = self._find_first_grip_frame(eidx, si, seg)
                        prev_obj_id = str(prev_seg.get("primary_object_id", ""))
                        curr_obj_id = str(seg.get("primary_object_id", ""))
                        reverse_goal_mask = self._decode_object_mask(
                            ep, start, prev_obj_id, is_post=False,
                            end_frame=end, start_frame=start,
                        )
                        if reverse_goal_mask is None:
                            n_rev_missing_mask += (end - first_grip + 1)
                            continue

                        for obs in range(first_grip, end + 1):
                            current_obj_mask = self._decode_object_mask(
                                ep, obs, curr_obj_id, is_post=False,
                                end_frame=end, start_frame=start,
                            )
                            if current_obj_mask is None:
                                n_rev_missing_mask += 1
                                continue

                            overlap = self._mask_overlap_ratio(
                                reverse_goal_mask, current_obj_mask)
                            if overlap > self.reverse_max_mask_overlap:
                                n_rev_overlap_skip += 1
                                continue
                            samples.append(self._sample_mode_tuple(
                                eidx, si, obs, SAMPLE_MODE_REVERSE, int(si - 1)))

        logger.info("Sample filtering: %d total -> %d kept, %d uncertain excluded (%.1f%%)",
                     n_total, len([s for s in samples if self._parse_sample_tuple(s)[3] == SAMPLE_MODE_NORMAL]),
                     n_uncertain, 100 * n_uncertain / max(n_total, 1))
        if n_tail_excluded > 0:
            logger.info("Tail exclusion (ratio=%.3f): %d frames excluded (%.1f%%)",
                         tail_ratio, n_tail_excluded, 100 * n_tail_excluded / max(n_total, 1))
        if self.use_reverse_aug:
            n_rev = sum(1 for s in samples if self._parse_sample_tuple(s)[3] == SAMPLE_MODE_REVERSE)
            logger.info("Reverse augmentation: %d samples", n_rev)
            logger.info(
                "Reverse mask filter: %d skipped by overlap(>%.3f), %d skipped by missing mask",
                n_rev_overlap_skip, self.reverse_max_mask_overlap, n_rev_missing_mask,
            )
        if self.use_retarget_aug:
            logger.info(
                "Retarget augmentation[%s]: %d samples (%d skipped)",
                self.retarget_version, n_retarget, n_retarget_skip,
            )
        return samples

    def _effective_target_subtask_idx(
        self,
        eidx: int,
        si: int,
        mode: int,
        target_si: int,
    ) -> Optional[int]:
        segs = self.episodes.get(eidx, {}).get("subtask_segments", [])
        if si < 0 or si >= len(segs):
            return None
        if mode == SAMPLE_MODE_REVERSE:
            tsi = si - 1
            return tsi if 0 <= tsi < len(segs) else None
        if mode == SAMPLE_MODE_RETARGET and 0 <= target_si < len(segs):
            return int(target_si)
        return int(si)

    def _build_goal_consistency_groups(self) -> None:
        """Build (episode, target-subtask) groups from normal+augmentation samples.

        - Triplet pools: non-post samples grouped by same target.
        - Post pools: any post sample grouped by episode.
        """
        groups: Dict[Tuple[int, int], List[int]] = {}
        sample_group: List[Optional[Tuple[int, int]]] = [None] * len(self.samples)
        post_by_episode: Dict[int, List[int]] = {}
        all_samples: List[int] = list(range(len(self.samples)))

        for idx, sample in enumerate(self.samples):
            eidx, si, obs_frame, mode, target_si = self._parse_sample_tuple(sample)
            segs = self.episodes.get(eidx, {}).get("subtask_segments", [])
            if si < 0 or si >= len(segs):
                continue
            seg = segs[si]
            is_post = int(obs_frame) > int(seg.get("end_frame", obs_frame))
            if is_post:
                post_by_episode.setdefault(int(eidx), []).append(int(idx))
                continue

            tgt_si = self._effective_target_subtask_idx(eidx, si, mode, target_si)
            if tgt_si is None:
                continue
            key = (int(eidx), int(tgt_si))
            groups.setdefault(key, []).append(int(idx))
            sample_group[idx] = key

        filtered: Dict[Tuple[int, int], List[int]] = {}
        anchors: List[int] = []
        for key, idxs in groups.items():
            eidx = int(key[0])
            if len(idxs) <= 0 or len(post_by_episode.get(eidx, [])) <= 0:
                for j in idxs:
                    sample_group[j] = None
                continue
            filtered[key] = idxs
            anchors.extend(idxs)

        self.goal_consistency_groups = filtered
        self.episode_post_indices = post_by_episode
        self.goal_consistency_group_of_sample = sample_group
        self.goal_consistency_anchor_indices = anchors
        self.goal_pair_batch_anchor_indices = all_samples

        key_to_gid = {k: i for i, k in enumerate(sorted(self.goal_consistency_groups.keys()))}
        gid_of_sample: List[int] = [-1] * len(self.samples)
        for i, key in enumerate(self.goal_consistency_group_of_sample):
            if key is not None and key in key_to_gid:
                gid_of_sample[i] = int(key_to_gid[key])
        self.goal_consistency_group_id_of_sample = gid_of_sample

        if self.is_training:
            logger.info(
                "Goal consistency: %d target groups, %d post episodes, %d grouped non-post, %d anchors",
                len(self.goal_consistency_groups),
                len(self.episode_post_indices),
                len(self.goal_consistency_anchor_indices),
                len(self.goal_pair_batch_anchor_indices),
            )

    def sample_goal_consistency_partner(
        self,
        anchor_idx: int,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Sample another frame from the same goal-consistency group."""
        if rng is None:
            rng = random

        if anchor_idx < 0 or anchor_idx >= len(self.goal_consistency_group_of_sample):
            return int(anchor_idx)
        key = self.goal_consistency_group_of_sample[anchor_idx]
        if key is None:
            pool = self.goal_pair_batch_anchor_indices
            if len(pool) <= 1:
                return int(anchor_idx)
            partner = int(anchor_idx)
            for _ in range(8):
                cand = int(pool[rng.randrange(len(pool))])
                if cand != anchor_idx:
                    partner = cand
                    break
            return partner

        members = self.goal_consistency_groups.get(key, [])
        if len(members) <= 1:
            return int(anchor_idx)
        if len(members) == 2:
            return int(members[0] if members[1] == anchor_idx else members[1])

        partner = int(anchor_idx)
        for _ in range(8):
            cand = int(members[rng.randrange(len(members))])
            if cand != anchor_idx:
                partner = cand
                break
        if partner == anchor_idx:
            # Deterministic fallback
            pos = members.index(anchor_idx)
            partner = int(members[(pos + 1) % len(members)])
        return partner

    def _find_first_grip_frame(self, eidx: int, si: int, seg: Dict) -> int:
        """Find first frame where gripper is firmly gripping (for reverse aug).

        Falls back to segment start when grip timing cannot be estimated.
        """
        start = seg["start_frame"]
        key = (eidx, si)
        if key not in self.segment_gripper_ranges:
            return start
        g_min, g_max, grip_traj = self.segment_gripper_ranges[key]
        for i in range(len(grip_traj)):
            if not np.isnan(grip_traj[i]):
                norm_pos = (grip_traj[i] - g_min) / max(g_max - g_min, 1e-6)
                # For release: gripper starts closed (low norm_pos)
                if norm_pos < 0.3:
                    return start + i
        return start

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        eidx, si, obs_frame, sample_mode, target_si = self._parse_sample_tuple(self.samples[idx])
        is_reverse = sample_mode == SAMPLE_MODE_REVERSE
        is_retarget = sample_mode == SAMPLE_MODE_RETARGET
        ep = self.episodes[eidx]
        seg = ep["subtask_segments"][si]
        target_seg = seg
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        primary_obj_id = str(seg["primary_object_id"])
        action_type_str = seg["action_type"]
        target_obj_id = primary_obj_id
        if is_reverse and si > 0:
            # Reverse samples use the previous (grasp) target object and action type.
            prev_seg = ep["subtask_segments"][si - 1]
            target_seg = prev_seg
            target_obj_id = str(prev_seg.get("primary_object_id", primary_obj_id))
            action_type_str = prev_seg.get("action_type", action_type_str)
        elif (is_retarget) and 0 <= target_si < len(ep["subtask_segments"]):
            inter_seg = ep["subtask_segments"][target_si]
            target_seg = inter_seg
            target_obj_id = str(inter_seg.get("primary_object_id", primary_obj_id))
            action_type_str = inter_seg.get("action_type", action_type_str)
        action_type_id = ACTION_TYPE_MAP.get(action_type_str, 0)
        goal_frame = self._get_assess_goal_frame(seg)
        is_post = obs_frame > end_frame

        # ── Subtask progress (0→1 within segment) ──
        seg_length = max(1, end_frame - start_frame)
        subtask_progress = max(0.0, min(1.0, float(obs_frame - start_frame) / seg_length))

        # ── Vision input ──
        # Online multi-scale mode reads compact RGB observations. The frozen
        # DINO backbone runs once on the concatenated third+wrist batch in the
        # model. Offline mode retains the packed single-scale feature path.
        if self.packed_images is not None:
            def get_image(view: str) -> torch.Tensor:
                image = self.packed_images.get(eidx, obs_frame, view)
                if image is None:
                    image = self.packed_images.get(eidx, end_frame, view)
                if image is None:
                    image = self.packed_images.get(eidx, start_frame, view)
                if image is None:
                    size = self.packed_images.input_size
                    image = torch.zeros(3, size, size, dtype=torch.float32)
                return image

            image_third = get_image("third")
            image_wrist = get_image("wrist")
        else:
        # Stage-2 progress uses only the final DINO scale. Skip the two
        # intermediate scales used only by the action decoder.
            load_multiscale = self.packed_dino.multi_scale and not self.use_progress
            feature_getter = (
                self.packed_dino.get_multiscale
                if load_multiscale
                else self.packed_dino.get
            )
            d3 = feature_getter(eidx, obs_frame, "third")
            if d3 is None:
                d3 = feature_getter(eidx, end_frame, "third")
            if d3 is None:
                d3 = feature_getter(eidx, start_frame, "third")
            zero_shape = (
                (self.packed_dino.num_scales, self.dino_num_tokens, self.dino_hidden_dim)
                if load_multiscale
                else (self.dino_num_tokens, self.dino_hidden_dim)
            )
            dino_third = (
                d3 if d3 is not None
                else torch.zeros(*zero_shape)
            )

            dw = feature_getter(eidx, obs_frame, "wrist")
            if dw is None:
                dw = feature_getter(eidx, end_frame, "wrist")
            if dw is None:
                dw = feature_getter(eidx, start_frame, "wrist")
            dino_wrist = (
                dw if dw is not None
                else torch.zeros(*zero_shape)
            )

        # ── Object mask ──
        # Reverse aug should point to the reverse-goal region (release start),
        # not the object's region at the current release frame.
        mask_frame = obs_frame
        mask_is_post = is_post
        if is_reverse:
            mask_frame = start_frame
            mask_is_post = False
        elif is_retarget:
            # Interaction/navigate augmentation prompts the target object/place in current scene.
            mask_frame = obs_frame
            mask_is_post = False
        mask_data = self._get_mask_data(
            ep, mask_frame, target_obj_id, mask_is_post, end_frame, start_frame)
        if is_reverse and not mask_data and mask_frame != obs_frame:
            # Fallback: keep sample valid even if goal-frame mask is missing.
            mask_data = self._get_mask_data(
                ep, obs_frame, target_obj_id, is_post, end_frame, start_frame)
        if (is_retarget) and not mask_data:
            # Fallback to target segment interaction frame when target is occluded now.
            t_start = int(target_seg.get("start_frame", obs_frame))
            t_end = int(target_seg.get("end_frame", t_start))
            t_goal = self._segment_target_frame(target_seg)
            mask_data = self._get_mask_data(
                ep, t_goal, target_obj_id, False, t_end, t_start)
            if not mask_data:
                mask_data = self._get_mask_data(
                    ep, t_start, target_obj_id, False, t_end, t_start)

        rle = mask_data.get("mask_visible_agent")
        if rle and "counts" in rle:
            third_mask = decode_rle_mask_with_bbox(rle, mask_data.get("bbox_agent"))
        else:
            third_mask = np.zeros((256, 256), dtype=np.uint8)
        wrist_mask = np.zeros_like(third_mask, dtype=np.uint8)
        if self.use_wrist_goal_mask:
            wrist_rle = _get_wrist_mask_rle(mask_data)
            if wrist_rle is not None:
                wrist_mask = decode_rle_mask_with_bbox(wrist_rle, _get_wrist_bbox(mask_data))

        # Target place mask: grasp/push uses the next subtask's object mask.
        target_place_third_raw = np.zeros_like(third_mask, dtype=np.uint8)
        target_place_wrist_raw = np.zeros_like(third_mask, dtype=np.uint8)
        if self.use_target_place and not is_reverse:
            next_seg = None
            segs = ep.get("subtask_segments", [])
            if not is_retarget:
                if action_type_str in ("grasp", "push") and si + 1 < len(segs):
                    next_seg = segs[si + 1]
            elif self.retarget_version == "stateful_v1":
                if action_type_str in ("grasp", "push") and target_si + 1 < len(segs):
                    next_seg = segs[target_si + 1]

            if next_seg is not None:
                next_obj_id = str(
                    next_seg.get("target_object_id")
                    or next_seg.get("primary_object_id", "")
                )
                next_end = int(next_seg.get("end_frame", end_frame))
                next_start = int(next_seg.get("start_frame", start_frame))
                place_is_post = is_post and not is_retarget
                place_data = self._get_mask_data(
                    ep, obs_frame, next_obj_id, place_is_post, next_end, next_start)
                if not place_data:
                    fallback_place_frame = next_start if is_retarget else start_frame
                    place_data = self._get_mask_data(
                        ep, fallback_place_frame, next_obj_id, False, next_end, next_start)
                place_rle = place_data.get("mask_visible_agent") if place_data else None
                if place_rle and "counts" in place_rle:
                    target_place_third_raw = decode_rle_mask(place_rle)
                if self.use_wrist_target_place:
                    place_rle_wrist = _get_wrist_mask_rle(place_data)
                    if place_rle_wrist is not None:
                        target_place_wrist_raw = decode_rle_mask_with_bbox(
                            place_rle_wrist, _get_wrist_bbox(place_data)
                        )

        mask_augmentation_mode = "clean"
        mask_augmentation_changed = False
        if self.packed_images is not None:
            # Online DINOv2-L uses a 14px patch size.
            patch_grid = int(self.packed_images.input_size) // 14
        else:
            patch_grid = dino_patch_grid_from_tokens(
                self.dino_num_tokens,
                patch_only=bool(self.packed_dino.patch_only),
            )
        if self.is_training and self.mask_augmentation != "none":
            augmented = augment_multiview_masks(
                {
                    "goal": (third_mask, wrist_mask),
                    "target_place": (target_place_third_raw, target_place_wrist_raw),
                },
                recipe=self.mask_augmentation,
                visibility_grid=patch_grid,
            )
            third_mask, wrist_mask = augmented.pairs["goal"]
            target_place_third_raw, target_place_wrist_raw = augmented.pairs["target_place"]
            mask_augmentation_mode = augmented.mode
            mask_augmentation_changed = augmented.changed

        goal_mask_third = downsample_mask_to_patches(third_mask, grid=patch_grid)
        goal_mask_wrist = downsample_mask_to_patches(wrist_mask, grid=patch_grid)
        target_place_mask_third = downsample_mask_to_patches(
            target_place_third_raw, grid=patch_grid
        )
        target_place_mask_wrist = downsample_mask_to_patches(
            target_place_wrist_raw, grid=patch_grid
        )

        # ── Text feature ──
        text_feat_full = self.text_cache.get(target_seg.get("description", ""))
        text_feat = text_feat_full.copy()
        if is_reverse:
            drop_prob = self.reverse_text_dropout_prob
        elif is_retarget:
            drop_prob = self.retarget_text_dropout_prob
        else:
            drop_prob = self.text_dropout_prob
        if self.is_training and drop_prob > 0.0 and random.random() < drop_prob:
            text_feat = np.zeros(self.text_feature_dim, dtype=np.float32)

        # ── State (8D from parquet observation.state) ──
        full_states = self.episode_full_states.get(eidx)
        if full_states is not None and obs_frame < len(full_states):
            state = full_states[obs_frame].copy()
        elif is_post and full_states is not None and end_frame < len(full_states):
            state = full_states[end_frame].copy()
        else:
            state = np.zeros(8, dtype=np.float32)

        # obs_xyz, obs_rpy, obs_gripper are needed for progress GT and retarget aug
        obs_xyz = state[:3].copy()
        obs_rpy = state[3:6].copy()
        obs_gripper = float(state[6]) if len(state) > 6 else 0.0

        # GT labels: distance, alignment, task_completion
        gt_distance = 1.0
        gt_alignment = 1.0
        gt_task_completion = 1.0

        if self.use_progress:
            if is_reverse:
                # Reverse objective: return from current release frame back to release start.
                reverse_goal_frame = start_frame
                target_xyz = self._get_frame_world_pos(ep, reverse_goal_frame)
                if target_xyz is None:
                    target_xyz = obs_xyz
                gt_distance = compute_distance_score(obs_xyz, target_xyz, self.distance_alpha)
                obs_rotmat = self._get_rotmat_at_frame(eidx, obs_frame)
                target_rotmat = self._get_rotmat_at_frame(eidx, reverse_goal_frame)
                gt_alignment = compute_alignment_score(obs_rotmat, target_rotmat)
                gt_task_completion = compute_task_completion_score(obs_xyz, target_xyz)
            elif is_retarget:
                # Interaction/navigate target objective:
                # use target segment interaction pose from same episode.
                tgt_xyz, _, tgt_frame = self._segment_target_pose(
                    ep, eidx, target_seg, fallback_xyz=obs_xyz
                )
                gt_distance = compute_distance_score(obs_xyz, tgt_xyz, self.distance_alpha)
                obs_rotmat = self._get_rotmat_at_frame(eidx, obs_frame)
                target_rotmat = self._get_rotmat_at_frame(eidx, tgt_frame)
                gt_alignment = compute_alignment_score(obs_rotmat, target_rotmat)
                gt_task_completion = self._interaction_task_completion(
                    eidx=eidx,
                    obs_xyz=obs_xyz,
                    target_xyz=tgt_xyz,
                    obs_gripper=obs_gripper,
                    target_action_type=action_type_str,
                )
            elif action_type_str in ("grasp", "release"):
                # Target = goal_frame (move phase end)
                if obs_frame <= goal_frame:
                    target_xyz = self._get_frame_world_pos(ep, goal_frame)
                    if target_xyz is None:
                        target_xyz = obs_xyz
                    gt_distance = compute_distance_score(obs_xyz, target_xyz, self.distance_alpha)
                    obs_rotmat = self._get_rotmat_at_frame(eidx, obs_frame)
                    target_rotmat = self._get_rotmat_at_frame(eidx, goal_frame)
                    gt_alignment = compute_alignment_score(obs_rotmat, target_rotmat)

                # TC-event-v3 is the sole completion-label source for progress
                # training.  Do not run the legacy per-segment gripper
                # classifier first: valid post-event samples can extend beyond
                # that legacy trajectory slice and the value would be
                # overwritten by _tc_event_label below in any case.
                if self.tc_events:
                    gt_task_completion = 0.0
                else:
                    key = (eidx, si)
                    if key in self.segment_gripper_ranges:
                        g_min, g_max, grip_traj = self.segment_gripper_ranges[key]
                        obs_idx_in_seg = obs_frame - start_frame
                        label = classify_gripper_confident(
                            action_type_str, grip_traj, obs_idx_in_seg, g_min, g_max)
                        if (
                            action_type_str == "release"
                            and label == 1.0
                            and obs_frame <= goal_frame
                        ):
                            gt_task_completion = 0.0
                        else:
                            gt_task_completion = label if label is not None else 0.0
                    else:
                        gt_task_completion = 0.0
            else:
                # push/turn_on/close/open: target = end_frame
                if obs_frame <= end_frame:
                    target_xyz = self._get_frame_world_pos(ep, end_frame)
                    if target_xyz is None:
                        target_xyz = obs_xyz
                    gt_distance = compute_distance_score(obs_xyz, target_xyz, self.distance_alpha)
                    obs_rotmat = self._get_rotmat_at_frame(eidx, obs_frame)
                    target_rotmat = self._get_rotmat_at_frame(eidx, end_frame)
                    gt_alignment = compute_alignment_score(obs_rotmat, target_rotmat)

                    if self.tc_events:
                        gt_task_completion = 0.0
                    elif action_type_str == "turn_on":
                # Rotation-based TC for turn_on
                        key = (eidx, si)
                        if key in self.segment_rotation_progress:
                            rot_prog = self.segment_rotation_progress[key]
                            obs_idx = obs_frame - start_frame
                            if 0 <= obs_idx < len(rot_prog):
                                p = rot_prog[obs_idx]
                                if p < TURNON_ROT_LOW:
                                    gt_task_completion = 0.0
                                elif p > TURNON_ROT_HIGH:
                                    gt_task_completion = 1.0
                                else:
                                    gt_task_completion = 0.0  # filtered in _build_samples
                            else:
                                gt_task_completion = 0.0
                        else:
                            gt_task_completion = 0.0
                    else:
                # Distance-based TC for push/open/close
                        gt_task_completion = compute_task_completion_score(obs_xyz, target_xyz)

            # Post-only TC: override TC for normal (non-reverse, non-retarget)
            # samples. TC=0 during action, TC=1 only in post frames.
            # For episode-terminal segments (no room for post frames),
            # treat the last max_post_subtask_frames as TC=1.
            # When post_only_tc_types is set, only apply to those action types
            # (hybrid mode: e.g. gripper-based TC for grasp/release,
            #  post_only_tc for turn_on/push/open/close).
            apply_post_only = (
                self.post_only_tc and not is_reverse and not is_retarget
                and (self.post_only_tc_types is None or action_type_str in self.post_only_tc_types)
            )
            if apply_post_only:
                if is_post:
                    gt_task_completion = 1.0
                elif end_frame + 1 >= len(self.episode_actions[eidx]):
                    # Terminal segment: last N frames → TC=1
                    frames_to_end = end_frame - obs_frame
                    gt_task_completion = 1.0 if frames_to_end < self.max_post_subtask_frames else 0.0
                else:
                    gt_task_completion = 0.0

            if is_post:
                # Post-subtask frames are considered task-complete for distance/alignment.
                gt_distance = 1.0
                gt_alignment = 1.0

            if not is_reverse and not is_retarget:
                tc_label = self._tc_event_label(eidx, si, obs_frame)
                if tc_label is None:
                    raise RuntimeError(
                        "TC IGNORE frame reached __getitem__; sample filtering is inconsistent: "
                        f"episode={eidx}, subtask_index={si}, frame={obs_frame}"
                    )
                gt_task_completion = tc_label

        # ── Action + Plan targets ──
        actions_raw = self.episode_actions[eidx]
        plan_waypoints = np.zeros((self.num_plan_waypoints, 3), dtype=np.float32)
        if is_reverse:
            action_targets = self._compute_reverse_action_targets(
                actions_raw, obs_frame, start_frame)
            if self.use_plan:
                plan_waypoints = self._compute_reverse_plan(
                    eidx, obs_frame, start_frame, obs_xyz)
        elif is_retarget:
            # Unified: build one shared path, derive both plan and action
            plan_waypoints, action_targets = self._compute_interaction_plan_and_action(
                ep=ep,
                eidx=eidx,
                obs_frame=obs_frame,
                source_si=si,
                target_seg=target_seg,
                obs_xyz=obs_xyz,
                obs_rpy=obs_rpy,
                obs_gripper=obs_gripper,
                compute_plan=self.use_plan,
            )
        else:
            action_targets = np.zeros((self.num_action_steps, 7), dtype=np.float32)
            end_f = min(obs_frame + self.num_action_steps, len(actions_raw))
            n_avail = end_f - obs_frame
            if n_avail > 0:
                action_targets[:n_avail] = actions_raw[obs_frame:end_f, :7]
            if self.use_plan and not is_post:
                plan_goal_frame = self._get_plan_goal_frame(seg, obs_frame)
                plan_goal_frame = min(plan_goal_frame, len(actions_raw))
                plan_waypoints = self._compute_plan_waypoints(
                    eidx, obs_frame, plan_goal_frame, obs_xyz)

        prof = self.episode_motion_profiles.get(eidx, {})
        action_xyz_scale = CONTROLLER_XYZ_SCALE_MEAN
        gid = -1
        if idx < len(self.goal_consistency_group_id_of_sample):
            gid = int(self.goal_consistency_group_id_of_sample[idx])

        result = {
            "goal_mask_third": torch.from_numpy(goal_mask_third),
            "goal_mask_wrist": torch.from_numpy(goal_mask_wrist),
            "target_place_mask_third": torch.from_numpy(target_place_mask_third),
            "target_place_mask_wrist": torch.from_numpy(target_place_mask_wrist),
            "text_feat": torch.from_numpy(text_feat),
            "text_feat_full": torch.from_numpy(text_feat_full),
            "state": torch.from_numpy(state),
            "gt_action": torch.from_numpy(action_targets),
            "gt_plan": torch.from_numpy(plan_waypoints),
            "action_xyz_scale": torch.tensor(action_xyz_scale, dtype=torch.float32),
            "gt_distance": torch.tensor(gt_distance, dtype=torch.float32),
            "gt_alignment": torch.tensor(gt_alignment, dtype=torch.float32),
            "gt_task_completion": torch.tensor(gt_task_completion, dtype=torch.float32),
            "action_type": torch.tensor(action_type_id, dtype=torch.long),
            "is_reverse": torch.tensor(is_reverse, dtype=torch.bool),
            "is_post_subtask": torch.tensor(is_post, dtype=torch.bool),
            "is_retarget_aug": torch.tensor(is_retarget, dtype=torch.bool),
            "mask_augmentation_mode": torch.tensor(
                mask_augmentation_mode_id(mask_augmentation_mode), dtype=torch.long
            ),
            "mask_augmentation_changed": torch.tensor(
                mask_augmentation_changed, dtype=torch.bool
            ),
            "sample_index": torch.tensor(int(idx), dtype=torch.long),
            "goal_consistency_gid": torch.tensor(gid, dtype=torch.long),
            "subtask_progress": torch.tensor(subtask_progress, dtype=torch.float32),
        }
        if self.packed_images is not None:
            result["image_third"] = image_third
            result["image_wrist"] = image_wrist
        else:
            result["dino_third"] = dino_third
            result["dino_wrist"] = dino_wrist
        return result

    def _get_mask_data(self, ep, obs_frame, obj_id, is_post, end_frame, start_frame):
        """Get object mask data with fallback for post-subtask frames."""
        frame_data = ep["frames"].get(str(obs_frame), {})
        objects = frame_data.get("objects", {})
        obj_data = objects.get(obj_id, {})
        if not obj_data and is_post:
            for fallback_f in [end_frame, start_frame]:
                fd = ep["frames"].get(str(fallback_f), {})
                od = fd.get("objects", {}).get(obj_id, {})
                if od:
                    return od
        return obj_data

    def _decode_object_mask(
        self,
        ep: Dict[str, Any],
        frame: int,
        obj_id: str,
        is_post: bool,
        end_frame: int,
        start_frame: int,
    ) -> Optional[np.ndarray]:
        obj = self._get_mask_data(ep, frame, obj_id, is_post, end_frame, start_frame)
        rle = obj.get("mask_visible_agent") if obj else None
        if not (rle and "counts" in rle):
            return None
        m = decode_rle_mask_with_bbox(rle, obj.get("bbox_agent")).astype(np.uint8)
        if m.sum() == 0:
            return None
        return m

    @staticmethod
    def _mask_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = (mask_a > 0)
        b = (mask_b > 0)
        inter = float((a & b).sum())
        if inter <= 0.0:
            return 0.0
        den = float(min(a.sum(), b.sum()))
        if den <= 0.0:
            return 0.0
        return inter / den

    def _compute_plan_waypoints(
        self, eidx: int, obs_frame: int, goal_frame: int, obs_xyz: np.ndarray,
    ) -> np.ndarray:
        """Compute plan waypoints from obs_frame to goal_frame (relative to obs_xyz)."""
        ep = self.episodes[eidx]
        waypoints = []
        for f in range(obs_frame, goal_frame + 1):
            pos = self._get_frame_world_pos(ep, f)
            if pos is not None:
                waypoints.append(pos.astype(np.float32))

        if len(waypoints) < 2:
            return np.zeros((self.num_plan_waypoints, 3), dtype=np.float32)

        traj = np.array(waypoints)
        sampled = arc_length_sample(traj, self.num_plan_waypoints)
        return (sampled - obs_xyz).astype(np.float32)

    def _select_and_warp_template(
        self,
        obs_xyz: np.ndarray,
        target_xyz: np.ndarray,
        action_type: str,
    ) -> Optional[np.ndarray]:
        """Select best matching trajectory template and warp to current obs→target."""
        if self.retarget_templates is None:
            return None
        tkey = f"templates_{action_type}"
        dkey = f"directions_{action_type}"
        mkey = f"magnitudes_{action_type}"
        if tkey not in self.retarget_templates:
            return None
        templates = self.retarget_templates[tkey]
        directions = self.retarget_templates[dkey]
        magnitudes = self.retarget_templates[mkey]
        if len(templates) == 0:
            return None
        desired_disp = target_xyz - obs_xyz
        desired_mag = float(np.linalg.norm(desired_disp))
        if desired_mag < 1e-6:
            return None
        desired_dir = desired_disp / desired_mag
        cos_sims = directions @ desired_dir
        good_mask = cos_sims > 0.5
        mag_ratio = desired_mag / np.maximum(magnitudes, 1e-6)
        mag_ok = (mag_ratio > 0.3) & (mag_ratio < 3.0)
        best_mask = good_mask & mag_ok
        if best_mask.any():
            idxs = np.where(best_mask)[0]
            chosen = idxs[random.randint(0, len(idxs) - 1)]
        elif good_mask.any():
            idxs = np.where(good_mask)[0]
            chosen = idxs[random.randint(0, len(idxs) - 1)]
        else:
            chosen = int(np.argmax(cos_sims))
        R = rotation_between_vectors(directions[chosen], desired_dir)
        warped = (R @ templates[chosen].T).T * desired_mag + obs_xyz
        return warped.astype(np.float32)

    def _build_interaction_path(
        self,
        start_xyz: np.ndarray,
        target_xyz: np.ndarray,
        action_type: str,
        eidx: int,
    ) -> np.ndarray:
        # Try template warping first
        if self.retarget_templates is not None:
            warped = self._select_and_warp_template(start_xyz, target_xyz, action_type)
            if warped is not None and len(warped) >= 2:
                return warped

        # Fallback: linear + safe lift
        prof = self.episode_motion_profiles.get(eidx, {})
        lift = float(np.clip(4.0 * prof.get("median_world_step", 0.005), 0.04, 0.18))
        start = start_xyz.astype(np.float32)
        target = target_xyz.astype(np.float32)
        z_safe = float(max(start[2], target[2]) + lift)

        waypoints: List[np.ndarray] = [start]
        if z_safe - start[2] > 0.012:
            waypoints.append(np.asarray([start[0], start[1], z_safe], dtype=np.float32))
        if float(np.linalg.norm(target[:2] - start[:2])) > 0.015:
            waypoints.append(np.asarray([target[0], target[1], z_safe], dtype=np.float32))
        waypoints.append(target)

        dense: List[np.ndarray] = [waypoints[0]]
        desired_step = float(
            np.clip(1.9 * prof.get("median_world_step", 0.005), 0.01, 0.05)
        )
        for i in range(len(waypoints) - 1):
            a = waypoints[i]
            b = waypoints[i + 1]
            seg_len = float(np.linalg.norm(b - a))
            n = int(max(2, np.ceil(seg_len / max(desired_step, 1e-4))))
            for j in range(1, n + 1):
                dense.append((a + (b - a) * (j / n)).astype(np.float32))
        return np.asarray(dense, dtype=np.float32)

    def _compute_interaction_plan_and_action(
        self,
        ep: Dict[str, Any],
        eidx: int,
        obs_frame: int,
        source_si: int,
        target_seg: Dict[str, Any],
        obs_xyz: np.ndarray,
        obs_rpy: np.ndarray,
        obs_gripper: float,
        compute_plan: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build interaction plan and action using approach + replay structure.

        Approach phase: synthetic path from obs to closest point on reference trajectory.
        Replay phase: actual recorded action commands from the reference trajectory.

        Falls back to pure synthetic path when reference trajectory is unavailable.

        Returns:
            (plan_waypoints (num_plan_waypoints, 3),
             action_targets (num_action_steps, 7))
        """
        if self.retarget_rollout_mode == "linear_interp_v1":
            return self._compute_interaction_linear_interp_baseline(
                ep=ep,
                eidx=eidx,
                obs_frame=obs_frame,
                source_si=source_si,
                target_seg=target_seg,
                obs_xyz=obs_xyz,
                obs_rpy=obs_rpy,
                obs_gripper=obs_gripper,
                compute_plan=compute_plan,
            )

        target_xyz, target_rpy, _ = self._segment_target_pose(
            ep, eidx, target_seg, fallback_xyz=obs_xyz
        )
        target_action_type = str(target_seg.get("action_type", "unknown")).lower()

        zero_plan = np.zeros((self.num_plan_waypoints, 3), dtype=np.float32)
        zero_action = np.zeros((self.num_action_steps, 7), dtype=np.float32)

        prof = self.episode_motion_profiles.get(eidx, {})
        med_world_step = float(prof.get("median_world_step", 0.005))
        xyz_gain = float(np.clip(prof.get("xyz_gain", 1.0), 0.01, 1000.0))

        # ── Reference trajectory from target segment ──
        ref_traj = self._get_segment_trajectory(eidx, target_seg)
        actions_raw = self.episode_actions.get(eidx)

        # Try approach+replay; fall back to pure synthetic on failure.
        result = None
        if len(ref_traj) >= 2 and actions_raw is not None:
            result = self._try_approach_replay(
                ep=ep, eidx=eidx, obs_frame=obs_frame, source_si=source_si,
                target_seg=target_seg, obs_xyz=obs_xyz, obs_rpy=obs_rpy,
                obs_gripper=obs_gripper, compute_plan=compute_plan,
                target_xyz=target_xyz, target_rpy=target_rpy,
                target_action_type=target_action_type,
                ref_traj=ref_traj, actions_raw=actions_raw,
                prof=prof,
            )
        if result is not None:
            self._debug_splice_ok = getattr(self, "_debug_splice_ok", 0) + 1
            return result
        self._debug_splice_fail = getattr(self, "_debug_splice_fail", 0) + 1

        # ── Fallback: pure synthetic path ──
        return self._compute_interaction_synthetic_fallback(
            ep=ep, eidx=eidx, obs_frame=obs_frame, source_si=source_si,
            target_seg=target_seg, obs_xyz=obs_xyz, obs_rpy=obs_rpy,
            obs_gripper=obs_gripper, compute_plan=compute_plan,
            target_xyz=target_xyz, target_rpy=target_rpy,
            target_action_type=target_action_type, prof=prof,
        )

    def _compute_interaction_linear_interp_baseline(
        self,
        ep: Dict[str, Any],
        eidx: int,
        obs_frame: int,
        source_si: int,
        target_seg: Dict[str, Any],
        obs_xyz: np.ndarray,
        obs_rpy: np.ndarray,
        obs_gripper: float,
        compute_plan: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Naive baseline: linearly interpolate current state to target interaction state.

        This intentionally ignores trajectory replay and junction selection.
        It uses only the current state and the target segment's interaction state,
        then converts state deltas back into controller-space action commands.
        """
        target_xyz, target_rpy, target_gripper, _ = self._segment_target_state(
            ep=ep,
            eidx=eidx,
            seg=target_seg,
            fallback_xyz=obs_xyz,
            fallback_rpy=obs_rpy,
            fallback_gripper=float(obs_gripper),
        )

        start_xyz = np.asarray(obs_xyz, dtype=np.float32)
        start_rpy = np.asarray(obs_rpy, dtype=np.float32)
        start_gripper = float(obs_gripper)

        t = np.linspace(0.0, 1.0, self.num_action_steps + 1, dtype=np.float32)
        xyz_traj = start_xyz[None, :] + (target_xyz - start_xyz)[None, :] * t[:, None]
        rpy_traj = start_rpy[None, :] + (target_rpy - start_rpy)[None, :] * t[:, None]
        gripper_traj = start_gripper + (target_gripper - start_gripper) * t

        xyz_cmd = np.clip(
            (xyz_traj[1:] - xyz_traj[:-1]) * CONTROLLER_XYZ_GAIN,
            -1.0,
            1.0,
        ).astype(np.float32)
        rpy_cmd = (rpy_traj[1:] - rpy_traj[:-1]).astype(np.float32)
        gripper_cmd = self._gripper_delta_to_action(
            eidx=eidx,
            delta=(gripper_traj[1:] - gripper_traj[:-1]),
        )

        action_targets = np.zeros((self.num_action_steps, 7), dtype=np.float32)
        action_targets[:, :3] = xyz_cmd
        action_targets[:, 3:6] = rpy_cmd
        action_targets[:, 6] = gripper_cmd

        if compute_plan:
            plan_sampled = arc_length_sample(xyz_traj, self.num_plan_waypoints)
            plan_waypoints = (plan_sampled - start_xyz).astype(np.float32)
        else:
            plan_waypoints = np.zeros((self.num_plan_waypoints, 3), dtype=np.float32)

        return plan_waypoints, action_targets.astype(np.float32)

    @staticmethod
    def _find_tangent_junction(
        obs_xyz: np.ndarray,
        ref_positions: np.ndarray,
    ) -> int:
        """Find tangent junction: point on ref trajectory where obs→ref direction
        aligns with the trajectory's tangent direction.

        For each ref point i:
        - t_i = normalize(ref[i+1] - ref[i-1])  # central-difference tangent
        - v_i = normalize(ref[i] - obs)          # obs→ref[i] direction
        - tangency = dot(v_i, t_i)               # positive = forward-facing

        Score = tangency * distance_weight * position_weight
        - Prefer earlier points (more replay remaining)
        - Prefer closer points (shorter approach)
        - Require forward tangency (approach direction matches ref flow)

        Returns index of best tangent point.
        """
        M = len(ref_positions)
        if M < 3:
            return int(np.argmin(np.linalg.norm(ref_positions - obs_xyz, axis=1)))

        # Central-difference tangent vectors
        tangents = np.zeros_like(ref_positions)
        tangents[0] = ref_positions[1] - ref_positions[0]
        tangents[-1] = ref_positions[-1] - ref_positions[-2]
        tangents[1:-1] = ref_positions[2:] - ref_positions[:-2]

        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, 1e-8)

        # obs→ref[i] direction vectors
        vecs = ref_positions - obs_xyz
        dists = np.linalg.norm(vecs, axis=1, keepdims=True).flatten()
        vecs_norm = vecs / np.maximum(dists[:, None], 1e-8)

        # Forward tangency: positive = obs→ref aligns with trajectory forward
        dot_signed = np.sum(vecs_norm * tangents, axis=1)

        # Validity: exclude too-close points and backward tangencies
        min_dist = float(dists.min()) * 0.3
        valid = (dists > min_dist) & (dot_signed > 0.1)

        # Reserve at least 2 replay points
        valid[M - 2:] = False

        if not np.any(valid):
            # Fallback: closest point in the first 80% of trajectory
            limit = max(1, int(M * 0.8))
            return int(np.argmin(dists[:limit]))

        # Composite score: tangency * inverse_distance * early_position_bonus
        max_dist = float(dists.max()) + 1e-8
        dist_weight = 1.0 - (dists / max_dist)  # closer = higher weight

        # Position weight: prefer first half of trajectory (more replay)
        position = np.arange(M, dtype=np.float32) / float(M)
        pos_weight = 1.0 - position  # earlier = higher

        score = dot_signed * dist_weight * pos_weight
        score[~valid] = -np.inf
        return int(np.argmax(score))

    @staticmethod
    def _build_hermite_approach(
        obs_xyz: np.ndarray,
        junction_xyz: np.ndarray,
        junction_tangent: np.ndarray,
        xyz_gain: float,
    ) -> np.ndarray:
        """Cubic Hermite spline: obs → junction, tangent-continuous at junction.

        H(t) = (2t³-3t²+1)P0 + (t³-2t²+t)M0 + (-2t³+3t²)P1 + (t³-t²)M1

        P0 = obs_xyz, P1 = junction_xyz
        M0 = chord direction * chord_len (depart toward junction)
        M1 = junction_tangent * chord_len (arrive along ref tangent)

        Step count is adaptively increased until no per-step action
        exceeds [-1, 1] (i.e., no clipping of delta * CONTROLLER_XYZ_GAIN).
        """
        chord = junction_xyz - obs_xyz
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-6:
            return obs_xyz.reshape(1, 3).astype(np.float32)

        M0 = chord  # departure tangent: toward junction
        M1 = junction_tangent * chord_len  # arrival tangent: ref trajectory direction

        # Initial step count from chord (straight-line estimate)
        n_steps = max(2, int(np.ceil(np.max(np.abs(chord) * CONTROLLER_XYZ_GAIN))))

        # Generate spline, check for clipping, increase steps if needed
        for _ in range(5):  # max 5 refinement iterations
            t = np.linspace(0, 1, n_steps + 1, dtype=np.float32)
            t2 = t * t
            t3 = t2 * t

            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2

            positions = (h00[:, None] * obs_xyz + h10[:, None] * M0 +
                         h01[:, None] * junction_xyz + h11[:, None] * M1)

            # Check if any per-step action would clip
            deltas = positions[1:] - positions[:-1]
            max_cmd = float(np.max(np.abs(deltas * CONTROLLER_XYZ_GAIN)))
            if max_cmd <= 1.0:
                break
            # Increase steps proportionally to the worst overshoot
            n_steps = max(n_steps + 1, int(np.ceil(n_steps * max_cmd)))

        return positions.astype(np.float32)

    def _try_approach_replay(
        self,
        ep: Dict[str, Any],
        eidx: int,
        obs_frame: int,
        source_si: int,
        target_seg: Dict[str, Any],
        obs_xyz: np.ndarray,
        obs_rpy: np.ndarray,
        obs_gripper: float,
        compute_plan: bool,
        target_xyz: np.ndarray,
        target_rpy: np.ndarray,
        target_action_type: str,
        ref_traj: List[Tuple[int, np.ndarray]],
        actions_raw: np.ndarray,
        prof: Dict[str, Any],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Tangent-junction approach + replay.

        1. Find tangent junction on ref trajectory.
        2. Build Hermite spline approach: obs→junction (smooth curve).
        3. Append ref positions junction→end as-is (replay).
        4. Derive action commands from consecutive XYZ deltas.
        5. GT action = first N steps.  GT plan = arc-length sample of full path.
        """
        M = len(ref_traj)
        if M < 2:
            return None

        ref_frames = np.array([t[0] for t in ref_traj], dtype=np.int64)
        ref_positions = np.array([t[1] for t in ref_traj], dtype=np.float32)

        # ── 1. Tangent junction ──
        junction_idx = self._find_tangent_junction(obs_xyz, ref_positions)

        # Quality check: junction too close to end → insufficient replay
        if junction_idx > M - 3:
            return None

        junction_xyz = ref_positions[junction_idx].copy()
        junction_frame = int(ref_frames[junction_idx])

        # ── 2. Junction tangent vector ──
        if 0 < junction_idx < M - 1:
            tangent = ref_positions[junction_idx + 1] - ref_positions[junction_idx - 1]
        elif junction_idx < M - 1:
            tangent = ref_positions[junction_idx + 1] - ref_positions[junction_idx]
        else:
            tangent = ref_positions[junction_idx] - ref_positions[junction_idx - 1]
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-8)

        # ── 3. Hermite spline approach: obs → junction ──
        approach_positions = self._build_hermite_approach(
            obs_xyz, junction_xyz, tangent,
            float(np.mean(CONTROLLER_XYZ_GAIN)))

        # Workspace bounds clipping
        ws_min = prof.get("workspace_min")
        ws_max = prof.get("workspace_max")
        if ws_min is not None and ws_max is not None:
            approach_positions = np.clip(approach_positions, ws_min, ws_max)

        approach_steps = len(approach_positions) - 1  # first point is obs

        # ── 4. Replay: ref positions after junction ──
        replay_positions = ref_positions[junction_idx + 1:]
        replay_len = len(replay_positions)

        if replay_len > 0:
            full_positions = np.concatenate(
                [approach_positions, replay_positions], axis=0
            )
        else:
            full_positions = approach_positions

        full_steps = len(full_positions) - 1
        if full_steps < 1:
            return None

        # ── 5. Derive action commands ──
        # Approach: per-axis gain converts EEF deltas to action commands
        approach_deltas = full_positions[1:approach_steps+1] - full_positions[:approach_steps]
        approach_cmds = np.clip(
            (approach_deltas * CONTROLLER_XYZ_GAIN).astype(np.float32), -1.0, 1.0
        )
        # Replay: use original parquet action XYZ (zero-error)
        replay_cmds = np.zeros((max(replay_len, 0), 3), dtype=np.float32)
        for i in range(replay_len):
            src_f = junction_frame + i
            if 0 <= src_f < len(actions_raw):
                replay_cmds[i] = actions_raw[src_f, :3]
        xyz_commands = np.concatenate([approach_cmds, replay_cmds], axis=0)[:full_steps]

        # ── 6. RPY & gripper ──
        src_seg = ep.get("subtask_segments", [])[source_si]
        if self.retarget_version == "stateful_v1":
            holding, _, _ = self._infer_retarget_source_state_v1(
                eidx=eidx,
                seg=src_seg,
                si=source_si,
                obs_frame=obs_frame,
                obs_gripper=float(obs_gripper),
            )
        else:
            holding = self._infer_holding_state(
                eidx=eidx, si=source_si, obs_frame=obs_frame,
                action_type=str(src_seg.get("action_type", "")),
                obs_gripper=float(obs_gripper),
            )
        # Travel gripper: maintain current holding state during approach
        travel_cmd = 1.0 if holding else -1.0

        junction_rpy = self._get_frame_abs_rpy(eidx, junction_frame)

        # ── 7. Build full action array ──
        full_actions = np.zeros((full_steps, 7), dtype=np.float32)
        full_actions[:, :3] = xyz_commands

        # Approach RPY: interpolate obs_rpy → junction_rpy
        rpy_prev = obs_rpy.copy()
        for i in range(approach_steps):
            t = float(i + 1) / float(approach_steps)
            rpy_i = obs_rpy + t * (junction_rpy - obs_rpy)
            full_actions[i, 3:6] = rpy_i - rpy_prev
            rpy_prev = rpy_i.copy()
            full_actions[i, 6] = travel_cmd

        # Replay: use original parquet action (XYZ + RPY + gripper)
        for i in range(replay_len):
            src_f = junction_frame + i
            if 0 <= src_f < len(actions_raw):
                full_actions[approach_steps + i, :7] = actions_raw[src_f, :7]
            elif i > 0:
                full_actions[approach_steps + i] = full_actions[approach_steps + i - 1]

        full_actions = np.clip(full_actions, -1.0, 1.0).astype(np.float32)

        # ── 8. Truncate/pad to fixed action chunk size ──
        N = self.num_action_steps
        if full_steps >= N:
            full_actions = full_actions[:N]
        else:
            padded = np.zeros((N, 7), dtype=np.float32)
            padded[:full_steps] = full_actions
            if full_steps > 0:
                padded[full_steps:, 6] = full_actions[-1, 6]  # hold last gripper
            full_actions = padded

        # ── 9. Plan from full world-space path ──
        if compute_plan:
            plan_sampled = arc_length_sample(
                full_positions, self.num_plan_waypoints
            )
            plan_waypoints = (plan_sampled - obs_xyz).astype(np.float32)
        else:
            plan_waypoints = np.zeros(
                (self.num_plan_waypoints, 3), dtype=np.float32
            )

        return plan_waypoints, full_actions

    def _compute_interaction_synthetic_fallback(
        self,
        ep: Dict[str, Any],
        eidx: int,
        obs_frame: int,
        source_si: int,
        target_seg: Dict[str, Any],
        obs_xyz: np.ndarray,
        obs_rpy: np.ndarray,
        obs_gripper: float,
        compute_plan: bool,
        target_xyz: np.ndarray,
        target_rpy: np.ndarray,
        target_action_type: str,
        prof: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: pure synthetic path when reference trajectory is unavailable."""
        zero_plan = np.zeros((self.num_plan_waypoints, 3), dtype=np.float32)
        zero_action = np.zeros((self.num_action_steps, 7), dtype=np.float32)

        path = self._build_interaction_path(
            start_xyz=obs_xyz,
            target_xyz=target_xyz,
            action_type=target_action_type,
            eidx=eidx,
        )
        if len(path) < 2:
            return zero_plan, zero_action

        # Plan
        if compute_plan:
            plan_sampled = arc_length_sample(path, self.num_plan_waypoints)
            plan_waypoints = (plan_sampled - obs_xyz).astype(np.float32)
        else:
            plan_waypoints = zero_plan

        # Action
        med_world_step = float(prof.get("median_world_step", 0.005))
        episode_action_reach = med_world_step * float(self.num_action_steps)

        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_path_len = float(np.sum(diffs))
        action_arc = min(episode_action_reach, total_path_len)
        cumlen = np.concatenate([[0.0], np.cumsum(diffs)])
        cut_idx = max(2, min(int(np.searchsorted(cumlen, action_arc)) + 1, len(path)))
        action_path = path[:cut_idx]

        move_steps = int(max(1, self.num_action_steps))
        traj = arc_length_sample(action_path, move_steps + 1)
        cmd_xyz_world = np.diff(traj, axis=0).astype(np.float32)
        cmd_xyz_raw = (cmd_xyz_world * CONTROLLER_XYZ_GAIN).astype(np.float32)

        # Clipping quality check
        cmd_magnitudes = np.linalg.norm(cmd_xyz_raw, axis=1)
        clipped_magnitudes = np.linalg.norm(np.clip(cmd_xyz_raw, -1.0, 1.0), axis=1)
        active_mask = cmd_magnitudes > 1e-6
        if active_mask.any():
            clip_ratios = np.where(
                active_mask,
                1.0 - clipped_magnitudes / np.maximum(cmd_magnitudes, 1e-8),
                0.0,
            )
            if float(np.mean(clip_ratios[active_mask] > 0.2)) > 0.25:
                return zero_plan, zero_action

        cmd_xyz = np.clip(cmd_xyz_raw, -1.0, 1.0).astype(np.float32)

        # Plan-action consistency
        if compute_plan and total_path_len > 1e-5:
            action_endpoint_world = np.sum(cmd_xyz_world, axis=0)
            plan_endpoint = plan_waypoints[-1]
            a_norm = float(np.linalg.norm(action_endpoint_world))
            p_norm = float(np.linalg.norm(plan_endpoint))
            if a_norm > 1e-6 and p_norm > 1e-6:
                cos_sim = float(
                    np.dot(action_endpoint_world, plan_endpoint) / (a_norm * p_norm)
                )
                if cos_sim < 0.3:
                    return zero_plan, zero_action

        diff_rpy = target_rpy.astype(np.float32) - obs_rpy.astype(np.float32)
        per_rpy = (diff_rpy / float(max(move_steps, 1))).astype(np.float32)

        # Gripper command logic
        src_seg = ep.get("subtask_segments", [])[source_si]
        if self.retarget_version == "stateful_v1":
            holding, _, _ = self._infer_retarget_source_state_v1(
                eidx=eidx,
                seg=src_seg,
                si=source_si,
                obs_frame=obs_frame,
                obs_gripper=float(obs_gripper),
            )
        else:
            holding = self._infer_holding_state(
                eidx=eidx, si=source_si, obs_frame=obs_frame,
                action_type=str(src_seg.get("action_type", "")),
                obs_gripper=float(obs_gripper),
            )
        # Travel gripper: maintain current holding state during approach
        travel_cmd = 1.0 if holding else -1.0

        targets = np.zeros((self.num_action_steps, 7), dtype=np.float32)
        for i in range(self.num_action_steps):
            if i < len(cmd_xyz):
                targets[i, :3] = cmd_xyz[i]
                targets[i, 3:6] = per_rpy
                targets[i, 6] = travel_cmd  # maintain holding state during travel
            else:
                # Post-travel: transition to target action's final gripper state
                if target_action_type == "grasp":
                    targets[i, 6] = 1.0    # grasp complete: close gripper
                elif target_action_type == "release":
                    targets[i, 6] = -1.0   # release complete: open gripper
                else:
                    targets[i, 6] = travel_cmd  # other: maintain
        action_targets = np.clip(targets, -1.0, 1.0).astype(np.float32)

        return plan_waypoints, action_targets

    def _compute_reverse_plan(
        self, eidx: int, obs_frame: int, seg_start_frame: int, obs_xyz: np.ndarray,
    ) -> np.ndarray:
        """Compute reversed plan: trajectory from obs_frame back to seg_start."""
        ep = self.episodes[eidx]
        waypoints = []
        for f in range(seg_start_frame, obs_frame + 1):
            pos = self._get_frame_world_pos(ep, f)
            if pos is not None:
                waypoints.append(pos.astype(np.float32))

        if len(waypoints) < 2:
            return np.zeros((self.num_plan_waypoints, 3), dtype=np.float32)

        traj = np.array(waypoints)[::-1]  # reverse
        sampled = arc_length_sample(traj, self.num_plan_waypoints)
        return (sampled - obs_xyz).astype(np.float32)

    def _compute_reverse_action_targets(
        self, actions_raw: np.ndarray, obs_frame: int, seg_start_frame: int,
    ) -> np.ndarray:
        """Build inverse action chunk from current frame toward seg_start_frame.

        Action format is delta control (xyz/rpy/gripper), so reverse uses sign inversion.
        """
        targets = np.zeros((self.num_action_steps, 7), dtype=np.float32)
        for i in range(self.num_action_steps):
            src = obs_frame - 1 - i
            if src < seg_start_frame or src < 0 or src >= len(actions_raw):
                break
            targets[i] = -actions_raw[src, :7]
        return targets


# ---------------------------------------------------------------------------
# Train / Val split
# ---------------------------------------------------------------------------

def build_datasets(
    episodes_json: str,
    frames_dir: str,
    parquet_dir: str,
    packed_features_dir: str,
    tc_event_manifest: str = "",
    images_dir: str = "",
    num_action_steps: int = 16,
    num_plan_waypoints: int = 8,
    distance_alpha: float = 10.0,
    max_post_subtask_frames: int = 15,
    use_progress: bool = True,
    use_dit: bool = True,
    use_plan: bool = False,
    use_reverse_aug: bool = False,
    use_retarget_aug: bool = False,
    retarget_version: str = "legacy",
    retarget_rollout_mode: str = "tangent_replay",
    retarget_text_dropout_prob: float = 0.8,
    retarget_template_path: str = "",
    text_dropout_prob: float = 0.0,
    reverse_text_dropout_prob: float = 1.0,
    reverse_max_mask_overlap: float = 0.05,
    xyz_source: str = "action_delta",
    post_only_tc: bool = False,
    post_only_tc_types: Optional[List[str]] = None,
    tail_exclude_ratio: float = 0.0,
    use_target_place: bool = False,
    use_wrist_goal_mask: bool = False,
    use_wrist_target_place: bool = False,
    clip_model_name: str = DEFAULT_CLIP_TEXT_MODEL,
    text_feature_dim: int = 768,
    mask_augmentation: str = "none",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple["RAINDataset", "RAINDataset"]:
    """Build train/val datasets with episode-level split."""
    episodes = load_episode_records(episodes_json)
    all_ids = sorted(episodes.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    split = int(len(all_ids) * train_ratio)

    common = dict(
        episodes_json=episodes_json, frames_dir=frames_dir,
        tc_event_manifest=tc_event_manifest,
        parquet_dir=parquet_dir, packed_features_dir=packed_features_dir,
        images_dir=images_dir,
        num_action_steps=num_action_steps, num_plan_waypoints=num_plan_waypoints,
        distance_alpha=distance_alpha, max_post_subtask_frames=max_post_subtask_frames,
        use_progress=use_progress, use_dit=use_dit,
        use_plan=use_plan,
        retarget_version=retarget_version,
        retarget_rollout_mode=retarget_rollout_mode,
        retarget_template_path=retarget_template_path,
        text_dropout_prob=text_dropout_prob,
        reverse_text_dropout_prob=reverse_text_dropout_prob,
        reverse_max_mask_overlap=reverse_max_mask_overlap,
        xyz_source=xyz_source,
        retarget_text_dropout_prob=retarget_text_dropout_prob,
        post_only_tc=post_only_tc,
        post_only_tc_types=post_only_tc_types,
        tail_exclude_ratio=tail_exclude_ratio,
        use_target_place=use_target_place,
        use_wrist_goal_mask=use_wrist_goal_mask,
        use_wrist_target_place=use_wrist_target_place,
        clip_model_name=clip_model_name,
        text_feature_dim=text_feature_dim,
        mask_augmentation=mask_augmentation,
    )

    train_ds = RAINDataset(
        **common, use_reverse_aug=use_reverse_aug,
        use_retarget_aug=use_retarget_aug,
        episode_ids=all_ids[:split], is_training=True,
    )
    val_ds = RAINDataset(
        **common, use_reverse_aug=False, use_retarget_aug=False,
        episode_ids=all_ids[split:], is_training=False,
    )
    return train_ds, val_ds
