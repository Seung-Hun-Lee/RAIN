#!/usr/bin/env python3
"""Stage 1: DDP training for RAIN action generation.

Target-adaptive Cross-view Encoder (TCE) + PlanDiT.
Progress head is created but frozen; only TCE + PlanDiT are trained.

Usage:
    torchrun --nproc_per_node=8 -m rain.train_action \
        --episodes-json DATA/episodes.json \
        --parquet-dir DATA/parquets \
        --packed-features-dir DATA/packed_224 \
        --use-plan --use-retarget-aug \
        --experiment-name rain_action_v1
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from rainv2.configs.config import (
    ThirdEncoderConfig, WristEncoderConfig, DiTConfig, ProgressConfig, DataConfig, TrainingConfig,
)
from shared.data.dataset import (
    GoalConsistencyBatchSampler,
    build_datasets,
    collate_fn,
)
from rainv2.models.model import RAINModel
from shared.clip_utils import DEFAULT_CLIP_TEXT_MODEL, get_clip_text_feature_dim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TeeStream:
    """Mirror writes to terminal stream and an experiment log file."""

    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_LOG_FILE_HANDLE = None


def setup_experiment_log_tee(output_path: Path) -> Path:
    global _LOG_FILE_HANDLE
    log_path = output_path / "train.log"
    _LOG_FILE_HANDLE = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, _LOG_FILE_HANDLE)
    sys.stderr = TeeStream(sys.stderr, _LOG_FILE_HANDLE)
    return log_path


class TrainLogger:
    FIELDS = [
        "global_step", "epoch",
        "train_loss", "train_dit_loss",
        "train_action_loss", "train_plan_loss",
        "train_goal_xyz_consistency_loss",
        "lr", "grad_norm", "epoch_time_s", "timestamp",
    ]

    def __init__(self, log_path: str, append: bool = False):
        self.log_path = log_path
        self.append = append
        should_write_header = (
            not append
            or not os.path.exists(self.log_path)
            or os.path.getsize(self.log_path) == 0
        )
        if should_write_header:
            mode = "a" if append else "w"
            with open(self.log_path, mode, newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **kwargs):
        kwargs["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(
                {k: kwargs.get(k, "") for k in self.FIELDS})


def restore_best_metric_from_csv(log_path: str, field: str, mode: str, default: float) -> float:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return default

    best = default
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(field, "")
            if raw in ("", None):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if mode == "min":
                best = min(best, value)
            elif mode == "max":
                best = max(best, value)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
    return best


# ---------------------------------------------------------------------------
# Training utils
# ---------------------------------------------------------------------------
def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log_print(msg):
    if is_main():
        print(msg, flush=True)


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embed" in name or "_pos" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def create_scheduler(optimizer, warmup_steps, total_steps, lr):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=lr * 0.01)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def apply_goal_consistency_group_mapping(batch, config, group_size=4):
    """All N frames in each block share the same target for goal xyz consistency."""
    weight = float(getattr(config.dit, "goal_xyz_consistency_weight", 0.0))
    if weight <= 0.0:
        batch["goal_group_index"] = None
        batch["goal_ref_index"] = None
        return batch

    state = batch.get("state")
    if state is None or state.ndim != 2:
        batch["goal_group_index"] = None
        batch["goal_ref_index"] = None
        return batch

    B = int(state.shape[0])
    if B < group_size or (B % group_size) != 0:
        batch["goal_group_index"] = None
        batch["goal_ref_index"] = None
        return batch

    group_idx = torch.full((B,), -1, dtype=torch.long, device=state.device)
    ref_idx = torch.full((B,), -1, dtype=torch.long, device=state.device)
    gid = batch.get("goal_consistency_gid")
    gt_distance = batch.get("gt_distance")

    if gid is not None and isinstance(gid, torch.Tensor) and gid.shape[0] == B:
        gid = gid.long().to(state.device)
    else:
        gid = None
    if gt_distance is not None and isinstance(gt_distance, torch.Tensor) and gt_distance.shape[0] == B:
        gt_distance = gt_distance.float().to(state.device)
    else:
        gt_distance = None

    n_blocks = B // group_size
    for b in range(n_blocks):
        i0 = group_size * b
        members = torch.arange(i0, i0 + group_size, device=state.device, dtype=torch.long)

        valid = True
        if gid is not None:
            g = gid[members]
            valid = bool((g >= 0).all() and (g == g[0]).all())

        if valid:
            group_idx[members] = int(b)
            if gt_distance is not None:
                local_ref = int(torch.argmax(gt_distance[members]).item())
                anchor = int(members[local_ref].item())
            else:
                anchor = int(members[-1].item())
            ref_idx[members] = anchor

    batch["goal_group_index"] = group_idx
    batch["goal_ref_index"] = ref_idx
    return batch


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, scaler,
                device, epoch, config, global_step,
                consistency_group_size=4, max_steps=0):
    model.train()
    totals: Dict[str, float] = {}
    num_batches = 0
    consecutive_nan = 0
    last_grad_norm = 0.0
    accum_steps = config.gradient_accumulation_steps

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    elif hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main())
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        batch = move_to_device(batch, device)
        batch = apply_goal_consistency_group_mapping(
            batch, config, group_size=consistency_group_size)

        with autocast("cuda", enabled=config.mixed_precision):
            vision_inputs = {}
            if "image_third" in batch:
                vision_inputs = {
                    "image_third": batch["image_third"],
                    "image_wrist": batch["image_wrist"],
                }
            else:
                vision_inputs = {
                    "dino_third": batch["dino_third"],
                    "dino_wrist": batch["dino_wrist"],
                }
            output = model(
                **vision_inputs,
                goal_mask_third=batch.get("goal_mask_third"),
                goal_mask_wrist=batch.get("goal_mask_wrist"),
                target_place_mask=batch.get("target_place_mask_third"),
                target_place_mask_wrist=batch.get("target_place_mask_wrist"),
                text_feat=batch.get("text_feat"),
                state=batch.get("state"),
                action_type=batch.get("action_type"),
                gt_action=batch.get("gt_action"),
                gt_plan=batch.get("gt_plan"),
                is_post_subtask=batch.get("is_post_subtask"),
                goal_group_index=batch.get("goal_group_index"),
                goal_ref_index=batch.get("goal_ref_index"),
            )
            loss = output["loss"] / accum_steps

        loss_val = loss.item() * accum_steps

        # NaN detection
        is_nan = not np.isfinite(loss_val)
        if dist.is_initialized():
            nan_flag = torch.tensor(1.0 if is_nan else 0.0, device=device)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            is_nan = nan_flag.item() > 0.5

        if is_nan:
            consecutive_nan += 1
            log_print(f"[NaN guard] Skipping batch ({consecutive_nan}), loss={loss_val}")
            if consecutive_nan >= 10:
                n = max(num_batches, 1)
                return {k: v / n for k, v in totals.items()}, global_step, True, last_grad_norm
            optimizer.zero_grad()
            continue
        consecutive_nan = 0

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            last_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        for k, v in output.items():
            if isinstance(v, torch.Tensor) and "loss" in k:
                totals[k] = totals.get(k, 0.0) + v.item()
        num_batches += 1

        if is_main():
            postfix = {"loss": f"{loss_val:.4f}"}
            for k, v in output.items():
                if isinstance(v, torch.Tensor) and "loss" in k and k != "loss":
                    short = k.replace("_loss", "").replace("consistency_", "c")
                    postfix[short] = f"{v.item():.3f}"
            postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix(**postfix)

        if max_steps > 0 and global_step >= max_steps:
            break

    n = max(num_batches, 1)
    return {k: v / n for k, v in totals.items()}, global_step, False, last_grad_norm


def save_checkpoint(model, optimizer, scheduler, scaler, global_step, epoch, path):
    if not is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    # Frozen DINOv2 is reconstructed from the pinned hub checkpoint. Excluding
    # it keeps each RAIN checkpoint hundreds of MB smaller without losing any
    # learned state.
    model_state = {
        key: value
        for key, value in model_state.items()
        if not key.startswith("online_dino.backbone.")
    }
    ckpt = {
        "global_step": global_step,
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(ckpt, path)
    log_print(f"Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAIN Action Training (Stage 1)")

    # Data
    parser.add_argument("--episodes-json", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, default="")
    parser.add_argument("--parquet-dir", type=str, required=True)
    parser.add_argument("--packed-features-dir", type=str, required=True)
    parser.add_argument(
        "--images-dir",
        type=str,
        default="",
        help="224px packed RGB observations used by online frozen DINO.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--text-dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--mask-augmentation",
        type=str,
        choices=["none", "vlm_sam2_v1"],
        default="none",
        help="Training-only corruption recipe for VLM+SAM2 mask robustness.",
    )
    parser.add_argument("--reverse-text-dropout-prob", type=float, default=1.0)
    parser.add_argument("--reverse-max-mask-overlap", type=float, default=0.05)
    parser.add_argument("--xyz-source", type=str, choices=["action_delta"], default="action_delta")
    parser.add_argument(
        "--online-dino",
        action="store_true",
        help="Extract frozen DINOv2-L block 11/17/23 features online.",
    )
    parser.add_argument("--dino-input-size", type=int, default=224)
    parser.add_argument("--num-scales", type=int, choices=[1, 3], default=1)

    # Mode flags
    parser.add_argument("--condition-mode", type=str, choices=["text", "action_type"], default="action_type")
    parser.add_argument(
        "--cross-attn-mode",
        type=str,
        choices=["bidirectional", "third_to_wrist", "wrist_to_third"],
        default="bidirectional",
        help="Cross-attention mode between third and wrist encoders.",
    )
    parser.add_argument("--use-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no-plan-action-causal",
        action="store_true",
        help="Disable blockwise self-attention mask so plan tokens can attend action tokens.",
    )
    parser.add_argument(
        "--plan-attn-mode",
        type=str,
        choices=["causal", "bidirectional"],
        default=None,
        help="Explicit plan/action self-attention mode. Overrides --no-plan-action-causal when set.",
    )
    parser.add_argument("--use-reverse-aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-retarget-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retarget-version",
        type=str,
        choices=["legacy", "stateful_v1"],
        default="legacy",
        help="Retarget augmentation selector/masking version.",
    )
    parser.add_argument(
        "--retarget-rollout-mode",
        type=str,
        choices=["tangent_replay", "linear_interp_v1"],
        default="tangent_replay",
        help="How retarget action chunks are synthesized once a target subtask is selected.",
    )
    parser.add_argument("--retarget-text-dropout-prob", type=float, default=0.8)
    parser.add_argument("--retarget-template-path", type=str, default="")
    parser.add_argument("--goal-xyz-consistency-weight", type=float, default=0.1)
    parser.add_argument("--consistency-group-size", type=int, default=4)
    parser.add_argument("--encoder-hidden-dim", type=int, default=448,
                        help="Hidden dim for third/wrist encoders (448 for 224px, 1024 for 448px)")
    parser.add_argument(
        "--dino-feature-dim",
        type=int,
        default=1024,
        help="Packed DINO feature dim (1024 for DINOv2-large, 384 for DINOv2-small).",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default=DEFAULT_CLIP_TEXT_MODEL,
        help="CLIP text model used for cached text features and action-type embeddings.",
    )
    parser.add_argument(
        "--force-goal-pair-batch",
        action="store_true",
        help="Force goal-consistency grouped batch sampling even when consistency loss is disabled.",
    )
    parser.add_argument("--use-target-place", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable target place embedding for pick & place tasks.")
    parser.add_argument(
        "--use-wrist-goal-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load wrist-view object masks and use them for wrist mask conditioning.",
    )
    parser.add_argument(
        "--use-wrist-target-place",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load wrist-view target-place masks when target-place conditioning is enabled.",
    )
    parser.add_argument(
        "--wrist-conditioning",
        type=str,
        choices=["corr", "mask", "none"],
        default="corr",
        help="Conditioning applied to wrist-view patches.",
    )
    parser.add_argument(
        "--use-view-transformer-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the per-view post-FiLM transformer blocks (cross-attn/self-attn/FFN).",
    )
    parser.add_argument(
        "--zero-third-view-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace third-view goal/target-place masks with zeros while keeping mask-conditioned modules active.",
    )
    parser.add_argument(
        "--zero-wrist-view-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace wrist-view goal/target-place masks with zeros while keeping mask-conditioned modules active.",
    )

    # Training
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--expected-config", type=str, default="",
                        help="Validate the complete config before loading training data.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop training after this many optimizer steps (0=no limit)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Initialize model weights from a checkpoint but start optimizer/scheduler/step from scratch.",
    )

    args = parser.parse_args()

    if args.online_dino:
        if args.num_scales != 3:
            parser.error("--online-dino requires --num-scales 3")
        if not args.images_dir:
            parser.error("--online-dino requires --images-dir")
        if args.dino_input_size != 224:
            parser.error("Online multi-scale training requires --dino-input-size 224")
        if args.dino_feature_dim != 1024:
            parser.error("Online multi-scale training requires --dino-feature-dim 1024")
    elif args.images_dir:
        parser.error("--images-dir is only used with --online-dino")

    use_goal_pair_batch = bool(args.goal_xyz_consistency_weight > 0.0 or args.force_goal_pair_batch)
    if use_goal_pair_batch:
        block_sz = args.consistency_group_size
        if args.batch_size % block_sz != 0:
            parser.error(f"goal-consistency batch requires --batch-size multiple of {block_sz}")
    if args.goal_xyz_consistency_weight > 0.0 and not args.use_plan:
        parser.error("--goal-xyz-consistency-weight requires --use-plan")

    experiment_name = args.experiment_name or f"rain_action_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    text_feature_dim = get_clip_text_feature_dim(args.clip_model_name)

    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed, rank)
    device = torch.device(f"cuda:{local_rank}")

    output_path = Path(args.output_dir) / experiment_name
    if is_main():
        output_path.mkdir(parents=True, exist_ok=True)
        log_path = setup_experiment_log_tee(output_path)
        print(f"[Log] tee -> {log_path}", flush=True)

    logging.basicConfig(level=logging.INFO if is_main() else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Build config
    if args.plan_attn_mode is not None:
        plan_action_causal = (args.plan_attn_mode == "causal")
    else:
        plan_action_causal = not args.no_plan_action_causal

    dit_config = DiTConfig(
        use_plan=args.use_plan,
        goal_xyz_consistency_weight=args.goal_xyz_consistency_weight,
        plan_action_causal=plan_action_causal,
        text_dim=text_feature_dim,
        vision_input_dim=args.encoder_hidden_dim,
        num_scales=args.num_scales,
    )

    data_config = DataConfig(
        episodes_json=args.episodes_json,
        frames_dir=args.frames_dir,
        parquet_dir=args.parquet_dir,
        packed_features_dir=args.packed_features_dir,
        images_dir=args.images_dir if args.online_dino else "",
        clip_model_name=args.clip_model_name,
        text_feature_dim=text_feature_dim,
        text_dropout_prob=args.text_dropout_prob,
        mask_augmentation=args.mask_augmentation,
        reverse_text_dropout_prob=args.reverse_text_dropout_prob,
        reverse_max_mask_overlap=args.reverse_max_mask_overlap,
        xyz_source=args.xyz_source,
        use_retarget_aug=args.use_retarget_aug,
        retarget_version=args.retarget_version,
        retarget_rollout_mode=args.retarget_rollout_mode,
        retarget_text_dropout_prob=args.retarget_text_dropout_prob,
        use_goal_consistency_pair_batch=use_goal_pair_batch,
        retarget_template_path=args.retarget_template_path,
        consistency_group_size=args.consistency_group_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    config = TrainingConfig(
        third_encoder=ThirdEncoderConfig(
            dino_dim=args.dino_feature_dim,
            hidden_dim=args.encoder_hidden_dim,
            use_target_place=args.use_target_place,
            use_transformer_blocks=args.use_view_transformer_blocks,
            zero_view_masks=args.zero_third_view_masks,
        ),
        wrist_encoder=WristEncoderConfig(
            dino_dim=args.dino_feature_dim,
            hidden_dim=args.encoder_hidden_dim,
            conditioning_mode=args.wrist_conditioning,
            use_corr_modulation=(args.wrist_conditioning == "corr"),
            use_goal_mask=args.use_wrist_goal_mask,
            use_target_place=args.use_wrist_target_place,
            use_transformer_blocks=args.use_view_transformer_blocks,
            zero_view_masks=args.zero_wrist_view_masks,
        ),
        dit=dit_config,
        progress=ProgressConfig(
            hidden_dim=args.encoder_hidden_dim,
            text_dim=text_feature_dim,
            clip_model_name=args.clip_model_name,
        ),
        data=data_config,
        condition_mode=args.condition_mode,
        cross_attn_mode=args.cross_attn_mode,
        use_reverse_aug=args.use_reverse_aug,
        online_dino=args.online_dino,
        dino_input_size=args.dino_input_size,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        batch_size_per_gpu=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
    )

    if args.expected_config:
        from shared.training_config_check import check_training_config
        check_training_config(config, args.expected_config)

    eff_batch = args.batch_size * world_size * args.gradient_accumulation_steps

    log_print("=== RAIN Action Training (Stage 1) ===")
    log_print(f"Modes: CondMode={config.condition_mode} "
              f"DiT=True Plan={config.dit.use_plan} "
              f"PlanActionCausal={config.dit.plan_action_causal} "
              f"CrossAttnMode={config.cross_attn_mode} "
              f"GoalXYZW={config.dit.goal_xyz_consistency_weight} "
              f"ReverseAug={config.use_reverse_aug} "
              f"RetargetAug={config.data.use_retarget_aug} "
              f"RetargetVersion={config.data.retarget_version} "
              f"RetargetRollout={config.data.retarget_rollout_mode} "
              f"GoalPairBatch={config.data.use_goal_consistency_pair_batch} "
              f"ForceGoalPairBatch={args.force_goal_pair_batch} "
              f"ConsistGroupSize={config.data.consistency_group_size} "
              f"XYZSource={config.data.xyz_source} "
              f"TargetPlace={args.use_target_place} "
              f"WristCond={config.wrist_encoder.conditioning_mode} "
              f"ViewBlocks={config.third_encoder.use_transformer_blocks} "
              f"ZeroThirdMasks={config.third_encoder.zero_view_masks} "
              f"ZeroWristMasks={config.wrist_encoder.zero_view_masks} "
              f"WristGoalMask={config.wrist_encoder.use_goal_mask} "
              f"WristTargetPlace={config.wrist_encoder.use_target_place} "
              f"MaskAug={config.data.mask_augmentation} "
              f"OnlineDINO={config.online_dino} "
              f"NumScales={config.dit.num_scales} "
              f"TextDrop={config.data.text_dropout_prob}")
    log_print(f"World: {world_size}, Batch/GPU: {args.batch_size}, "
              f"Accum: {args.gradient_accumulation_steps}, Effective: {eff_batch}")

    if is_main():
        with open(output_path / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

    # Dataset (shared data pipeline)
    log_print("Loading dataset...")
    train_ds, val_ds = build_datasets(
        episodes_json=data_config.episodes_json,
        frames_dir=data_config.frames_dir,
        parquet_dir=data_config.parquet_dir,
        packed_features_dir=data_config.packed_features_dir,
        images_dir=data_config.images_dir,
        num_action_steps=dit_config.num_action_tokens,
        num_plan_waypoints=dit_config.num_plan_tokens,
        distance_alpha=data_config.distance_alpha,
        max_post_subtask_frames=data_config.max_post_subtask_frames,
        use_progress=False,  # action-only: skip uncertain filtering, include all frames
        use_dit=True,
        use_plan=config.dit.use_plan,
        use_reverse_aug=config.use_reverse_aug,
        use_retarget_aug=data_config.use_retarget_aug,
        retarget_version=data_config.retarget_version,
        retarget_rollout_mode=data_config.retarget_rollout_mode,
        retarget_text_dropout_prob=data_config.retarget_text_dropout_prob,
        retarget_template_path=data_config.retarget_template_path,
        text_dropout_prob=data_config.text_dropout_prob,
        mask_augmentation=data_config.mask_augmentation,
        reverse_text_dropout_prob=data_config.reverse_text_dropout_prob,
        reverse_max_mask_overlap=data_config.reverse_max_mask_overlap,
        xyz_source=data_config.xyz_source,
        use_target_place=args.use_target_place,
        use_wrist_goal_mask=config.wrist_encoder.use_goal_mask,
        use_wrist_target_place=config.wrist_encoder.use_target_place,
        clip_model_name=data_config.clip_model_name,
        text_feature_dim=data_config.text_feature_dim,
        train_ratio=data_config.train_ratio,
        seed=data_config.seed,
    )
    log_print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    if train_ds.dino_hidden_dim != args.dino_feature_dim:
        raise ValueError(
            "Packed DINO hidden dim mismatch: "
            f"dataset={train_ds.dino_hidden_dim}, config={args.dino_feature_dim}. "
            "Check --packed-features-dir and --dino-feature-dim."
        )
    if not args.online_dino and train_ds.packed_dino.num_scales != args.num_scales:
        raise ValueError(
            "Packed DINO scale mismatch: "
            f"dataset={train_ds.packed_dino.num_scales}, config={args.num_scales}. "
            "Check --packed-features-dir and --num-scales."
        )

    if data_config.use_goal_consistency_pair_batch:
        train_sampler = GoalConsistencyBatchSampler(
            train_ds,
            batch_size=args.batch_size,
            group_size=args.consistency_group_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
            drop_last=False,
        )
    else:
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if world_size > 1 else None
        )
    if data_config.use_goal_consistency_pair_batch:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )

    steps_per_epoch = max(len(train_loader) // config.gradient_accumulation_steps, 1)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * config.max_epochs
    log_print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    # Model
    log_print("Creating model...")
    model = RAINModel(config).to(device)
    raw_model = model
    raw_model.set_training_stage("action")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    optimizer = create_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = create_scheduler(optimizer, config.warmup_steps, total_steps, config.learning_rate)
    scaler = GradScaler("cuda", enabled=config.mixed_precision)

    global_step = 0
    start_epoch = 0

    if args.resume and args.init_model:
        raise ValueError("--resume and --init-model are mutually exclusive")

    if args.init_model:
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_action_checkpoint(args.init_model)
        log_print(f"Initialized model weights from {args.init_model} (fresh optimizer/scheduler/step)")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        raw_model = model.module if isinstance(model, DDP) else model
        resume_state = {
            k: v
            for k, v in ckpt["model_state_dict"].items()
            if not k.startswith("fusion_branch.")
        }
        missing, unexpected = raw_model.load_state_dict(resume_state, strict=False)
        real_missing = [
            k for k in missing
            if not k.startswith(("fusion_branch.", "online_dino.backbone."))
        ]
        if real_missing:
            log_print(f"[WARN] Resume missing non-progress keys: {real_missing}")
        if unexpected:
            log_print(f"[WARN] Resume unexpected keys: {unexpected}")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt.get("epoch", 0)
        log_print(f"Resumed from {args.resume} at step {global_step}, epoch {start_epoch}")

    if world_size > 1:
        dist.barrier()

    best_metric = float("inf")
    train_logger = None
    if is_main():
        train_log_csv = str(output_path / "train_log.csv")
        train_logger = TrainLogger(train_log_csv, append=bool(args.resume))
        if args.resume:
            best_metric = restore_best_metric_from_csv(
                train_log_csv, field="train_loss", mode="min", default=best_metric)

    if args.max_steps > 0 and global_step >= args.max_steps:
        log_print(f"Resume checkpoint already reached max_steps={args.max_steps}. Saving final checkpoint and exiting.")
        if is_main():
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                global_step, start_epoch,
                output_path / "checkpoints" / "checkpoint_final.pt",
            )
            log_print(f"Training complete! Best train_loss: {best_metric:.4f}")
        cleanup_distributed()
        return

    log_print("Starting training...")

    last_epoch = start_epoch
    for epoch in range(start_epoch + 1, config.max_epochs + 1):
        last_epoch = epoch
        epoch_t0 = time.time()

        train_metrics, global_step, nan_stop, grad_norm = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config, global_step,
            consistency_group_size=data_config.consistency_group_size,
            max_steps=args.max_steps,
        )
        epoch_time = time.time() - epoch_t0

        if nan_stop:
            log_print("Training stopped due to repeated NaN losses.")
            break

        if args.max_steps > 0 and global_step >= args.max_steps:
            log_print(f"Reached max_steps={args.max_steps} at epoch {epoch}. Stopping.")
            break

        if is_main():
            lr = scheduler.get_last_lr()[0]

            parts = [f"Epoch {epoch}"]
            parts.append(f"Loss: {train_metrics.get('loss', 0):.4f}")
            if "dit_loss" in train_metrics:
                parts.append(f"DiT: {train_metrics['dit_loss']:.4f}")
            if "action_loss" in train_metrics:
                parts.append(f"Act: {train_metrics['action_loss']:.4f}")
            if "plan_loss" in train_metrics:
                parts.append(f"Plan: {train_metrics['plan_loss']:.4f}")
            if "goal_xyz_consistency_loss" in train_metrics:
                parts.append(f"GoalXYZ: {train_metrics['goal_xyz_consistency_loss']:.4f}")
            parts.append(f"LR: {lr:.2e} GN: {grad_norm:.3f} Step: {global_step} T: {epoch_time:.1f}s")
            log_print(" | ".join(parts))

            if train_logger:
                log_kwargs = {
                    "global_step": global_step, "epoch": epoch,
                    "lr": f"{lr:.2e}", "grad_norm": f"{grad_norm:.4f}",
                    "epoch_time_s": f"{epoch_time:.1f}",
                }
                for k, v in train_metrics.items():
                    log_kwargs[f"train_{k}"] = f"{v:.6f}"
                train_logger.log(**log_kwargs)

            # Save best checkpoint (based on train_loss)
            current = train_metrics.get("loss", float("inf"))
            if current < best_metric:
                best_metric = current
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    global_step, epoch,
                    output_path / "checkpoints" / "checkpoint_best.pt",
                )
                log_print(f"  -> New best train_loss: {current:.4f}")

            # Save latest checkpoint every epoch
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                global_step, epoch,
                output_path / "checkpoints" / "checkpoint_latest.pt",
            )

    # Final save
    if is_main():
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            global_step, last_epoch,
            output_path / "checkpoints" / "checkpoint_final.pt",
        )
        log_print(f"Training complete! Best train_loss: {best_metric:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
