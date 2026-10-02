#!/usr/bin/env python3
"""Stage 2: DDP training for RAIN progress prediction.

Loads TCE + PlanDiT weights from Stage 1 checkpoint, freezes them,
and trains only the progress head (gated fusion).

Usage:
    torchrun --nproc_per_node=8 -m rain.train_progress \
        --action-checkpoint outputs/rain_action/checkpoints/checkpoint_final.pt \
        --episodes-json DATA/episodes.json \
        --parquet-dir DATA/parquets \
        --packed-features-dir DATA/packed_224 \
        --experiment-name rain_progress
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    ThirdEncoderConfig, WristEncoderConfig, DiTConfig, ProgressConfig,
    DataConfig, TrainingConfig,
)
from shared.data.dataset import build_datasets, collate_fn
from rainv2.models.progress_heads import SingleViewProgressHead
from rainv2.models.model import RAINModel
from shared.clip_utils import DEFAULT_CLIP_TEXT_MODEL

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
        "train_loss", "train_progress_loss",
        "train_fusion_progress_loss",
        "train_fusion_loss_dist", "train_fusion_loss_align", "train_fusion_loss_task_comp",
        "val_loss", "val_progress_loss",
        "val_fusion_progress_loss",
        # Fusion metrics
        "val_task_comp_accuracy", "val_task_comp_precision",
        "val_task_comp_recall", "val_task_comp_f1",
        "val_dist_mae", "val_dist_r", "val_align_mae", "val_align_r",
        # Fusion gate stats
        "val_fusion_gate_global_mean", "val_fusion_gate_obj_mean",
        "lr", "grad_norm", "epoch_time_s", "timestamp",
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **kwargs):
        kwargs["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(
                {k: kwargs.get(k, "") for k in self.FIELDS})


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
    """Create optimizer for trainable parameters only."""
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


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, scaler,
                device, epoch, config, global_step):
    model.train()
    totals: Dict[str, float] = {}
    num_batches = 0
    consecutive_nan = 0
    last_grad_norm = 0.0
    accum_steps = config.gradient_accumulation_steps

    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main())
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        batch = move_to_device(batch, device)

        with autocast("cuda", enabled=config.mixed_precision):
            output = model(
                dino_third=batch["dino_third"],
                dino_wrist=batch["dino_wrist"],
                goal_mask_third=batch.get("goal_mask_third"),
                goal_mask_wrist=batch.get("goal_mask_wrist"),
                target_place_mask=batch.get("target_place_mask_third"),
                target_place_mask_wrist=batch.get("target_place_mask_wrist"),
                text_feat=batch.get("text_feat"),
                text_feat_full=batch.get("text_feat_full"),
                action_type=batch.get("action_type"),
                gt_distance=batch.get("gt_distance"),
                gt_alignment=batch.get("gt_alignment"),
                gt_task_completion=batch.get("gt_task_completion"),
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

        # Accumulate losses
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and "loss" in k:
                totals[k] = totals.get(k, 0.0) + v.item()
        num_batches += 1

        if is_main():
            postfix = {"loss": f"{loss_val:.4f}"}
            for k, v in output.items():
                if isinstance(v, torch.Tensor) and "loss" in k and k != "loss":
                    short = k.replace("_loss", "").replace("progress", "p").replace("fusion_", "f_")
                    postfix[short] = f"{v.item():.3f}"
            postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix(**postfix)

    n = max(num_batches, 1)
    return {k: v / n for k, v in totals.items()}, global_step, False, last_grad_norm


@torch.no_grad()
def validate(model, dataloader, device, config):
    model.eval()
    totals: Dict[str, float] = {}
    num_batches = 0

    all_tc_probs = []
    all_tc_gt = []
    all_dist_preds = []
    all_dist_gt = []
    all_align_preds = []
    all_align_gt = []
    all_fusion_gates = []

    for batch in tqdm(dataloader, desc="Val", disable=not is_main()):
        batch = move_to_device(batch, device)

        with autocast("cuda", enabled=config.mixed_precision):
            output = model(
                dino_third=batch["dino_third"],
                dino_wrist=batch["dino_wrist"],
                goal_mask_third=batch.get("goal_mask_third"),
                goal_mask_wrist=batch.get("goal_mask_wrist"),
                target_place_mask=batch.get("target_place_mask_third"),
                target_place_mask_wrist=batch.get("target_place_mask_wrist"),
                text_feat=batch.get("text_feat"),
                text_feat_full=batch.get("text_feat_full"),
                action_type=batch.get("action_type"),
                gt_distance=batch.get("gt_distance"),
                gt_alignment=batch.get("gt_alignment"),
                gt_task_completion=batch.get("gt_task_completion"),
            )

        for k, v in output.items():
            if isinstance(v, torch.Tensor) and "loss" in k:
                totals[k] = totals.get(k, 0.0) + v.item()
        num_batches += 1

        # Predictions
        raw_model = model.module if isinstance(model, DDP) else model
        with autocast("cuda", enabled=config.mixed_precision):
            pred_out = raw_model.predict(
                dino_third=batch["dino_third"],
                dino_wrist=batch["dino_wrist"],
                goal_mask_third=batch.get("goal_mask_third"),
                goal_mask_wrist=batch.get("goal_mask_wrist"),
                target_place_mask=batch.get("target_place_mask_third"),
                target_place_mask_wrist=batch.get("target_place_mask_wrist"),
                text_feat=batch.get("text_feat"),
                action_type=batch.get("action_type"),
            )

        all_tc_probs.append(pred_out["task_comp_prob"].cpu())
        all_tc_gt.append(batch["gt_task_completion"].cpu())
        all_dist_preds.append(pred_out["pred_distance"].cpu())
        all_dist_gt.append(batch["gt_distance"].cpu())
        all_align_preds.append(pred_out["pred_alignment"].cpu())
        all_align_gt.append(batch["gt_alignment"].cpu())
        all_fusion_gates.append(pred_out["fusion_gate"].cpu())

    n = max(num_batches, 1)
    metrics = {k: v / n for k, v in totals.items()}

    # TC metrics
    if all_tc_probs:
        tc_probs = torch.cat(all_tc_probs)
        tc_gt = torch.cat(all_tc_gt)
        dist_gt_all = torch.cat(all_dist_gt)

        # Mask out noisy labels
        tc_mask_thresh = config.progress.tc_mask_dist_thresh
        if tc_mask_thresh > 0:
            tc_eval_mask = (dist_gt_all <= tc_mask_thresh) | (tc_gt > 0.5)
            tc_probs_m = tc_probs[tc_eval_mask]
            tc_gt_m = tc_gt[tc_eval_mask]
        else:
            tc_probs_m = tc_probs
            tc_gt_m = tc_gt

        tc_pred = (tc_probs_m > 0.5).float()
        tc_correct = (tc_pred == (tc_gt_m > 0.5).float()).float()
        metrics["task_comp_accuracy"] = tc_correct.mean().item()

        tp = ((tc_pred == 1) & (tc_gt_m > 0.5)).sum().float()
        fp = ((tc_pred == 1) & (tc_gt_m <= 0.5)).sum().float()
        fn = ((tc_pred == 0) & (tc_gt_m > 0.5)).sum().float()
        prec = (tp / (tp + fp).clamp_min(1)).item()
        rec = (tp / (tp + fn).clamp_min(1)).item()
        metrics["task_comp_precision"] = prec
        metrics["task_comp_recall"] = rec
        metrics["task_comp_f1"] = 2 * prec * rec / max(prec + rec, 1e-8)

    # Distance metrics
    if all_dist_preds:
        dist_preds = torch.cat(all_dist_preds)
        dist_gt = torch.cat(all_dist_gt)
        metrics["dist_mae"] = (dist_preds - dist_gt).abs().mean().item()
        if len(dist_gt) > 2:
            r = torch.corrcoef(torch.stack([dist_preds, dist_gt]))[0, 1].item()
            metrics["dist_r"] = r if np.isfinite(r) else 0.0

    # Alignment metrics
    if all_align_preds:
        align_preds = torch.cat(all_align_preds)
        align_gt = torch.cat(all_align_gt)
        metrics["align_mae"] = (align_preds - align_gt).abs().mean().item()
        if len(align_gt) > 2:
            r = torch.corrcoef(torch.stack([align_preds, align_gt]))[0, 1].item()
            metrics["align_r"] = r if np.isfinite(r) else 0.0

    # Fusion gate stats
    if all_fusion_gates:
        gates = torch.cat(all_fusion_gates)  # (N, 2)
        metrics["fusion_gate_global_mean"] = gates[:, 0].mean().item()
        metrics["fusion_gate_obj_mean"] = gates[:, 1].mean().item()

    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, global_step, epoch,
                    path, val_metrics=None):
    if not is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    ckpt = {
        "global_step": global_step,
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if val_metrics:
        ckpt["val_metrics"] = val_metrics
    torch.save(ckpt, path)
    log_print(f"Saved checkpoint: {path}")


def _filter_dict(src: Dict[str, Any], cls) -> Dict[str, Any]:
    """Keep only dataclass fields present in cls."""
    if not isinstance(src, dict):
        return {}
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in src.items() if k in valid}


def _resolve_action_config_path(
    action_checkpoint: str,
    action_config_arg: str,
) -> Optional[Path]:
    """Resolve action config path (explicit arg > checkpoint-adjacent config)."""
    if action_config_arg:
        cfg_path = Path(action_config_arg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"--action-config not found: {cfg_path}")
        return cfg_path

    inferred = Path(action_checkpoint).parent.parent / "config.json"
    if inferred.exists():
        return inferred
    return None


def _load_action_config(
    action_checkpoint: str,
    action_config_arg: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load action-stage config JSON for trunk/DiT inheritance."""
    cfg_path = _resolve_action_config_path(action_checkpoint, action_config_arg)
    if cfg_path is None:
        return {}, None

    with open(cfg_path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Action config must be a JSON object: {cfg_path}")
    return cfg, str(cfg_path)


def _checkpoint_has_plan_tokens(action_checkpoint: str) -> bool:
    """Infer whether action checkpoint contains plan-token weights."""
    ckpt = torch.load(action_checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    return any(k.startswith("dit.plan_") for k in state.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAIN Stage 2: Progress Training")

    # Required: action checkpoint
    parser.add_argument("--action-checkpoint", type=str, required=True,
                        help="Path to Stage 1 action checkpoint (encoder + DiT weights)")
    parser.add_argument("--action-config", type=str, default="",
                        help="Optional action-stage config.json. "
                             "If omitted, auto-resolve from checkpoint parent (../config.json)")

    # Data
    parser.add_argument("--episodes-json", type=str, required=True)
    parser.add_argument(
        "--tc-event-manifest",
        type=str,
        required=True,
        help="TC v3 event manifest used for progress labels and IGNORE filtering.",
    )
    parser.add_argument("--frames-dir", type=str, default="")
    parser.add_argument("--parquet-dir", type=str, required=True)
    parser.add_argument("--packed-features-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--text-dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--mask-augmentation",
        type=str,
        choices=["none", "vlm_sam2_v1"],
        default=None,
        help="Training-only mask corruption; defaults to the action-stage recipe.",
    )
    parser.add_argument("--reverse-text-dropout-prob", type=float, default=1.0)
    parser.add_argument("--reverse-max-mask-overlap", type=float, default=0.05)

    # Mode
    parser.add_argument("--condition-mode", type=str, choices=["text", "action_type"],
                        default="action_type")
    parser.add_argument("--use-reverse-aug", action="store_true")

    # Training
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--expected-config", type=str, default="",
                        help="Validate the complete config before loading training data.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--encoder-hidden-dim", type=int, default=None,
                        help="Hidden dim for encoders + progress head. "
                             "If omitted, inherited from action config.")
    parser.add_argument(
        "--use-wrist-goal-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override inherited wrist goal-mask loading/conditioning behavior.",
    )
    parser.add_argument(
        "--use-wrist-target-place",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override inherited wrist target-place conditioning behavior.",
    )

    # TC labels
    parser.add_argument("--post-only-tc", action=argparse.BooleanOptionalAction, default=False,
                        help="Legacy label mode; incompatible with --tc-event-manifest.")
    parser.add_argument("--post-only-tc-types", type=str, default=None,
                        help="Comma-separated action types for post_only_tc (e.g. turn_on,push,close,open). "
                             "If not set, applies to ALL types when --post-only-tc is on.")
    parser.add_argument("--tail-exclude-ratio", type=float, default=0.0,
                        help="Legacy filtering; TC v3 uses its explicit IGNORE interval.")

    # TC loss masking
    parser.add_argument("--tc-mask-dist-thresh", type=float, default=0.8,
                        help="Mask TC loss for action frames with gt_dist > thresh (0=disable)")
    parser.add_argument(
        "--progress-lambda-dist",
        type=float,
        default=None,
        help="Override ProgressConfig.lambda_dist for this stage-2 run.",
    )
    parser.add_argument(
        "--progress-lambda-align",
        type=float,
        default=None,
        help="Override ProgressConfig.lambda_align for this stage-2 run.",
    )
    parser.add_argument(
        "--progress-lambda-tc",
        type=float,
        default=None,
        help="Override ProgressConfig.lambda_tc for this stage-2 run.",
    )

    args = parser.parse_args()

    experiment_name = args.experiment_name or f"rain_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
    # Inherit model settings from the action-stage configuration.
    action_cfg_dict, action_cfg_src = _load_action_config(
        action_checkpoint=args.action_checkpoint,
        action_config_arg=args.action_config,
    )

    inherited_condition_mode = None
    if isinstance(action_cfg_dict.get("condition_mode"), str):
        candidate_mode = action_cfg_dict["condition_mode"].strip().lower()
        if candidate_mode in {"text", "action_type"}:
            inherited_condition_mode = candidate_mode

    final_condition_mode = inherited_condition_mode or args.condition_mode
    if inherited_condition_mode and inherited_condition_mode != args.condition_mode:
        log_print(
            f"[INFO] condition_mode overridden by action config: "
            f"{args.condition_mode} -> {inherited_condition_mode}"
        )

    # Inherit cross_attn_mode from action config
    inherited_cross_attn_mode = "bidirectional"  # default
    if isinstance(action_cfg_dict.get("cross_attn_mode"), str):
        inherited_cross_attn_mode = action_cfg_dict["cross_attn_mode"].strip().lower()
    elif isinstance(action_cfg_dict.get("wrist_encoder", {}).get("cross_attn_mode"), str):
        # Accept the nested cross-attention field in older config files.
        inherited_cross_attn_mode = action_cfg_dict["wrist_encoder"]["cross_attn_mode"].strip().lower()

    third_cfg = ThirdEncoderConfig(**_filter_dict(
        action_cfg_dict.get("third_encoder", {}), ThirdEncoderConfig
    ))
    wrist_cfg = WristEncoderConfig(**_filter_dict(
        action_cfg_dict.get("wrist_encoder", {}), WristEncoderConfig
    ))
    dit_cfg = DiTConfig(**_filter_dict(
        action_cfg_dict.get("dit", {}), DiTConfig
    ))
    inherited_data_cfg = DataConfig(**_filter_dict(
        action_cfg_dict.get("data", {}), DataConfig
    ))
    inherited_clip_model_name = getattr(
        inherited_data_cfg, "clip_model_name", DEFAULT_CLIP_TEXT_MODEL
    )
    mask_augmentation = (
        args.mask_augmentation
        if args.mask_augmentation is not None
        else inherited_data_cfg.mask_augmentation
    )

    # Override encoder hidden_dim if explicitly specified
    if args.encoder_hidden_dim is not None:
        third_cfg.hidden_dim = args.encoder_hidden_dim
        wrist_cfg.hidden_dim = args.encoder_hidden_dim
    if args.use_wrist_goal_mask is not None:
        wrist_cfg.use_goal_mask = bool(args.use_wrist_goal_mask)
    if args.use_wrist_target_place is not None:
        wrist_cfg.use_target_place = bool(args.use_wrist_target_place)

    # Progress head hidden_dim must match encoder output dim
    progress_hidden_dim = third_cfg.hidden_dim
    default_progress_cfg = ProgressConfig()
    progress_config = ProgressConfig(
        hidden_dim=progress_hidden_dim,
        text_dim=dit_cfg.text_dim,
        clip_model_name=inherited_clip_model_name,
        lambda_dist=(
            default_progress_cfg.lambda_dist
            if args.progress_lambda_dist is None
            else args.progress_lambda_dist
        ),
        lambda_align=(
            default_progress_cfg.lambda_align
            if args.progress_lambda_align is None
            else args.progress_lambda_align
        ),
        lambda_tc=(
            default_progress_cfg.lambda_tc
            if args.progress_lambda_tc is None
            else args.progress_lambda_tc
        ),
        tc_mask_dist_thresh=args.tc_mask_dist_thresh,
    )

    data_config = DataConfig(
        episodes_json=args.episodes_json,
        tc_event_manifest=args.tc_event_manifest,
        frames_dir=args.frames_dir,
        parquet_dir=args.parquet_dir,
        packed_features_dir=args.packed_features_dir,
        clip_model_name=inherited_clip_model_name,
        text_feature_dim=dit_cfg.text_dim,
        text_dropout_prob=args.text_dropout_prob,
        mask_augmentation=mask_augmentation,
        reverse_text_dropout_prob=args.reverse_text_dropout_prob,
        reverse_max_mask_overlap=args.reverse_max_mask_overlap,
        tail_exclude_ratio=args.tail_exclude_ratio,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    config = TrainingConfig(
        third_encoder=third_cfg,
        wrist_encoder=wrist_cfg,
        dit=dit_cfg,
        progress=progress_config,
        data=data_config,
        condition_mode=final_condition_mode,
        cross_attn_mode=inherited_cross_attn_mode,
        use_reverse_aug=args.use_reverse_aug,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        batch_size_per_gpu=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        early_stop_patience=args.early_stop_patience,
    )

    if args.expected_config:
        from shared.training_config_check import check_training_config
        check_training_config(config, args.expected_config)

    eff_batch = args.batch_size * world_size * args.gradient_accumulation_steps

    log_print("=== RAIN Stage 2: Progress Training ===")
    log_print(f"Action checkpoint: {args.action_checkpoint}")
    if action_cfg_src:
        log_print(f"Action config: {action_cfg_src} (trunk/DiT/condition inherited)")
    else:
        log_print("Action config: <none> (using default trunk/DiT config)")
    post_only_tc_types = [t.strip() for t in args.post_only_tc_types.split(",")] if args.post_only_tc_types else None
    log_print(f"CondMode={config.condition_mode} ReverseAug={config.use_reverse_aug} "
              f"PostOnlyTC={args.post_only_tc} PostOnlyTypes={post_only_tc_types} "
              f"TailExclude={args.tail_exclude_ratio:.3f} TCMask={args.tc_mask_dist_thresh} "
              f"MaskAug={config.data.mask_augmentation}")
    log_print(f"TCEventManifest={config.data.tc_event_manifest}")
    log_print(
        "Action trunk flags: "
        f"use_plan={config.dit.use_plan}, "
        f"cross_attn_mode={config.cross_attn_mode}, "
        f"use_corr_mod={config.wrist_encoder.use_corr_modulation}, "
        f"wrist_cond={config.wrist_encoder.conditioning_mode}, "
        f"wrist_goal_mask={config.wrist_encoder.use_goal_mask}, "
        f"wrist_target_place={config.wrist_encoder.use_target_place}"
    )
    log_print(
        "Progress loss weights: "
        f"dist={config.progress.lambda_dist}, "
        f"align={config.progress.lambda_align}, "
        f"tc={config.progress.lambda_tc}"
    )
    log_print(f"World: {world_size}, Batch/GPU: {args.batch_size}, "
              f"Accum: {args.gradient_accumulation_steps}, Effective: {eff_batch}")

    ckpt_has_plan = _checkpoint_has_plan_tokens(args.action_checkpoint)
    if ckpt_has_plan != bool(config.dit.use_plan):
        if action_cfg_src:
            raise RuntimeError(
                "Action config and checkpoint mismatch for use_plan: "
                f"config={config.dit.use_plan}, checkpoint_has_plan={ckpt_has_plan}. "
                f"action_config={action_cfg_src}"
            )
        log_print(
            "[WARN] action config missing and use_plan mismatch inferred from checkpoint. "
            f"Auto-fixing use_plan to {ckpt_has_plan}."
        )
        config.dit.use_plan = ckpt_has_plan

    if is_main():
        with open(output_path / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        with open(output_path / "config_sources.json", "w") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "action_checkpoint": args.action_checkpoint,
                    "action_config": action_cfg_src,
                    "action_config_auto_inferred": (not bool(args.action_config) and action_cfg_src is not None),
                    "merge_policy": {
                        "from_action_config": [
                            "third_encoder",
                            "wrist_encoder",
                            "dit",
                            "condition_mode",
                        ],
                        "from_stage2_args": [
                            "progress",
                            "data",
                            "use_reverse_aug",
                            "training_hparams",
                            "output_dir",
                            "experiment_name",
                        ],
                    },
                },
                f,
                indent=2,
            )

    # Dataset (progress labels)
    log_print("Loading dataset...")
    train_ds, val_ds = build_datasets(
        episodes_json=data_config.episodes_json,
        tc_event_manifest=data_config.tc_event_manifest,
        frames_dir=data_config.frames_dir,
        parquet_dir=data_config.parquet_dir,
        packed_features_dir=data_config.packed_features_dir,
        use_progress=True,
        use_dit=False,
        use_plan=False,
        use_reverse_aug=config.use_reverse_aug,
        text_dropout_prob=data_config.text_dropout_prob,
        mask_augmentation=data_config.mask_augmentation,
        reverse_text_dropout_prob=data_config.reverse_text_dropout_prob,
        reverse_max_mask_overlap=data_config.reverse_max_mask_overlap,
        post_only_tc=args.post_only_tc,
        post_only_tc_types=post_only_tc_types,
        tail_exclude_ratio=data_config.tail_exclude_ratio,
        use_target_place=getattr(config.third_encoder, "use_target_place", False),
        use_wrist_goal_mask=getattr(config.wrist_encoder, "use_goal_mask", False),
        use_wrist_target_place=getattr(config.wrist_encoder, "use_target_place", False),
        clip_model_name=data_config.clip_model_name,
        text_feature_dim=data_config.text_feature_dim,
        train_ratio=data_config.train_ratio,
        seed=data_config.seed,
    )
    log_print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    if train_ds.dino_hidden_dim != third_cfg.dino_dim:
        raise ValueError(
            "Packed DINO hidden dim mismatch: "
            f"dataset={train_ds.dino_hidden_dim}, config={third_cfg.dino_dim}. "
            "Check --packed-features-dir and inherited action config."
        )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1 else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    steps_per_epoch = max(len(train_loader) // config.gradient_accumulation_steps, 1)
    total_steps = steps_per_epoch * config.max_epochs
    log_print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    # Model
    log_print("Creating RAINModel...")
    model = RAINModel(config).to(device)

    # Load action checkpoint (encoder + DiT)
    log_print(f"Loading action checkpoint: {args.action_checkpoint}")
    model.load_action_checkpoint(args.action_checkpoint)

    # Set training stage: freeze encoder+DiT, unfreeze active progress head
    model.set_training_stage("progress")

    # Verify only active progress-head params are trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    expected_prefixes = ("fusion_branch.",)
    non_expected = [n for n in trainable_names if not n.startswith(expected_prefixes)]
    if non_expected:
        log_print(f"[WARN] Non-progress-head params are trainable: {non_expected}")
    log_print(
        f"Trainable params: {len(trainable_names)} "
        f"(all expected progress head: {all(n.startswith(expected_prefixes) for n in trainable_names)})"
    )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = create_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = create_scheduler(optimizer, config.warmup_steps, total_steps, config.learning_rate)
    scaler = GradScaler("cuda", enabled=config.mixed_precision)

    global_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        # Re-freeze encoder+DiT after loading full state_dict
        raw_model.set_training_stage("progress")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt.get("epoch", 0)
        log_print(f"Resumed from {args.resume} at step {global_step}, epoch {start_epoch}")

    if world_size > 1:
        dist.barrier()

    train_logger = None
    if is_main():
        train_logger = TrainLogger(str(output_path / "train_log.csv"))

    # Early stopping
    best_metric = 0.0  # task_comp_accuracy (higher is better)
    patience_counter = 0

    log_print("Starting training...")

    for epoch in range(start_epoch + 1, config.max_epochs + 1):
        epoch_t0 = time.time()

        train_metrics, global_step, nan_stop, grad_norm = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config, global_step,
        )
        epoch_time = time.time() - epoch_t0

        if nan_stop:
            log_print("Training stopped due to repeated NaN losses.")
            break

        # Validation
        val_metrics = {}
        if epoch % config.eval_every_n_epochs == 0:
            val_metrics = validate(model, val_loader, device, config)

        if is_main():
            lr = scheduler.get_last_lr()[0]

            # Log
            parts = [f"Epoch {epoch}"]
            parts.append(f"Loss: {train_metrics.get('loss', 0):.4f}")
            if "progress_loss" in train_metrics:
                parts.append(f"Prog: {train_metrics['progress_loss']:.4f}")
            if "fusion_progress_loss" in train_metrics:
                parts.append(f"Fus: {train_metrics['fusion_progress_loss']:.4f}")
            if val_metrics:
                parts.append(f"|| Val: {val_metrics.get('loss', 0):.4f}")
                if "task_comp_accuracy" in val_metrics:
                    parts.append(f"TC: {val_metrics['task_comp_accuracy']:.4f}")
                if "task_comp_f1" in val_metrics:
                    parts.append(f"F1: {val_metrics['task_comp_f1']:.4f}")
                if "dist_mae" in val_metrics:
                    parts.append(f"Dist: {val_metrics['dist_mae']:.4f}")
                if "fusion_gate_global_mean" in val_metrics:
                    parts.append(f"Gate: G={val_metrics['fusion_gate_global_mean']:.2f} "
                                 f"O={val_metrics['fusion_gate_obj_mean']:.2f}")
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
                for k, v in val_metrics.items():
                    log_kwargs[f"val_{k}"] = f"{v:.6f}" if isinstance(v, float) else str(v)
                train_logger.log(**log_kwargs)

            # Save best checkpoint
            if val_metrics:
                current = val_metrics.get("task_comp_accuracy", 0)
                improved = current > best_metric

                if improved:
                    best_metric = current
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, epoch,
                        output_path / "checkpoints" / "checkpoint_best.pt",
                        val_metrics,
                    )
                    log_print(f"  -> New best task_comp_accuracy: {current:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Save latest checkpoint every epoch
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                global_step, epoch,
                output_path / "checkpoints" / "checkpoint_latest.pt",
                val_metrics,
            )

        # Early stopping (broadcast from rank 0)
        if dist.is_initialized():
            stop_flag = torch.tensor(1.0 if patience_counter >= config.early_stop_patience else 0.0, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() > 0.5:
                log_print(f"Early stopping at epoch {epoch} (patience={config.early_stop_patience})")
                break
        elif patience_counter >= config.early_stop_patience:
            log_print(f"Early stopping at epoch {epoch} (patience={config.early_stop_patience})")
            break

    # Final save
    if is_main():
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            global_step, epoch,
            output_path / "checkpoints" / "checkpoint_final.pt",
            val_metrics if val_metrics else None,
        )
        log_print(f"Training complete! Best task_comp_accuracy: {best_metric:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
