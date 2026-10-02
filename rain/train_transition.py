"""Train a pooling Transition Head with the shared progress-training pipeline."""
import os
import argparse

import torch
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import rain.training.transition as original
from .model import PoolingModel


@torch.no_grad()
def validate(model, dataloader, device, config):
    """Compute per-rank TC loss, accuracy, precision, recall, and F1."""
    model.eval()
    totals = {}
    num_batches = 0
    all_tc_probs, all_tc_gt, all_dist_gt = [], [], []
    for batch in tqdm(dataloader, desc='Val', disable=not original.is_main()):
        batch = original.move_to_device(batch, device)
        with autocast('cuda', enabled=config.mixed_precision):
            output = model(
                dino_third=batch['dino_third'], dino_wrist=batch['dino_wrist'],
                goal_mask_third=batch.get('goal_mask_third'), goal_mask_wrist=batch.get('goal_mask_wrist'),
                target_place_mask=batch.get('target_place_mask_third'),
                target_place_mask_wrist=batch.get('target_place_mask_wrist'),
                text_feat=batch.get('text_feat'), text_feat_full=batch.get('text_feat_full'),
                action_type=batch.get('action_type'), gt_distance=batch.get('gt_distance'),
                gt_alignment=batch.get('gt_alignment'), gt_task_completion=batch.get('gt_task_completion'),
            )
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and 'loss' in k:
                totals[k] = totals.get(k, 0.0) + v.item()
        num_batches += 1
        raw_model = model.module if isinstance(model, DDP) else model
        with autocast('cuda', enabled=config.mixed_precision):
            pred_out = raw_model.predict(
                dino_third=batch['dino_third'], dino_wrist=batch['dino_wrist'],
                goal_mask_third=batch.get('goal_mask_third'), goal_mask_wrist=batch.get('goal_mask_wrist'),
                target_place_mask=batch.get('target_place_mask_third'),
                target_place_mask_wrist=batch.get('target_place_mask_wrist'),
                text_feat=batch.get('text_feat'), action_type=batch.get('action_type'),
            )
        all_tc_probs.append(pred_out['task_comp_prob'].cpu())
        all_tc_gt.append(batch['gt_task_completion'].cpu())
        all_dist_gt.append(batch['gt_distance'].cpu())
    metrics = {k: v / max(num_batches, 1) for k, v in totals.items()}
    if all_tc_probs:
        tc_probs = torch.cat(all_tc_probs)
        tc_gt = torch.cat(all_tc_gt)
        dist_gt_all = torch.cat(all_dist_gt)
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
        metrics['task_comp_accuracy'] = tc_correct.mean().item()
        tp = ((tc_pred == 1) & (tc_gt_m > 0.5)).sum().float()
        fp = ((tc_pred == 1) & (tc_gt_m <= 0.5)).sum().float()
        fn = ((tc_pred == 0) & (tc_gt_m > 0.5)).sum().float()
        prec = (tp / (tp + fp).clamp_min(1)).item()
        rec = (tp / (tp + fn).clamp_min(1)).item()
        metrics['task_comp_precision'] = prec
        metrics['task_comp_recall'] = rec
        metrics['task_comp_f1'] = 2 * prec * rec / max(prec + rec, 1e-8)
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, global_step, epoch, path, val_metrics=None):
    if not original.is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    ckpt = dict(global_step=global_step, epoch=epoch, model_state_dict=raw.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(), progress_architecture=raw.progress_architecture)
    if val_metrics:
        ckpt['val_metrics'] = val_metrics
    torch.save(ckpt, path)
    original.log_print(f'Saved checkpoint: {path}')


def main():
    # The published recipe uses four ranks; explicit smaller runs are supported.
    assert 'RAIN_SKIP_CLIP_ACTION_TYPE_INIT' not in os.environ
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument('--resume')
    resume_args, _ = resume_parser.parse_known_args()
    if resume_args.resume:
        checkpoint = torch.load(resume_args.resume, map_location='cpu', weights_only=True, mmap=True)
        variant = os.environ.get('RAIN_POOLING_HEAD_VARIANT', 'region_gated')
        if checkpoint.get('progress_architecture') != variant:
            raise ValueError('Resume checkpoint head variant does not match the requested Transition Head')
    original.RAINModel = PoolingModel
    original.validate = validate
    original.save_checkpoint = save_checkpoint
    original.main()


if __name__ == '__main__':
    main()
