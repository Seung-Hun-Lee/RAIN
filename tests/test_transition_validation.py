"""Supervised validation uses the same TC loss without enabling training mode."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rain.configs.config import ProgressConfig
from rain.model import PoolingModel
from rain.train_transition import validate
from rain.transition_head import PoolingTCHead, VARIANTS, tc_loss


class EncodedViewModel(nn.Module):
    """Exercise the public forward/predict paths with already-encoded views."""
    forward = PoolingModel.forward_progress
    predict = PoolingModel.predict

    def __init__(self, variant):
        super().__init__()
        self.config = SimpleNamespace(
            dit=SimpleNamespace(num_scales=3), mixed_precision=False,
            progress=ProgressConfig(lambda_dist=0, lambda_align=0, lambda_tc=1.2,
                                    release_task_comp_weight=1.5, tc_mask_dist_thresh=0.8),
        )
        self.online_dino = None
        self.fusion_branch = PoolingTCHead(variant)

    def _encode_views_split(self, third, wrist, *args, **kwargs):
        return third, wrist, third.new_zeros(third.shape[:2]), None

    def _encode_vision_scales(self, scales, *args):
        return [self._encode_views_split(*scale) for scale in scales]

    def _build_condition_text_feat(self, text, *args):
        return text


@pytest.fixture
def batch():
    torch.manual_seed(42)
    return {
        "dino_third": torch.randn(4, 8, 1024, requires_grad=True),
        "dino_wrist": torch.randn(4, 8, 1024, requires_grad=True),
        "goal_mask_third": torch.ones(4, 8),
        "goal_mask_wrist": torch.zeros(4, 8),
        "text_feat": torch.randn(4, 768),
        "action_type": torch.tensor([0, 0, 1, 5]),
        "gt_distance": torch.tensor([0.2, 0.9, 0.95, 0.1]),
        "gt_task_completion": torch.tensor([0., 0., 1., 1.]),
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_eval_loss_matches_training_eligibility_and_release_weights(variant, batch):
    model = EncodedViewModel(variant)
    train_losses = model.train()(**batch)
    with torch.no_grad():
        eval_losses = model.eval()(**batch)
        predictions, _ = model.fusion_branch(
            batch["dino_third"], batch["dino_wrist"], batch["goal_mask_third"],
            batch["goal_mask_wrist"], batch["text_feat"],
        )
        expected = tc_loss(predictions["task_comp_logit"], batch["gt_distance"],
                           batch["gt_task_completion"], batch["action_type"], model.config.progress)
    assert eval_losses["loss"].item() > 0
    assert not eval_losses["loss"].requires_grad
    for key in train_losses:
        torch.testing.assert_close(eval_losses[key], train_losses[key], rtol=0, atol=0)
    torch.testing.assert_close(eval_losses["loss"], expected["progress_loss"], rtol=0, atol=0)
    train_losses["loss"].backward()
    assert batch["dino_third"].grad is None and batch["dino_wrist"].grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.fusion_branch.parameters())


def test_actual_validation_loop_records_supervised_loss_without_gradients(batch):
    model = EncodedViewModel("region_gated").eval()
    with torch.no_grad():
        expected = model(**batch)["loss"].item()
        predictions = model.predict(**batch)["task_comp_prob"]
    metrics = validate(model, [batch], torch.device("cpu"), model.config)
    assert not model.training
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["loss"] > 0
    eligible = (batch["gt_distance"] <= 0.8) | (batch["gt_task_completion"] > 0.5)
    expected_accuracy = ((predictions[eligible] > 0.5)
                         == (batch["gt_task_completion"][eligible] > 0.5)).float().mean().item()
    assert metrics["task_comp_accuracy"] == expected_accuracy
    assert all(p.grad is None for p in model.parameters())


def test_forward_without_supervision_keeps_existing_zero_placeholder(batch):
    model = EncodedViewModel("region_gated").eval()
    batch.pop("gt_distance")
    batch.pop("gt_task_completion")
    with torch.no_grad():
        assert model(**batch)["loss"].item() == 0
