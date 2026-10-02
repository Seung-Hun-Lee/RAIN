import argparse
import json
from pathlib import Path
from dataclasses import asdict
import sys

import pytest
import torch
from torch import nn

from rain.checkpoints import load_exact_action, load_exact_transition
from rain.inference import load_config
from rain.train import ABLATIONS, build_recipe


def args(stage="action", **kwargs):
    values = dict(stage=stage, data_root="/tmp/data", output_root="/tmp/output", action_checkpoint=None,
                  action_config=None, ablation="full", head_variant="region_gated", packed_features_dir=None,
                  batch_size=None, num_workers=None, nproc=4, resume=None)
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_saved_recipe_architecture():
    config, command, _ = build_recipe(args())
    assert config["dit"]["num_scales"] == 3
    assert config["dit"]["num_blocks"] == 12
    assert config["data"]["train_ratio"] == 1.0
    assert "--force-goal-pair-batch" in command
    assert config["max_steps"] == 100000


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_action_ablations(ablation):
    options = {"packed_features_dir": "/tmp/last_scale"} if ablation == "no_multistage" else {}
    config, command, _ = build_recipe(args(ablation=ablation, **options))
    for path, expected in ABLATIONS[ablation].items():
        actual = config
        for part in path.split("."):
            actual = actual[part]
        assert actual == expected
    assert "--expected-config" in command


def test_transition_recipe_accuracy_and_tc_only(tmp_path):
    config, _, _ = build_recipe(args())
    path = tmp_path / "action.json"
    path.write_text(json.dumps(config))
    transition, command, _ = build_recipe(args("transition", action_checkpoint="/tmp/action.pt", action_config=str(path)))
    assert transition["progress"]["lambda_dist"] == transition["progress"]["lambda_align"] == 0
    assert transition["early_stop_metric"] == "task_comp_accuracy"
    assert transition["max_epochs"] == 30
    assert transition["dit"] == config["dit"]
    assert "rain.train_transition" in command


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(3, 2)
        self.fusion_branch = nn.Linear(2, 1)


def test_strict_checkpoint_restoration_and_architecture(tmp_path):
    model = SmallPolicy()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": state, "progress_architecture": "region_gated"}, path)
    load_exact_action(model, path)
    load_exact_transition(model, path)
    with pytest.raises(ValueError, match="architecture"):
        load_exact_transition(model, path, "global_gated")
    state.pop("dit.bias")
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(ValueError, match="key mismatch"):
        load_exact_action(model, path)


def test_public_import_names():
    from rain import RAIN, TCE, TarLN, PlanDiT, TransitionHead
    assert TransitionHead().variant == "region_gated"
    assert all(issubclass(cls, nn.Module) for cls in (RAIN, TCE, TarLN, PlanDiT, TransitionHead))


def test_config_loader_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"dit":{"nonexistent":true}}')
    with pytest.raises(ValueError, match="Unknown dit"):
        load_config(path)


@pytest.mark.parametrize("stage", ["action", "transition"])
@pytest.mark.parametrize("ablation", ABLATIONS)
def test_portable_cli_matches_actual_trainer_config(stage, ablation, tmp_path, monkeypatch):
    import rainv2.train_action as action_trainer
    import rainv2.train_progress as transition_trainer
    import shared.training_config_check as checker
    import rain.train as launcher
    packed = str(tmp_path / "last_scale") if ablation == "no_multistage" else None
    action, _, _ = build_recipe(args(output_root=str(tmp_path), ablation=ablation, packed_features_dir=packed))
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action))
    options = args(stage, output_root=str(tmp_path), ablation=ablation, packed_features_dir=packed,
                   action_checkpoint=str(tmp_path / "action.pt"), action_config=str(action_path))
    expected, command, _ = build_recipe(options)
    trainer = action_trainer if stage == "action" else transition_trainer
    monkeypatch.setattr(trainer, "setup_distributed", lambda: (0, 1, 0))
    monkeypatch.setattr(trainer, "set_seed", lambda *a: None)
    monkeypatch.setattr(trainer, "setup_experiment_log_tee", lambda p: p / "train.log")
    module = "rainv2.train_action" if stage == "action" else "rain.train_transition"
    monkeypatch.setattr(sys, "argv", [module] + command[command.index(module) + 1:])
    class Checked(Exception):
        pass
    def check(actual, path):
        assert asdict(actual) == expected
        raise Checked()
    monkeypatch.setattr(checker, "check_training_config", check)
    with pytest.raises(Checked):
        trainer.main()


def test_final_condition_tc_protocol_boundary_and_rejection():
    from rain.transition_control import FinalConditionTCStop
    stop = FinalConditionTCStop()
    assert not stop.observe(0.7, final_condition=True)
    assert stop.observe(0.7, final_condition=True)
    assert not stop.observe(1.0, final_condition=False)
    assert not stop.observe(float("nan"), final_condition=True)
    assert not stop.observe(1.0, final_condition=True)


def test_bddl_exact_workspace_region(tmp_path):
    from rain.workspace_geometry import _region_bounds
    path = tmp_path / "task.bddl"
    path.write_text('(define (problem test) (:regions (goal (:target main_table) (:ranges ((0.1 0.2 0.3 0.4))))))')
    target, bounds = _region_bounds(path, "main_table_goal")
    assert target == "main_table"
    assert bounds.tolist() == [0.1, 0.2, 0.3, 0.4]
