"""Supported implementation and GT controller regression checks."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
INVARIANTS = json.loads(
    (Path(__file__).parent / "fixtures/gt_source_invariants.json").read_text()
)


def _canonical(node):
    if isinstance(node, ast.AST):
        return {
            "node": type(node).__name__,
            **{
                key: _canonical(value)
                for key, value in ast.iter_fields(node)
                if key != "type_params"
            },
        }
    if isinstance(node, list):
        return [_canonical(value) for value in node]
    if node is Ellipsis:
        return {"constant": "Ellipsis"}
    return node


def _digest(node):
    payload = json.dumps(_canonical(node), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize("relative", INVARIANTS["functions"])
def test_preserved_gt_function_asts(relative):
    tree = ast.parse((ROOT / relative).read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for name, expected in INVARIANTS["functions"][relative].items():
        assert _digest(functions[name]) == expected, f"{relative}:{name}"


def test_preserved_gt_worker_branch_ast():
    tree = ast.parse((ROOT / "eval/eval_libero.py").read_text())
    worker = next(node for node in tree.body if getattr(node, "name", "") == "_gpu_worker_process")
    branch = next(
        node for node in ast.walk(worker)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "use_sim_seg and all_episodes is not None"
    )
    assert _digest(branch) == INVARIANTS["gt_worker_branch"]


@pytest.mark.parametrize("relative", INVARIANTS["unchanged_training_files"])
def test_training_and_model_sources_unchanged(relative):
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == INVARIANTS["unchanged_training_files"][relative]


def test_private_implementations_not_shipped():
    excluded = [
        "rain_perception",
        "shared/data/tc_event_labels.py",
        "shared/strict_text_plans.py",
        "shared/sim_jf_audit.py",
        "shared/action_sampling.py",
        "final_libero_ex_eval/impl/evaluator.py",
        "final_libero_ex_eval/impl/online_rollout.py",
    ]
    assert all(not (ROOT / relative).exists() for relative in excluded)
    evaluator = (ROOT / "eval/eval_libero.py").read_text()
    assert "strict_online" not in evaluator
    assert "VLM_evaluation" not in evaluator
    assert "load_base_model" not in evaluator
    assert "load_base_plus_model" not in evaluator
    assert "rain_perception" not in (ROOT / "pyproject.toml").read_text()


@pytest.mark.parametrize("option", ["--strict-online-masks", "--strict-plans", "--strict-provider-source"])
def test_private_cli_options_rejected_before_execution(option, monkeypatch, capsys):
    from eval.eval_libero import main
    monkeypatch.setattr(sys, "argv", ["eval", "--model-type", "rain", "--checkpoint", "missing.pt", option])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("model_type", ["base", "base_plus"])
def test_unavailable_models_rejected(model_type, monkeypatch, capsys):
    from eval.eval_libero import main
    monkeypatch.setattr(sys, "argv", ["eval", "--model-type", model_type, "--checkpoint", "missing.pt"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_public_low_level_cli_requires_gt(monkeypatch, capsys):
    from eval.eval_libero import main
    monkeypatch.setattr(sys, "argv", ["eval", "--model-type", "rain", "--checkpoint", "missing.pt"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    assert "requires --use-sim-seg" in capsys.readouterr().err


def test_manifest_consumption_without_offline_generator(tmp_path):
    from shared.data.dataset import RAINDataset
    dataset = RAINDataset.__new__(RAINDataset)
    segment = {"subtask_id": 1, "action_type": "release", "start_frame": 0, "end_frame": 7}
    dataset.episodes = {3: {"num_frames": 12, "subtask_segments": [segment]}}
    dataset.tc_events = {}
    row = dict(segment, guard_start=6, event_frame=8)
    manifest = {"version": "tc_event_v3", "episodes": [{"episode_index": 3, "segments": [row]}]}
    path = tmp_path / "tc.json"
    path.write_text(json.dumps(manifest))
    dataset._load_tc_event_manifest(str(path))
    assert [dataset._tc_event_label(3, 0, frame) for frame in (5, 6, 7, 8)] == [0.0, None, None, 1.0]
    assert importlib.util.find_spec("shared.data.tc_event_labels") is None
