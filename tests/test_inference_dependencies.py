"""Inference cache isolation, action conditioning, and rendering failures."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rain.models.model import RAINModel


ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = ("eval/eval_libero.py", "final_libero_ex_eval/impl/runtime.py")


def function_node(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def isolated_function(path, name, namespace):
    """Load a helper without installing global hooks or starting a simulator."""
    node = function_node(path, name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("path", RUNTIMES)
def test_dino_uses_configured_torch_cache(path, tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr(torch.hub, "_hub_dir", None)
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    local_repo.mkdir(parents=True)
    original = Mock(return_value=object())
    load = isolated_function(path, "_patched_hub_load", {
        "torch": torch, "os": os, "_original_hub_load": original,
    })
    result = load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
    original.assert_called_once_with(
        str(local_repo), "dinov2_vitl14_reg", pretrained=True, source="local",
    )
    assert result is original.return_value


@pytest.mark.parametrize("path", RUNTIMES)
@pytest.mark.parametrize("repo,kwargs", [
    ("facebookresearch/dinov2", {}),
    ("another/repository", {}),
    ("/local/dinov2", {"source": "local"}),
])
def test_dino_cache_miss_and_other_sources_are_forwarded(path, repo, kwargs, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    original = Mock()
    load = isolated_function(path, "_patched_hub_load", {
        "torch": torch, "os": os, "_original_hub_load": original,
    })
    load(repo, "model", **kwargs)
    original.assert_called_once_with(repo, "model", **kwargs)


@pytest.mark.parametrize("config", ("action.json", "transition.json"))
def test_public_recipes_use_checkpoint_action_type_embeddings(config):
    assert json.loads((ROOT / "configs" / config).read_text())["condition_mode"] == "action_type"


@pytest.mark.parametrize("description", (None, "zeros", "random", "nan"))
def test_task_description_cannot_change_action_type_conditioning(description):
    # Every supported action selects the saved embedding, irrespective of the
    # description payload transported by the evaluation worker.
    embeddings = torch.arange(7 * 768, dtype=torch.float32).reshape(7, 768)
    model = SimpleNamespace(condition_mode="action_type", action_type_clip_features=embeddings)
    features = {
        None: None,
        "zeros": torch.zeros(7, 768),
        "random": torch.randn(7, 768),
        "nan": torch.full((7, 768), float("nan")),
    }[description]
    actual = RAINModel._build_condition_text_feat(
        model, features, torch.arange(7), 7, torch.device("cpu"),
    )
    torch.testing.assert_close(actual, embeddings, rtol=0, atol=0)


def test_libero_does_not_load_description_encoder():
    node = function_node("eval/eval_libero.py", "_gpu_worker_process")
    assert "TextFeatureCache" not in ast.unparse(node)
    assignment = next(child for child in node.body if isinstance(child, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "text_feat"
                              for target in child.targets))
    placeholder = eval(compile(ast.Expression(assignment.value), "placeholder", "eval"), {"np": np})
    assert placeholder.shape == (768,)
    assert placeholder.dtype == np.float32
    assert not placeholder.any()


def segmentation_helper():
    return isolated_function("eval/eval_libero.py", "sim_mask_for_object_id", {
        "np": np, "LIBERO_ENV_RESOLUTION": 256,
        "is_object_segmentable": lambda episode, object_id: object_id != "missing",
        "_get_task_specific_geom_ids": lambda *args, **kwargs: [3],
    })


def test_libero_renderer_failure_is_not_a_missing_target():
    failure = OverflowError("renderer failed")
    render = Mock(side_effect=failure)
    env = SimpleNamespace(sim=SimpleNamespace(render=render))
    with pytest.raises(RuntimeError, match="agentview.*target") as error:
        segmentation_helper()(env, {}, "target")
    assert error.value.__cause__ is failure


def test_libero_invalid_segmentation_shape_is_not_a_missing_target():
    env = SimpleNamespace(sim=SimpleNamespace(render=Mock(return_value=np.zeros((4, 4)))))
    with pytest.raises(ValueError, match="segmentation shape"):
        segmentation_helper()(env, {}, "target")


def test_libero_invisible_and_missing_targets_remain_optional():
    env = SimpleNamespace(sim=SimpleNamespace(render=Mock(return_value=np.zeros((4, 4, 2)))))
    mask = segmentation_helper()
    assert mask(env, {}, "target") is None
    assert mask(env, {}, "missing") is None
    assert mask(None, {}, "target") is None
    assert env.sim.render.call_count == 1


def test_libero_visible_mask_values_and_orientation_are_unchanged():
    seg = np.zeros((4, 4, 2), dtype=np.int32)
    seg[0, 1] = [5, 3]
    env = SimpleNamespace(sim=SimpleNamespace(render=Mock(return_value=seg)))
    actual = segmentation_helper()(env, {}, "target", image_size=4)
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[3, 2] = 1
    np.testing.assert_array_equal(actual, expected)
