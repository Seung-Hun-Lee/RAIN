"""Inference cache isolation, action conditioning, and rendering failures."""

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rain.models.model import RAINModel
from shared.dinov2 import DINO_REPOSITORY, DINO_REVISION, load_dinov2


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


def test_dino_uses_pinned_revision_in_configured_torch_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr(torch.hub, "_hub_dir", None)
    for revision in ("main", DINO_REVISION):
        cached_repo = Path(torch.hub.get_dir()) / f"facebookresearch_dinov2_{revision}"
        cached_repo.mkdir(parents=True)
        (cached_repo / "hubconf.py").write_text(
            f"def dinov2_vitl14_reg():\n    return {revision!r}\n"
        )
    download = Mock(side_effect=AssertionError("cached source should not be downloaded"))
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    assert load_dinov2() == DINO_REVISION
    download.assert_not_called()


def test_dino_cache_miss_requests_recorded_public_revision(monkeypatch):
    original = Mock(return_value=object())
    monkeypatch.setattr(torch.hub, "load", original)
    result = load_dinov2("dinov2_vits14_reg")
    original.assert_called_once_with(
        DINO_REPOSITORY, "dinov2_vits14_reg", verbose=False,
        trust_repo=True, skip_validation=True,
    )
    assert result is original.return_value


@pytest.mark.parametrize("path", RUNTIMES)
def test_evaluator_import_does_not_replace_torch_hub(path):
    original = torch.hub.load
    module = importlib.import_module(path[:-3].replace("/", "."))
    importlib.reload(module)
    assert torch.hub.load is original


def test_dino_manifest_and_all_loaders_share_one_source():
    identity = json.loads((ROOT / "configs/pretrained.json").read_text())["dinov2"]
    assert identity["source_git_commit"] == DINO_REVISION
    for module_name in ("rain.prepare_data", "rain.models.multiscale_vision", "shared.components"):
        module = importlib.import_module(module_name)
        assert module.load_dinov2 is load_dinov2
        source = Path(module.__file__).read_text()
        assert "torch.hub.load(" not in source
        assert "facebookresearch_dinov2_main" not in source


@pytest.mark.parametrize("multiscale", (False, True))
def test_online_dino_loaders_keep_backbone_frozen(multiscale, monkeypatch):
    from rain.models.multiscale_vision import FrozenDINOv2LargeMultiScale
    from shared.components import FrozenDINOv2

    backbone = torch.nn.Linear(3, 3)
    original = Mock(return_value=backbone)
    monkeypatch.setattr(torch.hub, "load", original)
    model = FrozenDINOv2LargeMultiScale() if multiscale else FrozenDINOv2()
    assert model.backbone is backbone
    original.assert_called_once_with(
        DINO_REPOSITORY, "dinov2_vitl14_reg", verbose=False,
        trust_repo=True, skip_validation=True,
    )
    model.train()
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


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
