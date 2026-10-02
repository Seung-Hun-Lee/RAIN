"""Renderer regressions that run without a GPU or OpenGL context."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = ["eval/eval_libero.py", "final_libero_ex_eval/impl/runtime.py"]


def source_function(relative, name, namespace, parent=None):
    tree = ast.parse((ROOT / relative).read_text())
    if parent:
        tree = next(node for node in tree.body if getattr(node, "name", None) == parent)
    node = next(node for node in ast.walk(tree) if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("relative", RUNTIMES)
@pytest.mark.parametrize("depth", [False, True])
def test_segmentation_decodes_ids_above_uint8_without_overflow(relative, depth):
    encoded_rgb = np.array([[[1, 1, 0], [0, 0, 0]]], dtype=np.uint8)

    def read_pixels(*, rgb, depth, **kwargs):
        rgb[:] = encoded_rgb
        if depth is not None:
            depth[:] = 0.5

    bindings = SimpleNamespace(np=np, mujoco=SimpleNamespace(
        MjrRect=lambda *args: args, mjr_readPixels=read_pixels))
    decode = source_function(relative, "_patched", {"binding_utils": bindings},
                             parent="_patch_robosuite_egl")
    geoms = [SimpleNamespace(segid=-1) for _ in range(257)]
    geoms[-1] = SimpleNamespace(segid=256, objtype=5, objid=17)
    context = SimpleNamespace(con=None, scn=SimpleNamespace(ngeom=257, geoms=geoms))
    result = decode(context, 2, 1, depth=depth, segmentation=True)
    segmentation = result[0] if depth else result
    np.testing.assert_array_equal(segmentation, [[[5, 17], [-1, -1]]])
    if depth:
        np.testing.assert_array_equal(result[1], [[0.5, 0.5]])


def mask_function(relative):
    return source_function(relative, "sim_mask_for_object_id", {
        "np": np, "LIBERO_ENV_RESOLUTION": 256,
        "is_object_segmentable": lambda *args: True,
        "_get_task_specific_geom_ids": lambda *args, **kwargs: [17],
    })


@pytest.mark.parametrize("relative", RUNTIMES)
def test_render_error_is_not_silently_treated_as_invisible_object(relative):
    def render(**kwargs):
        raise OverflowError("invalid segmentation decoding")

    env = SimpleNamespace(sim=SimpleNamespace(render=render))
    with pytest.raises(RuntimeError, match="Segmentation rendering failed") as error:
        mask_function(relative)(env, {}, "target")
    assert isinstance(error.value.__cause__, OverflowError)


@pytest.mark.parametrize("relative", RUNTIMES)
@pytest.mark.parametrize("visible", [False, True])
def test_visible_and_invisible_masks_keep_their_semantics(relative, visible):
    segmentation = np.full((2, 2, 2), -1, dtype=np.int32)
    if visible:
        segmentation[0, 0] = [5, 17]
    env = SimpleNamespace(sim=SimpleNamespace(render=lambda **kwargs: segmentation))
    mask = mask_function(relative)(env, {}, "target")
    if visible:
        np.testing.assert_array_equal(mask, [[0, 0], [0, 1]])
    else:
        assert mask is None


@pytest.mark.parametrize("relative", RUNTIMES)
def test_invalid_segmentation_shape_is_an_error(relative):
    env = SimpleNamespace(sim=SimpleNamespace(render=lambda **kwargs: np.zeros((2, 2))))
    with pytest.raises(ValueError, match="Expected segmentation shape"):
        mask_function(relative)(env, {}, "target")
