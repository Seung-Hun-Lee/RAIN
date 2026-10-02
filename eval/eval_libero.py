#!/usr/bin/env python3
"""RAIN GT-mask LIBERO evaluation.

Use ``python -m rain.eval_libero`` for the public interface.
Rollouts combine the action model, region-gated Transition Head, GT mask
construction, and subtask switching controller.
"""

import argparse
import importlib.util
import json
import os
import queue as pyqueue
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Process, Queue, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shared.data.episode_io import load_episode_records
from shared.eval_transition import (
    subtask_completion_ready,
)

# ---------------------------------------------------------------------------
# Use the local DINO torch.hub cache when available.
# ---------------------------------------------------------------------------
import torch
_original_hub_load = torch.hub.load

def _patched_hub_load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2" and kwargs.get("source", "github") == "github":
        local_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.isdir(local_path):
            kwargs["source"] = "local"
            return _original_hub_load(local_path, model, *args, **kwargs)
    return _original_hub_load(repo_or_dir, model, *args, **kwargs)

torch.hub.load = _patched_hub_load

LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
MIDDLE_DRAWER_HANDLE_GEOM_IDS = [189, 190, 191, 192]
TOP_DRAWER_HANDLE_GEOM_IDS = [178, 179, 180, 181]
CABINET_TOP_RELEASE_TASKS = {
    "put the wine bottle on top of the cabinet",
    "put the bowl on top of the cabinet",
}
DEFAULT_REGION_SIM_MASK_TARGETS = {
    "main_table_stove_front_region",
    "living_room_table_plate_right_region",
}





ACTION_TYPE_TO_ID = {
    "grasp": 0, "release": 1, "push": 2,
    "turn_on": 3, "close": 4, "open": 5,
    "approach": 6,
}


def action_type_to_id(action_type) -> int:
    t = str(action_type).strip().lower()
    alias = {
        "pickup": "grasp", "pick": "grasp", "lift": "grasp",
        "approach": "approach",
        "place": "release", "drop": "release", "put": "release",
        "turnon": "turn_on", "turnoff": "turn_on",
        "turn_off": "turn_on", "switch_off": "turn_on",
    }
    return ACTION_TYPE_TO_ID.get(alias.get(t, t), 0)


def infer_action_type_from_language(task_language: str, default="grasp") -> str:
    text = str(task_language or "").strip().lower()
    if not text:
        return default
    keyword_map = {
        "approach": ["approach"],
        "turn_on": ["turn on", "switch on"],
        "close": ["close", "shut"],
        "open": ["open"],
        "push": ["push", "slide"],
        "release": ["place", "put", "drop", "release", "set down"],
        "grasp": ["pick up", "pickup", "pick", "grab", "lift", "take"],
    }
    best_type, best_pos = None, None
    for at, keywords in keyword_map.items():
        for kw in keywords:
            pos = text.find(kw)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos, best_type = pos, at
    return best_type or default


def _quat2axisangle(q):
    """Quaternion (xyzw) to axis-angle (3D)."""
    import math
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[3] < 0:
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def extract_state(obs, state_dim=8):
    """Extract robot state matching parquet observation.state format."""
    xyz = obs["robot0_eef_pos"].astype(np.float32)
    aa = _quat2axisangle(obs["robot0_eef_quat"])
    if state_dim == 7:
        gripper = obs["robot0_gripper_qpos"][:1].astype(np.float32)
    else:
        gripper = obs["robot0_gripper_qpos"].astype(np.float32)
    return np.concatenate([xyz, aa, gripper])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------








def load_rainv2_model(checkpoint_path, progress_checkpoint, device):
    """Restore every model tensor and require an adjacent configuration file."""
    from rain.inference import load_policy
    path = Path(checkpoint_path)
    candidates = (path.parent / "config.json", path.parent.parent / "config.json")
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        raise FileNotFoundError("Action config.json is required next to the checkpoint or its checkpoints directory")
    model = load_policy(checkpoint_path, progress_checkpoint, config_path, device)
    return model, model.config



def _resolve_checkpoint_path(path_or_dir: str, checkpoint_type: str = "latest") -> Path:
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")
    ckpt_dir = p / "checkpoints" if (p / "checkpoints").is_dir() else p
    name_map = {"latest": "checkpoint_latest.pt", "best": "checkpoint_best.pt"}
    ckpt = ckpt_dir / name_map[checkpoint_type]
    if ckpt.exists():
        return ckpt
    available = sorted(x.name for x in ckpt_dir.glob("checkpoint_*.pt"))
    raise FileNotFoundError(f"Not found: {ckpt}. Available: {available}")


def _infer_dino_hub_name(config) -> str:
    """Infer the DINO hub model name from checkpoint config or packed metadata.

    Defaults to DINOv2 ViT-L/14 with register tokens.
    """
    default = "dinov2_vitl14_reg"

    data_cfg = getattr(config, "data", None)
    packed_dir = getattr(data_cfg, "packed_features_dir", "") if data_cfg is not None else ""
    if packed_dir:
        meta_path = Path(str(packed_dir)) / "dino_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                hub_name = str(meta.get("hub_name", "")).strip()
                if hub_name:
                    return hub_name
                variant = str(meta.get("variant", "")).strip().lower()
                if variant == "small":
                    return "dinov2_vits14_reg"
                if variant == "base":
                    return "dinov2_vitb14_reg"
                if variant == "large":
                    return "dinov2_vitl14_reg"
                if variant == "giant":
                    return "dinov2_vitg14_reg"
            except Exception:
                pass

    third_cfg = getattr(config, "third_encoder", None)
    dino_dim = int(getattr(third_cfg, "dino_dim", 1024)) if third_cfg is not None else 1024
    dim_to_hub = {
        384: "dinov2_vits14_reg",
        768: "dinov2_vitb14_reg",
        1024: "dinov2_vitl14_reg",
        1536: "dinov2_vitg14_reg",
    }
    return dim_to_hub.get(dino_dim, default)


# ---------------------------------------------------------------------------
# GPU Inference Worker (CUDA subprocess)
# ---------------------------------------------------------------------------


def _gpu_inference_loop(
    checkpoint, progress_checkpoint, model_type, gpu_id, dino_input_size,
    req_q, resp_q,
):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    try:
        print(f"  [GPU{gpu_id}] CUDA worker: starting model load", flush=True)
        if model_type not in ("rain", "rainv2"):
            raise ValueError(f"Unsupported public model type: {model_type}")
        model, config = load_rainv2_model(checkpoint, progress_checkpoint, device)
        print(f"  [GPU{gpu_id}] CUDA worker: model ready on device", flush=True)

        dino_model_name = _infer_dino_hub_name(config)
        dino_num_scales = int(
            getattr(getattr(config, "dit", None), "num_scales", 1)
        )
        print(
            f"  [GPU{gpu_id}] CUDA worker: DINO input={dino_input_size} "
            f"model={dino_model_name} scales={dino_num_scales}",
            flush=True,
        )
        if dino_num_scales > 1:
            if dino_num_scales != 3:
                raise ValueError(
                    f"RAIN multi-scale eval requires exactly 3 scales, got {dino_num_scales}"
                )
            if dino_input_size != 224 or dino_model_name != "dinov2_vitl14_reg":
                raise ValueError(
                    "RAIN multi-scale eval requires DINOv2-L at 224px, got "
                    f"model={dino_model_name} input={dino_input_size}"
                )
            from rainv2.models.multiscale_vision import FrozenDINOv2LargeMultiScale
            dino = FrozenDINOv2LargeMultiScale(
                input_size=dino_input_size,
            ).to(device).eval()
        else:
            from shared.components import FrozenDINOv2
            dino = FrozenDINOv2(
                input_size=dino_input_size,
                dino_model=dino_model_name,
            ).to(device).eval()
        print(f"  [GPU{gpu_id}] CUDA worker: DINO loaded", flush=True)

        dummy = torch.randn(2, 3, dino_input_size, dino_input_size, device=device)
        with torch.no_grad():
            dino(dummy)
        torch.cuda.synchronize()
        del dummy
        print(f"  [GPU{gpu_id}] CUDA worker: warmup complete", flush=True)
    except Exception as e:
        import traceback
        resp_q.put(f"error: {e}\n{traceback.format_exc()}")
        return

    resp_q.put("ready")

    while True:
        request = req_q.get()
        if request is None:
            break
        if request.get("control") == "seed":
            import random

            inference_seed = int(request["seed"])
            random.seed(inference_seed)
            np.random.seed(inference_seed)
            torch.manual_seed(inference_seed)
            torch.cuda.manual_seed_all(inference_seed)
            resp_q.put({"seeded": inference_seed})
            continue

        imgs_t = request["imgs_third"]
        imgs_w = request["imgs_wrist"]
        num_steps = request.get("num_inference_steps", 4)
        B = len(imgs_t)

        batch_t = torch.stack([
            torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            for im in imgs_t
        ]).to(device)
        batch_w = torch.stack([
            torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            for im in imgs_w
        ]).to(device)
        batch_state = torch.from_numpy(
            np.asarray(request["states"], dtype=np.float32)).to(device)
        batch_text = torch.from_numpy(
            np.asarray(request["text_feat"], dtype=np.float32)).to(device)
        batch_action_type = request.get("action_type")
        if batch_action_type is not None:
            batch_action_type = torch.from_numpy(
                np.asarray(batch_action_type, dtype=np.int64)).to(device)
        batch_mask = torch.from_numpy(
            np.asarray(request["masks"], dtype=np.float32)).to(device)
        batch_wrist_mask = request.get("wrist_masks")
        if batch_wrist_mask is not None:
            batch_wrist_mask = torch.from_numpy(
                np.asarray(batch_wrist_mask, dtype=np.float32)).to(device)
        else:
            batch_wrist_mask = torch.zeros_like(batch_mask)
        batch_place_mask = request.get("target_place_mask")
        if batch_place_mask is not None:
            batch_place_mask = torch.from_numpy(
                np.asarray(batch_place_mask, dtype=np.float32)).to(device)
        batch_place_mask_wrist = request.get("target_place_mask_wrist")
        if batch_place_mask_wrist is not None:
            batch_place_mask_wrist = torch.from_numpy(
                np.asarray(batch_place_mask_wrist, dtype=np.float32)).to(device)

        sampling_generator = None
        if "action_sampling_seed" in request:
            action_sampling_seed = int(request["action_sampling_seed"])
            if not 0 <= action_sampling_seed < (1 << 63):
                raise ValueError("action_sampling_seed must be in [0, 2**63)")
            sampling_generator = torch.Generator(device=device)
            sampling_generator.manual_seed(action_sampling_seed)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            dino_both = dino(torch.cat([batch_t, batch_w], dim=0))
            if dino_num_scales > 1:
                if len(dino_both) != dino_num_scales:
                    raise RuntimeError(
                        f"DINO returned {len(dino_both)} scales, expected {dino_num_scales}"
                    )
                # Match the packed training contract: (B, scale, patch, dim).
                dino_t = torch.stack(
                    [features[:B] for features in dino_both], dim=1
                )
                dino_w = torch.stack(
                    [features[B:] for features in dino_both], dim=1
                )
            else:
                dino_t, dino_w = dino_both[:B], dino_both[B:]

            output = model.predict(
                dino_third=dino_t, dino_wrist=dino_w,
                goal_mask_third=batch_mask,
                goal_mask_wrist=batch_wrist_mask,
                target_place_mask=batch_place_mask,
                target_place_mask_wrist=batch_place_mask_wrist,
                text_feat=batch_text,
                action_type=batch_action_type,
                state=batch_state, num_steps=num_steps,
                sampling_generator=sampling_generator,
            )

        result = {"action": output["action"].float().cpu().numpy()}
        for key in ("plan", "task_comp_prob", "pred_distance", "pred_alignment"):
            if key in output and output[key] is not None:
                result[key] = output[key].float().cpu().numpy()

        resp_q.put(result)


class GPUInferenceWorker:
    """GPU inference in a subprocess. Must be created BEFORE any EGL imports."""

    def __init__(
        self, checkpoint, progress_checkpoint, model_type,
        gpu_id=0, dino_input_size=448, startup_timeout_s=300,
    ):
        mp_ctx = get_context("spawn")
        self._req_q = mp_ctx.Queue()
        self._resp_q = mp_ctx.Queue()
        self._proc = mp_ctx.Process(
            target=_gpu_inference_loop,
            args=(checkpoint, progress_checkpoint, model_type,
                  gpu_id, dino_input_size,
                  self._req_q, self._resp_q),
            daemon=True,
        )
        self._proc.start()
        msg = self._resp_q.get(timeout=startup_timeout_s)
        if msg != "ready":
            raise RuntimeError(f"GPU worker failed: {msg}")

    def infer(self, imgs_third, imgs_wrist, states, masks, text_feat,
              action_type=None, num_inference_steps=4,
              target_place_mask=None, target_place_mask_wrist=None,
              wrist_masks=None, action_sampling_seed=None):
        req = {
            "imgs_third": imgs_third,
            "imgs_wrist": imgs_wrist,
            "states": np.asarray(states, dtype=np.float32),
            "masks": np.asarray(masks, dtype=np.float32),
            "text_feat": np.asarray(text_feat, dtype=np.float32),
            "action_type": (
                None if action_type is None
                else np.asarray(action_type, dtype=np.int64)
            ),
            "num_inference_steps": num_inference_steps,
        }
        if action_sampling_seed is not None:
            if (
                isinstance(action_sampling_seed, (bool, np.bool_))
                or not isinstance(action_sampling_seed, (int, np.integer))
            ):
                raise TypeError("action_sampling_seed must be an integer")
            action_sampling_seed = int(action_sampling_seed)
            if not 0 <= action_sampling_seed < (1 << 63):
                raise ValueError("action_sampling_seed must be in [0, 2**63)")
            req["action_sampling_seed"] = action_sampling_seed
        if wrist_masks is not None:
            req["wrist_masks"] = np.asarray(wrist_masks, dtype=np.float32)
        if target_place_mask is not None:
            req["target_place_mask"] = np.asarray(target_place_mask, dtype=np.float32)
        if target_place_mask_wrist is not None:
            req["target_place_mask_wrist"] = np.asarray(
                target_place_mask_wrist, dtype=np.float32)
        self._req_q.put(req)
        return self._resp_q.get()

    def set_seed(self, seed: int) -> None:
        """Reset stochastic flow-matching noise at an episode boundary."""
        expected = int(seed)
        self._req_q.put({"control": "seed", "seed": expected})
        response = self._resp_q.get()
        if response != {"seeded": expected}:
            raise RuntimeError(f"GPU worker seed acknowledgement failed: {response}")

    def close(self):
        try:
            self._req_q.put(None)
            self._proc.join(timeout=10)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.terminate()


# ---------------------------------------------------------------------------
# EGL patching
# ---------------------------------------------------------------------------


def _disable_robosuite_file_logging():
    """Force robosuite to skip /tmp/robosuite.log file writes.

    Some shared machines leave an existing /tmp/robosuite.log owned by another
    user, which makes robosuite import fail before any evaluation starts.
    """
    module_name = "robosuite.macros"
    macros_mod = sys.modules.get(module_name)
    if macros_mod is None:
        for entry in sys.path:
            candidate = Path(entry) / "robosuite" / "macros.py"
            if not candidate.is_file():
                continue
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue
            macros_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(macros_mod)
            sys.modules[module_name] = macros_mod
            break
    if macros_mod is None:
        return

    setattr(macros_mod, "FILE_LOGGING_LEVEL", None)

    private_name = "robosuite.macros_private"
    if private_name not in sys.modules:
        private_mod = type(sys)(private_name)
        for name in dir(macros_mod):
            if not name.startswith("__"):
                setattr(private_mod, name, getattr(macros_mod, name))
        setattr(private_mod, "FILE_LOGGING_LEVEL", None)
        sys.modules[private_name] = private_mod


def _patch_robosuite_egl():
    """Patch EGL init to iterate over all GPU devices."""
    _disable_robosuite_file_logging()
    import robosuite.renderers.context.egl_context as egl_ctx
    from mujoco.egl import egl_ext as EGL
    from OpenGL import error

    def _patched_create_display(device_id=0):
        all_devices = EGL.eglQueryDevicesEXT()
        if not all_devices:
            return EGL.EGL_NO_DISPLAY
        pref = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        try:
            pref_idx = int(pref) if pref is not None else int(device_id)
        except Exception:
            pref_idx = int(device_id)
        ordered = []
        if 0 <= pref_idx < len(all_devices):
            ordered.append(all_devices[pref_idx])
        for i, dev in enumerate(all_devices):
            if i != pref_idx:
                ordered.append(dev)
        for device in ordered:
            display = EGL.eglGetPlatformDisplayEXT(
                EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
            if display != EGL.EGL_NO_DISPLAY and EGL.eglGetError() == EGL.EGL_SUCCESS:
                try:
                    initialized = EGL.eglInitialize(display, None, None)
                except error.GLError:
                    continue
                if initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS:
                    return display
        return EGL.EGL_NO_DISPLAY

    egl_ctx.create_initialized_egl_device_display = _patched_create_display
    import robosuite.utils.binding_utils as binding_utils  # noqa

    if getattr(binding_utils.MjRenderContext.read_pixels, "__name__", "") != "_patched":
        _orig_read_pixels = binding_utils.MjRenderContext.read_pixels

        def _patched(self, width, height, depth=False, segmentation=False):
            if not segmentation:
                return _orig_read_pixels(self, width, height, depth=depth, segmentation=segmentation)
            viewport = binding_utils.mujoco.MjrRect(0, 0, width, height)
            rgb_img = binding_utils.np.empty((height, width, 3), dtype=binding_utils.np.uint8)
            depth_img = binding_utils.np.empty((height, width), dtype=binding_utils.np.float32) if depth else None
            binding_utils.mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)
            rgb_i32 = rgb_img.astype(binding_utils.np.int32, copy=False)
            seg_img = rgb_i32[:, :, 0] + (rgb_i32[:, :, 1] << 8) + (rgb_i32[:, :, 2] << 16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = binding_utils.np.full((self.scn.ngeom + 1, 2), -1, dtype=binding_utils.np.int32)
            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    sid = int(geom.segid) + 1
                    if 0 <= sid < seg_ids.shape[0]:
                        seg_ids[sid, 0] = int(geom.objtype)
                        seg_ids[sid, 1] = int(geom.objid)
            ret_img = seg_ids[seg_img]
            return (ret_img, depth_img) if depth else ret_img

        binding_utils.MjRenderContext.read_pixels = _patched
        binding_utils.MjRenderContextOffscreen.read_pixels = _patched


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _render_views_once(env, h, w):
    img_agent = env.sim.render(
        camera_name="agentview", width=w, height=h
    )[::-1, ::-1].copy()
    img_wrist = env.sim.render(
        camera_name="robot0_eye_in_hand", width=w, height=h
    )[::-1, ::-1].copy()
    return np.ascontiguousarray(img_agent), np.ascontiguousarray(img_wrist)




def _set_camera_observables(env, enabled):
    base_env = getattr(env, "env", None)
    observables = getattr(base_env, "_observables", None)
    if observables is None:
        return
    for key, obs_obj in observables.items():
        if any(x in key for x in ("image", "depth", "segmentation")):
            obs_obj.set_enabled(enabled)
            obs_obj.set_active(enabled)


def save_video(frames, path, fps=15):
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                output_params=["-crf", "23"])
    for frame in frames:
        writer.append_data(frame)
    writer.close()


# ---------------------------------------------------------------------------
# Mask-sequence helpers
# ---------------------------------------------------------------------------


def _decode_rle_mask(rle_dict):
    """Decode RLE mask dict to (H, W) uint8 binary mask."""
    counts = rle_dict["counts"]
    h, w = rle_dict["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + c] = 1
        pos += c
    return flat.reshape((h, w), order="F")


def _bbox_to_mask(bbox, h=256, w=256):
    """Convert bounding box [x0, y0, x1, y1] to binary mask."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def downsample_mask_to_patches(mask, grid=16):
    """(H,W) binary mask -> (grid*grid,) float patches."""
    import torch
    import torch.nn.functional as F
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


def is_object_segmentable(episode_data, object_id):
    """Check objects[oid].segmentable (default True)."""
    objects = episode_data.get("objects", {})
    obj = objects.get(str(object_id))
    if obj is None:
        return False
    if "segmentable" not in obj:
        return True
    return bool(obj.get("segmentable"))


def get_object_body_ids(episode_data, object_id):
    """Get objects[oid].body_ids list."""
    objects = episode_data.get("objects", {})
    obj = objects.get(str(object_id))
    if obj is None:
        return []
    return list(obj.get("body_ids") or [])


def get_object_geom_ids(episode_data, object_id, action_type=None):
    """Get explicit geom_ids override for segmentation.

    Default behavior for drawer-open goal tasks is to use drawer body masks
    (whole drawer), not handle-only geoms. To use handle-only masks,
    set env `RAIN_DRAWER_MASK_SCOPE=handle`.
    """
    objects = episode_data.get("objects", {})
    obj = objects.get(str(object_id))
    if obj is None:
        return []
    geom_ids = list(obj.get("geom_ids") or [])
    if geom_ids:
        return geom_ids
    action = str(action_type or "").strip().lower()
    task_desc = str(episode_data.get("task_description", "")).strip().lower()
    obj_name = str(obj.get("name", "")).strip().lower()
    drawer_scope = str(
        os.environ.get("RAIN_DRAWER_MASK_SCOPE", "whole")
    ).strip().lower()
    use_handle_only = drawer_scope in {"handle", "handle_only", "legacy"}
    if (
        action == "open"
        and
        task_desc == "open the middle drawer of the cabinet"
        and obj_name == "wooden_cabinet_1_cabinet_middle"
    ):
        if use_handle_only:
            return list(MIDDLE_DRAWER_HANDLE_GEOM_IDS)
        return []
    if (
        action == "open"
        and
        task_desc == "open the top drawer and put the bowl inside"
        and obj_name == "wooden_cabinet_1_cabinet_top"
    ):
        if use_handle_only:
            return list(TOP_DRAWER_HANDLE_GEOM_IDS)
        return []
    return []


@lru_cache(maxsize=256)
def _parse_bddl_region_ranges(bddl_path: str, region_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse BDDL region ranges as (x_min, y_min, x_max, y_max)."""
    if not bddl_path:
        return None
    path = Path(str(bddl_path))
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except Exception:
        return None
    short_name = str(region_name or "").strip()
    if not short_name:
        return None
    for prefix in ("main_table_", "kitchen_table_", "living_room_table_", "study_table_"):
        if short_name.startswith(prefix):
            short_name = short_name[len(prefix):]
            break
    pattern = rf"{re.escape(short_name)}.*?:ranges\s*\(\s*\(([\d\s\.\-eE]+)\)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        vals = [float(x) for x in match.group(1).split()]
    except Exception:
        return None
    if len(vals) < 4:
        return None
    return (vals[0], vals[1], vals[2], vals[3])


def _world_to_image(sim, cam_name: str, world_pos, img_h: int, img_w: int) -> Optional[Tuple[int, int]]:
    """Project world XYZ to image pixel using MuJoCo camera pose/FOV."""
    try:
        cam_id = sim.model.camera_name2id(cam_name)
    except Exception:
        return None
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
    cam_mat = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    rel = np.asarray(world_pos, dtype=np.float64) - cam_pos
    cam_frame = cam_mat.T @ rel
    depth = -cam_frame[2]  # MuJoCo camera looks along -Z
    if depth <= 1e-8:
        return None
    fovy = np.deg2rad(float(sim.model.cam_fovy[cam_id]))
    f = float(img_h) / (2.0 * np.tan(fovy / 2.0))
    u = f * cam_frame[0] / depth + img_w / 2.0
    v = f * cam_frame[1] / depth + img_h / 2.0
    # Match MuJoCo render orientation used elsewhere in this evaluator.
    u = img_w - 1 - u
    v = img_h - 1 - v
    return (
        int(np.clip(u, 0, img_w - 1)),
        int(np.clip(v, 0, img_h - 1)),
    )


def _estimate_table_surface_z(sim, episode_data: Dict) -> float:
    """Heuristic table surface Z for BDDL region projection."""
    z_vals: List[float] = []
    objects = (episode_data or {}).get("objects", {}) or {}
    for obj in objects.values():
        if not bool(obj.get("segmentable", True)):
            continue
        for bid in (obj.get("body_ids") or []):
            try:
                bid_i = int(bid)
            except Exception:
                continue
            if 0 <= bid_i < sim.model.nbody:
                z_vals.append(float(sim.data.body_xpos[bid_i][2]))
                break
    if z_vals:
        return float(min(z_vals))
    for bname in ("main_table", "table", "kitchen_table", "living_room_table", "study_table"):
        try:
            bid = sim.model.body_name2id(bname)
            return float(sim.data.body_xpos[int(bid)][2])
        except Exception:
            continue
    return 0.9


def _region_to_bbox_mask(
    sim,
    cam_name: str,
    ranges: Tuple[float, float, float, float],
    table_z: float,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Project BDDL table-region ranges to camera and rasterize bbox mask."""
    table_bid = None
    for bid in range(sim.model.nbody):
        bname = sim.model.body_id2name(bid) or ""
        if bname in ("table", "main_table", "kitchen_table", "living_room_table", "study_table"):
            table_bid = bid
            break
    table_pos = (
        np.asarray(sim.data.body_xpos[table_bid], dtype=np.float64)
        if table_bid is not None
        else np.zeros(3, dtype=np.float64)
    )
    x0r, y0r, x1r, y1r = [float(x) for x in ranges[:4]]
    corners = [
        np.array([table_pos[0] + x0r, table_pos[1] + y0r, table_z], dtype=np.float64),
        np.array([table_pos[0] + x1r, table_pos[1] + y0r, table_z], dtype=np.float64),
        np.array([table_pos[0] + x0r, table_pos[1] + y1r, table_z], dtype=np.float64),
        np.array([table_pos[0] + x1r, table_pos[1] + y1r, table_z], dtype=np.float64),
    ]
    uvs = []
    for c in corners:
        uv = _world_to_image(sim, cam_name, c, img_h=img_h, img_w=img_w)
        if uv is not None:
            uvs.append(uv)
    if len(uvs) < 2:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    us = [u for u, _ in uvs]
    vs = [v for _, v in uvs]
    x0, x1 = int(min(us)), int(max(us))
    y0, y1 = int(min(vs)), int(max(vs))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def sim_region_mask_for_object_id(
    env,
    episode_data,
    object_id,
    bddl_path: str = "",
    image_size: int = LIBERO_ENV_RESOLUTION,
    camera_name: str = "agentview",
) -> Optional[np.ndarray]:
    """Render non-segmentable table-region mask via BDDL projection."""
    if env is None:
        return None
    objects = (episode_data or {}).get("objects", {}) or {}
    obj = objects.get(str(object_id)) or {}
    if not obj:
        return None
    if bool(obj.get("segmentable", True)):
        return None
    region_name = str(obj.get("name", "")).strip()
    if "region" not in region_name:
        return None
    # Guard region patch scope to known region tasks by default.
    allow_raw = os.environ.get("RAIN_REGION_SIM_MASK_TARGETS")
    if allow_raw is None or not str(allow_raw).strip():
        allow_list = sorted(DEFAULT_REGION_SIM_MASK_TARGETS)
    else:
        allow_l = str(allow_raw).strip().lower()
        if allow_l in {"off", "none", "0", "false"}:
            allow_list = []
        else:
            allow_list = [x.strip() for x in allow_l.split(",") if x.strip()]
    region_l = region_name.lower()
    if (not allow_list) or ("*" not in allow_list and region_l not in allow_list):
        return None
    ranges = _parse_bddl_region_ranges(str(bddl_path or ""), region_name)
    if ranges is None:
        return None
    try:
        table_z = _estimate_table_surface_z(env.sim, episode_data)
        mask = _region_to_bbox_mask(
            env.sim,
            cam_name=camera_name,
            ranges=ranges,
            table_z=table_z,
            img_h=image_size,
            img_w=image_size,
        )
    except Exception:
        return None
    if int(mask.sum()) <= 0:
        return None
    return np.ascontiguousarray(mask.astype(np.uint8))


def _get_task_specific_geom_ids(
    env,
    episode_data,
    object_id,
    action_type=None,
):
    """Task-specific geom override to match training-mask semantics."""
    action = str(action_type or "").strip().lower()
    if action != "release":
        return []

    task_desc = str(episode_data.get("task_description", "")).strip().lower()
    if task_desc not in CABINET_TOP_RELEASE_TASKS:
        return []

    obj = (episode_data.get("objects") or {}).get(str(object_id)) or {}
    obj_name = str(obj.get("name", "")).strip().lower()
    if obj_name != "wooden_cabinet_1_main":
        return []

    # Training masks for surface_z on cabinet-main targets use the first
    # geom on wooden_cabinet_1_base.
    try:
        base_bid = env.sim.model.body_name2id("wooden_cabinet_1_base")
        geom_body = np.asarray(env.sim.model.geom_bodyid, dtype=np.int32)
    except Exception:
        return []
    base_geoms = np.flatnonzero(geom_body == int(base_bid))
    if len(base_geoms) == 0:
        return []
    return [int(base_geoms[0])]


def verify_and_correct_body_ids(env, episode_data):
    """Validate JSON body_ids against sim body names.

    Returns dict[oid_str, list[int]] with corrected mappings.
    """
    objects = episode_data.get("objects", {})
    corrections = {}
    sim_name2id = {}
    for i in range(env.sim.model.nbody):
        sim_name2id[env.sim.model.body_id2name(i)] = i

    for oid, obj in objects.items():
        obj_name = obj.get("name", "")
        json_bids = obj.get("body_ids")
        segmentable = obj.get("segmentable", True)
        if not segmentable or json_bids is None:
            continue

        verified = True
        for bid in json_bids:
            if bid >= env.sim.model.nbody:
                verified = False
                break
            norm_obj = obj_name.lower().replace(" ", "_")
            norm_sim = env.sim.model.body_id2name(bid).lower()
            if norm_obj not in norm_sim and norm_sim not in norm_obj:
                verified = False
                break

        if verified:
            corrections[oid] = json_bids
            continue

        norm_obj = obj_name.lower().replace(" ", "_")
        found = None
        for bname, bid in sim_name2id.items():
            if norm_obj in bname.lower():
                found = [bid]
                break

        if found is not None:
            old_names = [env.sim.model.body_id2name(b)
                         for b in json_bids if b < env.sim.model.nbody]
            new_name = env.sim.model.body_id2name(found[0])
            print(f"  FIXED obj={obj_name} (id={oid}): body_id "
                  f"{json_bids}({','.join(old_names)}) -> {found}({new_name})",
                  flush=True)
            corrections[oid] = found
        else:
            corrections[oid] = json_bids

    return corrections


def sim_mask_for_object_id(
    env,
    episode_data,
    object_id,
    image_size=LIBERO_ENV_RESOLUTION,
    corrected_bids=None,
    camera_name="agentview",
    action_type=None,
):
    """Render sim segmentation mask for one object. Returns (H,W) uint8 or None."""
    if env is None or not is_object_segmentable(episode_data, object_id):
        return None
    try:
        seg = env.sim.render(
            camera_name=camera_name, width=image_size, height=image_size,
            segmentation=True,
        )
    except Exception:
        return None
    seg_np = np.asarray(seg)
    if seg_np.ndim != 3 or seg_np.shape[-1] < 2:
        return None

    # Prefer explicit geom_ids (e.g. drawer handle) over body_ids
    explicit_geom_ids = _get_task_specific_geom_ids(
        env, episode_data, object_id, action_type=action_type
    )
    if not explicit_geom_ids:
        explicit_geom_ids = get_object_geom_ids(
            episode_data, object_id, action_type=action_type
        )
    if explicit_geom_ids:
        geom_ids = np.asarray(explicit_geom_ids, dtype=np.int32)
    else:
        if corrected_bids and str(object_id) in corrected_bids:
            body_ids = corrected_bids[str(object_id)]
        else:
            body_ids = get_object_body_ids(episode_data, object_id)
        if not body_ids:
            return None
        try:
            geom_body = np.asarray(env.sim.model.geom_bodyid, dtype=np.int32)
        except Exception:
            return None
        body_ids_arr = np.asarray(body_ids, dtype=np.int32)
        geom_ids = np.flatnonzero(np.isin(geom_body, body_ids_arr))
        if len(geom_ids) == 0:
            # Some root bodies (e.g. cabinet main) can have zero direct geoms.
            # Fall back to descendant bodies so segmentation can still be rendered.
            try:
                parents = np.asarray(env.sim.model.body_parentid, dtype=np.int32)
                body_set = set(int(b) for b in body_ids_arr.tolist())
                changed = True
                while changed:
                    changed = False
                    for cid, pid in enumerate(parents):
                        if int(pid) in body_set and cid not in body_set:
                            body_set.add(cid)
                            changed = True
                geom_ids = np.flatnonzero(np.isin(
                    geom_body, np.asarray(sorted(body_set), dtype=np.int32)))
            except Exception:
                pass
    if len(geom_ids) == 0:
        return None
    obj_type, obj_id = seg_np[..., 0], seg_np[..., 1]
    if np.any(obj_type == 5):
        mask = np.logical_and(obj_type == 5, np.isin(obj_id, geom_ids)).astype(np.uint8)
    else:
        mask = np.isin(obj_id, geom_ids).astype(np.uint8)
    mask = np.ascontiguousarray(mask[::-1, ::-1])
    return mask if int(mask.sum()) > 0 else None


def _make_prompt_mask(obj_data, prompt_type="visible", view="agent"):
    """Build binary mask from object annotation data."""
    if prompt_type == "bbox":
        bbox = obj_data.get(f"bbox_{view}")
        if bbox and len(bbox) == 4:
            return _bbox_to_mask(bbox)
        return None
    key = f"mask_{prompt_type}_{view}"
    rle = obj_data.get(key)
    if rle is None:
        return None
    try:
        return _decode_rle_mask(rle)
    except Exception:
        return None


def load_subtask_mask(
    episode_data,
    seg,
    prompt_type="visible",
    mask_object_id=None,
    patch_grid=16,
    view="agent",
    prefer_nonzero=False,
):
    """Load mask from episode JSON for non-segmentable objects."""
    obj_id = str(mask_object_id) if mask_object_id else str(seg["primary_object_id"])
    start_frame = int(seg["start_frame"])
    end_frame = int(seg.get("end_frame", start_frame))
    first_found = None

    mode = str(
        os.environ.get("RAIN_MASK_PICK_MODE", "first_nonzero")
    ).strip().lower()
    use_max_nonzero = mode in {"max_nonzero", "largest_nonzero"}

    # Optional mode: choose largest non-zero mask across the segment window.
    best_nonzero = None
    best_nonzero_area = -1

    def _visit(frame_idx):
        nonlocal first_found, best_nonzero, best_nonzero_area
        frame_data = episode_data.get("frames", {}).get(str(frame_idx), {})
        obj_data = frame_data.get("objects", {}).get(obj_id, {})
        mask_raw = _make_prompt_mask(obj_data, prompt_type, view=view)
        if mask_raw is None:
            return
        patches = downsample_mask_to_patches(mask_raw, grid=patch_grid)
        area = int(mask_raw.sum())
        if first_found is None:
            first_found = (mask_raw, patches)
        if use_max_nonzero and area > 0 and area > best_nonzero_area:
            best_nonzero_area = area
            best_nonzero = (mask_raw, patches)
        if not prefer_nonzero:
            # The default path returns the first available mask.
            raise StopIteration

    seed_frames = [start_frame, start_frame + 1, start_frame - 1]
    visited = set()
    try:
        for frame_idx in seed_frames:
            if frame_idx in visited:
                continue
            visited.add(frame_idx)
            _visit(frame_idx)
    except StopIteration:
        return first_found

    if prefer_nonzero and end_frame >= start_frame:
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in visited:
                continue
            _visit(frame_idx)

    if use_max_nonzero and best_nonzero is not None:
        return best_nonzero
    if first_found is not None:
        return first_found
    return None, np.zeros(patch_grid * patch_grid, dtype=np.float32)


def _find_episode(episodes, task_desc, benchmark_name):
    """Match LIBERO task description to episode JSON entry."""
    want_desc = str(task_desc or "").strip().lower()
    cands = []
    for ek, ep in episodes.items():
        ep_desc = str(ep.get("task_description", "")).strip().lower()
        if ep_desc == want_desc:
            cands.append((ek, ep))
    if not cands:
        raise KeyError(f"No episode for: {task_desc}")

    # Prefer benchmark-tagged candidates when hdf5 paths are available.
    # Some merged JSONs (e.g. final_full.json) do not carry hdf5_path.
    tag = f"/{benchmark_name}/"
    tagged = []
    for ek, ep in cands:
        hdf5_path = str(ep.get("hdf5_path", ""))
        if tag in hdf5_path:
            tagged.append((ek, ep))
    if tagged:
        cands = tagged

    cands.sort(key=lambda x: (int(x[1].get("episode_index", 10**9)), str(x[0])))
    return cands[0]


# ---------------------------------------------------------------------------
# SubtaskCondition
# ---------------------------------------------------------------------------


@dataclass
class SubtaskCondition:
    subtask_id: int
    action_type: str
    action_type_id: int
    object_id: str
    object_name: str
    mask_raw: Optional[np.ndarray]      # (H,W) uint8
    mask_patches: np.ndarray            # (num_patches,) float32
    wrist_mask_raw: Optional[np.ndarray]
    wrist_mask_patches: np.ndarray
    mask_source: str                    # "sim_seg_deferred", "json", etc.
    target_place_patches: Optional[np.ndarray] = None  # (num_patches,) float32
    target_place_wrist_patches: Optional[np.ndarray] = None  # (num_patches,) float32


def _resolve_object_ref_to_known_id(
    episode: Dict,
    object_ref: str,
) -> str:
    """Resolve target/object alias text to an existing episode object id.

    Some labels use placement aliases like `*_plate` while episode objects
    only include the parent body name (e.g. `flat_stove_1_burner`).
    """
    objects = episode.get("objects", {}) or {}
    ref = str(object_ref or "").strip()
    if not ref:
        return ""
    if ref in objects:
        return ref

    ref_l = ref.lower()
    by_name: List[Tuple[str, str]] = []
    for oid, obj in objects.items():
        name = str(obj.get("name", "")).strip().lower()
        body = str(obj.get("body_name", "")).strip().lower()
        if name:
            by_name.append((oid, name))
        if body and body != name:
            by_name.append((oid, body))

    for oid, cand in by_name:
        if cand == ref_l:
            return oid

    for suffix in ("_plate", "_surface", "_area", "_region", "_slot", "_top"):
        if ref_l.endswith(suffix):
            base = ref_l[: -len(suffix)]
            for oid, cand in by_name:
                if cand == base:
                    return oid

    best_oid = ""
    best_len = -1
    for oid, cand in by_name:
        if ref_l.startswith(cand) or cand.startswith(ref_l):
            if len(cand) > best_len:
                best_len = len(cand)
                best_oid = oid
    return best_oid


def build_subtask_conditions(episode, dino_input_size=448,
                             prompt_type="visible"):
    """Build list of SubtaskCondition from episode JSON subtask_segments."""
    segs = sorted(episode.get("subtask_segments", []),
                  key=lambda x: int(x.get("subtask_id", 0)))
    objects_meta = episode.get("objects", {})
    patch_grid = dino_input_size // 14
    num_patches = patch_grid * patch_grid
    conditions = []

    for seg in segs:
        sid = int(seg.get("subtask_id", len(conditions)))
        atype = str(seg.get("action_type", "grasp"))
        obj_id = str(seg.get("primary_object_id", ""))
        if atype in ("release", "push"):
            target_ref = str(seg.get("target_object_id", ""))
            resolved_target_obj_id = _resolve_object_ref_to_known_id(
                episode, target_ref
            )
            if resolved_target_obj_id:
                obj_id = resolved_target_obj_id
        segmentable = is_object_segmentable(episode, obj_id)

        mask_raw = None
        mask_patches = np.zeros(num_patches, dtype=np.float32)
        wrist_mask_raw = None
        wrist_mask_patches = np.zeros(num_patches, dtype=np.float32)

        if segmentable:
            mask_source = "sim_seg_deferred"
            # Precompute JSON fallback
            mr, mp = load_subtask_mask(episode, seg, prompt_type, obj_id,
                                       patch_grid=patch_grid, view="agent")
            wr, wp = load_subtask_mask(episode, seg, prompt_type, obj_id,
                                       patch_grid=patch_grid, view="wrist",
                                       prefer_nonzero=True)
            if mr is None:
                mr, mp = load_subtask_mask(episode, seg, "bbox", obj_id,
                                           patch_grid=patch_grid, view="agent")
                if mr is not None:
                    mask_source = "sim_seg_deferred+bbox_fallback"
            if wr is None:
                wr, wp = load_subtask_mask(episode, seg, "bbox", obj_id,
                                           patch_grid=patch_grid, view="wrist",
                                           prefer_nonzero=True)
            if mr is not None:
                mask_raw = mr.astype(np.uint8)
                mask_patches = mp.astype(np.float32)
                if mask_source == "sim_seg_deferred":
                    mask_source = "sim_seg_deferred+json_fallback"
            if wr is not None:
                wrist_mask_raw = wr.astype(np.uint8)
                wrist_mask_patches = wp.astype(np.float32)
        else:
            mr, mp = load_subtask_mask(episode, seg, prompt_type, obj_id,
                                       patch_grid=patch_grid, view="agent")
            wr, wp = load_subtask_mask(episode, seg, prompt_type, obj_id,
                                       patch_grid=patch_grid, view="wrist",
                                       prefer_nonzero=True)
            if mr is not None:
                mask_raw = mr.astype(np.uint8)
                mask_patches = mp.astype(np.float32)
                mask_source = "json"
            else:
                mr, mp = load_subtask_mask(episode, seg, "bbox", obj_id,
                                           patch_grid=patch_grid, view="agent")
                if mr is not None:
                    mask_raw = mr.astype(np.uint8)
                    mask_patches = mp.astype(np.float32)
                    mask_source = "json_bbox"
                else:
                    mask_source = "missing"
            if wr is None:
                wr, wp = load_subtask_mask(episode, seg, "bbox", obj_id,
                                           patch_grid=patch_grid, view="wrist",
                                           prefer_nonzero=True)
            if wr is not None:
                wrist_mask_raw = wr.astype(np.uint8)
                wrist_mask_patches = wp.astype(np.float32)

        # Precompute target place mask for grasp/push subtasks
        target_place_patches = np.zeros(num_patches, dtype=np.float32)
        target_place_wrist_patches = np.zeros(num_patches, dtype=np.float32)
        si = segs.index(seg)
        if atype in ("grasp", "push") and si + 1 < len(segs):
            next_seg = segs[si + 1]
            next_obj_id = str(
                next_seg.get("target_object_id")
                or next_seg.get("primary_object_id", "")
            )
            pmr, pmp = load_subtask_mask(episode, next_seg, prompt_type,
                                         next_obj_id, patch_grid=patch_grid, view="agent")
            if pmr is None:
                pmr, pmp = load_subtask_mask(episode, next_seg, "bbox",
                                             next_obj_id, patch_grid=patch_grid, view="agent")
            if pmp is not None:
                target_place_patches = pmp.astype(np.float32)
            pwmr, pwmp = load_subtask_mask(episode, next_seg, prompt_type,
                                           next_obj_id, patch_grid=patch_grid, view="wrist",
                                           prefer_nonzero=True)
            if pwmr is None:
                pwmr, pwmp = load_subtask_mask(episode, next_seg, "bbox",
                                               next_obj_id, patch_grid=patch_grid, view="wrist",
                                               prefer_nonzero=True)
            if pwmp is not None:
                target_place_wrist_patches = pwmp.astype(np.float32)

        obj_name = objects_meta.get(obj_id, {}).get("name", f"obj_{obj_id}")
        conditions.append(SubtaskCondition(
            subtask_id=sid, action_type=atype,
            action_type_id=action_type_to_id(atype),
            object_id=obj_id, object_name=obj_name,
            mask_raw=mask_raw, mask_patches=mask_patches,
            wrist_mask_raw=wrist_mask_raw, wrist_mask_patches=wrist_mask_patches,
            mask_source=mask_source,
            target_place_patches=target_place_patches,
            target_place_wrist_patches=target_place_wrist_patches,
        ))
    return conditions












# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------


def _overlay_mask_on_view(view_bgr, mask_raw):
    """Draw a green overlay + contour for a binary mask on one view."""
    import cv2

    if mask_raw is None or not mask_raw.any():
        return view_bgr

    H, W = view_bgr.shape[:2]
    overlay = view_bgr.copy()
    m = mask_raw
    if m.shape[:2] != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST)
    overlay[m > 0] = (50, 255, 50)
    cv2.addWeighted(overlay, 0.3, view_bgr, 0.7, 0, view_bgr)
    contours, _ = cv2.findContours(m.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view_bgr, contours, -1, (50, 255, 50), 2)
    return view_bgr


def _overlay_place_mask_on_view(view_bgr, mask_raw):
    """Draw a yellow overlay + contour for a target_place_mask."""
    import cv2

    if mask_raw is None or not mask_raw.any():
        return view_bgr

    H, W = view_bgr.shape[:2]
    overlay = view_bgr.copy()
    m = mask_raw
    if m.shape[:2] != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST)
    overlay[m > 0] = (50, 230, 255)  # yellow (BGR)
    cv2.addWeighted(overlay, 0.3, view_bgr, 0.7, 0, view_bgr)
    contours, _ = cv2.findContours(m.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view_bgr, contours, -1, (50, 230, 255), 2)
    return view_bgr




def compose_vis_frame(agent_rgb, wrist_rgb, mask_raw, info_lines,
                      tc_val=0.0, success=False, wrist_mask_raw=None,
                      place_mask_raw=None, place_mask_wrist_raw=None,
                      reference_mask_raw=None, reference_wrist_mask_raw=None,
                      reference_place_mask_raw=None,
                      reference_place_wrist_mask_raw=None):
    """Render inference video or a 2(view)x2(prediction/GT) mask comparison."""
    import cv2
    H, W = agent_rgb.shape[:2]
    agent_base = cv2.cvtColor(agent_rgb, cv2.COLOR_RGB2BGR)
    wrist_base = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    agent_bgr = agent_base.copy()
    wrist_bgr = wrist_base.copy()

    agent_bgr = _overlay_mask_on_view(agent_bgr, mask_raw)
    agent_bgr = _overlay_place_mask_on_view(agent_bgr, place_mask_raw)
    wrist_bgr = _overlay_mask_on_view(wrist_bgr, wrist_mask_raw)
    wrist_bgr = _overlay_place_mask_on_view(wrist_bgr, place_mask_wrist_raw)
    has_reference = any(
        value is not None
        for value in (
            reference_mask_raw,
            reference_wrist_mask_raw,
            reference_place_mask_raw,
            reference_place_wrist_mask_raw,
        )
    )
    cv2.putText(agent_bgr, "agent / prediction", (4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(wrist_bgr, "wrist / prediction", (4, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    if has_reference:
        # Reference masks enter only this post-prediction visualization branch.
        # Separate prediction and GT panels expose overfilled masks. Green marks
        # the active interaction region; yellow marks the target-place region.
        agent_gt = _overlay_mask_on_view(
            agent_base.copy(), reference_mask_raw
        )
        agent_gt = _overlay_place_mask_on_view(
            agent_gt, reference_place_mask_raw
        )
        wrist_gt = _overlay_mask_on_view(
            wrist_base.copy(), reference_wrist_mask_raw
        )
        wrist_gt = _overlay_place_mask_on_view(
            wrist_gt, reference_place_wrist_mask_raw
        )
        cv2.putText(agent_gt, "agent / GT (reference only)", (4, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(wrist_gt, "wrist / GT (reference only)", (4, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                    cv2.LINE_AA)
        views = np.vstack([
            np.hstack([agent_bgr, agent_gt]),
            np.hstack([wrist_bgr, wrist_gt]),
        ])
    else:
        views = np.hstack([agent_bgr, wrist_bgr])
    info_h = 48
    bg = (0, 100, 0) if success else (40, 40, 40)
    bar = np.full((info_h, W * 2, 3), bg, dtype=np.uint8)
    for i, line in enumerate(info_lines[:3]):
        cv2.putText(bar, str(line)[:100], (4, 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)

    # TC progress bar
    bar_x = W * 2 - 120
    cv2.putText(bar, f"TC:{tc_val:.2f}", (bar_x, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)
    bar_w = 100
    cv2.rectangle(bar, (bar_x, 20), (bar_x + bar_w, 30), (80, 80, 80), -1)
    fill = int(bar_w * min(tc_val, 1.0))
    color = (0, 200, 0) if tc_val > 0.7 else (0, 200, 200)
    cv2.rectangle(bar, (bar_x, 20), (bar_x + fill, 30), color, -1)

    if success:
        cv2.putText(bar, "SUCCESS", (W * 2 - 80, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    combined = np.vstack([views, bar])
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# GT-mask rollout with subtask switching
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Single-episode mask-sequence rollout
# ---------------------------------------------------------------------------


GRIPPER_ACTIONS = {
    "grasp":   np.array([0, 0, 0, 0, 0, 0, +1], dtype=np.float32),
    "release": np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
    "turn_on": np.array([0, 0, 0, 0, 0, 0, +1], dtype=np.float32),
    "close":   np.array([0, 0, 0, 0, 0, 0, +1], dtype=np.float32),
    "open":    np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
    "push":    np.array([0, 0, 0, 0, 0, 0, +1], dtype=np.float32),
}
PERCEPTION_LIFT_ACTION = np.array(
    [0, 0, 0.5, 0, 0, 0, -1], dtype=np.float32
)
PERCEPTION_GRASP_RECOVERY_ACTION = np.array(
    [0, 0, 0.5, 0, 0, 0, +1], dtype=np.float32
)
GRASP_RGB_VERIFY_LIFT_STEPS = 4
GRASP_RGB_VERIFY_REOPEN_STEPS = 4


def run_single_episode_mask_sequence(
    gpu_worker, env, init_state, text_feat, episode_data, conditions,
    replan_steps=8, num_inference_steps=4, max_steps=520,
    feas_threshold=0.7, consecutive_stop=2,
    record_video=True, corrected_bids=None,
    dino_input_size=448, state_dim=8, bddl_path="",
):
    """Single-episode rollout with subtask mask switching and 3 policies.

    Policy 1: Prevent cascading switches (block switch if just_switched)
    Policy 2: Double-check previous subtask (one-time after switch, revert if TC < threshold)
    Policy 3: Gripper enforcement on switch (close for grasp, open for release)
    """
    H = W = LIBERO_ENV_RESOLUTION
    patch_grid = dino_input_size // 14
    num_patches = patch_grid * patch_grid

    # Reset env
    obs = env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    total_steps = 0
    sub_idx = 0
    consecutive_complete = 0
    success = False
    frames = [] if record_video else None

    # Policy 1: prevent cascading switches
    just_switched = False
    # Policy 2: double-check previous subtask after switch
    prev_check_info = None  # (object_id, action_type_id, action_type)

    turn_on_sim_only = str(
        os.environ.get("RAIN_TURN_ON_SIM_ONLY", "1")
    ).strip().lower() not in {"0", "false", "off", "no"}

    while total_steps < max_steps:
        cond = conditions[min(sub_idx, len(conditions) - 1)]

        # --- Get mask: sim seg for segmentable, stored for non-segmentable ---
        active_mask_raw = cond.mask_raw
        active_mask_patches = cond.mask_patches
        active_wrist_mask_raw = cond.wrist_mask_raw
        active_wrist_mask_patches = cond.wrist_mask_patches
        active_mask_source = cond.mask_source

        if is_object_segmentable(episode_data, cond.object_id):
            is_turn_on = str(cond.action_type).strip().lower() == "turn_on"
            sm = sim_mask_for_object_id(env, episode_data, cond.object_id,
                                        corrected_bids=corrected_bids,
                                        camera_name="agentview",
                                        action_type=cond.action_type)
            if sm is not None:
                active_mask_raw = sm
                active_mask_patches = downsample_mask_to_patches(
                    sm, grid=patch_grid).astype(np.float32)
                active_mask_source = "sim_seg"
            else:
                if is_turn_on and turn_on_sim_only:
                    active_mask_raw = None
                    active_mask_patches = np.zeros(num_patches, dtype=np.float32)
                    active_mask_source = "sim_missing_turn_on"
                else:
                    if (cond.mask_patches is not None
                            and float(cond.mask_patches.sum()) > 0.0):
                        active_mask_raw = cond.mask_raw
                        active_mask_patches = cond.mask_patches.astype(np.float32)
                        active_mask_source = "json_fallback"
                    else:
                        active_mask_raw = None
                        active_mask_patches = np.zeros(num_patches, dtype=np.float32)
                        active_mask_source = "sim_missing"
            wrist_sm = sim_mask_for_object_id(
                env, episode_data, cond.object_id,
                corrected_bids=corrected_bids,
                camera_name="robot0_eye_in_hand",
                action_type=cond.action_type,
            )
            if wrist_sm is not None:
                active_wrist_mask_raw = wrist_sm
                active_wrist_mask_patches = downsample_mask_to_patches(
                    wrist_sm, grid=patch_grid).astype(np.float32)
            else:
                if is_turn_on and turn_on_sim_only:
                    active_wrist_mask_raw = None
                    active_wrist_mask_patches = np.zeros(num_patches, dtype=np.float32)
                elif cond.wrist_mask_patches is not None and float(cond.wrist_mask_patches.sum()) > 0.0:
                    active_wrist_mask_raw = cond.wrist_mask_raw
                    active_wrist_mask_patches = cond.wrist_mask_patches.astype(np.float32)
                else:
                    active_wrist_mask_raw = None
                    active_wrist_mask_patches = np.zeros(num_patches, dtype=np.float32)
        else:
            # Non-segmentable region objects:
            # keep third-view mask path as-is (JSON) unless empty, and
            # refresh wrist-view mask from runtime BDDL projection.
            region_agent = sim_region_mask_for_object_id(
                env, episode_data, cond.object_id,
                bddl_path=bddl_path, image_size=H, camera_name="agentview",
            )
            if (
                region_agent is not None
                and float(active_mask_patches.sum()) <= 0.0
            ):
                active_mask_raw = region_agent
                active_mask_patches = downsample_mask_to_patches(
                    region_agent, grid=patch_grid).astype(np.float32)
                active_mask_source = "sim_region"
            region_wrist = sim_region_mask_for_object_id(
                env, episode_data, cond.object_id,
                bddl_path=bddl_path, image_size=H, camera_name="robot0_eye_in_hand",
            )
            if region_wrist is not None:
                active_wrist_mask_raw = region_wrist
                active_wrist_mask_patches = downsample_mask_to_patches(
                    region_wrist, grid=patch_grid).astype(np.float32)

        # Render views, extract state
        img_t, img_w = _render_views_once(env, H, W)
        state = extract_state(obs, state_dim=state_dim)

        # --- Target place mask: for grasp/push, use next subtask's object ---
        active_place_patches = cond.target_place_patches
        active_place_wrist_patches = cond.target_place_wrist_patches
        active_place_raw = None  # for visualization (agent view)
        active_place_wrist_raw = None  # for visualization (wrist view)
        if (active_place_patches is not None
                and cond.action_type in ("grasp", "push")
                and sub_idx + 1 < len(conditions)):
            next_cond = conditions[sub_idx + 1]
            if is_object_segmentable(episode_data, next_cond.object_id):
                place_sm = sim_mask_for_object_id(
                    env, episode_data, next_cond.object_id,
                    corrected_bids=corrected_bids,
                    action_type=next_cond.action_type)
                if place_sm is not None:
                    active_place_raw = place_sm
                    active_place_patches = downsample_mask_to_patches(
                        place_sm, grid=patch_grid).astype(np.float32)
                place_wrist_sm = sim_mask_for_object_id(
                    env, episode_data, next_cond.object_id,
                    corrected_bids=corrected_bids,
                    camera_name="robot0_eye_in_hand",
                    action_type=next_cond.action_type)
                if place_wrist_sm is not None:
                    active_place_wrist_raw = place_wrist_sm
                    active_place_wrist_patches = downsample_mask_to_patches(
                        place_wrist_sm, grid=patch_grid).astype(np.float32)
                elif (active_place_wrist_patches is None
                        or float(active_place_wrist_patches.sum()) <= 0.0):
                    active_place_wrist_patches = next_cond.wrist_mask_patches.astype(np.float32)
            else:
                # Non-segmentable target-place region: prefer runtime BDDL projection.
                place_region_agent = sim_region_mask_for_object_id(
                    env, episode_data, next_cond.object_id,
                    bddl_path=bddl_path, image_size=H, camera_name="agentview",
                )
                place_region_wrist = sim_region_mask_for_object_id(
                    env, episode_data, next_cond.object_id,
                    bddl_path=bddl_path, image_size=H, camera_name="robot0_eye_in_hand",
                )
                if place_region_agent is not None:
                    # Preserve third-view semantics from stored mask when available.
                    if active_place_raw is None:
                        active_place_raw = place_region_agent
                    if float(active_place_patches.sum()) <= 0.0:
                        active_place_patches = downsample_mask_to_patches(
                            place_region_agent, grid=patch_grid).astype(np.float32)
                active_place_raw = next_cond.mask_raw if next_cond.mask_raw is not None else active_place_raw
                if place_region_wrist is not None:
                    active_place_wrist_raw = place_region_wrist
                    active_place_wrist_patches = downsample_mask_to_patches(
                        place_region_wrist, grid=patch_grid).astype(np.float32)
                else:
                    active_place_wrist_raw = next_cond.wrist_mask_raw
                    if (active_place_wrist_patches is None
                            or float(active_place_wrist_patches.sum()) <= 0.0):
                        active_place_wrist_patches = next_cond.wrist_mask_patches.astype(np.float32)

        place_arg = None
        if active_place_patches is not None and float(active_place_patches.sum()) > 0:
            place_arg = active_place_patches[np.newaxis]
        place_wrist_arg = None
        if (active_place_wrist_patches is not None
                and float(active_place_wrist_patches.sum()) > 0):
            place_wrist_arg = active_place_wrist_patches[np.newaxis]

        # Always run inference; target-place is optional and may be None.
        output = gpu_worker.infer(
            imgs_third=[img_t],
            imgs_wrist=[img_w],
            states=state[np.newaxis],
            masks=active_mask_patches[np.newaxis],
            wrist_masks=active_wrist_mask_patches[np.newaxis],
            text_feat=text_feat[np.newaxis],
            action_type=np.array([cond.action_type_id], dtype=np.int64),
            num_inference_steps=num_inference_steps,
            target_place_mask=place_arg,
            target_place_mask_wrist=place_wrist_arg,
        )

        actions = output["action"][0]  # (16, 7)
        tc_val = float(output["task_comp_prob"][0]) if "task_comp_prob" in output else 0.0

        # --- Policy 2: double-check previous subtask (one-time after switch) ---
        if prev_check_info is not None:
            prev_oid, prev_atid, prev_action_type = prev_check_info
            prev_check_info = None  # only check once

            prev_patches = np.zeros(num_patches, dtype=np.float32)
            prev_wrist_patches = np.zeros(num_patches, dtype=np.float32)
            if is_object_segmentable(episode_data, prev_oid):
                prev_sm = sim_mask_for_object_id(
                    env, episode_data, prev_oid, corrected_bids=corrected_bids,
                    camera_name="agentview",
                    action_type=prev_action_type)
                if prev_sm is not None:
                    prev_patches = downsample_mask_to_patches(
                        prev_sm, grid=patch_grid).astype(np.float32)
                prev_wrist_sm = sim_mask_for_object_id(
                    env, episode_data, prev_oid, corrected_bids=corrected_bids,
                    camera_name="robot0_eye_in_hand",
                    action_type=prev_action_type)
                if prev_wrist_sm is not None:
                    prev_wrist_patches = downsample_mask_to_patches(
                        prev_wrist_sm, grid=patch_grid).astype(np.float32)
            if float(prev_patches.sum()) == 0.0:
                for c in conditions:
                    if c.object_id == prev_oid and float(c.mask_patches.sum()) > 0.0:
                        prev_patches = c.mask_patches.astype(np.float32)
                        break
            if float(prev_wrist_patches.sum()) == 0.0:
                for c in conditions:
                    if c.object_id == prev_oid and float(c.wrist_mask_patches.sum()) > 0.0:
                        prev_wrist_patches = c.wrist_mask_patches.astype(np.float32)
                        break

            prev_output = gpu_worker.infer(
                imgs_third=[img_t], imgs_wrist=[img_w],
                states=state[np.newaxis],
                masks=prev_patches[np.newaxis],
                wrist_masks=prev_wrist_patches[np.newaxis],
                text_feat=text_feat[np.newaxis],
                action_type=np.array([prev_atid], dtype=np.int64),
                num_inference_steps=num_inference_steps,
            )
            prev_tc = (float(prev_output["task_comp_prob"][0])
                       if "task_comp_prob" in prev_output else 0.0)

            if prev_tc < feas_threshold:
                print(f"  REVERT sub{sub_idx}->sub{sub_idx - 1}: "
                      f"prev_TC={prev_tc:.2f} < {feas_threshold}", flush=True)
                sub_idx -= 1
                consecutive_complete = 0
                # A revert happens before any action is executed in this
                # replan cycle.  Keep the switch guard set so the previous
                # subtask must execute at least one action chunk before it can
                # switch forward again.  Clearing it here creates a zero-step
                # sub0 -> sub1 -> sub0 loop in which total_steps never grows.
                just_switched = True
                continue

        # Subtask completion check
        if tc_val > feas_threshold:
            consecutive_complete += 1
        else:
            consecutive_complete = 0

        # --- Subtask switch (not last subtask, not if just switched) ---
        can_switch = (
            sub_idx < len(conditions) - 1
            and not just_switched
            and subtask_completion_ready(
                consecutive_complete,
                consecutive_stop,
            )
        )  # Policy 1 + causal temporal consensus

        if can_switch:
            # Policy 3: gripper enforcement on switch
            gripper_qpos = obs["robot0_gripper_qpos"][0]
            if cond.action_type == "grasp" and gripper_qpos > 0.01:
                grip_action = GRIPPER_ACTIONS["grasp"]
                max_grip = min(16, max_steps - total_steps)
                for gs in range(max_grip):
                    obs, _, _, _ = env.step(grip_action.tolist())
                    total_steps += 1
                    if obs["robot0_gripper_qpos"][0] <= 0.01:
                        break
            elif cond.action_type == "release":
                grip_action = GRIPPER_ACTIONS["release"]
                max_grip = min(16, max_steps - total_steps)
                for gs in range(max_grip):
                    obs, _, _, _ = env.step(grip_action.tolist())
                    total_steps += 1
                    if obs["robot0_gripper_qpos"][0] >= 0.035:
                        break

            if env.check_success():
                success = True
                break

            # Save for Policy 2 double-check on next cycle
            prev_check_info = (
                cond.object_id, cond.action_type_id, cond.action_type
            )

            # Switch to next subtask
            sub_idx += 1
            consecutive_complete = 0
            just_switched = True
            nxt = conditions[sub_idx]

            if record_video:
                nxt_mask = None
                if is_object_segmentable(episode_data, nxt.object_id):
                    nxt_mask = sim_mask_for_object_id(
                        env, episode_data, nxt.object_id,
                        corrected_bids=corrected_bids,
                        action_type=nxt.action_type)
                if nxt_mask is None:
                    nxt_mask = nxt.mask_raw
                info = [
                    f"SWITCH sub{cond.subtask_id} -> sub{nxt.subtask_id}",
                    f"obj: {cond.object_name} -> {nxt.object_name} [{nxt.action_type}]",
                    f"step={total_steps}/{max_steps} TC={tc_val:.2f}",
                ]
                nxt_wrist_mask = None
                if is_object_segmentable(episode_data, nxt.object_id):
                    nxt_wrist_mask = sim_mask_for_object_id(
                        env, episode_data, nxt.object_id,
                        corrected_bids=corrected_bids,
                        camera_name="robot0_eye_in_hand",
                        action_type=nxt.action_type,
                    )
                if nxt_wrist_mask is None:
                    nxt_wrist_mask = sim_region_mask_for_object_id(
                        env, episode_data, nxt.object_id,
                        bddl_path=bddl_path, image_size=H,
                        camera_name="robot0_eye_in_hand",
                    )
                if nxt_wrist_mask is None:
                    nxt_wrist_mask = nxt.wrist_mask_raw
                if nxt_mask is None:
                    nxt_mask = sim_region_mask_for_object_id(
                        env, episode_data, nxt.object_id,
                        bddl_path=bddl_path, image_size=H,
                        camera_name="agentview",
                    )
                frames.append(compose_vis_frame(
                    img_t, img_w, nxt_mask, info, tc_val=tc_val,
                    wrist_mask_raw=nxt_wrist_mask))
            continue

        # Record one frame per replan cycle (before executing chunk)
        if record_video:
            info = [
                f"sub{sub_idx + 1}/{len(conditions)} (id={cond.subtask_id}) [{cond.action_type}]",
                f"obj={cond.object_name} mask={active_mask_source}",
                f"step={total_steps}/{max_steps} TC={tc_val:.2f}",
            ]
            frames.append(compose_vis_frame(
                img_t, img_w, active_mask_raw, info, tc_val=tc_val,
                wrist_mask_raw=active_wrist_mask_raw,
                place_mask_raw=active_place_raw,
                place_mask_wrist_raw=active_place_wrist_raw))

        # --- Execute action chunk ---
        steps_to_exec = min(replan_steps, max_steps - total_steps)
        for si in range(steps_to_exec):
            action = np.clip(actions[si], -1.0, 1.0)
            obs, _, _, _ = env.step(action.tolist())
            total_steps += 1
            if env.check_success():
                success = True
                break

        # Clear just_switched after executing actions
        just_switched = False

        if success:
            break

    # Final success check (20 extra steps)
    if not success:
        for _ in range(20):
            obs, _, _, _ = env.step([0.0] * 7)
            if env.check_success():
                success = True
                break

    # Record final frame
    if record_video:
        fr_t, fr_w = _render_views_once(env, H, W)
        cond = conditions[min(sub_idx, len(conditions) - 1)]
        final_mask = cond.mask_raw
        final_wrist_mask = cond.wrist_mask_raw
        if is_object_segmentable(episode_data, cond.object_id):
            sm = sim_mask_for_object_id(
                env, episode_data, cond.object_id,
                corrected_bids=corrected_bids,
                camera_name="agentview",
                action_type=cond.action_type,
            )
            if sm is not None:
                final_mask = sm
            wrist_sm = sim_mask_for_object_id(
                env, episode_data, cond.object_id,
                corrected_bids=corrected_bids,
                camera_name="robot0_eye_in_hand",
                action_type=cond.action_type,
            )
            if wrist_sm is not None:
                final_wrist_mask = wrist_sm
        else:
            sm = sim_region_mask_for_object_id(
                env, episode_data, cond.object_id,
                bddl_path=bddl_path, image_size=H,
                camera_name="agentview",
            )
            if sm is not None:
                final_mask = sm
            wrist_sm = sim_region_mask_for_object_id(
                env, episode_data, cond.object_id,
                bddl_path=bddl_path, image_size=H,
                camera_name="robot0_eye_in_hand",
            )
            if wrist_sm is not None:
                final_wrist_mask = wrist_sm
        info = [
            f"DONE {'SUCCESS' if success else 'FAIL'}",
            f"subs={min(sub_idx + 1, len(conditions))}/{len(conditions)}",
            f"steps={total_steps}/{max_steps}",
        ]
        frames.append(compose_vis_frame(
            fr_t, fr_w, final_mask, info, tc_val=0.0, success=success,
            wrist_mask_raw=final_wrist_mask))

    return frames, success, {
        "subtasks_reached": min(sub_idx + 1, len(conditions)),
        "num_subtasks": len(conditions),
        "total_steps": total_steps,
    }




# ---------------------------------------------------------------------------
# Per-GPU worker process
# ---------------------------------------------------------------------------




def _gpu_worker_process(
    gpu_id, task_ids, checkpoint, progress_checkpoint, model_type,
    benchmark_name, episodes_per_task, num_parallel, replan_steps,
    num_inference_steps, seed, record_video, video_dir, result_queue,
    egl_gpu_id=None, dino_input_size=448, use_sim_seg=False,
    episodes_json_path='', feas_threshold=0.7, consecutive_stop=2,
    max_success_videos_per_task=-1, max_fail_videos_per_task=-1,
    episode_ids: Optional[List[int]] = None, max_steps_override=None,
):
    """Per-GPU worker: GPUInferenceWorker (CUDA subprocess) + envs in same process."""
    egl_gpu = egl_gpu_id if egl_gpu_id is not None else gpu_id
    prefix = f'[CUDA {gpu_id} EGL {egl_gpu}]'
    print(f'{prefix} Starting inference worker...', flush=True)
    gpu_worker = GPUInferenceWorker(checkpoint, progress_checkpoint, model_type, gpu_id=gpu_id, dino_input_size=dino_input_size)
    print(f'{prefix} Worker ready.', flush=True)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_gpu)
    _patch_robosuite_egl()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from shared.data.packed_features import TextFeatureCache
    max_steps = int(max_steps_override) if max_steps_override is not None else MAX_STEPS_MAP.get(benchmark_name, 520)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[benchmark_name]()
    config_path = Path(checkpoint).parent.parent / 'config.json'
    state_dim = 8
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        state_dim = cfg.get('dit', {}).get('state_dim', 8)
    print(f'{prefix} state_dim={state_dim}', flush=True)
    text_cache = TextFeatureCache()
    descs = [task_suite.get_task(tid).language for tid in task_ids]
    text_cache.precompute(descs)
    all_episodes = None
    if use_sim_seg and episodes_json_path:
        all_episodes = load_episode_records(episodes_json_path)
        purpose = 'policy masks'
        print(f'{prefix} Loaded {len(all_episodes)} episodes from JSON for {purpose}', flush=True)
    results = []
    H = W = LIBERO_ENV_RESOLUTION
    total_tasks_on_gpu = len(task_ids)
    tasks_done = 0
    gpu_total_success = 0
    gpu_total_episodes = 0
    t0_gpu = time.time()
    task_times = []
    for tid in task_ids:
        t0_task = time.time()
        task = task_suite.get_task(tid)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(tid)
        eval_episode_ids = list(episode_ids) if episode_ids is not None else list(range(episodes_per_task))
        text_feat = text_cache.get(task_description)
        bddl_file = str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
        if use_sim_seg and all_episodes is not None:
            (ep_key, episode) = _find_episode(all_episodes, task_description.strip(), benchmark_name)
            conditions = build_subtask_conditions(episode, dino_input_size=dino_input_size)
            print(f'{prefix} Task {tid} [{tasks_done + 1}/{total_tasks_on_gpu}]: {len(conditions)} subtasks | {task_description[:50]} | episodes {eval_episode_ids}', flush=True)
            print(f"{prefix}   episode_match: key={ep_key} episode_index={episode.get('episode_index')} task_index={episode.get('task_index')}", flush=True)
            for c in conditions:
                print(f'{prefix}   sub{c.subtask_id}: [{c.action_type}] obj={c.object_name} mask={c.mask_source}', flush=True)
            task_successes = 0
            saved_success_videos = 0
            saved_fail_videos = 0
            corrected_bids = None
            for ep_idx in eval_episode_ids:
                env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=H, camera_widths=W)
                env.seed(seed + tid * 100 + ep_idx)
                gpu_worker.set_seed(seed + tid * 100 + ep_idx)
                _set_camera_observables(env, enabled=False)
                if corrected_bids is None:
                    corrected_bids = verify_and_correct_body_ids(env, episode)
                try:
                    (frames, s, meta) = run_single_episode_mask_sequence(gpu_worker, env, initial_states[ep_idx % len(initial_states)], text_feat, episode, conditions, replan_steps=replan_steps, num_inference_steps=num_inference_steps, max_steps=max_steps, feas_threshold=feas_threshold, consecutive_stop=consecutive_stop, record_video=record_video, corrected_bids=corrected_bids, dino_input_size=dino_input_size, state_dim=state_dim, bddl_path=bddl_file)
                finally:
                    env.close()
                if s:
                    task_successes += 1
                results.append({'task_id': tid, 'task_description': task_description, 'episode_idx': ep_idx, 'success': s, 'meta': meta})
                tag = 'success' if s else 'fail'
                print(f"{prefix}   ep{ep_idx}: {('OK' if s else 'FAIL')} subs={meta['subtasks_reached']}/{meta['num_subtasks']} steps={meta['total_steps']}", flush=True)
                if record_video and video_dir and frames:
                    should_save = False
                    if s:
                        should_save = max_success_videos_per_task < 0 or saved_success_videos < max_success_videos_per_task
                    else:
                        should_save = max_fail_videos_per_task < 0 or saved_fail_videos < max_fail_videos_per_task
                    if should_save:
                        vpath = Path(video_dir) / f'task{tid}_ep{ep_idx}_{tag}.mp4'
                        save_video(frames, str(vpath), fps=5)
                        if s:
                            saved_success_videos += 1
                        else:
                            saved_fail_videos += 1
            task_elapsed = time.time() - t0_task
            task_times.append(task_elapsed)
            tasks_done += 1
            gpu_total_success += task_successes
            gpu_total_episodes += len(eval_episode_ids)
            gpu_sr = gpu_total_success / gpu_total_episodes * 100
            print(f'{prefix} Task {tid} DONE: {task_successes}/{len(eval_episode_ids)} ({task_successes / max(1, len(eval_episode_ids)) * 100:.0f}%) in {task_elapsed:.1f}s | GPU: {gpu_total_success}/{gpu_total_episodes} ({gpu_sr:.1f}%)', flush=True)
            continue
    gpu_elapsed = time.time() - t0_gpu
    print(f'{prefix} ALL DONE: {gpu_total_success}/{gpu_total_episodes} ({gpu_total_success / max(1, gpu_total_episodes) * 100:.1f}%) in {gpu_elapsed:.1f}s', flush=True)
    gpu_worker.close()
    result_queue.put({'episodes': results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='LIBERO evaluation')
    parser.add_argument('--model-type', type=str, required=True, choices=['rain', 'rainv2'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--progress-checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint-type', type=str, choices=['best', 'latest'], default='latest')
    parser.add_argument('--benchmark', type=str, default='libero_10')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU IDs (e.g. 0,1,2,3)')
    parser.add_argument('--episodes-per-task', type=int, default=50)
    parser.add_argument('--num-parallel', type=int, default=1)
    parser.add_argument('--replan-steps', type=int, default=8)
    parser.add_argument('--num-inference-steps', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=None, help='Optional episode horizon override; deviates from the benchmark protocol.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save-dir', type=str, default='', help='Override eval results dir (default: eval_results/{experiment}/{benchmark})')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--video-dir', type=str, default='', help='Override video dir (default: visualize/{experiment}/{benchmark})')
    parser.add_argument('--max-success-videos-per-task', type=int, default=-1, help='Max success videos saved per task (-1 = unlimited)')
    parser.add_argument('--max-fail-videos-per-task', type=int, default=-1, help='Max failure videos saved per task (-1 = unlimited)')
    parser.add_argument('--dino-input-size', type=int, default=224, help='DINOv2 input resolution (224, 448, 518, etc.)')
    parser.add_argument('--use-sim-seg', action='store_true', help='Required public mode: simulator GT masks')
    parser.add_argument('--egl-gpu-offset', type=int, default=0, help='EGL GPU = (cuda_gpu + offset) %% num_total_gpus. Set to 4 on node0 to separate EGL from CUDA.')
    parser.add_argument('--egl-gpu', type=int, default=None, help='Fixed EGL GPU for all workers (overrides --egl-gpu-offset)')
    parser.add_argument('--tasks', type=str, default=None, help='Comma-separated task IDs to evaluate (e.g. 2,3,5,6,7,9)')
    parser.add_argument('--episode-ids', type=str, default=None, help='Comma-separated episode indices to evaluate for each selected task (e.g. 0,7,13,22). Overrides --episodes-per-task when set.')
    parser.add_argument('--episodes-json', type=str, default='', help='Path to episode JSON; auto-resolved from config.json if empty')
    parser.add_argument('--feas-threshold', type=float, default=0.7, help='TC threshold for subtask completion')
    parser.add_argument('--consecutive-stop', type=int, default=2, help='Consecutive replan cycles with TC > threshold before switching')
    args = parser.parse_args()
    if not args.use_sim_seg:
        parser.error('Public evaluation requires --use-sim-seg; use python -m rain.eval_libero')
    if args.num_parallel != 1:
        parser.error('Public GT evaluation requires --num-parallel 1')
    if args.model_type in ('rain', 'rainv2') and args.progress_checkpoint is None:
        parser.error('--progress-checkpoint is required for --model-type rain/rainv2')
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    resolved_ckpt = str(_resolve_checkpoint_path(args.checkpoint, args.checkpoint_type))
    resolved_progress = None
    if args.progress_checkpoint:
        resolved_progress = str(_resolve_checkpoint_path(args.progress_checkpoint, args.checkpoint_type))
    episodes_json_path = args.episodes_json
    if args.use_sim_seg and (not episodes_json_path):
        config_path = Path(resolved_ckpt).parent.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            episodes_json_path = cfg.get('data', {}).get('episodes_json', '')
        if not episodes_json_path:
            parser.error('--episodes-json required for --use-sim-seg (could not auto-resolve from config.json)')
    model_labels = {'rain': 'RAIN', 'rainv2': 'RAIN'}
    model_label = model_labels[args.model_type]
    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.benchmark]()
    num_tasks = task_suite.get_num_tasks()
    del task_suite
    if args.tasks:
        eval_task_ids = [int(t) for t in args.tasks.split(',')]
    else:
        eval_task_ids = list(range(num_tasks))
    episode_ids = None
    if args.episode_ids:
        episode_ids = [int(ep) for ep in args.episode_ids.split(',') if str(ep).strip()]
        if not episode_ids:
            parser.error('--episode-ids provided but no valid indices parsed')
    num_gpus = len(gpu_ids)
    task_assignments = {i: [] for i in range(num_gpus)}
    for (idx, tid) in enumerate(eval_task_ids):
        task_assignments[idx % num_gpus].append(tid)
    import torch
    all_gpu_count = torch.cuda.device_count() or 8
    egl_offset = args.egl_gpu_offset
    fixed_egl_gpu = args.egl_gpu
    print('=' * 70)
    print(f'{args.benchmark.upper()} Evaluation ({model_label})')
    print('=' * 70)
    print(f'  Checkpoint: {resolved_ckpt}')
    if resolved_progress:
        print(f'  Progress: {resolved_progress}')
    print(f'  GPUs: {gpu_ids} | Envs/GPU: {args.num_parallel}')
    if args.record_video:
        print(f'  Video caps per task: success={args.max_success_videos_per_task}, fail={args.max_fail_videos_per_task}')
    if args.use_sim_seg:
        print(f'  Mode: mask-sequence (sim-seg)')
        print(f'  Episodes JSON: {episodes_json_path}')
        print(f'  Feas threshold: {args.feas_threshold} | Consecutive stop: {args.consecutive_stop}')
    if args.tasks:
        print(f'  Tasks: {eval_task_ids}')
    for (i, gid) in enumerate(gpu_ids):
        if fixed_egl_gpu is not None:
            egl_gid = fixed_egl_gpu
        elif egl_offset:
            egl_gid = (gid + egl_offset) % all_gpu_count
        else:
            egl_gid = gid
        print(f'  Worker {i}: CUDA={gid}, EGL={egl_gid} | tasks {task_assignments[i]}')
    print(f'  Episodes/task: {args.episodes_per_task}')
    if episode_ids is not None:
        print(f'  Episode IDs: {episode_ids}')
    print('  Rollout: sequential GT-mask episodes per worker')
    print(flush=True)
    _ckpt_parent = Path(resolved_ckpt).parent.parent
    _experiment_name = _ckpt_parent.name
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        ckpt_suffix = f'_{args.checkpoint_type}' if args.checkpoint_type != 'latest' else ''
        save_dir = Path('eval_results') / _experiment_name / f'{args.benchmark}{ckpt_suffix}'
    save_dir.mkdir(parents=True, exist_ok=True)
    video_dir = None
    if args.record_video:
        if args.video_dir:
            video_dir = Path(args.video_dir)
        elif args.save_dir:
            video_dir = save_dir / 'videos'
        else:
            video_dir = Path('visualize') / _experiment_name / args.benchmark
        video_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result_queue = Queue()
    processes = []
    for (i, gid) in enumerate(gpu_ids):
        if not task_assignments[i]:
            continue
        if fixed_egl_gpu is not None:
            egl_gid = fixed_egl_gpu
        elif egl_offset:
            egl_gid = (gid + egl_offset) % all_gpu_count
        else:
            egl_gid = None
        p = Process(target=_gpu_worker_process, args=(gid, task_assignments[i], resolved_ckpt, resolved_progress, args.model_type, args.benchmark, args.episodes_per_task, args.num_parallel, args.replan_steps, args.num_inference_steps, args.seed, args.record_video, str(video_dir) if video_dir else None, result_queue, egl_gid, args.dino_input_size, args.use_sim_seg, episodes_json_path, args.feas_threshold, args.consecutive_stop, args.max_success_videos_per_task, args.max_fail_videos_per_task, episode_ids, args.max_steps))
        p.start()
        processes.append(p)
    all_results = []
    pending = len(processes)
    while pending > 0:
        try:
            payload = result_queue.get(timeout=30)
            if not isinstance(payload, dict) or not isinstance(payload.get('episodes'), list):
                raise RuntimeError('GPU worker returned a malformed result payload')
            all_results.extend(payload['episodes'])
            pending -= 1
            continue
        except pyqueue.Empty:
            dead = [p for p in processes if not p.is_alive()]
            failed = [p for p in dead if p.exitcode not in (0, None)]
            if failed:
                details = ', '.join((f'pid={p.pid},exit={p.exitcode}' for p in failed))
                raise RuntimeError(f'One or more GPU workers failed before returning results: {details}')
            if len(dead) == len(processes):
                raise RuntimeError(f'All worker processes exited but not all results were received. received={len(processes) - pending}, expected={len(processes)}')
    for p in processes:
        p.join()
    elapsed = time.time() - t0
    all_results.sort(key=lambda r: (r['task_id'], r['episode_idx']))
    print('\n' + '=' * 80)
    print(f'{args.benchmark.upper()} EVALUATION RESULTS ({model_label})')
    print('=' * 80)
    print(f"{'Task':>4} | {'Success':>10} | Description")
    print('-' * 80)
    total_success = 0
    total_episodes = 0
    task_summaries = []
    for tid in range(num_tasks):
        task_results = [r for r in all_results if r['task_id'] == tid]
        if not task_results:
            continue
        successes = sum((1 for r in task_results if r['success']))
        n = len(task_results)
        desc = task_results[0]['task_description']
        sr = successes / n * 100 if n > 0 else 0.0
        print(f'{tid:>4} | {successes:>4}/{n:<4} | {desc}')
        task_summaries.append({'task_id': tid, 'task_description': desc, 'successes': successes, 'episodes': n, 'success_rate': sr, 'mask_quality': None})
        total_success += successes
        total_episodes += n
    print('-' * 80)
    rate = total_success / total_episodes * 100 if total_episodes > 0 else 0
    print(f"{'ALL':>4} | {total_success:>4}/{total_episodes:<4} | {rate:.1f}%")
    print(f'Time: {elapsed:.1f}s')
    print('=' * 80)
    output = {'benchmark': args.benchmark, 'model_type': args.model_type, 'checkpoint': resolved_ckpt, 'progress_checkpoint': resolved_progress, 'episodes_per_task': args.episodes_per_task, 'episode_ids': episode_ids, 'num_parallel': args.num_parallel, 'replan_steps': args.replan_steps, 'num_inference_steps': args.num_inference_steps, 'max_steps': args.max_steps, 'seed': args.seed, 'max_success_videos_per_task': args.max_success_videos_per_task, 'max_fail_videos_per_task': args.max_fail_videos_per_task, 'total_success': total_success, 'total_episodes': total_episodes, 'success_rate': rate, 'elapsed_seconds': elapsed, 'task_summaries': task_summaries, 'episodes': all_results, 'feas_threshold': args.feas_threshold, 'consecutive_stop': args.consecutive_stop}
    if args.use_sim_seg:
        output['episodes_json'] = episodes_json_path
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
