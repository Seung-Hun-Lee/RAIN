"""Project finite BDDL workspace regions from the current simulator geometry.

The kitchen workspace is called ``kitchen_table`` in BDDL but its simulator
body is ``table``.  Region sites and the workspace surface supply the frame
directly, avoiding assumptions about body names or source-scene coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bddl import sections


def _region_bounds(bddl_path: Path, full_name: str) -> tuple[str, np.ndarray]:
    text = Path(bddl_path).read_text(encoding="utf-8")
    for region in sections(text).get(":regions", []):
        attrs = {item[0]: item[1:] for item in region[1:]}
        name, target = region[0], attrs[":target"][0]
        if f"{target}_{name}" != full_name:
            continue
        ranges = attrs.get(":ranges")
        if not ranges:
            raise ValueError(f"Region {full_name!r} has no finite workspace rectangle")
        bounds = np.asarray([float(value) for value in ranges[0][0]], dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError(f"Invalid region bounds for {full_name!r}: {bounds}")
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(f"Empty region bounds for {full_name!r}: {bounds}")
        return target, bounds
    raise ValueError(f"Current BDDL does not define region {full_name!r}")


def _site_pose(sim, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        site_id = sim.model.site_name2id(name)
    except (ValueError, KeyError):
        return None
    if site_id < 0:
        return None
    return (
        np.asarray(sim.data.site_xpos[site_id], dtype=float).copy(),
        np.asarray(sim.data.site_xmat[site_id], dtype=float).reshape(3, 3).copy(),
    )


def region_world_corners(env, bddl_path: Path, full_name: str) -> np.ndarray:
    """Return the current rectangle corners on the actual workspace surface."""
    target, bounds = _region_bounds(bddl_path, full_name)
    inner = getattr(env, "env", env)
    sim = env.sim
    workspace_name = getattr(inner, "workspace_name", None)
    if workspace_name is not None and target != workspace_name:
        raise ValueError(f"{full_name!r} targets {target!r}, not workspace {workspace_name!r}")

    offset = getattr(inner, "workspace_offset", None)
    offset = None if offset is None else np.asarray(offset, dtype=float)
    # Generic tabletop and KitchenTableArena both expose table_top on body
    # table. Mesh workspaces (living room/study/floor) use workspace_offset.
    surface = _site_pose(sim, "table_top") if target in {"main_table", "kitchen_table"} else None
    if surface is not None:
        surface_z = float(surface[0][2])
    elif offset is not None and offset.shape == (3,):
        surface_z = float(offset[2])
    else:
        raise ValueError(f"Cannot resolve current surface height for {target!r}")

    x0, y0, x1, y1 = bounds
    pose = _site_pose(sim, full_name)
    if pose is not None:
        center, rotation = pose
        hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
        local = np.asarray([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])
        corners = center + local @ rotation.T
    elif offset is not None and offset.shape == (3,):
        # A workspace region without a named site still has an exact current
        # BDDL rectangle in the runtime workspace-offset frame.
        corners = np.asarray([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]) + offset
    else:
        raise ValueError(f"Cannot resolve current coordinate frame for {full_name!r}")
    corners[:, 2] = surface_z
    return corners


def render_region_mask(env, bddl_path: Path, full_name: str, image_size: int = 320, camera_name: str = "agentview") -> np.ndarray | None:
    """Drop-in replacement for base.render_region_mask, without GPU setup."""
    import cv2

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    corners = region_world_corners(env, bddl_path, full_name)
    sim = env.sim
    camera_id = sim.model.camera_name2id(camera_name)
    camera_pos = np.asarray(sim.data.cam_xpos[camera_id], dtype=float)
    camera_mat = np.asarray(sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    camera_points = (corners - camera_pos) @ camera_mat
    depth = -camera_points[:, 2]
    if np.any(depth <= 1e-8):
        return None
    focal = image_size / (2.0 * np.tan(np.deg2rad(float(sim.model.cam_fovy[camera_id])) / 2.0))
    points = np.column_stack((
        focal * camera_points[:, 0] / depth + image_size / 2.0,
        image_size - 1 - (focal * camera_points[:, 1] / depth + image_size / 2.0),
    ))
    if not np.isfinite(points).all():
        return None
    # Do not clamp individual corners to image edges: that can manufacture a
    # visible mask for a rectangle that lies entirely outside the camera view.
    polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    output = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillConvexPoly(output, polygon, 1)
    return output if output.any() else None
