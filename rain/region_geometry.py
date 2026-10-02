"""Project declared workspace regions or current native XML box sites.

Masks use the same top-left image coordinates as diverse_adapt_mask_geometry.
RAIN's additional horizontal image flip belongs to its evaluator adapter.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from . import workspace_geometry
from .bddl import sections, typed_declarations


def _declared_region(bddl_path: Path, full_name: str) -> bool:
    """Validate one exact current declaration; return whether it has ranges."""
    data = sections(Path(bddl_path).read_text(encoding="utf-8"))
    declared = {
        **typed_declarations(data.get(":fixtures", [])),
        **typed_declarations(data.get(":objects", [])),
    }
    matches = []
    for region in data.get(":regions", []):
        attrs = {entry[0]: entry[1:] for entry in region[1:]}
        target = attrs.get(":target", [])
        if len(target) != 1:
            raise ValueError(f"Malformed region target in current BDDL: {region[0]!r}")
        if target[0] + "_" + region[0] == full_name:
            if target[0] not in declared:
                raise ValueError(f"Region {full_name!r} targets an undeclared entity")
            matches.append(":ranges" in attrs)
    if len(matches) != 1:
        raise ValueError(f"Current BDDL must define region {full_name!r} exactly once; found {len(matches)}")
    return matches[0]


def site_world_corners(env, bddl_path: Path, full_name: str) -> np.ndarray:
    """Return eight exact corners of a declared current simulator box site."""
    if _declared_region(bddl_path, full_name):
        raise ValueError(f"{full_name!r} is a sampled rectangle, not a native XML site")
    sim = env.sim
    try:
        site_id = int(sim.model.site_name2id(full_name))
    except (ValueError, KeyError) as error:
        raise ValueError(f"Current simulator has no site {full_name!r}") from error
    if site_id < 0:
        raise ValueError(f"Current simulator has no site {full_name!r}")
    # MuJoCo mjtGeom.mjGEOM_BOX == 6. Avoid importing a simulator to project.
    if int(sim.model.site_type[site_id]) != 6:
        raise ValueError(f"Native region {full_name!r} must be an XML box site")
    center = np.asarray(sim.data.site_xpos[site_id], dtype=float)
    rotation = np.asarray(sim.data.site_xmat[site_id], dtype=float).reshape(3, 3)
    half_size = np.asarray(sim.model.site_size[site_id], dtype=float)
    if (center.shape != (3,) or half_size.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(rotation).all()
            or not np.isfinite(half_size).all() or np.any(half_size <= 0)):
        raise ValueError(f"Invalid current box geometry for {full_name!r}")
    # Do not enlarge a thin top_side or replace a compartment with its parent.
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    return center + (signs * half_size) @ rotation.T


def _clip_polygon(polygon: np.ndarray, image_size: int) -> np.ndarray:
    """Intersect a convex polygon with the image, without clamping vertices."""
    points = [point for point in polygon]
    for axis, bound, lower in ((0, 0., True), (0, image_size - 1., False),
                               (1, 0., True), (1, image_size - 1., False)):
        if not points:
            break
        clipped = []
        previous = points[-1]
        previous_in = previous[axis] >= bound if lower else previous[axis] <= bound
        for current in points:
            current_in = current[axis] >= bound if lower else current[axis] <= bound
            if current_in != previous_in:
                fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(previous + fraction * (current - previous))
            if current_in:
                clipped.append(current)
            previous, previous_in = current, current_in
        points = clipped
    return np.asarray(points, dtype=float).reshape(-1, 2)


def render_region_mask(env, bddl_path: Path, full_name: str, image_size: int = 320,
                       camera_name: str = "agentview") -> np.ndarray | None:
    """Render an exact region; reject camera-plane crossings and empty views.

    Native box masks are projected geometric silhouettes, not visibility or
    occlusion segmentations. Missing/undeclared/non-box sites are errors.
    Workspace rectangles retain the existing helper's implementation.
    """
    import cv2

    if not isinstance(image_size, (int, np.integer)) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if _declared_region(bddl_path, full_name):
        return workspace_geometry.render_region_mask(env, bddl_path, full_name, image_size, camera_name)
    corners = site_world_corners(env, bddl_path, full_name)
    sim = env.sim
    camera_id = int(sim.model.camera_name2id(camera_name))
    if camera_id < 0:
        raise ValueError(f"Current simulator has no camera {camera_name!r}")
    camera_pos = np.asarray(sim.data.cam_xpos[camera_id], dtype=float)
    camera_mat = np.asarray(sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    camera_points = (corners - camera_pos) @ camera_mat
    depth = -camera_points[:, 2]
    if not np.isfinite(camera_points).all() or np.any(depth <= 1e-8):
        return None
    fovy = float(sim.model.cam_fovy[camera_id])
    if not np.isfinite(fovy) or not 0 < fovy < 180:
        raise ValueError(f"Invalid camera field of view: {fovy}")
    focal = image_size / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    points = np.column_stack((
        focal * camera_points[:, 0] / depth + image_size / 2.0,
        image_size - 1 - (focal * camera_points[:, 1] / depth + image_size / 2.0),
    ))
    if not np.isfinite(points).all():
        return None
    polygon = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(float)
    polygon = _clip_polygon(polygon, image_size)
    if len(polygon) < 3:
        return None
    output = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillConvexPoly(output, np.rint(polygon).astype(np.int32).reshape(-1, 1, 2), 1)
    return output if output.any() else None
