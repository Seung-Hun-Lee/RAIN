"""Native bowl-drainer section poses without simulator or asset patching.

Robot forward is world +X and robot left is world +Y in the
LIBERO-Object floor scene.  At the native identity drainer orientation,
``left_region`` is robot-left and ``right_region`` is robot-right.  Rendered
camera pixels can reverse these names: task language always uses robot frame.
Only the fixture is translated, so the requested section center stays at the
original basket's learned destination (0, .26).  Native sites/geoms are intact.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from .novel_feedback_fixture_geometry import collision_boxes, quat_matrix


from ..assets import asset_root
ASSET_ROOT = asset_root()
DRAINER_XML = ASSET_ROOT / "turbosquid_objects/bowl_drainer/bowl_drainer.xml"
LEARNED_DESTINATION_XY = (0.0, 0.26)
# LIBERO's logical floor workspace reference is -.035, but the physical
# FloorArena plane is z=0.  Using the logical reference as support height
# buries the drainer base. Section tasks use the physical support height.
LOGICAL_FLOOR_REFERENCE_Z = -0.035
FLOOR_WORLD_Z = 0.0
SIDE_SPECS = {
    "left": {
        "native_region": "bowl_drainer_1_left_region",
        "site_local_position": (0.0, 0.06211, 0.06757),
        "fixture_center": (0.0, 0.19789),
        "fixture_translation_from_original": (0.0, -0.06211),
        "robot_side": "left",
    },
    "right": {
        "native_region": "bowl_drainer_1_right_region",
        "site_local_position": (0.0, -0.05153, 0.06757),
        "fixture_center": (0.0, 0.31153),
        "fixture_translation_from_original": (0.0, 0.05153),
        "robot_side": "right",
    },
}
OBJECT_KINDS = (
    "alphabet_soup", "cream_cheese", "tomato_sauce", "butter", "chocolate_pudding"
)
NATIVE_SOURCE_YAWS = {
    "alphabet_soup": np.pi / 2,
    "cream_cheese": 0.0,
    "tomato_sauce": np.pi / 2,
    "butter": 0.0,
    "chocolate_pudding": 0.0,
}
SIGNS = np.asarray(list(product((-1.0, 1.0), repeat=3)))


def target_region(side: str) -> str:
    return SIDE_SPECS[side]["native_region"]


def fixture_center(side: str) -> tuple[float, float]:
    return SIDE_SPECS[side]["fixture_center"]


def install_floor_drainer_support() -> None:
    """Set section-task fixture support in the current process.

    Reuses the shared fixture installer. Only a bowl_drainer owned by an exact
    fixed-fixture sampler whose logical floor reference is -.035 is translated
    vertically to the physical z=0 plane. Initialization and evaluation
    entrypoints must both call this installer.
    """
    from .novel_feedback_fixture_geometry import install_feedback_fixture_geometry
    install_feedback_fixture_geometry()
    from libero.libero.envs.regions.base_region_sampler import MultiRegionRandomSampler
    if getattr(MultiRegionRandomSampler.sample, "_bowl_drainer_sections_floor_support", False):
        return
    original = MultiRegionRandomSampler.sample
    min_z = min(float(box["lower"][2]) for box in collision_boxes(DRAINER_XML))

    def sample(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if type(self) is not MultiRegionRandomSampler:
            return result
        if not np.isclose(float(self.reference_pos[2]), LOGICAL_FLOOR_REFERENCE_Z, atol=1e-9):
            return result
        own_names = {obj.name for obj in self.mujoco_objects}
        for name, (position, quaternion, obj) in list(result.items()):
            if name not in own_names or getattr(obj, "category_name", "") != "bowl_drainer":
                continue
            corrected = (float(position[0]), float(position[1]), FLOOR_WORLD_Z-min_z)
            result[name] = (corrected, quaternion, obj)
        return result

    sample._bowl_drainer_sections_floor_support = True
    # Preserve installer guards, preventing another process-local wrapper
    # from later overwriting the corrected fixture's physical Z placement.
    sample._feedback_fixture_support = True
    sample._novel_scene_fixture_support = True
    MultiRegionRandomSampler.sample = sample


def _site_specs() -> dict:
    root = ET.parse(DRAINER_XML).find("./worldbody/body/body[@name='object']")
    result = {}
    for side, expected in SIDE_SPECS.items():
        site = root.find(f"./site[@name='{side}_region']")
        position = np.fromstring(site.get("pos"), sep=" ")
        rotation = quat_matrix(np.fromstring(site.get("quat"), sep=" "))
        half = np.fromstring(site.get("size"), sep=" ")
        if not np.allclose(position, expected["site_local_position"], atol=1e-9):
            raise ValueError("Native bowl-drainer site geometry changed")
        corners = position + (SIGNS * half) @ rotation.T
        result[side] = dict(position=position, rotation=rotation, half_size=half,
                            lower=corners.min(0), upper=corners.max(0))
    return result


def static_geometry_report() -> dict:
    """Read-only native box fit; a fit is a witness, not an SR prediction."""
    sites = _site_specs()
    drainer = collision_boxes(DRAINER_XML)
    floor = min(drainer, key=lambda box: box["center"][2])
    floor_upper_z = float(floor["upper"][2])
    root_z = FLOOR_WORLD_Z - min(float(box["lower"][2]) for box in drainer)
    objects = {}
    for kind in OBJECT_KINDS:
        path = ASSET_ROOT / "stable_hope_objects" / kind / f"{kind}.xml"
        boxes = collision_boxes(path)
        yaw = NATIVE_SOURCE_YAWS[kind]
        rotation = quat_matrix((np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        corners = np.concatenate([box["corners"] @ rotation.T for box in boxes])
        low, high = corners.min(0), corners.max(0)
        center = (low + high) / 2
        side_reports = {}
        for side, site in sites.items():
            position = np.array([site["position"][0] - center[0],
                                 site["position"][1] - center[1],
                                 floor_upper_z - low[2] + 0.0002])
            posed = corners + position
            margin = np.minimum(posed.min(0) - site["lower"], site["upper"] - posed.max(0))
            # This static test covers the exact native compartment envelope.
            # Actual side-wall/contact feasibility is tested by settled witnesses.
            side_reports[side] = dict(
                whole_collision_envelope_inside_native_site=bool(np.all(margin >= 0)),
                margin_xyz_m=margin.tolist(),
                object_root_in_drainer_frame=position.tolist(),
            )
        objects[kind] = dict(native_source_yaw_rad=float(yaw),
                             collision_envelope_xyz_m=(high-low).tolist(),
                             compartments=side_reports)
    return dict(
        robot_forward_world_axis="+X", robot_left_world_axis="+Y",
        robot_base_position_m=[-0.6, 0.0, 0.0],
        learned_destination_xy_m=list(LEARNED_DESTINATION_XY),
        fixture_root_z_m=root_z, native_floor_top_local_z_m=floor_upper_z,
        logical_floor_reference_z_m=LOGICAL_FLOOR_REFERENCE_Z,
        actual_physical_floor_z_m=FLOOR_WORLD_Z,
        new_fixture_vertical_correction_m=0.035,
        native_region_dimensions_xyz_m={side: (v["upper"]-v["lower"]).tolist()
                                        for side, v in sites.items()},
        sides=SIDE_SPECS, objects=objects,
        success_rule="Unchanged native In of only the selected region; the other compartment is failure.",
        mask_rule="Exact projected selected native box site; no whole fixture or union mask.",
    )


def _inner_env(env):
    while not hasattr(env, "objects_dict") and hasattr(env, "env"):
        env = env.env
    return env


def _body_ids(sim, root_id):
    parents = np.asarray(sim.model.body_parentid, dtype=int)
    selected = {int(root_id)}
    while True:
        expanded = selected | {i for i, parent in enumerate(parents) if int(parent) in selected}
        if expanded == selected:
            return selected
        selected = expanded


def validate_scene(env, metadata: dict) -> dict:
    """Gate a saved initial state without modifying simulator state.

    Metadata requires selected_compartment and target_object.  This checks
    geometry and goal falsity; it neither changes success nor executes policy.
    """
    from novel_scene_mask_geometry import render_region_mask

    inner = _inner_env(env)
    sim = inner.sim
    side = metadata["selected_compartment"]
    spec = SIDE_SPECS[side]
    target = metadata.get("target_object")
    if not target:
        atom = metadata["canonical_goal_atoms"][0]
        target = re.fullmatch(r"in\(([^,]+),\s*([^)]+)\)", atom).group(1)
    selected_name = target_region(side)
    other_name = target_region("right" if side == "left" else "left")
    fixture = inner.fixtures_dict["bowl_drainer_1"]
    root_id = int(sim.model.body_name2id(fixture.root_body))
    root_position = np.asarray(sim.data.body_xpos[root_id])
    root_rotation = np.asarray(sim.data.body_xmat[root_id]).reshape(3, 3)
    assert np.allclose(root_position[:2], spec["fixture_center"], atol=1e-6), root_position
    assert np.allclose(root_rotation, np.eye(3), atol=1e-6), root_rotation
    expected_z = static_geometry_report()["fixture_root_z_m"]
    assert abs(float(root_position[2])-expected_z) < 1e-6, (root_position, expected_z)
    floor_id = int(sim.model.geom_name2id("floor"))
    assert abs(float(sim.data.geom_xpos[floor_id][2])-FLOOR_WORLD_Z) < 1e-6
    selected_id = int(sim.model.site_name2id(selected_name))
    other_id = int(sim.model.site_name2id(other_name))
    selected_pos = np.asarray(sim.data.site_xpos[selected_id])
    other_pos = np.asarray(sim.data.site_xpos[other_id])
    assert np.allclose(selected_pos[:2], LEARNED_DESTINATION_XY, atol=1e-6), selected_pos
    assert (selected_pos[1] > other_pos[1]) == (side == "left"), (selected_pos, other_pos)
    robot_root = inner.robots[0].robot_model.root_body
    robot_id = int(sim.model.body_name2id(robot_root))
    robot_position = np.asarray(sim.data.body_xpos[robot_id])
    robot_rotation = np.asarray(sim.data.body_xmat[robot_id]).reshape(3, 3)
    assert np.allclose(robot_position[:2], (-0.6, 0.0), atol=1e-6), robot_position
    assert np.allclose(robot_rotation[:, 0], (1, 0, 0), atol=1e-6), robot_rotation
    assert np.allclose(robot_rotation[:, 1], (0, 1, 0), atol=1e-6), robot_rotation
    selected_truth = bool(inner._eval_predicate(["in", target, selected_name]))
    other_truth = bool(inner._eval_predicate(["in", target, other_name]))
    assert not selected_truth and not other_truth, (selected_truth, other_truth)
    assert not bool(inner._check_success()), "Initial task goal is true"
    bddl = Path(inner.bddl_file_name)
    selected_mask = render_region_mask(inner, bddl, selected_name)
    other_mask = render_region_mask(inner, bddl, other_name)
    assert selected_mask is not None and other_mask is not None
    selected_pixels = int(np.count_nonzero(selected_mask))
    other_pixels = int(np.count_nonzero(other_mask))
    different_pixels = int(np.count_nonzero(selected_mask != other_mask))
    union_extra_pixels = int(np.count_nonzero((other_mask != 0) & (selected_mask == 0)))
    assert selected_pixels > 0 and other_pixels > 0 and different_pixels > 0
    assert union_extra_pixels > 0, "Selected mask must not cover both compartment projections"
    drainer_bodies = _body_ids(sim, root_id)
    drainer_geoms = set(np.flatnonzero(np.isin(sim.model.geom_bodyid, list(drainer_bodies))))
    object_geoms = {}
    for object_name, obj in inner.objects_dict.items():
        bodies = _body_ids(sim, sim.model.body_name2id(obj.root_body))
        for geom_id in np.flatnonzero(np.isin(sim.model.geom_bodyid, list(bodies))):
            object_geoms[int(geom_id)] = object_name
    contacts = []
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        other_geom = second if first in drainer_geoms else first if second in drainer_geoms else None
        if other_geom in object_geoms:
            contacts.append(dict(object=object_geoms[other_geom], penetration_m=max(0., -float(contact.dist))))
    assert not contacts, f"Initial drainer/object interference: {contacts}"
    return dict(
        passed=True, robot_side=side, selected_native_region=selected_name,
        other_native_region=other_name, target_object=target,
        fixture_world_position_m=root_position.tolist(),
        selected_site_world_position_m=selected_pos.tolist(),
        other_site_world_position_m=other_pos.tolist(),
        robot_root_body=robot_root, robot_base_world_position_m=robot_position.tolist(),
        robot_forward_world_axis=robot_rotation[:, 0].tolist(),
        robot_left_world_axis=robot_rotation[:, 1].tolist(),
        selected_goal_initially_false=not selected_truth,
        other_goal_initially_false=not other_truth,
        selected_mask_pixels=selected_pixels, other_mask_pixels=other_pixels,
        mask_xor_pixels=different_pixels, union_pixels_excluded=union_extra_pixels,
        initial_drainer_object_contacts=contacts,
        success_predicate_unmodified=True, native_sites_unmodified=True,
    )
