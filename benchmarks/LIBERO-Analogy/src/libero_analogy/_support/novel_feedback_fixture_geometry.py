"""Physical fixture sites and fixed-fixture placement support.

Install explicitly before constructing initialization and evaluation environments.
Only in-memory object XML site annotations and fixed-root sampler placement are
changed. Meshes, collision geoms, joints and object rotations are not modified.
The drainer-plate and cubby In checks require full collision-envelope containment
and physical parent contact in addition to the native predicate.
No simulator is created by this module.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np


from ..assets import asset_root
ASSET_ROOT = asset_root()
FIXTURE_XML = {
    "short_fridge": "articulated_objects/short_fridge.xml",
    "short_cabinet": "articulated_objects/short_cabinet.xml",
    "white_cabinet": "articulated_objects/white_cabinet.xml",
    "wooden_cabinet": "articulated_objects/wooden_cabinet.xml",
    "microwave": "articulated_objects/microwave.xml",
    **{name: f"turbosquid_objects/{name}/{name}.xml" for name in (
        "white_storage_box", "wooden_shelf", "wooden_two_layer_shelf",
        "bowl_drainer", "dining_set_group", "wooden_tray", "desk_caddy",
    )},
}
AUDITED_ASSET_SHA256 = {
    "short_fridge": "4f928ee9c0dee7d14f2e95aef425a4f713c10c41a590a4407f501f0d5dc3376a",
    "short_cabinet": "410a5854aed734e046a092dbb54ad620abaad6c47ece3081e71afe0d11dc8f93",
    "white_cabinet": "b8ebf670ad0d5533d9fedbfdd7909d263792fc6725d4ca5d4df987660cee264a",
    "wooden_cabinet": "eca251258c7ff6a05fa181a316c767508801812f4ea870921f3ce28fd82ea7af",
    "microwave": "d06c1e44acb830c529a843e566deedfc5693a6e11f68a3d1d59d49021890fbdf",
    "white_storage_box": "db7b4ef7ef60d3a28bac7e6311601d33c42bc96a02da41e408188be8fc0ff544",
    "wooden_shelf": "a5f5662d2873cacd639064bef3802d6e1e763a94c15efd351844a5d855c9c22c",
    "wooden_two_layer_shelf": "f05e448de7985640152d7405f8a9bca65105d065268613ede321b02cb513fb28",
    "bowl_drainer": "b73fea1507c47cd797b4f5a133a7efca7542f1776fc8632fb7a636de20335e9a",
    "dining_set_group": "a80770ab226c5dcaeb2c9b436e13f646fcfe4d8e3e3d38dc1c117ddc89203e92",
    "wooden_tray": "d275a3ac94f12c633194f79248df724fe544ac053eaca7b3ee91e75685ec5dc3",
    "desk_caddy": "267de04e2191517700fced70d9f434958ba88d5e8d449e9def43aeaedd82dd85",
}
SIGNS = np.asarray(list(product((-1., 1.), repeat=3)))


def quat_matrix(quaternion):
    """MuJoCo XML wxyz quaternion, normalized before transforming geometry."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Invalid geometry quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _pose(element):
    if any(key in element.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Fixture geometry requires explicit quaternion or identity")
    return (np.fromstring(element.get("pos", "0 0 0"), sep=" "),
            quat_matrix(np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")))


def collision_boxes(path):
    """Compute q=0 box corners in the object's root frame, including children."""
    root = ET.parse(path).find("./worldbody/body/body[@name='object']")
    if root is None:
        raise ValueError(f"Missing object root: {path}")
    result = []

    def walk(body, position, rotation, names):
        for geom in body.findall("geom"):
            if int(geom.get("contype", "1")) == 0 and int(geom.get("conaffinity", "1")) == 0:
                continue
            if geom.get("type") != "box":
                raise ValueError(f"Non-box collision geometry requires a separate audit: {path}")
            local_pos, local_rot = _pose(geom)
            center, matrix = position + rotation @ local_pos, rotation @ local_rot
            half = np.fromstring(geom.get("size", ""), sep=" ")
            if half.shape != (3,) or np.any(half <= 0):
                raise ValueError(f"Invalid collision box: {path}")
            corners = center + (SIGNS * half) @ matrix.T
            result.append(dict(body_path=names, local_position=local_pos, center=center,
                               rotation=matrix, size=half, lower=corners.min(0),
                               upper=corners.max(0), corners=corners))
        for child in body.findall("body"):
            local_pos, local_rot = _pose(child)
            walk(child, position + rotation @ local_pos, rotation @ local_rot,
                 names + (child.get("name", ""),))

    walk(root, np.zeros(3), np.eye(3), ("object",))
    if not result:
        raise ValueError(f"No collision geometry: {path}")
    return result


def collision_min_z(path):
    return float(min(box["lower"][2] for box in collision_boxes(path)))


def _site(position, half_size, note):
    return dict(position=tuple(float(x) for x in position),
                half_size=tuple(float(x) for x in half_size), note=note)


@lru_cache(maxsize=1)
def functional_site_specs():
    """Conservative physical support/interior annotations, not synthetic floors."""
    cubby = collision_boxes(ASSET_ROOT / FIXTURE_XML["white_storage_box"])
    # Four existing collision panels: two side walls, tilted roof and floor.
    horizontal = sorted((box for box in cubby if box["upper"][2]-box["lower"][2] < .02),
                        key=lambda box: box["center"][2])
    if len(horizontal) != 2:
        raise ValueError("White cubby no longer has its audited floor and roof")
    floor, roof = horizontal
    walls = sorted((box for box in cubby if all(box is not panel for panel in horizontal)),
                   key=lambda box: box["center"][0])
    # Use outer envelopes of slightly tilted floor/roof. The resulting box lies
    # strictly within the actual four-panel tunnel, with both +/-Y ends open.
    lo = np.array([walls[0]["upper"][0], max(floor["lower"][1], roof["lower"][1]), floor["upper"][2]])
    hi = np.array([walls[-1]["lower"][0], min(floor["upper"][1], roof["upper"][1]), roof["lower"][2]])
    # The lower face is the actual floor-collision upper bound, not a raised
    # padding plane: adding 0.5mm there falsely excluded settled cup contacts.
    # Use the 1mm live contact tolerance.
    lo += [.001, .001, 0.]
    hi -= [.001, .001, .0005]
    if np.any(hi <= lo):
        raise ValueError("White storage box has no audited interior volume")

    fridge = collision_boxes(ASSET_ROOT / FIXTURE_XML["short_fridge"])
    fridge_roof = [box for box in fridge if box["body_path"] == ("object", "base")
                   and np.allclose(box["local_position"], [-.00066, .22434, -.00136], atol=1e-7)]
    if len(fridge_roof) != 1:
        raise ValueError("Short fridge roof collision panel changed")
    roof_box = fridge_roof[0]
    center = (roof_box["lower"] + roof_box["upper"]) / 2
    center[2] = roof_box["upper"][2]
    half = (roof_box["upper"] - roof_box["lower"]) / 2
    half[:2] -= .002
    half[2] = .001

    shelf = collision_boxes(ASSET_ROOT / FIXTURE_XML["wooden_two_layer_shelf"])
    shelf_roof = [box for box in shelf if np.allclose(box["local_position"], [.00498, .01635, .20815], atol=1e-7)]
    if len(shelf_roof) != 1:
        raise ValueError("Two-layer shelf roof collision panel changed")
    panel = shelf_roof[0]
    shelf_center = (panel["lower"] + panel["upper"]) / 2
    shelf_center[2] = panel["upper"][2]
    shelf_half = (panel["upper"] - panel["lower"]) / 2
    # Small inset excludes the sloped panel edges, retaining the real broad roof.
    shelf_half[:2] -= .003
    shelf_half[2] = .001
    return {
        "white_storage_box": {"feedback_contain_region": _site((lo+hi)/2, (hi-lo)/2,
            "Conservative open-ended cubby volume between existing floor, roof and side-wall collision boxes.")},
        "short_fridge": {"feedback_top_region": _site(center, half,
            "Actual existing roof collision panel, inset 2mm in XY; not fridge interior or floating placeholder top_site.")},
        "wooden_two_layer_shelf": {"feedback_roof_region": _site(shelf_center, shelf_half,
            "Broad physical roof at collision-panel upper envelope, inset 3mm; not native top_region interior.")},
        "dining_set_group": {"plate_support_region": _site((-.00413, 0., -.0195), (.07432, .075, .001),
            "Exposed central cloth/mat collision support between utensils; On retains parent-contact requirement.")},
    }


def _annotate_instance(instance, category):
    specs = functional_site_specs().get(category, {})
    if not specs:
        return
    roots = [instance.get_obj(), instance.worldbody.find("./body/body")]
    if any(root is None for root in roots):
        raise ValueError(f"Cannot attach feedback sites to {instance.name}")
    for suffix, spec in specs.items():
        name = instance.naming_prefix + suffix
        for root in roots:
            if root.find(f".//site[@name='{name}']") is not None:
                raise ValueError(f"Feedback site already exists: {name}")
            ET.SubElement(root, "site", name=name, type="box", quat="1 0 0 0",
                          pos=" ".join(f"{x:.12g}" for x in spec["position"]),
                          size=" ".join(f"{x:.12g}" for x in spec["half_size"]),
                          rgba="0 0 0 0", group="0")
        # _sites stores unprefixed names; its property prefixes them on access.
        instance._sites.append(suffix)


def full_containment_report(env, object_name, site_name, tolerance=.001):
    """Require every live collision-box corner inside a current native site.

    The 1mm tolerance is only boundary numerical/contact tolerance; it does not
    enlarge the underlying site or substitute root-point containment. A real
    contact with the site's physical parent is also required.
    """
    inner = env
    while not hasattr(inner, "get_object") and hasattr(inner, "env"):
        inner = inner.env
    sim = inner.sim
    obj = inner.get_object(object_name)
    root_id = int(sim.model.body_name2id(obj.root_body))
    selected = {root_id}
    parents = np.asarray(sim.model.body_parentid, dtype=int)
    changed = True
    while changed:
        before = len(selected)
        selected.update(i for i, parent in enumerate(parents) if int(parent) in selected)
        changed = len(selected) != before
    geom_body = np.asarray(sim.model.geom_bodyid, dtype=int)
    collision = ((np.asarray(sim.model.geom_contype) != 0)
                 | (np.asarray(sim.model.geom_conaffinity) != 0))
    geom_ids = np.flatnonzero(np.isin(geom_body, list(selected)) & collision)
    if not len(geom_ids) or np.any(np.asarray(sim.model.geom_type)[geom_ids] != 6):
        raise ValueError(f"Full containment requires audited live collision boxes: {object_name}")
    site_id = int(sim.model.site_name2id(site_name))
    if site_id < 0 or int(sim.model.site_type[site_id]) != 6:
        raise ValueError(f"Full containment requires an existing box site: {site_name}")
    corners = np.concatenate([
        sim.data.geom_xpos[i] + (SIGNS * sim.model.geom_size[i]) @ np.asarray(sim.data.geom_xmat[i]).reshape(3, 3).T
        for i in geom_ids
    ])
    frame = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3)
    local = (corners - sim.data.site_xpos[site_id]) @ frame
    half = np.asarray(sim.model.site_size[site_id], dtype=float)
    margin = np.minimum(local.min(0) + half, half - local.max(0))
    site = inner.object_sites_dict[site_name]
    contact = bool(inner.check_contact(inner.get_object(site.parent_name), obj))
    contained = bool(np.all(margin >= -float(tolerance)))
    return dict(contained=contained, parent_contact=contact, success=contained and contact,
                minimum_margin_xyz_m=margin.tolist(), tolerance_m=float(tolerance),
                collision_geom_count=len(geom_ids), collision_corner_count=len(corners))


def install_feedback_containment_guard():
    """Add containment and contact requirements to cubby and drainer-plate In."""
    from libero.libero.envs.object_states.base_object_states import SiteObjectState

    if getattr(SiteObjectState.check_contain, "_feedback_containment_guard", False):
        return
    original = SiteObjectState.check_contain

    def check_contain(self, other):
        ordinary = original(self, other)
        if not ordinary:
            return False
        plate_drainer = (other.object_name in {"plate_1", "plate_2"}
                         and self.object_name in {"bowl_drainer_1_left_region", "bowl_drainer_1_right_region"})
        cubby = self.object_name == "white_storage_box_1_feedback_contain_region"
        if not (plate_drainer or cubby):
            return ordinary
        return full_containment_report(self.env, other.object_name, self.object_name)["success"]

    check_contain._feedback_containment_guard = True
    SiteObjectState.check_contain = check_contain


def install_feedback_fixture_geometry():
    """Install process-local overlays, identically for build and inference."""
    for category, expected in AUDITED_ASSET_SHA256.items():
        path = ASSET_ROOT / FIXTURE_XML[category]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Feedback fixture asset changed since audit: {path}")
    from libero.libero.envs.objects import articulated_objects, turbosquid_objects
    from libero.libero.envs.regions.base_region_sampler import MultiRegionRandomSampler

    install_feedback_containment_guard()

    for module, class_name, category in (
        (articulated_objects, "ShortFridge", "short_fridge"),
        (turbosquid_objects, "WhiteStorageBox", "white_storage_box"),
        (turbosquid_objects, "WoodenTwoLayerShelf", "wooden_two_layer_shelf"),
        (turbosquid_objects, "DiningSetGroup", "dining_set_group"),
    ):
        cls = getattr(module, class_name)
        if getattr(cls.__init__, "_feedback_geometry", False):
            continue
        original_init = cls.__init__

        def initialize(self, *args, _original=original_init, _category=category, **kwargs):
            _original(self, *args, **kwargs)
            _annotate_instance(self, _category)

        initialize._feedback_geometry = True
        cls.__init__ = initialize

    if getattr(MultiRegionRandomSampler.sample, "_feedback_fixture_support", False):
        return
    # Microwave includes non-box handle geometry; its existing base collision
    # box spans z=0..0.187 and the handle/door do not extend below z=0.
    min_z = {name: (0. if name == "microwave" else collision_min_z(ASSET_ROOT / relative))
             for name, relative in FIXTURE_XML.items()}
    original_sample = MultiRegionRandomSampler.sample

    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        if type(self) is not MultiRegionRandomSampler:
            return result
        own = {obj.name for obj in self.mujoco_objects}
        for name, (position, quaternion, obj) in list(result.items()):
            category = getattr(obj, "category_name", "")
            if name not in own or category not in min_z:
                continue
            if len(self.x_ranges) != 1 or len(self.y_ranges) != 1:
                raise ValueError(f"Feedback fixture {name} must have one exact donor rectangle")
            corrected = (sum(self.x_ranges[0])/2 + float(self.reference_pos[0]),
                         sum(self.y_ranges[0])/2 + float(self.reference_pos[1]),
                         float(self.reference_pos[2]) - min_z[category])
            result[name] = (corrected, quaternion, obj)
        return result

    sample._feedback_fixture_support = True
    # The shared initialization wrapper also calls the older novel installer.
    # Our full replacement covers its categories, so prevent a second wrapper
    # with rounded legacy constants from creating an init/evaluation mismatch.
    sample._novel_scene_fixture_support = True
    MultiRegionRandomSampler.sample = sample
