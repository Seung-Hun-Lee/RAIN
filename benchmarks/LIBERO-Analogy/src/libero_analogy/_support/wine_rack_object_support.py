"""Read-only exact support scoring for Goal wine-rack object substitutions.

Original assets, native-site behavior, model poses and masks remain unchanged.
A transparent site describes the full upper-deck footprint; the native
wine-root strip is logged separately because it is unsuitable for some other
object root offsets. Scoring additionally requires exact upper-deck contact
and absence of gripper contact for five consecutive actual control steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .novel_feedback_fixture_geometry import ASSET_ROOT, collision_boxes, quat_matrix

FIXTURE = "wine_rack_1"
SOURCE_SITE = FIXTURE + "_top_region"
SITE = FIXTURE + "_adapt_top_region"
TASKS = {"GRACK_001": "alphabet_soup_1", "GRACK_002": "ketchup_1", "GRACK_003": "new_salad_dressing_1"}
OBJECTS = tuple(TASKS.values())
HOLD_STEPS = 5
TOLERANCE = .001
MIN_SUPPORT_FORCE = 1e-6
MIN_NORMAL_ALIGNMENT = .9
GRIPPER_TOUCH_TOLERANCE = 1e-7
RACK_XML = ASSET_ROOT / "turbosquid_objects/wine_rack/wine_rack.xml"
DECK_CENTER = np.array([0., -.00610, .25118])
DECK_HALF_SIZE = np.array([.00450, .07894, .13438])


def _object(value):
    if value not in OBJECTS:
        raise ValueError("Only the three requested native object identities are supported")
    return value


def target_xml(object_id):
    kind = _object(object_id).rsplit("_", 1)[0]
    return ASSET_ROOT / f"stable_hope_objects/{kind}/{kind}.xml"


@lru_cache(maxsize=1)
def geometry_spec():
    boxes = collision_boxes(RACK_XML)
    panels = [b for b in boxes if np.allclose(b["center"], DECK_CENTER, atol=1e-8, rtol=0)
              and np.allclose(b["size"], DECK_HALF_SIZE, atol=1e-8, rtol=0)]
    if len(boxes) != 9 or len(panels) != 1:
        raise RuntimeError("Expected the original rack's nine boxes and unique top deck")
    panel = panels[0]
    normal = -panel["rotation"][:, 0]
    if not np.allclose(normal, [0., -.51917052, .85467068], atol=1e-8, rtol=0):
        raise RuntimeError("The original rack top deck orientation changed")
    targets = {}
    for object_id in OBJECTS:
        path = target_xml(object_id)
        target = collision_boxes(path)
        low = np.min([b["lower"] for b in target], axis=0)
        high = np.max([b["upper"] for b in target], axis=0)
        targets[object_id] = dict(asset_xml=str(path), asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            collision_box_count=len(target), root_local_aabb_low_m=low.tolist(),
            root_local_aabb_high_m=high.tolist(), root_local_aabb_dimensions_m=(high-low).tolist())
    surface_axes = np.column_stack([panel["rotation"][:,2], panel["rotation"][:,1], normal])
    quaternion = Rotation.from_matrix(surface_axes).as_quat()[[3,0,1,2]]
    return dict(fixture_id=FIXTURE, source_native_destination=SOURCE_SITE,
        adapted_semantic_destination=SITE, mask_scope="original_whole_rack_body",
        asset_xml=str(RACK_XML), asset_sha256=hashlib.sha256(RACK_XML.read_bytes()).hexdigest(),
        rack_collision_box_count=len(boxes), target_assets=targets,
        top_deck=dict(collision_center_m=panel["center"].tolist(),
            collision_half_size_m=panel["size"].tolist(), collision_rotation=panel["rotation"].tolist(),
            upper_face_center_m=(panel["center"]+normal*panel["size"][0]).tolist(),
            upper_face_normal=normal.tolist(), normal_axis=0, normal_sign=-1,
            upper_face_full_dimensions_m=(2*panel["size"][[2, 1]]).tolist()),
        semantic_site=dict(name=SITE, position=(panel["center"]+normal*panel["size"][0]).tolist(),
            quaternion_wxyz=quaternion.tolist(), axes=surface_axes.tolist(),
            half_size_m=[float(panel["size"][2]),float(panel["size"][1]),.002],
            annotation_only=True, under_transform="world_delta @ actual_site_axes",
            maximum_body_root_height_m=.2),
        legacy_native_predicate_required=False, annotated_predicate_required=True, positive_normal_force_required=True,
        gripper_contact_forbidden=True, hold_control_steps=HOLD_STEPS,
        projected_point_semantics="Native object root/body origin, not collision or mass centroid",
        original_assets_modified=False, original_native_predicate_modified=False,
        new_destination_semantics="Actual full original upper-deck footprint; legacy wine-root strip auxiliary only")


def annotated_region_under(site_position, site_axes, point):
    """Check the point against the annotated support footprint."""
    site = geometry_spec()["semantic_site"]
    local = (np.asarray(point)-site_position)@np.asarray(site_axes).reshape(3,3)
    return bool(np.all(np.abs(local[:2]) <= np.asarray(site["half_size_m"][:2])+TOLERANCE)
                and -TOLERANCE <= local[2] <= site["maximum_body_root_height_m"])


def install_rack_adapt_geometry():
    """Add a transparent semantic site in memory, preserving all source assets.

    SiteObject.under uses the forward rotation in place of its inverse;
    the annotated site uses the actual orthonormal frame. Native site behavior
    is untouched, and the native wine-region truth is logged separately.
    """
    from libero.libero.envs.objects.turbosquid_objects import WineRack
    from libero.libero.envs.objects.site_object import SiteObject
    site = geometry_spec()["semantic_site"]
    if not getattr(WineRack.__init__, "_rack_adapt_semantic_site", False):
        previous = WineRack.__init__

        def initialize(self,*args,**kwargs):
            previous(self,*args,**kwargs)
            roots=(self.get_obj(),self.worldbody.find("./body/body"))
            if any(root is None for root in roots):
                raise RuntimeError("Missing original rack object roots")
            suffix="adapt_top_region"
            name=self.naming_prefix+suffix
            for root in roots:
                if root.find(f".//site[@name='{name}']") is not None:
                    raise RuntimeError("The rack semantic site already exists")
                ET.SubElement(root,"site",name=name,type="box",
                    pos=" ".join(f"{x:.15g}" for x in site["position"]),
                    quat=" ".join(f"{x:.15g}" for x in site["quaternion_wxyz"]),
                    size=" ".join(f"{x:.15g}" for x in site["half_size_m"]),rgba="0 0 0 0",group="0")
            self._sites.append(suffix)

        initialize._rack_adapt_semantic_site=True
        WineRack.__init__=initialize
    if not getattr(SiteObject.under,"_rack_adapt_semantic_site",False):
        original=SiteObject.under

        def under(self,this_position,this_mat,other_position,other_height=.10):
            if self.name == SITE:
                if not np.allclose(self.size,site["half_size_m"],atol=1e-9,rtol=0):
                    raise RuntimeError("The exact new rack footprint site dimensions changed")
                return annotated_region_under(this_position,this_mat,other_position)
            return original(self,this_position,this_mat,other_position,other_height)

        under._rack_adapt_semantic_site=True
        SiteObject.under=under


def classify_support_contact(contact_position, toward_object_normal, panel_center, panel_rotation,
                             panel_half_size, is_exact_panel):
    """Accept only force from the original slanted top-deck upper face."""
    axes = np.asarray(panel_rotation).reshape(3, 3)
    half = np.asarray(panel_half_size)
    local = (np.asarray(contact_position)-panel_center) @ axes
    height_error = -local[0]-half[0]
    within = bool(np.all(np.abs(local[[2, 1]]) <= half[[2, 1]] + TOLERANCE))
    alignment = float(np.asarray(toward_object_normal) @ -axes[:, 0])
    exact_face = bool(is_exact_panel and abs(height_error) <= TOLERANCE
                      and within and alignment >= MIN_NORMAL_ALIGNMENT)
    return dict(exact_top_deck_upper_face=exact_face, contact_deck_local_m=local.tolist(),
        upper_face_height_error_m=float(height_error), upward_normal_alignment=alignment,
        contact_inside_deck_footprint=within)


def strict_instant(facts):
    return bool(facts["annotated_on"] and facts["exact_top_deck_support"]
                and facts["body_center_over_deck"] and facts["body_center_above_deck"]
                and not facts["any_gripper_contact"])


def match_collision_box_multiset(expected, actual):
    """Match all native boxes bijectively, including intentionally repeated boxes.

    Some source XMLs contain geometrically identical collision boxes. Their
    identities are interchangeable, but their multiplicity is not: each source
    box consumes exactly one live geom.
    """
    if len(expected) != len(actual):
        raise RuntimeError("Live collision box multiplicity differs from original asset")
    remaining = {int(box["id"]): box for box in actual}
    if len(remaining) != len(actual):
        raise RuntimeError("Repeated live geom identity in collision audit")
    matched = []
    for box in expected:
        matches = sorted(identity for identity, live in remaining.items()
            if all(np.allclose(live[key], box[key], atol=1e-7, rtol=0)
                   for key in ("center", "size", "rotation")))
        if not matches:
            raise RuntimeError("Live collision box differs from original asset: expected="
                + repr({key:np.asarray(box[key]).tolist() for key in ("center", "size", "rotation")})
                + " remaining=" + repr([{key:(int(live[key]) if key == "id" else np.asarray(live[key]).tolist())
                    for key in ("id", "center", "size", "rotation")} for live in remaining.values()]))
        chosen = matches[0]
        matched.append(chosen)
        del remaining[chosen]
    if remaining:
        raise RuntimeError("Unaudited live collision box remains")
    return matched


def geom_box_in_root_frame(model, geom_id, root_body_id):
    """Accumulate static child-body poses; geom_pos is not always root-local."""
    center = np.asarray(model.geom_pos[geom_id]).copy()
    rotation = quat_matrix(model.geom_quat[geom_id])
    body = int(model.geom_bodyid[geom_id])
    visited = set()
    while body != int(root_body_id):
        if body <= 0 or body in visited:
            raise RuntimeError("Collision geom is not descended from its audited root")
        if int(model.body_jntnum[body]) != 0:
            raise RuntimeError("An audited rigid target/rack contains an unexpected articulated child")
        visited.add(body)
        axes = quat_matrix(model.body_quat[body])
        center = np.asarray(model.body_pos[body])+axes@center
        rotation = axes@rotation
        body = int(model.body_parentid[body])
    return dict(id=int(geom_id), center=center, rotation=rotation,
                size=np.asarray(model.geom_size[geom_id]).copy())


@dataclass
class RackPlacementObserver:
    object_id: str
    hold_steps: int = HOLD_STEPS
    last_step: int = -1
    consecutive: int = 0
    maximum_consecutive: int = 0
    records: list = field(default_factory=list)

    def __post_init__(self):
        _object(self.object_id)
        if self.hold_steps != HOLD_STEPS:
            raise ValueError("Exactly five consecutive actual control observations are required")

    def update(self, step, snapshot):
        if isinstance(step, bool) or int(step) != step or int(step) != self.last_step+1:
            raise RuntimeError("Every actual control must be observed once, starting at zero")
        if snapshot.get("object_id") != self.object_id:
            raise ValueError("Observed target identity changed")
        instant = strict_instant(snapshot)
        if step == 0 and (instant or snapshot["native_on"] or snapshot["annotated_on"]):
            raise RuntimeError("The rack goal must initially be false")
        self.consecutive = self.consecutive+1 if instant else 0
        self.maximum_consecutive = max(self.maximum_consecutive, self.consecutive)
        self.last_step = int(step)
        record = dict(snapshot)
        record.update(control_step=int(step), strict_instant=instant,
                      consecutive_supported_steps=self.consecutive)
        self.records.append(record)

    @property
    def complete(self):
        return self.consecutive >= self.hold_steps

    def as_dict(self):
        return dict(object_id=self.object_id, hold_control_steps=self.hold_steps,
            strict_released_supported_success=self.complete, native_predicate_required=False,
            original_native_on_auxiliary_only=True, annotated_predicate_required=True,
            projected_point_semantics="Native object root/body origin, not collision or mass centroid",
            actual_control_steps_observed=max(0, self.last_step), observations=len(self.records),
            maximum_consecutive_supported_steps=self.maximum_consecutive,
            final_consecutive_supported_steps=self.consecutive, records=self.records,
            velocity_gate_applied=False,
            stability_scope="Five consecutive full-deck-region On, released, exactly top-deck-supported controls")


class RackPlacementProbe:
    def __init__(self, env, object_id):
        self.object_id = _object(object_id)
        inner = env
        while not hasattr(inner, "objects_dict") and hasattr(inner, "env"):
            inner = inner.env
        self.inner, self.sim = inner, inner.sim
        sim = self.sim
        parents = np.asarray(sim.model.body_parentid)

        def subtree_geoms(name):
            root = int(sim.model.body_name2id(name))
            if root < 0:
                raise RuntimeError("Missing observed body: " + name)
            bodies = {root}
            while True:
                expanded = bodies | {int(i) for i, parent in enumerate(parents) if int(parent) in bodies}
                if expanded == bodies:
                    return set(np.flatnonzero(np.isin(sim.model.geom_bodyid, list(bodies))).tolist())
                bodies = expanded

        object_root = inner.get_object(self.object_id).root_body
        fixture_root = inner.get_object(FIXTURE).root_body
        self.object_body = int(sim.model.body_name2id(object_root))
        self.fixture_body = int(sim.model.body_name2id(fixture_root))
        self.object_geoms = subtree_geoms(object_root)
        self.fixture_geoms = subtree_geoms(fixture_root)
        collision = lambda i: bool(sim.model.geom_contype[i] or sim.model.geom_conaffinity[i])
        self.object_collision_geoms = sorted(i for i in self.object_geoms if collision(i))
        fixture_collision = sorted(i for i in self.fixture_geoms if collision(i))
        # Check every native collision box, including the support deck.
        for source, actual, root in ((target_xml(self.object_id), self.object_collision_geoms, self.object_body),
                                     (RACK_XML, fixture_collision, self.fixture_body)):
            expected = collision_boxes(source)
            if len(expected) != len(actual) or any(int(sim.model.geom_type[i]) != 6 for i in actual):
                raise RuntimeError("Live collision count/type differs from original asset: " + str(source))
            match_collision_box_multiset(expected, [geom_box_in_root_frame(sim.model, i, root) for i in actual])
        deck = geometry_spec()["top_deck"]
        matches = [i for i in fixture_collision
            if np.allclose(sim.model.geom_pos[i], deck["collision_center_m"], atol=1e-7, rtol=0)
            and np.allclose(sim.model.geom_size[i], deck["collision_half_size_m"], atol=1e-7, rtol=0)]
        if len(matches) != 1:
            raise RuntimeError("Could not identify the original rack top deck")
        self.deck_geom = matches[0]
        self.gripper_geoms = set()
        for robot in inner.robots:
            grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else [robot.gripper]
            for gripper in grippers:
                self.gripper_geoms.update(subtree_geoms(gripper.root_body))
                self.gripper_geoms.update(int(sim.model.geom_name2id(name)) for name in gripper.contact_geoms)
        if not self.gripper_geoms or any(i < 0 or i >= sim.model.ngeom for i in self.gripper_geoms):
            raise RuntimeError("Missing valid gripper collision identities")

    def snapshot(self):
        import mujoco
        sim = self.sim
        model, data = getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)
        center = np.asarray(sim.data.geom_xpos[self.deck_geom])
        axes = np.asarray(sim.data.geom_xmat[self.deck_geom]).reshape(3, 3)
        half = np.asarray(sim.model.geom_size[self.deck_geom])
        normal = -axes[:, 0]
        if normal[2] < .8:
            raise RuntimeError("The original rack must remain upright with its native angled deck")
        contacts = []
        for index in range(int(sim.data.ncon)):
            contact = sim.data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a in self.object_geoms else a if b in self.object_geoms else None
            if other is None or other in self.object_geoms:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            toward_object = np.asarray(contact.frame[:3]) * (-1 if a in self.object_geoms else 1)
            face = classify_support_contact(contact.pos, toward_object, center, axes, half, other == self.deck_geom)
            contacts.append(dict(other_geom_id=other, other_geom_name=sim.model.geom_id2name(other),
                rack=other in self.fixture_geoms, gripper=other in self.gripper_geoms,
                exact_top_deck=other == self.deck_geom, normal_force_n=float(force[0]),
                distance_m=float(contact.dist), contact_geom1_id=a, contact_geom2_id=b,
                target_is_geom1=a in self.object_geoms,
                contact_normal_geom1_to_geom2_world=np.asarray(contact.frame[:3]).tolist(),
                toward_object_normal_world=toward_object.tolist(),
                contact_world_position_m=np.asarray(contact.pos).tolist(), **face))
        support = any(c["exact_top_deck_upper_face"] and c["normal_force_n"] > MIN_SUPPORT_FORCE for c in contacts)
        held = any(c["gripper"] and (c["normal_force_n"] > MIN_SUPPORT_FORCE or c["distance_m"] <= GRIPPER_TOUCH_TOLERANCE)
                   for c in contacts)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.object_body, velocity, 0)
        position=np.asarray(sim.data.body_xpos[self.object_body])
        body_local=(position-center)@axes
        footprint_margin=half[[2,1]]-np.abs(body_local[[2,1]])
        root_height=-body_local[0]-half[0]
        facts = dict(object_id=self.object_id,
            projected_point_semantics="Native object root/body origin, not collision or mass centroid",
            native_on=bool(self.inner._eval_predicate(("on", self.object_id, SOURCE_SITE))),
            annotated_on=bool(self.inner._eval_predicate(("on", self.object_id, SITE))),
            body_center_over_deck=bool(np.all(footprint_margin>=-TOLERANCE)),
            body_center_above_deck=bool(root_height>=-TOLERANCE),
            body_center_footprint_margin_m=footprint_margin.tolist(),
            body_center_height_above_deck_m=float(root_height),
            exact_top_deck_support=bool(support), any_gripper_contact=bool(held), released=not held,
            contacts=contacts, object_position_m=np.asarray(sim.data.body_xpos[self.object_body]).tolist(),
            support_outward_normal_world=normal.tolist(),
            angular_speed_rad_s=float(np.linalg.norm(velocity[:3])), linear_speed_m_s=float(np.linalg.norm(velocity[3:])))
        facts["strict_instant"] = strict_instant(facts)
        return facts

    def geometry(self):
        result = dict(geometry_spec())
        result["observed_object_id"] = self.object_id
        result["live_binding"] = dict(object_body_id=self.object_body, fixture_body_id=self.fixture_body,
            object_collision_geom_ids=self.object_collision_geoms, top_deck_geom_id=self.deck_geom,
            top_deck_geom_name=self.sim.model.geom_id2name(self.deck_geom),
            gripper_geom_names=sorted(self.sim.model.geom_id2name(i) for i in self.gripper_geoms))
        return result
