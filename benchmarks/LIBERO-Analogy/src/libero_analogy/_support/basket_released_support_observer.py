"""Future-only basket scoring. All simulator operations here are read-only.

Native ordered predicates remain authoritative. This additional final gate
requires five actual controls of released, anchored and settled placements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

CONTRACT = dict(
    basket_completion_mode="ordered_native_and_released_supported_settled_v1",
    support_hold_control_steps=5,
    basket_support_graph_mode="required_targets_to_exact_basket_base",
    settle_linear_speed_max_m_s=.02,
    settle_angular_speed_max_rad_s=.5,
)
MIN_FORCE_N = 1e-6
TOUCH_TOLERANCE_M = 1e-7
MIN_UP_NORMAL = .5
FACE_TOLERANCE_M = .001


def validate_contract(meta, rules):
    for key, expected in CONTRACT.items():
        if meta.get(key) != expected or isinstance(meta.get(key), bool):
            raise RuntimeError(f"Explicit future basket contract required: {key}={expected!r}")
    atoms = rules.get("ordered_event_atoms") or []
    if (len(atoms) < 2 or rules.get("required_goal_atoms") != atoms
            or meta.get("canonical_goal_atoms") != atoms
            or rules.get("custom_eval_needed") is not True
            or rules.get("order_sensitive") is not True
            or rules.get("requires_transition") is not True
            or rules.get("strict_event_order") is not True
            or rules.get("final_success_requires_all_bddl_goals") is not True
            or rules.get("environment_ignore_done") is not True
            or rules.get("downstream_preservation_required") is not True
            or rules.get("final_tc_gate") is not False
            or rules.get("compose_final_tc_gate") is not False
            or rules.get("continue_after_success")
            or rules.get("forbidden_goal_atoms")):
        raise RuntimeError("Basket gate requires unchanged strict ordered native final rules")
    pairs = []
    for atom in atoms:
        match = re.fullmatch(r"in\(\s*([a-zA-Z0-9_]+)\s*,\s*(basket_[12])_contain_region\s*\)", str(atom))
        if not match:
            raise RuntimeError("Future basket gate only accepts native basket In placements")
        pairs.append(match.groups())
    if len({obj for obj, _ in pairs}) != len(pairs) or any(obj.startswith("basket_") for obj, _ in pairs):
        raise RuntimeError("Each basket goal must place a distinct non-basket object")
    return dict(pairs)


def _settled(row):
    speeds = [row["linear_speed_m_s"], row["angular_speed_rad_s"]]
    return bool(all(math.isfinite(float(x)) and float(x) >= 0 for x in speeds)
                and speeds[0] <= CONTRACT["settle_linear_speed_max_m_s"]
                and speeds[1] <= CONTRACT["settle_angular_speed_max_rad_s"])


def physical_final(snapshot, destinations):
    """Evaluate a directed support graph; cycles cannot anchor themselves."""
    entities = snapshot["entities"]
    baskets = set(destinations.values())
    expected = set(destinations) | baskets
    if set(entities) != expected:
        raise RuntimeError("Physical snapshot does not contain exactly all targets and baskets")
    safe = {name: _settled(row) and not row["any_robot_contact"] for name, row in entities.items()}
    anchors = {name for name in baskets if safe[name] and entities[name]["upright"]
               and entities[name]["original_support_contact"]}
    edges = {name: set() for name in destinations}
    for row in snapshot["support_edges"]:
        child, parent = row["child"], row["support"]
        if child not in destinations or not row["positive_upward_contact"]:
            continue
        if parent == destinations[child] and row["exact_basket_base_upper_face"]:
            edges[child].add(parent)
        elif parent in destinations and destinations[parent] == destinations[child]:
            edges[child].add(parent)
    supported = set(anchors)
    while True:
        added = {name for name in destinations if safe[name] and entities[name]["native_in"]
                 and any(parent in supported for parent in edges[name])} - supported
        if not added:
            break
        supported.update(added)
    facts = {name: bool(safe[name] and entities[name]["native_in"] and name in supported)
             for name in destinations}
    return bool(all(facts.values())), dict(target_strict_instant=facts,
        anchored_baskets=sorted(anchors), supported_entities=sorted(supported))


@dataclass
class BasketSupportObserver:
    destinations: dict
    last_step: int | None = None
    consecutive: int = 0
    maximum: int = 0
    first_complete_step: int | None = None
    records: list = field(default_factory=list)

    @property
    def complete(self):
        return self.consecutive >= CONTRACT["support_hold_control_steps"]

    def update(self, step, snapshot):
        step = int(step)
        if self.last_step == step:
            if self.records[-1]["snapshot"] != snapshot:
                raise RuntimeError("Duplicate control has inconsistent basket facts")
            return
        if step != (0 if self.last_step is None else self.last_step + 1):
            raise RuntimeError("Basket observer missed an actual control")
        instant, facts = physical_final(snapshot, self.destinations)
        if step == 0 and any(snapshot["entities"][name]["native_in"] for name in self.destinations):
            raise RuntimeError("Basket placement starts native-true")
        # The initial observation is not an executed policy control.
        self.consecutive = self.consecutive + 1 if instant and step > 0 else 0
        self.maximum = max(self.maximum, self.consecutive)
        if self.complete and self.first_complete_step is None:
            self.first_complete_step = step
        self.last_step = step
        self.records.append(dict(control_step=step, snapshot=snapshot, strict_instant=instant,
            consecutive_supported_settled_controls=self.consecutive, **facts))

    def as_dict(self):
        return dict(contract=CONTRACT, final_hold_complete=self.complete,
            final_consecutive_controls=self.consecutive, maximum_consecutive_controls=self.maximum,
            first_complete_step=self.first_complete_step, actual_control_steps_observed=self.last_step,
            records=self.records, native_order_replaced=False, velocity_gate_applied=True,
            action_generation_modified=False, tc_switching_modified=False,
            additional_wait_controls=0, policy_control_budget_extended=False)


def _subtree(model, name):
    root = int(model.body_name2id(name))
    if root < 0:
        raise RuntimeError(f"Missing body: {name}")
    bodies = {root}
    while True:
        expanded = bodies | {i for i, parent in enumerate(model.body_parentid) if int(parent) in bodies}
        if bodies == expanded:
            break
        bodies = expanded
    return root, {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies
                  and (model.geom_contype[i] or model.geom_conaffinity[i])}


class BasketContactProbe:
    """Exact native basket bases, force contacts, robot subtrees and velocities."""
    def __init__(self, env, destinations):
        inner = env
        while not hasattr(inner, "objects_dict") and hasattr(inner, "env"):
            inner = inner.env
        self.inner, self.sim = inner, inner.sim
        self.destinations = dict(destinations)
        self.baskets = set(destinations.values())
        self.roots, self.geoms, self.owners = {}, {}, {}
        m = self.sim.model
        for name in sorted(set(destinations) | self.baskets):
            self.roots[name], self.geoms[name] = _subtree(m, inner.get_object(name).root_body)
            if not self.geoms[name]:
                raise RuntimeError(f"Entity has no collision geometry: {name}")
            for geom in self.geoms[name]:
                if geom in self.owners:
                    raise RuntimeError("Overlapping collision ownership")
                self.owners[geom] = name
        self.robot_geoms = set()
        for robot in inner.robots:
            self.robot_geoms.update(_subtree(m, robot.robot_model.root_body)[1])
            grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else [robot.gripper]
            for gripper in grippers:
                self.robot_geoms.update(_subtree(m, gripper.root_body)[1])
        if not self.robot_geoms:
            raise RuntimeError("Missing robot/gripper collision identities")
        from libero.libero import get_libero_path
        asset = Path(get_libero_path("assets")) / "stable_scanned_objects/basket/basket.xml"
        if not asset.exists():
            raise RuntimeError("Cannot validate exact original basket asset")
        collision = [g for g in ET.parse(asset).getroot().findall(".//geom")
                     if g.get("type") == "box" and not (g.get("contype") == "0" and g.get("conaffinity") == "0")]
        self.asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
        self.base_geoms, self.base_faces = {}, {}
        for name in self.baskets:
            actual = sorted(self.geoms[name])
            if len(actual) != len(collision) or any(int(m.geom_type[g]) != 6 for g in actual):
                raise RuntimeError("Basket live collision count/type differs from native asset")
            unmatched = set(actual)
            matched = []
            for xml in collision:
                pos = np.fromstring(xml.get("pos", "0 0 0"), sep=" ")
                size = np.fromstring(xml.get("size"), sep=" ")
                quat = np.fromstring(xml.get("quat", "1 0 0 0"), sep=" ")
                quat /= np.linalg.norm(quat)
                hits = [g for g in unmatched if np.allclose(m.geom_pos[g], pos, atol=1e-7, rtol=0)
                        and np.allclose(m.geom_size[g], size, atol=1e-7, rtol=0)
                        and abs(float(np.dot(m.geom_quat[g], quat))) > 1-1e-7]
                if len(hits) != 1:
                    raise RuntimeError("Basket collision pose/size differs from original asset")
                matched.append(hits[0]); unmatched.remove(hits[0])
            base_index = min(range(len(collision)), key=lambda i: float(collision[i].get("pos").split()[2]))
            self.base_geoms[name] = matched[base_index]
            import mujoco
            base = self.base_geoms[name]
            local_axes = np.zeros(9)
            mujoco.mju_quat2Mat(local_axes, np.asarray(m.geom_quat[base]))
            thin = int(np.argmin(m.geom_size[base]))
            native_z = local_axes.reshape(3, 3)[2, thin]
            if abs(native_z) < .9:
                raise RuntimeError("Native basket bottom lacks a root-up thin face")
            self.base_faces[name] = (thin, 1 if native_z > 0 else -1)
        # Freeze exact floor/table supports at the ordinary post-warmup state.
        self.anchor_geoms = {name: set() for name in self.baskets}
        for c in self._contacts():
            for name in self.baskets:
                own = self.geoms[name]
                if c["geom1"] in own and c["geom2"] not in own:
                    other, up = c["geom2"], -c["normal_world"][2]
                elif c["geom2"] in own and c["geom1"] not in own:
                    other, up = c["geom1"], c["normal_world"][2]
                else:
                    continue
                if c["normal_force_n"] > MIN_FORCE_N and up >= MIN_UP_NORMAL and self._floor_or_table(other):
                    self.anchor_geoms[name].add(other)
        if any(not value for value in self.anchor_geoms.values()):
            raise RuntimeError("Basket lacks an exact positive-force initial floor/table anchor")

    def _floor_or_table(self, geom):
        m = self.sim.model
        body = int(m.geom_bodyid[geom])
        names = [str(m.geom_id2name(geom))]
        while body:
            if m.body_jntnum[body]:
                return False
            names.append(str(m.body_id2name(body)))
            body = int(m.body_parentid[body])
        allowed = {"floor", "floor_collision", "table", "table_collision", "main_table",
                   "kitchen_table", "living_room_table", "living_room_table_col"}
        return bool(allowed.intersection(names))

    def _contacts(self):
        import mujoco
        m, d = self.sim.model, self.sim.data
        rows = []
        for index in range(int(d.ncon)):
            c = d.contact[index]
            a, b = int(c.geom1), int(c.geom2)
            if a not in self.owners and b not in self.owners:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(getattr(m, "_model", m), getattr(d, "_data", d), index, force)
            rows.append(dict(geom1=a, geom2=b, geom1_name=str(m.geom_id2name(a)),
                geom2_name=str(m.geom_id2name(b)), normal_force_n=float(force[0]),
                distance_m=float(c.dist), normal_world=np.asarray(c.frame[:3]).tolist(),
                contact_position_m=np.asarray(c.pos).tolist()))
        return rows

    def _base_face(self, basket, other, point, toward_child):
        if other != self.base_geoms[basket]:
            return False
        m, d = self.sim.model, self.sim.data
        axes = np.asarray(d.geom_xmat[other]).reshape(3, 3)
        half = np.asarray(m.geom_size[other])
        thin, sign = self.base_faces[basket]
        local = (np.asarray(point)-np.asarray(d.geom_xpos[other])) @ axes
        return bool(sign*axes[2, thin] >= .9 and toward_child[2] >= MIN_UP_NORMAL
                    and abs(local[thin]-sign*half[thin]) <= FACE_TOLERANCE_M
                    and np.all(np.abs(local) <= half + FACE_TOLERANCE_M))

    def snapshot(self):
        import mujoco
        m, d = self.sim.model, self.sim.data
        entities = {}
        for name, root in self.roots.items():
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(getattr(m, "_model", m), getattr(d, "_data", d),
                                    mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
            entities[name] = dict(world_position_m=np.asarray(d.body_xpos[root]).tolist(),
                linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
                angular_speed_rad_s=float(np.linalg.norm(velocity[:3])), any_robot_contact=False)
            if name in self.destinations:
                entities[name]["native_in"] = bool(self.inner._eval_predicate(
                    ("in", name, self.destinations[name]+"_contain_region")))
            else:
                base = self.base_geoms[name]
                axes = np.asarray(d.geom_xmat[base]).reshape(3, 3)
                thin, sign = self.base_faces[name]
                entities[name].update(upright=bool(sign*axes[2, thin] >= .9), original_support_contact=False)
        contacts, edges = self._contacts(), []
        for c in contacts:
            for own, other, sign in ((c["geom1"], c["geom2"], -1), (c["geom2"], c["geom1"], 1)):
                name = self.owners.get(own)
                if name is None or self.owners.get(other) == name:
                    continue
                touch = c["normal_force_n"] > MIN_FORCE_N or c["distance_m"] <= TOUCH_TOLERANCE_M
                if other in self.robot_geoms and touch:
                    entities[name]["any_robot_contact"] = True
                toward = sign*np.asarray(c["normal_world"])
                upward = bool(c["normal_force_n"] > MIN_FORCE_N and toward[2] >= MIN_UP_NORMAL)
                if name in self.baskets and other in self.anchor_geoms[name] and upward:
                    entities[name]["original_support_contact"] = True
                parent = self.owners.get(other)
                if name in self.destinations and parent and upward:
                    edges.append(dict(child=name, support=parent, positive_upward_contact=True,
                        exact_basket_base_upper_face=(parent in self.baskets and
                            self._base_face(parent, other, c["contact_position_m"], toward)),
                        normal_force_n=c["normal_force_n"], normal_toward_child_world=toward.tolist(),
                        child_geom=own, support_geom=other))
        return dict(entities=entities, support_edges=edges, contacts=contacts)

    def geometry(self):
        m = self.sim.model
        return dict(original_basket_asset_sha256=self.asset_sha256,
            target_destinations=self.destinations,
            basket_base_geoms={k:dict(id=v, name=str(m.geom_id2name(v))) for k,v in self.base_geoms.items()},
            basket_native_upper_face_axes={k:dict(axis=v[0], sign=v[1]) for k,v in self.base_faces.items()},
            original_floor_table_anchor_geoms={k:[dict(id=g,name=str(m.geom_id2name(g))) for g in sorted(v)]
                                                for k,v in self.anchor_geoms.items()},
            robot_collision_geom_names=[str(m.geom_id2name(g)) for g in sorted(self.robot_geoms)],
            contact_force_min_n=MIN_FORCE_N, upward_normal_min=MIN_UP_NORMAL,
            basket_upper_face_tolerance_m=FACE_TOLERANCE_M,
            robot_touch_tolerance_m=TOUCH_TOLERANCE_M)
