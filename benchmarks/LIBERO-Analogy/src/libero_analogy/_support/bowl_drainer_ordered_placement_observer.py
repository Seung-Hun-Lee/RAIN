"""Observer-only scoring for two ordered bowl-drainer placements.

This module never moves the robot, advances RAIN's action index, changes a
predicate, or edits simulator assets.  ``native_in`` and ``released_supported``
are explicit alternative scoring modes, not interchangeable success labels.
The latter uses exact bottom-box contact and absence of any gripper contact;
it does not require the whole object to fit beneath the native site's top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .composition_batch_order import OrderedEvents, parse_atom


NATIVE_REGIONS = (
    "bowl_drainer_1_left_region",
    "bowl_drainer_1_right_region",
)
COMPLETION_MODES = ("native_in", "released_supported")
MIN_SUPPORT_NORMAL_FORCE_N = 1e-6
GRIPPER_CONTACT_DISTANCE_TOLERANCE_M = 1e-7


def physical_contact_flags(contacts):
    """A force-free near-contact is not support; any actual finger touch excludes release."""
    supported = any(row["native_base"] and row["normal_force_n"] > MIN_SUPPORT_NORMAL_FORCE_N
                    for row in contacts)
    gripper = any(row["gripper"] and (row["normal_force_n"] > MIN_SUPPORT_NORMAL_FORCE_N
                                     or row["distance_m"] <= GRIPPER_CONTACT_DISTANCE_TOLERANCE_M)
                  for row in contacts)
    return supported, gripper


@dataclass
class OrderedPlacementObserver:
    atoms: list[str]
    support_hold_control_steps: int = 5
    native: OrderedEvents = field(init=False)
    released_supported: OrderedEvents = field(init=False)
    last_step: int | None = field(default=None, init=False)
    last_inputs: tuple | None = field(default=None, init=False)
    support_runs: list[int] = field(default_factory=lambda: [0, 0], init=False)
    support_run_maximum: list[int] = field(default_factory=lambda: [0, 0], init=False)
    selected_final: list[bool] = field(default_factory=lambda: [False, False], init=False)
    opposite_final: list[bool] = field(default_factory=lambda: [False, False], init=False)
    supported_final: list[bool] = field(default_factory=lambda: [False, False], init=False)
    supported_instant_final: list[bool] = field(default_factory=lambda: [False, False], init=False)
    opposite_first_steps: list[int | None] = field(default_factory=lambda: [None, None], init=False)
    both_native_first_step: int | None = field(default=None, init=False)
    both_supported_first_step: int | None = field(default=None, init=False)
    changes: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        parsed = [parse_atom(atom) for atom in self.atoms]
        if (len(parsed) != 2 or any(len(atom) != 3 or atom[0] != "in" for atom in parsed)
                or len({atom[2] for atom in parsed}) != 2
                or set(atom[2] for atom in parsed) != set(NATIVE_REGIONS)
                or parsed[0][1] == parsed[1][1]):
            raise ValueError("Require two different objects and one use of each native left/right compartment")
        if not isinstance(self.support_hold_control_steps, int) or self.support_hold_control_steps < 1:
            raise ValueError("support_hold_control_steps must be a positive integer")
        self.native = OrderedEvents(self.atoms)
        self.released_supported = OrderedEvents(self.atoms)

    def update(self, step: int, selected, opposite, base_contacts, gripper_contacts):
        """Observe each control step once; repeated identical observations are safe."""
        values = tuple(tuple(bool(value) for value in row)
                       for row in (selected, opposite, base_contacts, gripper_contacts))
        if any(len(row) != 2 for row in values):
            raise ValueError("Two object observations are required in instruction order")
        step = int(step)
        if self.last_step is not None and step == self.last_step:
            if values != self.last_inputs:
                raise RuntimeError("The same control step has inconsistent physical observations")
            return
        expected = 0 if self.last_step is None else self.last_step + 1
        if step != expected:
            raise RuntimeError(f"Missing control-step observation: expected {expected}, got {step}")
        selected, opposite, base_contacts, gripper_contacts = values
        if step == 0 and (any(selected) or any(opposite)):
            raise RuntimeError("A target starts in one of the drainer compartments")
        instant = [inside and base and not gripper
                   for inside, base, gripper in zip(selected, base_contacts, gripper_contacts)]
        for index, value in enumerate(instant):
            self.support_runs[index] = self.support_runs[index] + 1 if value else 0
            self.support_run_maximum[index] = max(self.support_run_maximum[index], self.support_runs[index])
        supported = [count >= self.support_hold_control_steps for count in self.support_runs]
        self.native.update(step, selected)
        self.released_supported.update(step, supported)
        for index, value in enumerate(opposite):
            if value and self.opposite_first_steps[index] is None:
                self.opposite_first_steps[index] = step
        if all(selected) and self.both_native_first_step is None:
            self.both_native_first_step = step
        if all(supported) and self.both_supported_first_step is None:
            self.both_supported_first_step = step
        if values != self.last_inputs or supported != self.supported_final:
            self.changes.append(dict(
                control_step=step, selected_native=list(selected), opposite_native=list(opposite),
                exact_base_contact=list(base_contacts), any_gripper_contact=list(gripper_contacts),
                supported_instant=instant, consecutive_supported_steps=list(self.support_runs),
                supported_hold_complete=supported,
            ))
        self.selected_final = list(selected)
        self.opposite_final = list(opposite)
        self.supported_instant_final = instant
        self.supported_final = supported
        self.last_inputs, self.last_step = values, step

    def complete(self, mode: str) -> bool:
        if mode == "native_in":
            return bool(self.native.complete and all(self.selected_final))
        if mode == "released_supported":
            return bool(self.released_supported.complete and all(self.supported_final)
                        and all(self.selected_final))
        raise ValueError(f"An explicit mode is required: {COMPLETION_MODES}")

    def violation(self, mode: str):
        if mode == "native_in":
            return self.native.violation
        if mode == "released_supported":
            return self.released_supported.violation
        raise ValueError(f"An explicit mode is required: {COMPLETION_MODES}")

    def as_dict(self):
        return dict(
            native_order=self.native.as_dict(), released_supported_order=self.released_supported.as_dict(),
            native_ordered_final_success=self.complete("native_in"),
            released_supported_ordered_final_success=self.complete("released_supported"),
            selected_native_final=list(self.selected_final), opposite_native_final=list(self.opposite_final),
            supported_instant_final=list(self.supported_instant_final),
            supported_hold_final=list(self.supported_final),
            support_hold_control_steps=self.support_hold_control_steps,
            final_consecutive_supported_steps=list(self.support_runs),
            maximum_consecutive_supported_steps=list(self.support_run_maximum),
            opposite_native_first_steps=list(self.opposite_first_steps),
            both_native_first_step=self.both_native_first_step,
            both_supported_first_step=self.both_supported_first_step,
            actual_control_steps_observed=self.last_step,
            physical_state_changes=self.changes,
            action_generation_modified=False,
            tc_subtask_switching_modified=False,
            whole_object_inside_native_site_required=False,
        )


class DrainerContactProbe:
    """Read-only contact identities for two target objects and the native base."""

    def __init__(self, env, object_ids):
        import numpy as np

        inner = env
        while not hasattr(inner, "objects_dict") and hasattr(inner, "env"):
            inner = inner.env
        self.inner, self.sim = inner, inner.sim
        self.object_ids = tuple(object_ids)
        if len(self.object_ids) != 2 or len(set(self.object_ids)) != 2:
            raise ValueError("Contact probe needs two distinct target objects")
        parents = np.asarray(self.sim.model.body_parentid, dtype=int)

        def body_geoms(name):
            bodies = {int(self.sim.model.body_name2id(name))}
            while True:
                expanded = bodies | {index for index, parent in enumerate(parents) if int(parent) in bodies}
                if expanded == bodies:
                    return set(int(index) for index in np.flatnonzero(
                        np.isin(self.sim.model.geom_bodyid, list(bodies))))
                bodies = expanded

        self.target_geoms = {name: body_geoms(inner.objects_dict[name].root_body) for name in self.object_ids}
        self.drainer_geoms = body_geoms(inner.fixtures_dict["bowl_drainer_1"].root_body)
        collidable = [index for index in self.drainer_geoms
                      if int(self.sim.model.geom_contype[index]) or int(self.sim.model.geom_conaffinity[index])]
        if not collidable:
            raise RuntimeError("Native drainer has no collision geoms")
        ordered = sorted(collidable, key=lambda index: float(self.sim.data.geom_xpos[index][2]))
        self.base_geom = ordered[0]
        if len(ordered) < 2 or float(self.sim.data.geom_xpos[ordered[1]][2] - self.sim.data.geom_xpos[ordered[0]][2]) < .02:
            raise RuntimeError("Cannot uniquely identify the native lowest drainer support box")
        self.gripper_geoms = set()
        for robot in inner.robots:
            grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else [robot.gripper]
            for gripper in grippers:
                self.gripper_geoms.update(int(self.sim.model.geom_name2id(name)) for name in gripper.contact_geoms)
        if not self.gripper_geoms:
            raise RuntimeError("Robot gripper collision geoms were not resolved")

    def snapshot(self):
        import mujoco
        import numpy as np

        rows = []
        for object_id in self.object_ids:
            own = self.target_geoms[object_id]
            contacts = []
            for index in range(int(self.sim.data.ncon)):
                contact = self.sim.data.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                other = second if first in own else first if second in own else None
                if other is None or other in own:
                    continue
                force = np.zeros(6, dtype=float)
                mujoco.mj_contactForce(getattr(self.sim.model, "_model", self.sim.model),
                                       getattr(self.sim.data, "_data", self.sim.data), index, force)
                contacts.append(dict(
                    other_geom_id=other, other_geom_name=str(self.sim.model.geom_id2name(other)),
                    native_base=other == self.base_geom, drainer=other in self.drainer_geoms,
                    gripper=other in self.gripper_geoms, distance_m=float(contact.dist),
                    normal_force_n=float(force[0]),
                ))
            native = {region: bool(self.inner._eval_predicate(("in", object_id, region)))
                      for region in NATIVE_REGIONS}
            root_id = int(self.sim.model.body_name2id(self.inner.objects_dict[object_id].root_body))
            base_contact, gripper_contact = physical_contact_flags(contacts)
            rows.append(dict(
                object_id=object_id, native_predicates=native,
                native_base_contact=base_contact,
                any_gripper_contact=gripper_contact,
                world_position_m=np.asarray(self.sim.data.body_xpos[root_id], dtype=float).tolist(),
                contacts=contacts,
            ))
        return dict(
            native_base_geom_id=self.base_geom,
            native_base_geom_name=str(self.sim.model.geom_id2name(self.base_geom)),
            minimum_base_support_normal_force_n=MIN_SUPPORT_NORMAL_FORCE_N,
            gripper_contact_distance_tolerance_m=GRIPPER_CONTACT_DISTANCE_TOLERANCE_M,
            support_definition="positive_normal_force_on_exact_native_bottom_box",
            support_is_not_a_velocity_based_stability_certificate=True,
            gripper_geom_names=sorted(str(self.sim.model.geom_id2name(index)) for index in self.gripper_geoms),
            objects=rows,
        )


def self_test():
    atoms = [f"in(a, {NATIVE_REGIONS[0]})", f"in(b, {NATIVE_REGIONS[1]})"]
    assert physical_contact_flags([dict(native_base=True, gripper=False, normal_force_n=0., distance_m=.0001)]) == (False, False)
    assert physical_contact_flags([dict(native_base=True, gripper=False, normal_force_n=.2, distance_m=-.0001)]) == (True, False)
    assert physical_contact_flags([dict(native_base=False, gripper=True, normal_force_n=0., distance_m=0.)]) == (False, True)

    def fresh():
        observer = OrderedPlacementObserver(atoms, support_hold_control_steps=2)
        observer.update(0, [False, False], [False, False], [False, False], [False, False])
        return observer

    observer = fresh()
    # A enters the correct site while held: native partial, never strict completion.
    observer.update(1, [True, False], [False, False], [True, False], [True, False])
    assert observer.native.stage_index == 1 and observer.released_supported.stage_index == 0
    observer.update(2, [True, False], [False, False], [True, False], [False, False])
    observer.update(2, [True, False], [False, False], [True, False], [False, False])
    assert observer.support_runs == [1, 0], "Duplicate audits must not double-count hold duration"
    observer.update(3, [True, False], [False, False], [True, False], [False, False])
    assert observer.released_supported.event_steps == [3, None]
    observer.update(4, [True, True], [False, False], [True, True], [False, True])
    assert observer.complete("native_in") and not observer.complete("released_supported")
    observer.update(5, [True, True], [False, False], [True, True], [False, False])
    observer.update(6, [True, True], [False, False], [True, True], [False, False])
    assert observer.complete("released_supported") and observer.released_supported.event_steps == [3, 6]
    # Previously completed A falling out invalidates the final conjunction.
    observer.update(7, [False, True], [True, False], [True, True], [False, False])
    assert not observer.complete("released_supported") and not observer.complete("native_in")
    wrong = fresh()
    wrong.update(1, [False, False], [True, False], [True, False], [False, False])
    wrong.update(2, [False, False], [True, False], [True, False], [False, False])
    assert wrong.released_supported.stage_index == 0 and not wrong.complete("native_in")
    wall_only = fresh()
    wall_only.update(1, [True, False], [False, False], [False, False], [False, False])
    wall_only.update(2, [True, False], [False, False], [False, False], [False, False])
    assert wall_only.released_supported.stage_index == 0
    out_of_order = fresh()
    out_of_order.update(1, [False, True], [False, False], [False, True], [False, False])
    out_of_order.update(2, [False, True], [False, False], [False, True], [False, False])
    assert out_of_order.violation("native_in") and out_of_order.violation("released_supported")
    together = fresh()
    together.update(1, [True, True], [False, False], [True, True], [False, False])
    together.update(2, [True, True], [False, False], [True, True], [False, False])
    assert together.violation("native_in") and together.violation("released_supported")
    broken_hold = fresh()
    broken_hold.update(1, [True, False], [False, False], [True, False], [False, False])
    broken_hold.update(2, [True, False], [False, False], [False, False], [False, False])
    broken_hold.update(3, [True, False], [False, False], [True, False], [False, False])
    assert broken_hold.support_runs == [1, 0] and broken_hold.released_supported.stage_index == 0
    try:
        broken_hold.complete("")
    except ValueError:
        pass
    else:
        raise AssertionError("Scoring mode must be chosen explicitly")
    reverse_atoms = [f"in(b, {NATIVE_REGIONS[1]})", f"in(a, {NATIVE_REGIONS[0]})"]
    reverse = OrderedPlacementObserver(reverse_atoms, support_hold_control_steps=2)
    reverse.update(0, [False, False], [False, False], [False, False], [False, False])
    reverse.update(1, [True, False], [False, False], [True, False], [False, False])
    reverse.update(2, [True, False], [False, False], [True, False], [False, False])
    reverse.update(3, [True, True], [False, False], [True, True], [False, False])
    reverse.update(4, [True, True], [False, False], [True, True], [False, False])
    assert reverse.complete("released_supported")
    print("Ordered drainer observer tests passed: held, wrong-side, either order, wall-only, hold, final conjunction")


if __name__ == "__main__":
    self_test()
