"""Read-only physical scoring for one placement into the native wooden tray.

LIBERO's ordinary site ``In`` test checks only the object's root position.
This observer strengthens *evaluation only*: the complete collision envelope
must be in the native site, the object must have positive-force contact with
the actual tray, and no gripper collision may remain for five consecutive
control steps.  It never moves an object or changes policy actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .novel_feedback_fixture_geometry import full_containment_report
from .wooden_tray_object_choices_support import TARGET_REGION


MIN_TRAY_CONTACT_NORMAL_FORCE_N = 1e-6
GRIPPER_CONTACT_DISTANCE_TOLERANCE_M = 1e-7
FULL_CONTAINMENT_TOLERANCE_M = 1e-3


def _inner_env(env):
    inner = env
    while not hasattr(inner, "objects_dict") and hasattr(inner, "env"):
        inner = inner.env
    if not hasattr(inner, "objects_dict"):
        raise TypeError("Could not resolve the underlying LIBERO environment")
    return inner


def _descendant_geom_ids(sim, root_body_name: str) -> set[int]:
    root_id = int(sim.model.body_name2id(root_body_name))
    if root_id < 0:
        raise ValueError(f"Unknown body: {root_body_name}")
    parents = np.asarray(sim.model.body_parentid, dtype=int)
    bodies = {root_id}
    while True:
        expanded = bodies | {
            index for index, parent in enumerate(parents) if int(parent) in bodies
        }
        if expanded == bodies:
            break
        bodies = expanded
    return set(
        int(index)
        for index in np.flatnonzero(
            np.isin(np.asarray(sim.model.geom_bodyid, dtype=int), list(bodies))
        )
    )


class WoodenTrayContactProbe:
    """Read current target/tray/gripper contacts without changing simulation."""

    def __init__(self, env, object_id: str):
        self.inner = _inner_env(env)
        self.sim = self.inner.sim
        self.object_id = str(object_id)
        target = self.inner.get_object(self.object_id)
        tray = self.inner.get_object("wooden_tray_1")
        if target is None or tray is None:
            raise ValueError("Target object or wooden_tray_1 is missing")
        self.target_geoms = _descendant_geom_ids(self.sim, target.root_body)
        self.tray_geoms = _descendant_geom_ids(self.sim, tray.root_body)
        collision = (
            (np.asarray(self.sim.model.geom_contype, dtype=int) != 0)
            | (np.asarray(self.sim.model.geom_conaffinity, dtype=int) != 0)
        )
        self.target_geoms = {
            geom for geom in self.target_geoms if bool(collision[geom])
        }
        self.tray_geoms = {
            geom for geom in self.tray_geoms if bool(collision[geom])
        }
        if not self.target_geoms or not self.tray_geoms:
            raise RuntimeError("Target or tray has no physical collision geometry")

        self.gripper_geoms: set[int] = set()
        for robot in self.inner.robots:
            grippers = (
                robot.gripper.values()
                if isinstance(robot.gripper, dict)
                else [robot.gripper]
            )
            for gripper in grippers:
                # Include the whole gripper subtree, not only robosuite's
                # convenience contact-name list, so a palm / finger-housing
                # collision cannot be mistaken for a released object.
                self.gripper_geoms.update(
                    _descendant_geom_ids(self.sim, gripper.root_body)
                )
                for name in gripper.contact_geoms:
                    geom_id = int(self.sim.model.geom_name2id(name))
                    if geom_id >= 0:
                        self.gripper_geoms.add(geom_id)
        self.gripper_geoms = {
            geom for geom in self.gripper_geoms if bool(collision[geom])
        }
        if not self.gripper_geoms:
            raise RuntimeError("Robot gripper collision geoms were not resolved")

    def snapshot(self) -> dict:
        import mujoco

        contacts = []
        positive_tray_contact = False
        any_gripper_contact = False
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.target_geoms and second not in self.target_geoms:
                other = second
            elif second in self.target_geoms and first not in self.target_geoms:
                other = first
            else:
                continue
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(
                getattr(self.sim.model, "_model", self.sim.model),
                getattr(self.sim.data, "_data", self.sim.data),
                index,
                force,
            )
            normal_force = float(force[0])
            distance = float(contact.dist)
            tray_contact = other in self.tray_geoms
            gripper_contact = other in self.gripper_geoms
            if tray_contact and normal_force > MIN_TRAY_CONTACT_NORMAL_FORCE_N:
                positive_tray_contact = True
            if gripper_contact and (
                normal_force > MIN_TRAY_CONTACT_NORMAL_FORCE_N
                or distance <= GRIPPER_CONTACT_DISTANCE_TOLERANCE_M
            ):
                any_gripper_contact = True
            if tray_contact or gripper_contact:
                contacts.append(
                    {
                        "other_geom_id": other,
                        "other_geom_name": str(self.sim.model.geom_id2name(other)),
                        "tray": tray_contact,
                        "gripper": gripper_contact,
                        "distance_m": distance,
                        "normal_force_n": normal_force,
                    }
                )

        native_in = bool(
            self.inner._eval_predicate(("in", self.object_id, TARGET_REGION))
        )
        containment = full_containment_report(
            self.inner,
            self.object_id,
            TARGET_REGION,
            tolerance=FULL_CONTAINMENT_TOLERANCE_M,
        )
        strict_instant = bool(
            native_in
            and containment["contained"]
            and containment["parent_contact"]
            and positive_tray_contact
            and not any_gripper_contact
        )
        root_id = int(
            self.sim.model.body_name2id(
                self.inner.get_object(self.object_id).root_body
            )
        )
        return {
            "object_id": self.object_id,
            "target_region": TARGET_REGION,
            "native_in": native_in,
            "full_containment": containment,
            "positive_force_tray_contact": positive_tray_contact,
            "any_gripper_contact": any_gripper_contact,
            "strict_instant": strict_instant,
            "world_position_m": np.asarray(
                self.sim.data.body_xpos[root_id], dtype=float
            ).tolist(),
            "contacts": contacts,
        }


@dataclass
class StrictTrayPlacementObserver:
    """Require a consecutive released, contained, supported hold."""

    support_hold_control_steps: int = 5
    last_step: int | None = field(default=None, init=False)
    last_snapshot: dict | None = field(default=None, init=False)
    support_run: int = field(default=0, init=False)
    maximum_support_run: int = field(default=0, init=False)
    first_complete_step: int | None = field(default=None, init=False)
    changes: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.support_hold_control_steps, int)
            or self.support_hold_control_steps < 1
        ):
            raise ValueError("support_hold_control_steps must be positive")

    def update(self, step: int, snapshot: dict) -> None:
        step = int(step)
        previous_complete = self.complete
        expected = 0 if self.last_step is None else self.last_step + 1
        if self.last_step is not None and step == self.last_step:
            if snapshot != self.last_snapshot:
                raise RuntimeError(
                    "The same control step has inconsistent physical observations"
                )
            return
        if step != expected:
            raise RuntimeError(
                f"Missing control-step observation: expected {expected}, got {step}"
            )
        if step == 0 and bool(snapshot.get("native_in")):
            raise RuntimeError("The selected object starts inside the wooden tray")

        strict_instant = bool(snapshot.get("strict_instant"))
        self.support_run = self.support_run + 1 if strict_instant else 0
        self.maximum_support_run = max(self.maximum_support_run, self.support_run)
        complete_now = self.support_run >= self.support_hold_control_steps
        if complete_now and self.first_complete_step is None:
            self.first_complete_step = step
        if (
            self.last_snapshot is None
            or strict_instant != bool(self.last_snapshot.get("strict_instant"))
            or complete_now != previous_complete
        ):
            self.changes.append(
                {
                    "control_step": step,
                    "native_in": bool(snapshot.get("native_in")),
                    "contained": bool(
                        snapshot.get("full_containment", {}).get("contained")
                    ),
                    "parent_contact": bool(
                        snapshot.get("full_containment", {}).get("parent_contact")
                    ),
                    "positive_force_tray_contact": bool(
                        snapshot.get("positive_force_tray_contact")
                    ),
                    "any_gripper_contact": bool(
                        snapshot.get("any_gripper_contact")
                    ),
                    "strict_instant": strict_instant,
                    "consecutive_strict_steps": self.support_run,
                    "hold_complete": complete_now,
                }
            )
        self.last_step = step
        self.last_snapshot = snapshot

    @property
    def complete(self) -> bool:
        return bool(
            self.last_snapshot
            and self.last_snapshot.get("strict_instant")
            and self.support_run >= self.support_hold_control_steps
        )

    def as_dict(self) -> dict:
        return {
            "strict_final_success": self.complete,
            "support_hold_control_steps": self.support_hold_control_steps,
            "final_consecutive_strict_steps": self.support_run,
            "maximum_consecutive_strict_steps": self.maximum_support_run,
            "first_complete_step": self.first_complete_step,
            "actual_control_steps_observed": self.last_step,
            "final_snapshot": self.last_snapshot,
            "physical_state_changes": self.changes,
            "full_collision_containment_required": True,
            "full_containment_tolerance_m": FULL_CONTAINMENT_TOLERANCE_M,
            "positive_force_tray_contact_required": True,
            "minimum_tray_contact_normal_force_n": (
                MIN_TRAY_CONTACT_NORMAL_FORCE_N
            ),
            "no_gripper_contact_required": True,
            "gripper_contact_distance_tolerance_m": (
                GRIPPER_CONTACT_DISTANCE_TOLERANCE_M
            ),
            "action_generation_modified": False,
            "tc_subtask_switching_modified": False,
        }


def self_test() -> None:
    def snapshot(strict: bool) -> dict:
        return {
            "native_in": strict,
            "strict_instant": strict,
            "full_containment": {
                "contained": strict,
                "parent_contact": strict,
            },
            "positive_force_tray_contact": strict,
            "any_gripper_contact": False,
        }

    observer = StrictTrayPlacementObserver(2)
    observer.update(0, snapshot(False))
    observer.update(1, snapshot(True))
    assert not observer.complete
    observer.update(1, snapshot(True))
    observer.update(2, snapshot(True))
    assert observer.complete and observer.first_complete_step == 2
    observer.update(3, snapshot(False))
    assert not observer.complete and observer.maximum_support_run == 2
    print("Strict wooden-tray placement observer self-test passed")


if __name__ == "__main__":
    self_test()
