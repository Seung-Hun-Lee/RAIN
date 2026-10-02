import numpy as np
MIN_DIRECT_CONTACT_NORMAL_FORCE_N = 1e-6
def contact_certifies(efc_address: int, normal_force_n: float) -> bool:
    """True only for an active exact-pair constraint carrying normal force."""
    return bool(
        int(efc_address) >= 0
        and float(normal_force_n) > MIN_DIRECT_CONTACT_NORMAL_FORCE_N
    )

def descendants(model, root: int) -> set[int]:
    output = {int(root)}
    changed = True
    while changed:
        changed = False
        for body_id, parent in enumerate(model.body_parentid):
            if int(parent) in output and body_id not in output:
                output.add(body_id)
                changed = True
    return output

def contact_sets(env) -> tuple[set[int], set[int]]:
    inner = env.env
    model = env.sim.model
    door_root = int(model.body_name2id("microwave_1_microdoorroot"))
    door_bodies = descendants(model, door_root)
    door_geoms = {int(index) for index, body_id in enumerate(model.geom_bodyid) if int(body_id) in door_bodies}
    gripper_geoms = set()
    for robot in inner.robots:
        grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else (robot.gripper,)
        for gripper in grippers:
            for name in gripper.contact_geoms:
                gripper_geoms.add(int(model.geom_name2id(name)))
    if not door_geoms or not gripper_geoms:
        raise RuntimeError("failed to resolve microwave-door/gripper contact geoms")
    return door_geoms, gripper_geoms

def direct_contacts(env, door_geoms: set[int], gripper_geoms: set[int]) -> list[dict]:
    import mujoco

    model, data = env.sim.model, env.sim.data
    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in gripper_geoms and second in door_geoms:
            gripper, door = first, second
        elif second in gripper_geoms and first in door_geoms:
            gripper, door = second, first
        else:
            continue
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(
            getattr(model, "_model", model),
            getattr(data, "_data", data),
            index,
            force,
        )
        efc_address = int(contact.efc_address)
        normal_force = float(force[0])
        rows.append(
            {
                "contact_index": index,
                "gripper_geom": str(model.geom_id2name(gripper)),
                "door_geom": str(model.geom_id2name(door)),
                "distance_m": float(contact.dist),
                "efc_address": efc_address,
                "solver_constraint_active": efc_address >= 0,
                "normal_force_n": normal_force,
                "minimum_certifying_normal_force_n": MIN_DIRECT_CONTACT_NORMAL_FORCE_N,
                "certifying_physical_contact": contact_certifies(efc_address, normal_force),
            }
        )
    return rows

def overlap(first, second) -> bool:
    low_a, high_a = map(np.asarray, first)
    low_b, high_b = map(np.asarray, second)
    return bool(np.all(np.minimum(high_a, high_b) - np.maximum(low_a, low_b) > 0.0))
