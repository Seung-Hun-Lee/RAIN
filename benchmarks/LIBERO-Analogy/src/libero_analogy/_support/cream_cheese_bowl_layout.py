"""Exact fixed-fixture replay for the Goal cream-cheese scene."""
import hashlib
import numpy as np


def state_hash(state):
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def fixture_snapshot(env):
    result = {}
    for obj in env.env.fixtures_dict.values():
        body = obj.root_body
        bid = env.sim.model.body_name2id(body)
        result[body] = {
            "model_body_pos": env.sim.model.body_pos[bid].tolist(),
            "model_body_quat": env.sim.model.body_quat[bid].tolist(),
        }
    return result


def apply_fixtures(env, snapshot):
    for name, pose in snapshot.items():
        bid = env.sim.model.body_name2id(name)
        env.sim.model.body_pos[bid] = pose["model_body_pos"]
        env.sim.model.body_quat[bid] = pose["model_body_quat"]
    env.sim.forward()


def body_positions(env):
    return {str(env.sim.model.body_id2name(i)): env.sim.data.body_xpos[i].tolist()
            for i in range(env.sim.model.nbody)}


def position_error(env, reference):
    return max(float(np.max(np.abs(env.sim.data.body_xpos[env.sim.model.body_name2id(name)] - value)))
               for name, value in reference.items())
