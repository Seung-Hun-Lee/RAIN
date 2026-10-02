"""pi0.5 input preprocessing for two physical cameras."""
import math
from typing import Any
RESIZE_SIZE = 224
def quaternion_to_axis_angle(quaternion):
    import numpy as np

    q = np.asarray(quaternion, dtype=np.float64).copy()
    q /= np.linalg.norm(q) + 1e-12
    if q[3] < 0:
        q = -q
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(q[3]) ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(float(q[3])) / denominator).astype(
        np.float32
    )

def policy_observation(obs: dict[str, Any], prompt: str) -> tuple[dict[str, Any], Any]:
    import numpy as np
    from ._vendor.openpi import image_tools

    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    agent = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent, RESIZE_SIZE, RESIZE_SIZE)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE)
    )
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quaternion_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid LIBERO state: shape={state.shape}")
    return {
        "observation/image": agent,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }, agent
